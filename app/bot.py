"""Телеграм-бот: вопрос из Телеграма — локальной модели, ответ — обратно, на глазах.

Телеграм здесь только почта. Бот забирает сообщения запросом getUpdates и передаёт их
модели через Ollama на этом компьютере (app/llm.py). Пока модель пишет, ответ виден
черновиком (sendRichMessageDraft), готовый — сохраняется «богатым» сообщением
(sendRichMessage): Markdown, таблицы и формулы модели Телеграм рисует сам.
"""
import asyncio
import base64
import html
import itertools
import re
import sys
import time
from collections import defaultdict
from contextlib import aclosing, suppress

import httpx

from . import config, llm, machine

API = "https://api.telegram.org"
MEMORY = 10          # сколько последних сообщений переписки видит модель: 5 вопросов с ответами
PHOTO_SIDE = 1000    # фото — самый крупный вариант не больше 1000 px: 800 × 600 ≈ 500 токенов, ~6 с
TICK = 0.6           # как часто обновлять черновик, с
TAIL = 600           # сколько последних знаков хода мысли видно, пока модель думает

GREETING = f"""**Локальная LLM**

Отвечает {config.MODEL} — модель работает на компьютере владельца бота, без облака.

- ✍️ Вопрос — ответ печатается на глазах, ⏹ останавливает
- 🖼 Фото — модель расскажет, что на нём
- 💭 Думать — сначала рассуждает: точнее, но дольше
- 🆕 Новый диалог — модель забудет переписку"""


def log(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def num(x: float) -> str:
    return f"{x:.1f}".removesuffix(".0").replace(".", ",")


def secs(s: float) -> str:
    return f"{num(s)} с" if s < 10 else f"{round(s)} с" if s < 60 else f"{round(s) // 60} мин {round(s) % 60} с"


def plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def tex(text: str) -> str:
    """Формулы модели \\[…\\] и \\(…\\) → $$…$$ и $…$: в таком виде их рисует Телеграм."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m[1].strip()}$$", text, flags=re.S)
    return re.sub(r"\\\((.+?)\\\)", lambda m: f"${m[1].strip()}$", text, flags=re.S)


class TelegramError(RuntimeError):
    """Телеграм не выполнил запрос — описание из его ответа."""


class Telegram:
    """Bot API: POST https://api.telegram.org/bot<токен>/<метод>, параметры — JSON."""

    def __init__(self, token: str):
        self.token = token
        # Телеграм — в интернете: прокси из системы (HTTPS_PROXY), если он задан, используется.
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=70))

    async def __call__(self, method: str, **params):
        response = await self.http.post(f"{API}/bot{self.token}/{method}", json=params)
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(f"{method}: HTTP {response.status_code}") from None
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description')}")
        return data["result"]

    async def download(self, file_id: str) -> bytes:
        path = (await self("getFile", file_id=file_id))["file_path"]
        response = await self.http.get(f"{API}/file/bot{self.token}/{path}")
        if response.status_code != 200:     # без raise_for_status: его текст содержит адрес с токеном
            raise TelegramError(f"файл не скачан: HTTP {response.status_code}")
        return response.content


class Live:
    """Ответ, пока модель его пишет: ход мысли, текст, время."""

    def __init__(self):
        self.thought = self.text = ""
        self.started = time.monotonic()
        self.think_started = self.think_ended = 0.0
        self.thought_sent = self.over = False

    def thinking(self) -> float:
        """Сколько секунд модель думала — или думает сейчас."""
        return (self.think_ended or time.monotonic()) - self.think_started

    async def listen(self, messages: list[dict], think: bool) -> dict:
        """Читает ответ модели по кусочкам; возвращает счётчики из последнего события."""
        async with aclosing(llm.ask(config.MODEL, messages, think)) as events:
            async for event in events:
                if event["type"] == "request":
                    n = len(messages)
                    extra = (" · фото" if messages[-1].get("images") else "") + (" · думать" if think else "")
                    log(f"→ POST {event['url']} · {config.MODEL} · {n} {plural(n, 'сообщение', 'сообщения', 'сообщений')}{extra}")
                elif event["type"] == "think":
                    self.think_started = self.think_started or time.monotonic()
                    self.thought += event["text"]
                elif event["type"] == "text":
                    if self.think_started and not self.think_ended:
                        self.think_ended = time.monotonic()
                    self.text += event["text"]
                elif event["type"] == "done":
                    return event
        raise llm.LLMError("Ответ оборвался")

    def draft(self) -> str:
        """Что видно в черновике (Markdown). Пусто — Телеграм покажет «Думает…»."""
        if self.text:
            return tex(self.text)
        if not self.thought:
            return ""
        tail = self.thought if len(self.thought) <= TAIL else "…" + self.thought[-TAIL:].split(" ", 1)[-1]
        quote = "\n".join(">" + line for line in tex(tail).strip().splitlines())
        return f"💭 **Думает · {secs(self.thinking())}**\n\n{quote}"


class Bot:
    def __init__(self, telegram: Telegram):
        self.tg = telegram
        self.dialogs: dict[int, list[dict]] = defaultdict(list)   # чат → переписка, которую видит модель
        self.thinking: set[int] = set()                           # чаты, где включено «думать»
        self.buttons: dict[int, int] = {}                         # чат → сообщение, под которым кнопки
        self.running: dict[int, asyncio.Task] = {}                # чат → модель пишет ответ; ⏹ её отменяет
        self.locks = defaultdict(asyncio.Lock)                    # в одном чате — один ответ за раз
        self.drafts = itertools.count(1)                          # номера черновиков

    async def handle(self, update: dict) -> None:
        try:
            if stop := update.get("stopped_message_generation"):
                if task := self.running.get(stop["chat"]["id"]):
                    task.cancel()
            elif query := update.get("callback_query"):
                await self.on_button(query)
            elif (message := update.get("message")) and message["chat"]["type"] == "private":
                await self.on_message(message)
        except Exception as e:      # сбой с одним сообщением не останавливает бота
            log(f"Ошибка: {e!r}")

    async def on_message(self, message: dict) -> None:
        chat = message["chat"]["id"]
        text = (message.get("text") or message.get("caption") or "").strip()
        photos = message.get("photo")
        if text.startswith("/start"):
            self.dialogs.pop(chat, None)
            return await self.send(chat, GREETING)
        if not text and not photos:
            return await self.send(chat, "Понимаю текст и фото.")
        async with self.locks[chat]:
            dialog = self.dialogs[chat]
            question, note = {"role": "user", "content": text or "Что на фото?"}, ""
            if photos:              # Телеграм хранит фото в нескольких размерах, от маленького к большому
                photo = ([p for p in photos if max(p["width"], p["height"]) <= PHOTO_SIDE] or photos[:1])[-1]
                question["images"] = [base64.b64encode(await self.tg.download(photo["file_id"])).decode()]
                note = f" [фото {photo['width']}×{photo['height']}]"
                for old in dialog:  # модель видит только последнее фото: каждое — сотни токенов
                    old.pop("images", None)
            log(f"{message['from'].get('first_name', '')}: {question['content']}{note}")
            reply = await self.answer(chat, dialog + [question])
            if reply:
                dialog += [question, {"role": "assistant", "content": reply}]
                del dialog[:-MEMORY]

    async def on_button(self, query: dict) -> None:
        chat, message_id = query["message"]["chat"]["id"], query["message"]["message_id"]
        if query.get("data") == "think":
            self.thinking ^= {chat}
            on = chat in self.thinking
            log("Думать: " + ("включено" if on else "выключено"))
            await self.tg("answerCallbackQuery", callback_query_id=query["id"],
                          text="Думать включено: точнее, но дольше" if on else "Думать выключено")
            await self.tg("editMessageReplyMarkup", chat_id=chat, message_id=message_id,
                          reply_markup=self.keyboard(chat))
        elif query.get("data") == "new":
            self.dialogs.pop(chat, None)
            log("Новый диалог")
            await self.tg("answerCallbackQuery", callback_query_id=query["id"])
            await self.send(chat, "🆕 **Новый диалог** — прошлую переписку модель не помнит")

    async def answer(self, chat: int, messages: list[dict]) -> str:
        """Модель отвечает, Телеграм показывает ответ по мере печати. Вернёт ответ, если он дописан."""
        live = Live()
        task = asyncio.create_task(live.listen(messages, chat in self.thinking))
        self.running[chat] = task
        shown = asyncio.create_task(self.show(chat, live))
        try:
            await asyncio.wait({task})
        finally:
            task.cancel()
            live.over = True
            self.running.pop(chat, None)
            await shown                         # черновик больше не меняется — можно отправлять итог
        if live.thought and not live.thought_sent:
            await self.send_thought(chat, live)
        if task.cancelled():
            log("← остановлено")
            status = f"⏹ остановлено · {secs(time.monotonic() - live.started)}"
        elif error := task.exception():
            log(f"← {error}")
            status = f"⚠️ {error}"
        else:
            done = task.result()
            speed = f"{num(done['speed'])} токенов/сек · {secs(done['seconds'])}"
            log(f"← {done['tokens']} {plural(done['tokens'], 'токен', 'токена', 'токенов')} · {speed}")
            status = f"💻 {config.MODEL} · {machine.CPU_NAME} · {speed}"
            if done["load"] > 1:
                status += f" · загрузка {secs(done['load'])}"
        text = tex(live.text).strip()
        await self.send(chat, f"{text}\n\n---\n\n*{status}*" if text else f"*{status}*")
        return "" if task.cancelled() or task.exception() else live.text

    async def show(self, chat: int, live: Live) -> None:
        """Черновик ответа: обновляется, пока модель пишет; ⏹ на нём останавливает модель."""
        draft, shown, refresh, hold, failed = next(self.drafts), None, 0.0, 0.0, False
        while not live.over:
            if live.text and live.thought and not live.thought_sent:
                await self.send_thought(chat, live)     # ход мысли — отдельным сообщением над ответом
                draft = next(self.drafts)               # отправленное сообщение убирает черновик
            markdown, now = live.draft(), time.monotonic()
            if now >= hold and (markdown != shown or now >= refresh):   # черновик живёт 30 с — освежаем
                try:
                    if markdown:
                        await self.tg("sendRichMessageDraft", chat_id=chat, draft_id=draft,
                                      rich_message={"markdown": markdown}, can_stop=True, keep_on_stop=True)
                    else:                               # пустой текст — Телеграм покажет «Думает…»
                        await self.tg("sendMessageDraft", chat_id=chat, draft_id=draft, text="",
                                      can_stop=True, keep_on_stop=True)
                    shown, refresh = markdown, now + 10
                except (TelegramError, httpx.HTTPError) as e:
                    if not failed:
                        log(f"Черновик не показан: {e!r}")
                    failed, hold = True, now + 3        # пауза в показе, а не в ответе
            await asyncio.sleep(TICK)

    async def send_thought(self, chat: int, live: Live) -> None:
        """Ход мысли — свёрнутым блоком: виден заголовок с временем, текст — по нажатию."""
        live.thought_sent = True
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(live.thought.strip(), quote=False))
        paragraphs = "".join("<p>" + p.strip().replace("\n", "<br>") + "</p>" for p in text.split("\n\n") if p.strip())
        try:
            await self.tg("sendRichMessage", chat_id=chat, rich_message={
                "html": f"<details><summary>💭 Ход мысли · {secs(live.thinking())}</summary>{paragraphs}</details>"})
        except (TelegramError, httpx.HTTPError) as e:
            log(f"Ход мысли не отправлен: {e!r}")

    async def send(self, chat: int, markdown: str) -> None:
        """Сообщение с кнопками. У прошлого сообщения кнопки убираются: они всегда под последним."""
        markup = self.keyboard(chat)
        try:
            sent = await self.tg("sendRichMessage", chat_id=chat, rich_message={"markdown": markdown},
                                 reply_markup=markup)
        except TelegramError as e:      # разметку не приняли — тот же текст без оформления
            log(f"Без оформления: {e}")
            sent = await self.tg("sendMessage", chat_id=chat, text=markdown[:4096], reply_markup=markup)
        if (old := self.buttons.get(chat)) is not None:
            with suppress(TelegramError, httpx.HTTPError):
                await self.tg("editMessageReplyMarkup", chat_id=chat, message_id=old)
        self.buttons[chat] = sent["message_id"]

    def keyboard(self, chat: int) -> dict:
        think = {"text": "💭 Думать", "callback_data": "think"}
        if chat in self.thinking:
            think.update(text="💭 Думать ✓", style="primary")    # включено — кнопка синяя
        return {"inline_keyboard": [[think, {"text": "🆕 Новый диалог", "callback_data": "new"}]]}


async def main() -> None:
    if not config.TELEGRAM_TOKEN:
        sys.exit("Нет токена: создайте бота у @BotFather и впишите в файл .env строку TELEGRAM_TOKEN=…")
    telegram = Telegram(config.TELEGRAM_TOKEN)
    try:
        me = await telegram("getMe")
    except TelegramError as e:
        sys.exit(f"Телеграм не принял токен из .env — {e}")
    except httpx.HTTPError as e:
        sys.exit(f"Нет связи с Телеграмом — {e!r}")
    status = await llm.status()
    ollama = f"Ollama {status['version']} · {config.OLLAMA_URL}" if status["running"] \
        else "⚠️ Ollama не запущен — откройте его из меню «Пуск»"
    log(f"Бот @{me['username']} запущен · модель {config.MODEL} · {ollama} · Ctrl+C — остановить")
    bot, tasks, offset = Bot(telegram), set(), 0
    while True:
        try:
            updates = await telegram("getUpdates", offset=offset, timeout=50,
                                     allowed_updates=["message", "callback_query", "stopped_message_generation"])
        except (TelegramError, httpx.HTTPError) as e:
            log(f"Нет связи с Телеграмом — {e!r}, повтор через 5 с")
            await asyncio.sleep(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            task = asyncio.create_task(bot.handle(update))
            tasks.add(task)                 # держим ссылку, пока задача не закончится
            task.add_done_callback(tasks.discard)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")
