"""Приватный сервис (день 30): локальная модель по сети — HTTP API и чат.

Наружу смотрит только этот сервис, и то через HTTPS-прокси Caddy; Ollama слушает лишь 127.0.0.1.
У каждого запроса к модели четыре проверки:
  ключ     — заголовок Authorization: Bearer <ключ>, иначе 401;
  частота  — не больше RATE запросов в минуту на ключ, иначе 429 и сколько подождать;
  очередь  — отвечают PARALLEL сразу, ждут не больше QUEUE, иначе 503;
  контекст — переписка не длиннее CONTEXT токенов, иначе 413 (токены считает сам Ollama).
Ответ — не длиннее ANSWER токенов.
"""
import asyncio
import json
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator
from math import ceil
from pathlib import Path
from typing import Literal

import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import config, llm, machine

STATIC = Path(__file__).resolve().parent.parent / "static"
# Рекомендация Qwen для ответа без размышления; контекст и длина ответа — лимиты сервиса.
OPTIONS = {"temperature": 0.7, "top_p": 0.8, "num_ctx": config.CONTEXT, "num_predict": config.ANSWER}

app = FastAPI(title="Приватный сервис локальной LLM")
_bearer = HTTPBearer(auto_error=False)
_calls: dict[str, deque[float]] = {}            # ключ → когда были его запросы за последнюю минуту
_turn = asyncio.Semaphore(config.PARALLEL)      # места у модели
_now = {"writing": 0, "waiting": 0}             # сколько запросов модель пишет и сколько ждут очереди


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Chat(BaseModel):
    messages: list[Message] = Field(min_length=1)
    stream: bool = True


def _key(given: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """Ключ из заголовка Authorization: Bearer <ключ>. Нет ключа или чужой — 401."""
    secret = given.credentials.encode() if given else b""
    for key in config.API_KEYS:
        if secrets.compare_digest(secret, key.encode()):
            return key
    raise HTTPException(401, "Нужен ключ: заголовок Authorization: Bearer <ключ>",
                        headers={"WWW-Authenticate": "Bearer"})


def _minute(key: str) -> deque[float]:
    """Запросы ключа за последние 60 секунд."""
    calls = _calls.setdefault(key, deque())
    while calls and time.monotonic() - calls[0] >= 60:
        calls.popleft()
    return calls


def _admit(key: str) -> None:
    """Пустить запрос к модели или сразу отказать: сначала частота на ключ, потом место в очереди."""
    calls = _minute(key)
    if len(calls) >= config.RATE:
        wait = ceil(60 - (time.monotonic() - calls[0]))
        raise HTTPException(429, f"Не больше {config.RATE} запросов в минуту — подождите {wait} с",
                            headers={"Retry-After": str(wait)})
    if _now["writing"] + _now["waiting"] >= config.PARALLEL + config.QUEUE:
        raise HTTPException(503, f"Сервер занят: модель пишет {_now['writing']}, ждут {_now['waiting']} — "
                                 "повторите позже", headers={"Retry-After": "10"})
    calls.append(time.monotonic())


async def _answer(messages: list[dict]) -> AsyncIterator[dict]:
    """Ответ модели, когда подойдёт очередь: text… → done; в done ещё и сколько ждал очереди."""
    arrived = time.monotonic()
    _now["waiting"] += 1
    try:
        await _turn.acquire()
    finally:
        _now["waiting"] -= 1
    _now["writing"] += 1
    try:
        queue = time.monotonic() - arrived
        async for event in llm.ask(config.MODEL, messages, think=False, options=OPTIONS, truncate=False):
            if event["type"] == "text":
                yield event
            elif event["type"] == "done":
                yield {**event, "queue": queue}
    finally:
        _now["writing"] -= 1
        _turn.release()


@app.get("/")
def page():
    return FileResponse(STATIC / "chat.html")


@app.get("/api/status")
async def status(key: str = Depends(_key)):
    """Модель и сервер, лимиты и что сейчас: сколько пишет модель, сколько ждут, сколько запросов осталось ключу."""
    ollama = await llm.status()
    calls = _minute(key)
    return {
        "model": config.MODEL, "ollama": ollama.get("version"),
        "loaded": any(m["name"] == config.MODEL for m in ollama.get("loaded", [])),
        "machine": {"cpu": machine.CPU_NAME, "threads": machine.THREADS, "ram": psutil.virtual_memory().total,
                    "load": psutil.cpu_percent()},
        "limits": {"rate": config.RATE, "parallel": config.PARALLEL, "queue": config.QUEUE,
                   "context": config.CONTEXT, "answer": config.ANSWER},
        "now": {**_now, "left": config.RATE - len(calls),
                "reset": ceil(60 - (time.monotonic() - calls[0])) if calls else 0},
    }


@app.post("/api/chat")
async def chat(body: Chat, key: str = Depends(_key)):
    """Ответ на переписку: stream — строками JSON по мере генерации (text… → done), иначе одним JSON."""
    _admit(key)
    events = _answer([m.model_dump() for m in body.messages])
    try:
        first = await anext(events)     # очередь и первый кусочек: длинный запрос — 413 до начала ответа
    except llm.TooLong as e:
        raise HTTPException(413, str(e)) from e
    except llm.LLMError as e:
        raise HTTPException(503, str(e)) from e

    async def rest() -> AsyncIterator[dict]:
        yield first
        async for event in events:
            yield event

    if not body.stream:
        text, done = "", {}
        try:
            async for event in rest():
                if event["type"] == "text":
                    text += event["text"]
                else:
                    done = event
        except llm.LLMError as e:
            raise HTTPException(503, str(e)) from e
        return {"answer": text, **{k: v for k, v in done.items() if k != "type"}}

    async def lines():
        try:
            async for event in rest():
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except llm.LLMError as e:
            yield json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


if __name__ == "__main__":
    if not config.API_KEYS:
        raise SystemExit("Нет ключей доступа: задайте API_KEYS (через запятую) в .env или в окружении")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
