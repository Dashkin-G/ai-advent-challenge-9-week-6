"""Локальная модель через HTTP API Ollama.

Ollama — сервер на этом же компьютере. Вопрос уходит POST-запросом на /api/chat,
ответ приходит потоком: строка JSON на каждый кусочек текста, в последней —
сколько токенов сгенерировано и сколько это заняло. Векторы для поиска по индексу
считает модель эмбеддингов там же — POST /api/embed.
"""
import json
import re
from collections.abc import AsyncIterator

import httpx

from . import config


class LLMError(RuntimeError):
    """Ошибка на пути к ответу (Ollama, облако, индекс) — уже человеческими словами."""


def _client(read: float | None = 5) -> httpx.AsyncClient:
    # Ollama — на этом компьютере: системный прокси (HTTP_PROXY) к нему не применяем.
    return httpx.AsyncClient(base_url=config.OLLAMA_URL, trust_env=False,
                             timeout=httpx.Timeout(5, read=read))


async def status() -> dict:
    """Запущен ли Ollama, какие модели скачаны и какие сейчас в памяти."""
    try:
        async with _client() as client:
            version = (await client.get("/api/version")).json()["version"]
            models = (await client.get("/api/tags")).json()["models"]
            loaded = (await client.get("/api/ps")).json()["models"]
    except httpx.HTTPError:
        return {"running": False}
    # bge-m3 умеет только векторы для поиска — в выбор модели для ответа не попадает.
    chat = [m for m in models if "completion" in m.get("capabilities", ["completion"])]
    return {
        "running": True, "version": version, "default": config.MODEL,
        "models": [{"name": m["name"], "size": m["size"], "quant": m.get("details", {}).get("quantization_level")}
                   for m in chat],
        # vram — какая доля модели на видеокарте: 0 — считает процессор.
        "loaded": [{"name": m["name"], "size": m["size"], "vram": round(m["size_vram"] / m["size"], 2)} for m in loaded],
    }


async def ask(model: str, messages: list[dict], think: bool, options: dict | None = None) -> AsyncIterator[dict]:
    """Ответ по кусочкам на диалог: messages — [{role, content, images?}], последнее — вопрос.
    options — параметры генерации Ollama (temperature, num_predict, num_ctx…); не заданы —
    рекомендованные Qwen. События: request — что ушло в Ollama; think — ход мысли; text — ответ;
    done — счётчики. Ошибка — LLMError."""
    body = {"model": model, "messages": messages, "think": think, "stream": True}
    if options is not None:
        body["options"] = options
    elif not think:
        # Рекомендация Qwen для ответа без размышления; с размышлением — настройки модели.
        body["options"] = {"temperature": 0.7, "top_p": 0.8}
    yield {"type": "request", "url": f"{config.OLLAMA_URL}/api/chat", "body": body}
    try:
        # Чтение без тайм-аута: первая порция ждёт, пока модель загрузится в память.
        async with _client(read=None) as client, client.stream("POST", "/api/chat", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                raise LLMError(_explain(response, model))
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(f"Ollama прервал ответ: {chunk['error']}")
                message = chunk.get("message", {})
                if message.get("thinking"):
                    yield {"type": "think", "text": message["thinking"]}
                if message.get("content"):
                    yield {"type": "text", "text": message["content"]}
                if chunk.get("done"):
                    yield {"type": "done", **_counters(chunk)}
    except httpx.ConnectError as e:
        raise LLMError("Ollama не запущен") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama не отвечает: {e.__class__.__name__}") from e


async def embed(texts: list[str]) -> list[list[float]]:
    """Векторы текстов от модели эмбеддингов (bge-m3) — по ним ищется ближайшее в индексе."""
    try:
        async with _client(read=60) as client:
            response = await client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": texts})
    except httpx.ConnectError as e:
        raise LLMError("Ollama не запущен") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama не отвечает: {e.__class__.__name__}") from e
    if response.status_code != 200:
        raise LLMError(_explain(response, config.EMBED_MODEL))
    return response.json()["embeddings"]


async def load(model: str) -> None:
    """Загрузить модель в память заранее: запрос без текста только загружает её."""
    try:
        async with _client(read=None) as client:
            await client.post("/api/generate", json={"model": model})
    except httpx.HTTPError:
        pass                    # не запущен Ollama — скажет первый же вопрос


async def unload(model: str) -> None:
    """Выгрузить модель из памяти — вместе с ней Ollama забывает и кэш промптов."""
    try:
        async with _client(read=60) as client:
            await client.post("/api/generate", json={"model": model, "keep_alive": 0})
    except httpx.HTTPError:
        pass


def _explain(response: httpx.Response, model: str) -> str:
    if response.status_code == 404:
        return f"Модели {model} нет — скачайте: ollama pull {model}"
    try:
        detail = response.json()["error"]
    except ValueError:
        detail = response.text[:200]
    need = re.search(r"more system memory \(([\d.]+) GiB\) than is available \(([\d.]+) GiB\)", detail)
    if need:
        return f"Модели {model} нужно {need[1]} ГБ памяти, свободно {need[2]} ГБ — закройте лишние программы"
    return f"Ollama ответил {response.status_code}: {detail}"


def _counters(chunk: dict) -> dict:
    """Счётчики Ollama приходят в наносекундах. Время ответа — загрузка модели, чтение промпта
    (из кэша Ollama — доли секунды, если такой же промпт был недавно) и генерация."""
    tokens = chunk.get("eval_count", 0)
    generation = chunk.get("eval_duration", 0) / 1e9
    return {
        "tokens": tokens,
        "speed": tokens / generation if generation else 0,
        "seconds": chunk.get("total_duration", 0) / 1e9,
        "load": chunk.get("load_duration", 0) / 1e9,
        "prompt": chunk.get("prompt_eval_count", 0),
        "reading": chunk.get("prompt_eval_duration", 0) / 1e9,
        "writing": generation,
        "cut": chunk.get("done_reason") == "length",        # ответ оборван лимитом токенов
    }
