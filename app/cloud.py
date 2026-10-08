"""Облачная модель для сравнения: qwen3.8 в Alibaba Model Studio (DashScope).

Режим совместим с OpenAI: POST /chat/completions, ответ приходит потоком SSE — строка
«data: {…}» на каждый кусочек, расход токенов — в последней. Запрос идёт через системный
прокси, если он настроен: облако — в интернете, в отличие от Ollama.
"""
import json
import time
from collections.abc import AsyncIterator

import httpx

from . import config
from .llm import LLMError

# Кнопка «Облако» на странице: выключили — запрос в облако не уходит, как без интернета.
enabled = True


async def ask(messages: list[dict]) -> AsyncIterator[dict]:
    """События те же, что у локальной модели: request → think… → text… → done. Ошибка — LLMError."""
    if not enabled:
        raise LLMError("Облако отключено — отвечает только этот компьютер")
    if not config.CLOUD_KEY:
        raise LLMError("Нет ключа облака — впишите DASHSCOPE_API_KEY в .env")
    url = f"{config.CLOUD_URL}/chat/completions"
    body = {"model": config.CLOUD_MODEL, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "thinking_budget": config.CLOUD_THINKING}
    yield {"type": "request", "url": url, "body": body}
    started, first, tokens = time.perf_counter(), 0.0, 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15)) as client, client.stream(
                "POST", url, json=body, headers={"Authorization": f"Bearer {config.CLOUD_KEY}"}) as response:
            if response.status_code != 200:
                await response.aread()
                raise LLMError(f"Облако ответило {response.status_code}: {_detail(response)}")
            async for line in response.aiter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:])
                tokens = (chunk.get("usage") or {}).get("completion_tokens", tokens)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    for kind, key in (("think", "reasoning_content"), ("text", "content")):
                        if delta.get(key):
                            first = first or time.perf_counter()
                            yield {"type": kind, "text": delta[key]}
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError) as e:
        raise LLMError("Нет связи с облаком") from e
    except httpx.HTTPError as e:
        raise LLMError(f"Облако не отвечает: {e.__class__.__name__}") from e
    end = time.perf_counter()
    yield {"type": "done", "tokens": tokens, "speed": tokens / (end - first) if 0 < first < end else 0,
           "seconds": end - started}


def _detail(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]
