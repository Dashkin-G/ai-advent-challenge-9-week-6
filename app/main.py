"""HTTP: две страницы и их методы. «/» — вопросы локальной модели (день 26),
«/rag» — RAG по ПДД: локальная модель против облачной (день 28)."""
import asyncio
import json
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import check, cloud, config, llm, machine, rag

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Локальная LLM")


class Ask(BaseModel):
    question: str
    model: str = config.MODEL
    think: bool = False


class RagAsk(BaseModel):
    question: str
    where: Literal["local", "cloud"]


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/rag")
def rag_page():
    return FileResponse(STATIC / "rag.html")


@app.get("/api/status")
async def status():
    return await llm.status()


@app.post("/api/ask")
async def ask(body: Ask):
    """Ответ потоком строк JSON: request → think… → text… → done → usage (или error)."""
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")

    async def lines():
        # Первый снимок ресурсов — в потоке: поиск процессов Ollama занимает до полсекунды.
        usage = asyncio.create_task(asyncio.to_thread(machine.Usage))
        try:
            async for event in llm.ask(body.model, [{"role": "user", "content": question}], body.think):
                yield _line(event)
                if event["type"] == "done":
                    spent = await asyncio.to_thread((await usage).result)
                    loaded = next((m for m in (await llm.status()).get("loaded", []) if m["name"] == body.model), None)
                    yield _line({"type": "usage", **spent, "gpu": loaded["size"] * loaded["vram"] if loaded else 0})
        except llm.LLMError as e:
            yield _line({"type": "error", "text": str(e)})
        finally:
            usage.cancel()

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/api/rag")
def rag_info():
    return rag.info()


class Cloud(BaseModel):
    on: bool


@app.post("/api/rag/cloud")
def rag_cloud(body: Cloud):
    """Включить или отключить облако: отключённое отвечает ошибкой, не выходя в сеть."""
    cloud.enabled = body.on
    return {"cloud_on": cloud.enabled}


@app.post("/api/rag/ask")
async def rag_ask(body: RagAsk):
    """Поиск и ответ потоком строк JSON: found → request → think… → text… → done (или error).
    На контрольный вопрос в done — проверка: все ли ключевые факты на месте."""
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")

    async def lines():
        text = ""
        try:
            async for event in rag.answer(question, body.where):
                if event["type"] == "text":
                    text += event["text"]
                elif event["type"] == "done":
                    event["check"] = check.verdict(question, text)
                yield _line(event)
        except llm.LLMError as e:
            yield _line({"type": "error", "text": str(e)})

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/api/rag/check")
async def rag_check():
    return check.report()


@app.post("/api/rag/check")
async def rag_check_start():
    check.start()
    return check.report()


@app.delete("/api/rag/check")
async def rag_check_stop():
    check.stop()
    return check.report()


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
