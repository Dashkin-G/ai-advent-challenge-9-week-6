"""HTTP: страница и два метода — состояние модели и вопрос к ней."""
import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config, llm, machine

PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"

app = FastAPI(title="Локальная LLM")


class Ask(BaseModel):
    question: str
    model: str = config.MODEL
    think: bool = False


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.get("/")
def page():
    return FileResponse(PAGE)


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


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
