"""Оптимизация локальной модели под RAG по ПДД (день 29): «до» и «после» на одних вопросах.

«До» — как в дне 28: qwen3.5:4b (Q4_K_M), temperature 0.7 и top_p 0.8, остальное не задано —
Ollama берёт своё (UNSET), промпт недели 5. «После» — настройки, подобранные опытами на контрольных
вопросах (README); на странице их можно менять.

Проверка: 10 контрольных вопросов × 3 прогона для «до» и для «после». Ответы идут по очереди: две
модели сразу делили бы процессор и мешали замеру скорости. Перед каждым прогоном модели выгружаются —
иначе тот же промпт Ollama взял бы из кэша и время ответа было бы занижено. Время ответа — без
загрузки модели: поиск, чтение промпта и генерация.
"""
import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from . import check, config, llm, rag

RUNS = check.RUNS
SIDES = ("before", "after")
# Что берётся, если параметр не задан: штраф — из настроек модели (ollama show qwen3.5:4b),
# контекст — Ollama по умолчанию (4096 без видеокарты), длина ответа — без лимита.
UNSET = {"presence_penalty": 1.5, "num_ctx": 4096, "num_predict": None}


class Settings(BaseModel):
    """Настройки ответа. None — параметр не задан."""
    model: str
    temperature: float = Field(ge=0, le=2)
    presence_penalty: float | None = Field(None, ge=0, le=2)   # штраф за слова, которые уже были в ответе
    num_predict: int | None = Field(None, ge=1, le=8192)        # не больше стольких токенов ответа
    num_ctx: int | None = Field(None, ge=256, le=262144)        # контекст: промпт и ответ вместе
    prompt: Literal["week5", "task"] = "week5"

    def options(self) -> dict:
        """Параметры для Ollama: температура, top_p как в дне 28 и всё, что задано явно."""
        given = self.model_dump(include={"presence_penalty", "num_predict", "num_ctx"}, exclude_none=True)
        return {"temperature": self.temperature, "top_p": 0.8, **given}


BEFORE = Settings(model=config.MODEL, temperature=0.7)
AFTER = Settings(model=config.MODEL, temperature=0.1, presence_penalty=0, num_predict=512, num_ctx=2048,
                 prompt="task")

_task: asyncio.Task | None = None
_live: dict | None = None               # идущая проверка; None — показывается сохранённая


async def info() -> dict:
    """Настройки «до» и «после», скачанные модели, тексты промптов и итог облака в дне 28 — для страницы."""
    cloud = check.report()["summary"]["cloud"]
    return {"before": BEFORE.model_dump(), "after": AFTER.model_dump(), "unset": UNSET,
            "models": (await llm.status()).get("models", []),
            "cloud": {"correct": cloud["correct"], "done": cloud["done"]} if cloud["done"] else None,
            "prompts": {key: {"title": title, "system": system} for key, (title, system) in rag.PROMPTS.items()}}


async def answer(question: str, settings: Settings) -> AsyncIterator[dict]:
    """Поиск и ответ с этими настройками: found → request → text… → done (+ память модели)."""
    found = await rag.search(question)
    yield {"type": "found", **found}
    prompt = rag.messages(question, found["hits"], settings.prompt)
    async for event in llm.ask(settings.model, prompt, think=False, options=settings.options()):
        if event["type"] == "done":
            loaded = (await llm.status()).get("loaded", [])
            event["memory"] = next((m["size"] for m in loaded if m["name"] == settings.model), 0)
        yield event


async def _one(settings: Settings, question: str) -> dict:
    """Один ответ: текст, время без загрузки модели, скорость, память — или ошибка."""
    started, text, done = time.perf_counter(), "", {}
    try:
        async for event in answer(question, settings):
            if event["type"] == "text":
                text += event["text"]
            elif event["type"] == "done":
                done = event
    except Exception as e:      # сбой одного ответа не роняет всю проверку — он виден в таблице
        return {"error": str(e) or e.__class__.__name__, "seconds": round(time.perf_counter() - started, 1)}
    return {"answer": text, "seconds": round(time.perf_counter() - started - done.get("load", 0), 1),
            "speed": round(done.get("speed", 0), 1), "tokens": done.get("tokens", 0), "memory": done.get("memory", 0)}


async def _run(settings: dict[str, Settings]) -> None:
    global _live
    try:
        for run in range(RUNS):
            for side in SIDES:
                for model in {s.model for s in settings.values()}:
                    await llm.unload(model)
                for i, q in enumerate(check.QUESTIONS):
                    _live["results"][side][i][run] = await _one(settings[side], q["question"])
        _live["finished"] = datetime.now().isoformat(timespec="seconds")
        config.TUNE_PATH.parent.mkdir(exist_ok=True)
        config.TUNE_PATH.write_text(json.dumps(_live, ensure_ascii=False), encoding="utf-8")
    finally:
        _live = None


def running() -> bool:
    return _task is not None and not _task.done()


def start(after: Settings) -> None:
    global _task, _live
    if not running():
        _live = {"started": datetime.now().isoformat(timespec="seconds"), "finished": None,
                 "settings": {"before": BEFORE.model_dump(), "after": after.model_dump()},
                 "results": {side: [[None] * RUNS for _ in check.QUESTIONS] for side in SIDES}}
        _task = asyncio.create_task(_run({"before": BEFORE, "after": after}))


def stop() -> None:
    if running():
        _task.cancel()          # ответы обрываются, прежний итог остаётся на диске


def report() -> dict:
    """Идущая или последняя проверка: настройки, ответы по прогонам с итогом и сводка «до» и «после»."""
    data = _live
    if data is None and config.TUNE_PATH.exists():
        data = json.loads(config.TUNE_PATH.read_text(encoding="utf-8"))
    data = data or {"started": None, "finished": None, "settings": None,
                    "results": {side: [[None] * RUNS for _ in check.QUESTIONS] for side in SIDES}}
    results = check.judged(data["results"])
    summary = {}
    for side in SIDES:
        memory = [a["memory"] for row in results[side] for a in row if a and a.get("memory")]
        summary[side] = {**check.summary(results[side]), "memory": max(memory, default=None)}
    return {"running": running(), "runs": RUNS, "started": data["started"], "finished": data["finished"],
            "settings": data["settings"], "results": results, "questions": check.questions(), "summary": summary}
