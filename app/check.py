"""Проверка на 10 контрольных вопросах недели 5: качество, скорость, стабильность.

Каждый вопрос задаётся обеим моделям по 3 раза с одним и тем же промптом. Ответ верный,
если в нём есть все ключевые факты из ожидания: факт — регулярное выражение по тексту
ответа (control_questions.json). Итог по каждой модели:
- качество — сколько ответов из 30 верные;
- скорость — сколько в среднем ждать ответа: от вопроса до последнего слова, с поиском;
- стабильность — на скольких вопросах из 10 все три прогона дали один и тот же итог.

Локальная модель и облако отвечают одновременно, каждая — на один вопрос за раз. Прогоны
идут по кругу (1–10, 1–10, 1–10): тот же промпт сразу следом Ollama взял бы из кэша.
Ответы пишутся на диск в конце; остановленная проверка не затирает прежнюю. Верен ли ответ,
считается при чтении: поправили ожидание — итог пересчитается без нового похода к моделям.
"""
import asyncio
import json
import re
import statistics
import time
from datetime import datetime

from . import config, llm, rag

RUNS = 3
SIDES = ("local", "cloud")
QUESTIONS = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))

_task: asyncio.Task | None = None
_live: dict | None = None               # идущая проверка; None — показывается сохранённая


def _facts(q: dict, answer: str) -> dict:
    """Ответ против ожидания: верный ли и каких фактов нет. У факта с absent наоборот:
    совпадение — ошибка (сумма штрафа, которой в Правилах нет)."""
    missing = [f["text"] for f in q["facts"]
               if bool(re.search(f["re"], answer, re.IGNORECASE)) == bool(f.get("absent"))]
    return {"ok": not missing, "missing": missing}


def verdict(question: str, answer: str) -> dict | None:
    """Итог для контрольного вопроса; на любой другой вопрос — None."""
    q = next((q for q in QUESTIONS if q["question"] == question), None)
    return _facts(q, answer) if q else None


async def _one(side: str, question: str) -> dict:
    """Один ответ целиком: текст и время от вопроса до последнего слова — или ошибка."""
    started, text, done = time.perf_counter(), "", {}
    try:
        async for event in rag.answer(question, side):
            if event["type"] == "text":
                text += event["text"]
            elif event["type"] == "done":
                done = event
    except Exception as e:      # сбой одного ответа не роняет всю проверку — он виден в таблице
        return {"error": str(e) or e.__class__.__name__, "seconds": round(time.perf_counter() - started, 1)}
    return {"answer": text, "seconds": round(time.perf_counter() - started, 1), "speed": round(done.get("speed", 0), 1)}


async def _side(side: str) -> None:
    for run in range(RUNS):
        for i, q in enumerate(QUESTIONS):
            _live["results"][side][i][run] = await _one(side, q["question"])


async def _run() -> None:
    global _live
    try:
        try:        # обе модели — в память до замеров, иначе первый ответ ждал бы загрузку
            await asyncio.gather(llm.load(config.MODEL), rag.search("прогрев"))
        except llm.LLMError:
            pass    # Ollama не запущен — это скажет каждый ответ
        await asyncio.gather(*(_side(side) for side in SIDES))
        _live["finished"] = datetime.now().isoformat(timespec="seconds")
        config.CHECK_PATH.parent.mkdir(exist_ok=True)
        config.CHECK_PATH.write_text(json.dumps(_live, ensure_ascii=False), encoding="utf-8")
    finally:
        _live = None


def running() -> bool:
    return _task is not None and not _task.done()


def start() -> None:
    global _task, _live
    if not running():
        _live = {"started": datetime.now().isoformat(timespec="seconds"), "finished": None,
                 "results": {side: [[None] * RUNS for _ in QUESTIONS] for side in SIDES}}
        _task = asyncio.create_task(_run())


def stop() -> None:
    if running():
        _task.cancel()          # ответы обрываются, прежний итог остаётся на диске


def summary(rows: list[list[dict | None]]) -> dict:
    """Итог одной модели: верных из готовых, среднее время и скорость, стабильные вопросы, сбои."""
    done = [a for row in rows for a in row if a]
    answered = [a for a in done if "error" not in a]
    full = [row for row in rows if all(row)]            # вопросы, где прошли все прогоны
    return {
        "done": len(done), "total": len(rows) * RUNS, "full": len(full),
        "correct": sum(a["ok"] for a in answered),
        "errors": len(done) - len(answered),
        "seconds": round(statistics.mean(a["seconds"] for a in answered), 1) if answered else None,
        "speed": round(statistics.mean(a["speed"] for a in answered), 1) if answered else None,
        "stable": sum(all("ok" in a for a in row) and len({a["ok"] for a in row}) == 1 for row in full),
    }


def report() -> dict:
    """Идущая или последняя проверка: вопросы, ответы по прогонам с итогом и сводка по моделям."""
    data = _live
    if data is None and config.CHECK_PATH.exists():
        data = json.loads(config.CHECK_PATH.read_text(encoding="utf-8"))
    data = data or {"started": None, "finished": None,
                    "results": {side: [[None] * RUNS for _ in QUESTIONS] for side in SIDES}}
    # Итог — по нынешнему ожиданию: у ответа (не у сбоя) появляются ok и missing.
    results = {side: [[{**a, **_facts(q, a["answer"])} if a and "answer" in a else a for a in row]
                      for q, row in zip(QUESTIONS, rows)] for side, rows in data["results"].items()}
    return {"running": running(), "runs": RUNS, "started": data["started"], "finished": data["finished"],
            "results": results, "questions": [{"question": q["question"], "expect": q["expect"]} for q in QUESTIONS],
            "summary": {side: summary(results[side]) for side in SIDES}}
