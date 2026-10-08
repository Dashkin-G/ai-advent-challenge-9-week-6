"""RAG по ПДД: поиск по индексу недели 5 и ответ модели по найденному.

Индекс построен в неделе 5: Правила дорожного движения с приложениями нарезаны по пунктам
на 274 фрагмента, их векторы bge-m3 лежат в FAISS, текст и подписи — в SQLite. Здесь индекс
только читается.

Вопрос → вектор той же bge-m3 (в Ollama) → 3 ближайших фрагмента по косинусу (FAISS) →
промпт «фрагменты + вопрос» → модель: локальная (Ollama) или облачная (DashScope).
Промпт у обеих один и тот же — сравниваются модели, а не контекст.
"""
import json
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import closing

import faiss
import numpy as np

from . import cloud, config, llm

STRATEGY = "structure"      # нарезка по пунктам — она выиграла сравнение нарезок в неделе 5
# Инструкция недели 5: прямой ответ первым, после утверждения — пункт, только по фрагментам.
SYSTEM = (
    "Ты — справочник по Правилам дорожного движения РФ. Отвечай по-русски и коротко: первое "
    "предложение — прямой ответ на вопрос, дальше, если нужно, условия и исключения; всего не больше "
    "пяти предложений. Пиши обычным текстом, без Markdown. После каждого утверждения указывай "
    "в квадратных скобках пункт, на котором оно основано, например [п. 10.2 ПДД] или "
    "[п. 5.4 Перечня неисправностей]. Отвечай только по фрагментам правил из сообщения, ничего "
    "не добавляй от себя. Если во фрагментах ответа нет, ответь одной фразой: "
    "«В найденных пунктах правил ответа нет.»"
)
# В приложениях пункт — это знак или линия разметки.
KIND = {"Приложение 1. Дорожные знаки": ("знак", "знаки"), "Приложение 2. Дорожная разметка": ("разметка", "разметка")}

_index: faiss.Index | None = None       # индекс читается с диска один раз
_chunks: dict[int, dict] = {}           # номер вектора в FAISS → фрагмент


def _label(title: str, points: list[str]) -> str:
    """«п. 10.2», «п. 17.1–17.3», «знак 3.20» — по номерам пунктов фрагмента."""
    one, many = KIND.get(title, ("п.", "п."))
    if len(points) < 3:
        return f"{one if len(points) == 1 else many} {', '.join(points)}"
    return f"{many} {points[0]}–{points[-1].split('-')[-1]}"


def _load() -> tuple[faiss.Index, dict[int, dict]]:
    """Индекс недели 5: векторы — из FAISS, текст и подписи фрагментов — из SQLite, только чтение."""
    global _index
    if _index is None:
        try:
            raw = (config.INDEX_DIR / f"{STRATEGY}.faiss").read_bytes()
            db = (config.INDEX_DIR / "index.db").resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(db, uri=True)) as con:
                rows = con.execute("SELECT row, title, section, points, text FROM chunks WHERE strategy = ?",
                                   (STRATEGY,)).fetchall()
        except (OSError, sqlite3.Error) as e:
            raise llm.LLMError(f"Нет индекса недели 5 в {config.INDEX_DIR}") from e
        for row, title, section, points, text in rows:
            _chunks[row] = {"label": _label(title, json.loads(points)), "title": title, "section": section,
                            "text": text}
        _index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
    return _index, _chunks


def info() -> dict:
    """Что за индекс и какие модели — для шапки страницы."""
    try:
        _, chunks = _load()
    except llm.LLMError as e:
        return {"error": str(e)}
    return {"chunks": len(chunks), "top": config.TOP, "local": config.MODEL, "embed": config.EMBED_MODEL,
            "cloud": config.CLOUD_MODEL, "cloud_on": cloud.enabled}


async def search(question: str) -> dict:
    """Ближайшие к вопросу фрагменты (score — косинус) и сколько занял поиск."""
    started = time.perf_counter()
    index, chunks = _load()
    vector = np.array(await llm.embed([question]), dtype=np.float32)
    faiss.normalize_L2(vector)          # у нормированных векторов скалярное произведение — косинус
    scores, rows = index.search(vector, config.TOP)
    hits = [{**chunks[int(r)], "score": round(float(s), 2)} for s, r in zip(scores[0], rows[0]) if r >= 0]
    return {"hits": hits, "seconds": time.perf_counter() - started}


def messages(question: str, hits: list[dict]) -> list[dict]:
    """Промпт: инструкция, фрагменты с заголовком «документ › раздел» и вопрос."""
    fragments = "\n\n".join(f"Фрагмент {i} — {' › '.join(filter(None, (h['title'], h['section'])))}\n{h['text']}"
                            for i, h in enumerate(hits, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Фрагменты правил:\n\n{fragments}\n\nВопрос: {question}"}]


async def answer(question: str, where: str) -> AsyncIterator[dict]:
    """Поиск и ответ событиями: found → request → think… → text… → done.
    where — local (Ollama на этом компьютере) или cloud (DashScope). Ошибка — llm.LLMError."""
    found = await search(question)
    yield {"type": "found", **found}
    prompt = messages(question, found["hits"])
    model = llm.ask(config.MODEL, prompt, think=False) if where == "local" else cloud.ask(prompt)
    async for event in model:
        yield event
