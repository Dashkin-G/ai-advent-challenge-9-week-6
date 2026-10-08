"""Настройки: где работает Ollama, какой моделью отвечать, на каком адресе интерфейс,
токен Телеграм-бота, индекс и облачная модель для RAG. Всё переопределяется переменными
окружения или файлом .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("MODEL", "qwen3.5:4b")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Токен бота выдаёт @BotFather в Телеграме. Хранится в .env — этот файл не попадает в git.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

# --- RAG по ПДД ---
# Индекс построен в неделе 5 (python -m app.pipeline там) и здесь только читается.
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT.parent / "ai-advent-challenge-9-week-5" / "data" / "index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")    # та же модель, что строила индекс, — в Ollama
TOP = int(os.getenv("TOP", "3"))                     # столько фрагментов уходит модели
QUESTIONS_PATH = ROOT / "control_questions.json"     # 10 контрольных вопросов недели 5
CHECK_PATH = ROOT / "data" / "rag_check.json"        # итог последней проверки

# Облачная модель для сравнения — та, что отвечала в неделе 5 (Alibaba Model Studio).
CLOUD_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CLOUD_URL = os.getenv("DASHSCOPE_BASE_URL",
                      "https://ws-q1vxaj37wm4fa9q8.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "qwen3.8-2.4t-a95b")
# qwen3.8 думает перед ответом всегда. 64 токена размышления — и она, как локальная, отвечает сразу.
CLOUD_THINKING = 64
