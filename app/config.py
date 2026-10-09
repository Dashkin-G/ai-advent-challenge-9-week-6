"""Настройки: где работает Ollama, какой моделью отвечать, на каком адресе интерфейс,
токен Телеграм-бота, индекс и облачная модель для RAG, ключи и лимиты приватного сервиса.
Всё переопределяется переменными окружения или файлом .env."""
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
TUNE_PATH = ROOT / "data" / "tune_check.json"        # итог проверки «до и после оптимизации» (день 29)

# Облачная модель для сравнения — та, что отвечала в неделе 5 (Alibaba Model Studio).
CLOUD_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CLOUD_URL = os.getenv("DASHSCOPE_BASE_URL",
                      "https://ws-q1vxaj37wm4fa9q8.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "qwen3.8-2.4t-a95b")
# qwen3.8 думает перед ответом всегда. 64 токена размышления — и она, как локальная, отвечает сразу.
CLOUD_THINKING = 64

# --- Приватный сервис (день 30) ---
# Ключи доступа через запятую. На сервере они в /etc/llm.env — его пишет deploy/setup.sh.
API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()]
RATE = int(os.getenv("RATE", "10"))             # запросов в минуту на ключ
# Отвечают сразу. Ollama 0.40 запускает qwen3.5 с одним слотом (llama-server -np 1) даже при
# OLLAMA_NUM_PARALLEL=2: второй ответ всё равно ждал бы первого.
PARALLEL = int(os.getenv("PARALLEL", "1"))
QUEUE = int(os.getenv("QUEUE", "10"))           # ждут своей очереди не больше стольких
CONTEXT = int(os.getenv("CONTEXT", "4096"))     # контекст модели в токенах: переписка и ответ вместе
ANSWER = int(os.getenv("ANSWER", "512"))        # ответ не длиннее, токенов
