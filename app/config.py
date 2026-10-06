"""Настройки: где работает Ollama, какой моделью отвечать, на каком адресе интерфейс,
токен Телеграм-бота. Всё переопределяется переменными окружения или файлом .env."""
import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("MODEL", "qwen3.5:4b")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Токен бота выдаёт @BotFather в Телеграме. Хранится в .env — этот файл не попадает в git.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
