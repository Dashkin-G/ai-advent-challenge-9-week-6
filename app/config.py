"""Настройки: где работает Ollama, какой моделью отвечать, на каком адресе интерфейс.
Всё переопределяется переменными окружения."""
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("MODEL", "qwen3.5:4b")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
