#!/usr/bin/env bash
# Приватный сервис на чистой Ubuntu одной командой: Ollama с моделью, сервис, HTTPS.
#   sudo bash /opt/llm/deploy/setup.sh 185.12.34.56
# Адрес сервиса — https://185-12-34-56.sslip.io: sslip.io отвечает на такое имя этим же IP,
# и Caddy получает на него настоящий сертификат Let's Encrypt — свой домен не нужен.
# Повторный запуск ничего не ломает: ключ и скачанная модель остаются.
set -euo pipefail
IP=${1:?Укажите внешний IP сервера: sudo bash deploy/setup.sh 185.12.34.56}
HOST=${IP//./-}.sslip.io
DIR=$(cd "$(dirname "$0")/.." && pwd)
MODEL=qwen3.5:4b

echo "== 1. Ollama: только 127.0.0.1, по одному ответу, модель всегда в памяти"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
mkdir -p /etc/systemd/system/ollama.service.d
# Кэш промптов llama-server по умолчанию растёт до 8 ГБ — у машины 8 ГБ всего. Ограничен 1 ГБ:
# llama-server наследует окружение Ollama и читает LLAMA_ARG_CACHE_RAM (МБ).
cat > /etc/systemd/system/ollama.service.d/service.conf <<EOF
[Service]
Environment=OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KEEP_ALIVE=-1 LLAMA_ARG_CACHE_RAM=1024
EOF
systemctl daemon-reload
systemctl restart ollama
until ollama list >/dev/null 2>&1; do sleep 1; done
ollama pull "$MODEL"

echo "== 2. Сервис: ключ в /etc/llm.env (читает только root), systemd перезапускает при сбое"
apt-get update -q
apt-get install -yq python3-venv caddy
python3 -m venv "$DIR/.venv"
# Сервису нужны только эти пакеты: numpy и faiss из requirements.txt — для RAG дня 28.
"$DIR/.venv/bin/pip" install -q fastapi uvicorn httpx psutil python-dotenv
[ -f /etc/llm.env ] || (umask 077; echo "API_KEYS=$(openssl rand -hex 16)" > /etc/llm.env)
cat > /etc/systemd/system/llm.service <<EOF
[Unit]
Description=Приватный сервис локальной LLM
After=ollama.service

[Service]
EnvironmentFile=/etc/llm.env
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python -m app.service
DynamicUser=yes
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable llm
systemctl restart llm

echo "== 3. HTTPS: Caddy сам получает сертификат и передаёт запросы сервису"
cat > /etc/caddy/Caddyfile <<EOF
$HOST {
	reverse_proxy 127.0.0.1:8000 {
		flush_interval -1
	}
}
EOF
systemctl restart caddy

echo
echo "Готово: https://$HOST"
echo "Ключ:   $(grep -oP '(?<=API_KEYS=)[^,]+' /etc/llm.env)"
