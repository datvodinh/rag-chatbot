#!/usr/bin/env bash
set -euo pipefail

# Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && apt update && apt install ngrok

# Python (uv)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation failed or is not on PATH. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync --locked
