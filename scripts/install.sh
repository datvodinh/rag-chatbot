#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync --locked
