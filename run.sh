#!/usr/bin/env bash
# Sets up both uv environments (if needed) and starts the TTS Reader.
#
#   .venv              -> FastAPI + Kokoro (English), transformers 5.x  (this project)
#   xtts_engine/.venv  -> coqui-tts / XTTS-v2 (Polish), transformers 4.x  (uv sub-project)
#
# Bound to 127.0.0.1 only. Override the port with PORT=xxxx ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8765}"

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: 'ffmpeg' not found on PATH. Install it: brew install ffmpeg"
  exit 1
}

echo "==> Syncing app environment (.venv): FastAPI + Kokoro English engine…"
uv sync

echo "==> Syncing isolated Polish engine (xtts_engine/.venv): coqui-tts / XTTS-v2…"
uv sync --project xtts_engine

echo
echo "==> TTS Reader → http://127.0.0.1:${PORT}"
echo "    First Polish run downloads ~1.8 GB (XTTS); first English run ~0.4 GB (Kokoro)."
echo "    Watch this console for model load / generation progress."
echo
exec uv run python -m uvicorn server.main:app --host 127.0.0.1 --port "${PORT}"
