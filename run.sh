#!/usr/bin/env bash
# Sets up both Python environments (if needed) and starts the TTS Reader.
#
#   .venv       -> FastAPI + Kokoro (English), transformers 5.x
#   .venv-xtts  -> coqui-tts / XTTS-v2 (Polish), transformers 4.x  (isolated)
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
UV_PYTHON=3.11 uv sync

echo "==> Preparing isolated Polish engine (.venv-xtts): coqui-tts / XTTS-v2…"
[ -x ".venv-xtts/bin/python" ] || uv venv .venv-xtts --python 3.11
uv pip install --quiet --python .venv-xtts -r requirements-xtts.txt

echo
echo "==> TTS Reader → http://127.0.0.1:${PORT}"
echo "    First Polish run downloads ~1.8 GB (XTTS); first English run ~0.4 GB (Kokoro)."
echo "    Watch this console for model load / generation progress."
echo
exec .venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port "${PORT}"
