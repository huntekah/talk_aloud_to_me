#!/usr/bin/env bash
# ── TTS Reader ────────────────────────────────────────────────────────────────
# One command runs everything. No manual setup: this creates both uv
# environments on first run, then starts the server.  Just:  ./run.sh
#
# Optional env vars:
#   PORT=8765      port to serve on
#   HOST=0.0.0.0   bind address (default = reachable from other devices on your
#                  network; use HOST=127.0.0.1 to keep it to this machine only)
#   NO_OPEN=1      don't auto-open the browser
#
# Requires `uv` and `ffmpeg`  →  brew install uv ffmpeg
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: 'uv' not found → https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: 'ffmpeg' not found → brew install ffmpeg"; exit 1; }

echo "==> Setting up environments with uv (first run installs dependencies)…"
uv sync                        # app + English engine (Kokoro)
uv sync --project xtts_engine  # isolated Polish engine (XTTS-v2)

URL="http://localhost:${PORT}"
echo
echo "──────────────────────────────────────────────────────────────"
echo "   TTS Reader  →  ${URL}"
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  LANIP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
  [ -n "$LANIP" ] && echo "   on your network  →  http://${LANIP}:${PORT}   (no password — trusted networks only)"
fi
echo "   First run downloads models (~1.8 GB PL, ~0.4 GB EN) — watch below."
echo "   Stop with Ctrl-C."
echo "──────────────────────────────────────────────────────────────"
echo

# Open the browser once the server is up (skip with NO_OPEN=1).
if [ -z "${NO_OPEN:-}" ] && command -v open >/dev/null 2>&1; then
  ( sleep 3; open "$URL" >/dev/null 2>&1 || true ) &
fi

exec uv run python -m uvicorn server.main:app --host "$HOST" --port "$PORT"
