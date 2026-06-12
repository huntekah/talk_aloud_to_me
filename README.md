# 🎧 TTS Reader

[![CI](https://github.com/huntekah/talk_aloud_to_me/actions/workflows/ci.yml/badge.svg)](https://github.com/huntekah/talk_aloud_to_me/actions/workflows/ci.yml)

A local, single-user web app for turning pasted text into speech and listening
with podcast-style controls. Polish and English, a different model per language.
Runs natively on Apple Silicon (uses the GPU via MPS). Generation taking a
minute or two is fine — quality over latency.

- **Polish** → [XTTS-v2](https://github.com/idiap/coqui-ai-TTS) (voice cloning from a reference clip)
- **English** → [Kokoro-82M](https://github.com/Blaizzy/mlx-audio) via `mlx-audio` (fast, native MLX)
- **Frontend** → one static page, no framework, no build step
- **Backend** → FastAPI + a single background synthesis worker, SQLite + `.m4a` files

## Requirements

- macOS on Apple Silicon (tested on an M4)
- [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` (`brew install ffmpeg`)
- Python 3.11 (uv will provision it)
- ~3 GB free disk for model weights, downloaded on first use

## Run

```bash
./run.sh
```

That's the whole setup — `./run.sh` provisions both uv environments on first run,
starts the server, and opens your browser at **http://localhost:8765**. (Prereqs:
`uv` and `ffmpeg` → `brew install uv ffmpeg`.) The first Polish generation
downloads ~1.8 GB (XTTS) and the first English one ~0.4 GB (Kokoro); progress
shows in the console as “warming up model…”.

```bash
PORT=9000 ./run.sh         # different port
HOST=127.0.0.1 ./run.sh    # restrict to this machine only
NO_OPEN=1 ./run.sh         # don't auto-open the browser
```

> By default the server binds `0.0.0.0`, so it's reachable from other devices on
> your network (e.g. your phone) at `http://<your-mac-ip>:8765`. There's **no
> auth** — keep it to trusted networks, or set `HOST=127.0.0.1` for local-only.

## Why two uv projects?

`mlx-audio` (English) requires `transformers >= 5.5`, but `coqui-tts` (XTTS,
Polish) calls an API that `transformers 5.x` removed. They cannot share a
process, so each lives in its own locked uv environment:

| uv project      | venv                | Holds                                   | transformers |
|-----------------|---------------------|-----------------------------------------|--------------|
| `.` (this repo) | `.venv`             | FastAPI + Kokoro (English) + the server | 5.x          |
| `./xtts_engine` | `xtts_engine/.venv` | coqui-tts / XTTS-v2 (Polish)            | 4.57.x       |

`run.sh` runs `uv sync` for both (`uv sync` and `uv sync --project xtts_engine`).
The server (in `.venv`) talks to a long-lived **XTTS worker subprocess** (in
`xtts_engine/.venv`) over a small framed pipe protocol (`server/xtts_worker.py`).
The XTTS project pins `torch < 2.9` to avoid the `torchcodec`/FFmpeg-8 requirement.

## Voices

- **English:** pick from a few built-in Kokoro voices in the dropdown.
- **Polish:** drop a 6–20 s clean Polish speech `*.wav` into [`voices/`](voices/)
  and it becomes selectable (cloned). With no WAV present, Polish uses the
  built-in speaker *Ana Florence*. See [`voices/README.md`](voices/README.md).

## Layout

```
server/
  main.py          FastAPI app, routes, static mount
  jobs.py          queue, single worker, sqlite, caching, restart recovery
  pipeline.py      normalize · detect (lingua) · chunk · concat · ffmpeg encode
  xtts_worker.py   standalone XTTS process (runs in the xtts_engine env)
  engines/
    base.py        TTSEngine interface
    kokoro_en.py   English (Kokoro / mlx-audio)
    xtts_pl.py     Polish client → drives the XTTS worker subprocess
    router.py      lang → engine, lazy load, per-engine locks
web/               index.html · app.js · style.css
voices/            user-supplied Polish reference WAVs (+ README)
data/              jobs.sqlite + audio/*.m4a   (git-ignored)
tools/             voice-cloning / fine-tune utilities (run via uv)
xtts_engine/       isolated uv project for XTTS (pyproject.toml + uv.lock)
```

## API

| Method | Path                     | Notes                                              |
|--------|--------------------------|----------------------------------------------------|
| POST   | `/api/jobs`              | `{text, lang: auto\|pl\|en, voice?}` → 201 `{job_id}`; 409 `{job_id}` on cache hit |
| GET    | `/api/jobs/{id}`         | `{status, progress, lang, voice, title, duration_sec?, error?}` |
| GET    | `/api/jobs`              | recent jobs (newest first, max 50)                 |
| DELETE | `/api/jobs/{id}`         | delete job + audio                                 |
| GET    | `/api/audio/{id}.m4a`    | audio (supports HTTP Range → seeking)              |
| GET    | `/api/voices`            | `{pl: [...], en: [...]}`                            |
| POST   | `/api/engines/unload`    | `{name?}` free model RAM (`xtts`/`kokoro`/all)     |

## Player controls

Play/pause (space), −15 s / +15 s (← / →), draggable seek bar, speed cycle
(1.0–2.5× and 0.75×, pitch preserved), sleep timer (5/10/15/30 min), and
per-job resume (last position saved in `localStorage`).

## Development

```bash
make lint    # ruff --fix + ruff format (auto-fix)
make check   # ruff check + bandit security scan (read-only gate)
make test    # pytest (platform-independent pipeline tests)
```

CI runs the same checks on every push / PR to `main`
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): ruff, bandit, and
pytest. The test job installs only the lightweight `test` dependency group
(`uv sync --only-group test`) — no mlx/torch — so it runs on Linux runners even
though the app itself is macOS/Apple-Silicon only.

## Housekeeping

```bash
make clean         # delete generated jobs + audio
make clean-models  # delete both venvs (HF model cache is kept)
```

> XTTS-v2 is licensed CPML (non-commercial). Fine for personal use.
