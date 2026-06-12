"""FastAPI app: REST API + static frontend for the local TTS reader.

Binds to 127.0.0.1 only (see run.sh). No auth — single local user, same origin,
so no CORS needed. The frontend in ``web/`` is served at ``/``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engines.router import EngineRouter
from .jobs import JobManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tts.main")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
VOICES_DIR = PROJECT_ROOT / "voices"
WEB_DIR = PROJECT_ROOT / "web"
XTTS_PYTHON = PROJECT_ROOT / ".venv-xtts" / "bin" / "python"
XTTS_WORKER = PROJECT_ROOT / "server" / "xtts_worker.py"


class JobCreate(BaseModel):
    text: str = Field(min_length=1)
    lang: Literal["auto", "pl", "en"] = "auto"
    voice: Optional[str] = None


class UnloadRequest(BaseModel):
    name: Optional[str] = None  # "xtts" | "kokoro" | "pl" | "en" | None = all


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    router = EngineRouter(VOICES_DIR, XTTS_PYTHON, XTTS_WORKER)
    jobs = JobManager(DATA_DIR / "jobs.sqlite", AUDIO_DIR, router)
    jobs.init()
    await jobs.start()
    app.state.router = router
    app.state.jobs = jobs
    log.info("TTS reader ready. Engines: %s", router.status())
    try:
        yield
    finally:
        await jobs.stop()
        await router.unload()


app = FastAPI(title="Local TTS Reader", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
async def create_job(body: JobCreate):
    jobs: JobManager = app.state.jobs
    try:
        job_id, cached = jobs.create_job(body.text, body.lang, body.voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if cached:
        return JSONResponse(status_code=409, content={"job_id": job_id, "cached": True})
    return JSONResponse(status_code=201, content={"job_id": job_id, "cached": False})


@app.get("/api/jobs")
async def list_jobs():
    return app.state.jobs.list_jobs(limit=50)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = app.state.jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if not app.state.jobs.delete_job(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"deleted": True}


@app.get("/api/audio/{job_id}.m4a")
async def get_audio(job_id: str):
    path: Path = app.state.jobs.audio_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    # FileResponse honours Range requests, which the player needs for seeking.
    return FileResponse(path, media_type="audio/mp4", filename=f"{job_id}.m4a")


@app.get("/api/voices")
async def get_voices():
    return app.state.router.list_voices()


@app.get("/api/engines")
async def get_engines():
    return app.state.router.status()


@app.post("/api/engines/unload")
async def unload_engines(body: UnloadRequest):
    freed = await app.state.router.unload(body.name)
    return {"freed": freed}


# Static frontend mounted last so /api/* routes win. html=True serves index.html.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
