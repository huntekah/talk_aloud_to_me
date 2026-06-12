"""Job queue, single background worker, and SQLite persistence.

One synthesis runs at a time (a single asyncio worker pulling from a queue) so
the two engines never fight over RAM/GPU/thermals. Jobs and their metadata live
in SQLite; generated audio lives as ``data/audio/{id}.m4a``. Identical requests
hit a content-addressed cache. On startup any job left mid-flight by a crash is
marked failed so the library stays consistent.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

from . import pipeline
from .engines.router import EngineRouter

log = logging.getLogger("tts.jobs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    key           TEXT,
    status        TEXT NOT NULL,
    progress      REAL DEFAULT 0,
    lang          TEXT,
    voice         TEXT,
    title         TEXT,
    text          TEXT,
    duration_sec  REAL,
    error         TEXT,
    message       TEXT,
    chunks_done   INTEGER DEFAULT 0,
    chunks_total  INTEGER DEFAULT 0,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_key ON jobs(key);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""

# Public fields returned by the API (text/key are internal).
_PUBLIC_FIELDS = (
    "id", "status", "progress", "lang", "voice", "title",
    "duration_sec", "error", "message", "chunks_done", "chunks_total",
    "created_at",
)


class JobError(Exception):
    """Synthesis failure carrying the offending chunk for the user."""

    def __init__(self, message: str, chunk_index: int | None = None, excerpt: str = ""):
        super().__init__(message)
        self.chunk_index = chunk_index
        self.excerpt = excerpt


class JobManager:
    def __init__(self, db_path: Path, audio_dir: Path, router: EngineRouter) -> None:
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir)
        self.router = router
        self._db: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()
        self._queue: asyncio.Queue[str] | None = None
        self._worker_task: asyncio.Task | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # ----- lifecycle ---------------------------------------------------------
    def init(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        # Recover from an interrupted run: nothing can still be in flight.
        with self._db_lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='failed', "
                "error='interrupted by server restart', message=NULL "
                "WHERE status IN ('queued','running')"
            )
            self._db.commit()
            if cur.rowcount:
                log.info("Marked %d interrupted job(s) as failed on startup.", cur.rowcount)

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker(), name="tts-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._db:
            self._db.close()

    # ----- db helpers --------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._db_lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _row(self, job_id: str) -> sqlite3.Row | None:
        with self._db_lock:
            cur = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            return cur.fetchone()

    @staticmethod
    def _public(row: sqlite3.Row) -> dict:
        d = {k: row[k] for k in _PUBLIC_FIELDS}
        # Drop empties for a cleaner payload.
        if d.get("error") is None:
            d.pop("error")
        if not d.get("message"):
            d.pop("message", None)
        if d.get("duration_sec") is None:
            d.pop("duration_sec")
        return d

    def audio_path(self, job_id: str) -> Path:
        return self.audio_dir / f"{job_id}.m4a"

    # ----- public API used by routes ----------------------------------------
    def create_job(
        self, text: str, lang_req: str, voice_req: str | None
    ) -> tuple[str, bool]:
        """Create (or return cached) job. Returns (job_id, cached)."""
        normalized = pipeline.normalize(text)
        if not normalized:
            raise ValueError("No speakable text after normalization.")
        lang = lang_req if lang_req in ("pl", "en") else pipeline.detect_language(normalized)
        voice = voice_req or self.router.default_voice(lang) or ""
        engine_version = self.router.engine_version(lang)
        key = hashlib.sha256(
            "\x00".join([normalized, lang, voice, engine_version]).encode("utf-8")
        ).hexdigest()

        # Cache hit: an identical, completed job whose audio still exists.
        with self._db_lock:
            cur = self._db.execute(
                "SELECT id FROM jobs WHERE key=? AND status='done' "
                "ORDER BY created_at DESC LIMIT 1",
                (key,),
            )
            hit = cur.fetchone()
        if hit and self.audio_path(hit["id"]).exists():
            return hit["id"], True

        job_id = uuid.uuid4().hex
        self._exec(
            "INSERT INTO jobs (id, key, status, progress, lang, voice, title, text, "
            "chunks_done, chunks_total, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id, key, "queued", 0.0, lang, voice,
                pipeline.make_title(normalized), normalized, 0, 0, self._now(),
            ),
        )
        assert self._queue is not None
        self._queue.put_nowait(job_id)
        log.info("Queued job %s (lang=%s, voice=%s).", job_id, lang, voice)
        return job_id, False

    def get_job(self, job_id: str) -> dict | None:
        row = self._row(job_id)
        return self._public(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._db_lock:
            cur = self._db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._public(r) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        row = self._row(job_id)
        if not row:
            return False
        path = self.audio_path(job_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover
            log.warning("Could not delete audio for %s: %s", job_id, exc)
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))
        return True

    # ----- worker ------------------------------------------------------------
    async def _worker(self) -> None:
        assert self._queue is not None
        log.info("Synthesis worker started.")
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except Exception as exc:  # never let the worker die
                log.exception("Job %s crashed: %s", job_id, exc)
            finally:
                self._queue.task_done()

    def _set(self, job_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    async def _process(self, job_id: str) -> None:
        row = self._row(job_id)
        if row is None or row["status"] != "queued":
            return  # deleted or already handled
        lang, voice, text = row["lang"], row["voice"], row["text"]

        self._set(job_id, status="running", progress=0.0)
        try:
            if not self.router.is_loaded(lang):
                self._set(job_id, message="warming up model…")
                await self.router.ensure_loaded(lang)
                self._set(job_id, message=None)

            chunks = pipeline.chunk_text(text, pipeline.MAX_CHARS[lang])
            if not chunks:
                raise JobError("No speakable text to synthesize.")
            self._set(job_id, chunks_total=len(chunks))

            segments: list = []
            para_flags: list[bool] = []
            sample_rate = self.router.engine_for(lang).sample_rate
            for i, chunk in enumerate(chunks):
                audio, sr = await self._synth_with_retry(lang, chunk.text, voice, i)
                sample_rate = sr
                segments.append(audio)
                para_flags.append(chunk.para_end)
                self._set(
                    job_id, chunks_done=i + 1, progress=round((i + 1) / len(chunks), 4)
                )

            joined = pipeline.concat_audio(segments, para_flags, sample_rate)
            out = str(self.audio_path(job_id))
            duration = await asyncio.to_thread(
                pipeline.encode_to_m4a, joined, sample_rate, out
            )
            self._set(
                job_id, status="done", progress=1.0,
                duration_sec=round(duration, 2), message=None,
            )
            log.info("Job %s done (%.1fs audio).", job_id, duration)
        except JobError as exc:
            detail = str(exc)
            if exc.chunk_index is not None:
                detail += f" (chunk {exc.chunk_index}: {exc.excerpt!r})"
            self._set(job_id, status="failed", error=detail, message=None)
            log.warning("Job %s failed: %s", job_id, detail)
        except Exception as exc:
            self._set(job_id, status="failed", error=str(exc), message=None)
            log.exception("Job %s failed: %s", job_id, exc)

    async def _synth_with_retry(self, lang, text, voice, index):
        """Synthesize one chunk, retrying once before giving up."""
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                return await self.router.synthesize(lang, text, voice)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Chunk %d attempt %d failed: %s", index, attempt, exc)
        raise JobError(
            f"Synthesis failed: {last_exc}", chunk_index=index, excerpt=text[:80]
        )
