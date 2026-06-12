"""Polish engine: XTTS-v2 via coqui-tts, driven out-of-process.

coqui-tts needs transformers < 5 while mlx-audio (the English engine) needs
transformers >= 5.5, so XTTS cannot live in the app's venv. Instead a long-lived
worker (``server/xtts_worker.py``) runs under the ``xtts_engine`` uv project's
venv and this class is a thin client that spawns it, keeps it warm, and exchanges
framed messages with it.

Voice selection (listing ``voices/*.wav``, fallback speaker) stays here in the
parent; the worker only receives a resolved ``(kind, ref)`` pair.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from .base import TTSEngine

log = logging.getLogger("tts.xtts")

FALLBACK_SPEAKER = "Ana Florence"


class XTTSPolishEngine(TTSEngine):
    name = "xtts"
    version = "xtts_v2-1"
    sample_rate = 24000

    def __init__(self, voices_dir: Path, xtts_python: Path, worker_script: Path) -> None:
        self.voices_dir = Path(voices_dir)
        self.xtts_python = str(xtts_python)
        self.worker_script = str(worker_script)
        self._proc: subprocess.Popen | None = None

    # ----- process lifecycle -------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        if self.loaded:
            return
        if not Path(self.xtts_python).exists():
            raise RuntimeError(
                f"XTTS interpreter not found at {self.xtts_python}. "
                "Run ./run.sh (or: uv sync --project xtts_engine)."
            )
        env = os.environ.copy()
        env.setdefault("COQUI_TOS_AGREED", "1")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        log.info("Spawning XTTS worker (%s)…", self.xtts_python)
        # stderr=None → worker logs/coqui chatter inherit the server console.
        self._proc = subprocess.Popen(
            [self.xtts_python, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
        )

    def load(self) -> None:
        self._spawn()
        # Warm the model now so the first chunk isn't slow and load errors surface
        # while the job is still in the "warming up" stage.
        header = self._request({"cmd": "load"})
        sr = header.get("sample_rate")
        if sr:
            self.sample_rate = int(sr)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def unload(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(b'{"cmd": "shutdown"}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:
                self._kill()
                return
        self._proc = None

    # ----- protocol ----------------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        out = self._proc.stdout  # type: ignore[union-attr]
        while len(buf) < n:
            chunk = out.read(n - len(buf))
            if not chunk:
                raise RuntimeError("XTTS worker closed the pipe mid-response")
            buf.extend(chunk)
        return bytes(buf)

    def _request(self, req: dict) -> dict:
        if not self.loaded:
            self._spawn()
        try:
            self._proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
            header_line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not header_line:
                raise RuntimeError("XTTS worker died (no response)")
            header = json.loads(header_line.decode("utf-8").strip())
        except Exception:
            self._kill()  # drop the unhealthy worker; next call respawns
            raise
        if header.get("status") != "ok":
            raise RuntimeError(f"XTTS worker error: {header.get('error')}")
        return header

    # ----- voices ------------------------------------------------------------
    def _wav_files(self) -> list[Path]:
        if not self.voices_dir.exists():
            return []
        seen: set[str] = set()
        out: list[Path] = []
        for p in sorted(self.voices_dir.iterdir()):
            if p.suffix.lower() == ".wav" and p.name.lower() not in seen:
                seen.add(p.name.lower())
                out.append(p)
        return out

    def list_voices(self) -> list[str]:
        wavs = [p.name for p in self._wav_files()]
        return wavs if wavs else [FALLBACK_SPEAKER]

    def _resolve_voice(self, voice: str | None) -> tuple[str, str]:
        """Resolve to ("wav", abspath) or ("speaker", builtin_name)."""
        wavs = self._wav_files()
        if voice:
            for p in wavs:
                if voice in (p.name, p.stem):
                    return "wav", str(p)
            if voice != FALLBACK_SPEAKER and not voice.lower().endswith(".wav"):
                return "speaker", voice
        if wavs:
            return "wav", str(wavs[0])
        return "speaker", FALLBACK_SPEAKER

    # ----- synthesis ---------------------------------------------------------
    def synthesize(self, text_chunk: str, voice: str | None) -> tuple[np.ndarray, int]:
        kind, ref = self._resolve_voice(voice)
        header = self._request(
            {"cmd": "synthesize", "text": text_chunk, "voice_kind": kind, "voice_ref": ref}
        )
        n = int(header.get("samples", 0))
        sr = int(header.get("sample_rate", self.sample_rate))
        if n <= 0:
            return np.zeros(0, dtype=np.float32), sr
        raw = self._read_exact(n * 4)
        return np.frombuffer(raw, dtype="<f4").astype(np.float32), sr
