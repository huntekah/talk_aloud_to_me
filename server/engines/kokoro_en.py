"""English engine: Kokoro-82M via mlx-audio (Apple-Silicon native, fast)."""
from __future__ import annotations

import logging
import re

import numpy as np

from .base import TTSEngine

log = logging.getLogger("tts.kokoro")

KOKORO_REPO = "prince-canuma/Kokoro-82M"

# American-English voices (lang_code "a"). Kept to a curated handful per the spec.
BUILTIN_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam"]

# mlx-audio's Kokoro uses misaki for g2p; "a" = American English, "b" = British.
LANG_CODE = "a"


class KokoroEnglishEngine(TTSEngine):
    name = "kokoro"
    version = "kokoro-82m-1"
    sample_rate = 24000

    def __init__(self) -> None:
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model  # lazy heavy import

        log.info("Loading Kokoro-82M (%s)…", KOKORO_REPO)
        model = load_model(KOKORO_REPO)
        sr = getattr(model, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)
        self._model = model
        log.info("Kokoro-82M ready (sample_rate=%d).", self.sample_rate)

    def list_voices(self) -> list[str]:
        return list(BUILTIN_VOICES)

    def synthesize(self, text_chunk: str, voice: str | None) -> tuple[np.ndarray, int]:
        if self._model is None:
            self.load()
        v = voice or BUILTIN_VOICES[0]
        return self._generate_safe(text_chunk, v), self.sample_rate

    def _generate_once(self, text: str, voice: str) -> np.ndarray:
        segments: list[np.ndarray] = []
        for out in self._model.generate(
            text=text, voice=voice, speed=1.0, lang_code=LANG_CODE
        ):
            arr = np.asarray(out.audio, dtype=np.float32).reshape(-1)
            if arr.size:
                segments.append(arr)
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return segments[0] if len(segments) == 1 else np.concatenate(segments)

    def _generate_safe(self, text: str, voice: str, depth: int = 0) -> np.ndarray:
        """Generate audio, recursively splitting if Kokoro's ~510-token context is
        exceeded (mlx-audio raises a broadcast_shapes error on long/dense input)."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        try:
            return self._generate_once(text, voice)
        except Exception as exc:  # noqa: BLE001
            mid = self._split_point(text)
            if depth >= 8 or mid is None:
                raise  # genuinely un-synthesizable; let the job surface it
            log.warning("Kokoro chunk too long (%d chars); splitting. (%s)", len(text), exc)
            left = self._generate_safe(text[:mid], voice, depth + 1)
            right = self._generate_safe(text[mid:], voice, depth + 1)
            if not left.size:
                return right
            if not right.size:
                return left
            seam = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)
            return np.concatenate([left, seam, right])

    @staticmethod
    def _split_point(text: str) -> int | None:
        """Index to split a too-long chunk: nearest sentence end to the middle,
        else nearest whitespace; None if it can't be split further."""
        n = len(text)
        if n < 40:
            return None
        mid = n // 2
        best = None
        for m in re.finditer(r"[.!?…,;:]\s+", text):
            if 0 < m.end() < n and (best is None or abs(m.end() - mid) < abs(best - mid)):
                best = m.end()
        if best is not None:
            return best
        left = text.rfind(" ", 0, mid)
        right = text.find(" ", mid)
        cands = [c for c in (left, right) if c != -1]
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c - mid)) + 1

    def unload(self) -> None:
        self._model = None
        try:  # best-effort GPU cache release
            import gc

            import mlx.core as mx

            gc.collect()
            mx.clear_cache()
        except Exception:  # pragma: no cover
            pass
