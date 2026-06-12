"""Abstract TTS engine interface.

Every engine turns a *chunk* of text into a mono float32 waveform. The router
loads engines lazily and serialises access with a per-engine lock, so engines
themselves do not need to be thread-safe beyond their own ``load``/``synthesize``.

Keeping this interface tiny is deliberate: swapping XTTS for a future Polish
fine-tune, or Kokoro for Qwen3-TTS, should be a one-class change (see the design
doc, v2 notes).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    # Human-readable engine id (used in logs and the unload endpoint).
    name: str = "base"
    # Cache-key component. Bump when the model / its config changes so old
    # cached audio is not silently reused for a different-sounding engine.
    version: str = "0"
    # Native output sample rate; may be refined after the model loads.
    sample_rate: int = 24000

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Idempotent. Heavy imports live here."""

    @abstractmethod
    def synthesize(self, text_chunk: str, voice: str | None) -> tuple[np.ndarray, int]:
        """Synthesize one chunk → (mono float32 in [-1, 1], sample_rate)."""

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Return the selectable voice names for this engine."""

    def default_voice(self) -> str | None:
        """First listed voice, or None if the engine has no voices."""
        voices = self.list_voices()
        return voices[0] if voices else None

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """Whether the model is currently resident in memory."""

    def unload(self) -> None:  # pragma: no cover - overridden where meaningful
        """Free model memory. Default is a no-op."""
