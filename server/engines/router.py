"""Routes a language to its engine and serialises model access.

One asyncio lock per engine guards load + inference + unload, so a manual
``/api/engines/unload`` can never race a synthesis in flight. Engines are
loaded lazily on first use and stay resident (both fit easily in 36 GB).
Blocking model work is pushed to a worker thread via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from .base import TTSEngine
from .kokoro_en import KokoroEnglishEngine
from .xtts_pl import XTTSPolishEngine

log = logging.getLogger("tts.router")


class EngineRouter:
    def __init__(
        self, voices_dir: Path, xtts_python: Path, xtts_worker: Path
    ) -> None:
        self._engines: dict[str, TTSEngine] = {
            "pl": XTTSPolishEngine(voices_dir, xtts_python, xtts_worker),
            "en": KokoroEnglishEngine(),
        }
        self._locks: dict[str, asyncio.Lock] = {
            lang: asyncio.Lock() for lang in self._engines
        }

    def engine_for(self, lang: str) -> TTSEngine:
        if lang not in self._engines:
            raise ValueError(f"No engine for language {lang!r}")
        return self._engines[lang]

    def engine_version(self, lang: str) -> str:
        return self.engine_for(lang).version

    def default_voice(self, lang: str) -> str | None:
        return self.engine_for(lang).default_voice()

    def is_loaded(self, lang: str) -> bool:
        return self.engine_for(lang).loaded

    async def ensure_loaded(self, lang: str) -> None:
        eng = self.engine_for(lang)
        if eng.loaded:
            return
        async with self._locks[lang]:
            if not eng.loaded:
                await asyncio.to_thread(eng.load)

    async def synthesize(
        self, lang: str, text_chunk: str, voice: str | None
    ) -> tuple[np.ndarray, int]:
        eng = self.engine_for(lang)
        async with self._locks[lang]:
            if not eng.loaded:
                await asyncio.to_thread(eng.load)
            return await asyncio.to_thread(eng.synthesize, text_chunk, voice)

    def list_voices(self) -> dict[str, list[str]]:
        return {lang: eng.list_voices() for lang, eng in self._engines.items()}

    def status(self) -> dict[str, dict]:
        return {
            lang: {"name": eng.name, "loaded": eng.loaded, "version": eng.version}
            for lang, eng in self._engines.items()
        }

    async def unload(self, name: str | None = None) -> list[str]:
        """Unload an engine by name/lang, or all engines when name is None."""
        freed: list[str] = []
        for lang, eng in self._engines.items():
            if name is not None and name not in (lang, eng.name):
                continue
            async with self._locks[lang]:
                if eng.loaded:
                    await asyncio.to_thread(eng.unload)
                    freed.append(eng.name)
        return freed
