"""Text → audio pipeline helpers: normalize, detect, chunk, concat, encode.

These are pure/stateless functions (plus a lazily-built language detector). The
job worker orchestrates them around the engine router. Both engines emit 24 kHz
mono float32, so concatenation needs no resampling in practice; a guard handles
the unexpected case anyway.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("tts.pipeline")

# Per-engine chunk size ceilings (characters).
#   pl (XTTS): degrades/loops past ~250.
#   en (Kokoro): the model has a hard ~510 *token* context; ~800 chars of dense
#     text (numbers, code, symbols) overflows it and mlx-audio then mishandles the
#     internal split (broadcast_shapes error). 400 keeps a safe token margin; the
#     Kokoro engine also self-splits as a backstop for anything still too dense.
MAX_CHARS = {"pl": 230, "en": 400}
# Silence inserted between chunks and at paragraph boundaries.
GAP_SENTENCE_SEC = 0.25
GAP_PARAGRAPH_SEC = 0.60


@dataclass
class Chunk:
    text: str
    para_end: bool  # True if this is the last chunk of a paragraph


# --------------------------------------------------------------------------- #
# 1. Normalize
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")  # [text](url) / ![alt](url)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>+\s?", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_FENCE_RE = re.compile(r"```[^\n]*\n?")
_MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": " - ", "—": " - ", "−": "-",  # en/em/minus dashes
    " ": " ", "…": "…",
}


def normalize(text: str) -> str:
    """Strip markdown/HTML noise and tidy whitespace, preserving PL diacritics
    and paragraph boundaries (paragraphs come back separated by a blank line)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for src, dst in _QUOTE_MAP.items():
        text = text.replace(src, dst)
    text = _FENCE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)        # keep link/alt text
    text = _URL_RE.sub("link", text)           # speak bare URLs as "link"
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    # Apply emphasis stripping twice to catch nested **_x_**.
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = text.replace("`", "")

    paragraphs = []
    for para in _PARA_SPLIT_RE.split(text):
        collapsed = _MULTISPACE_RE.sub(" ", para.replace("\n", " ")).strip()
        if collapsed:
            paragraphs.append(collapsed)
    return "\n\n".join(paragraphs)


def make_title(normalized: str, limit: int = 60) -> str:
    flat = normalized.replace("\n", " ").strip()
    return flat[:limit].strip() if flat else "(untitled)"


# --------------------------------------------------------------------------- #
# 2. Language detection (lingua, restricted to PL/EN)
# --------------------------------------------------------------------------- #
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from lingua import Language, LanguageDetectorBuilder

        _detector = LanguageDetectorBuilder.from_languages(
            Language.POLISH, Language.ENGLISH
        ).build()
    return _detector


def detect_language(text: str) -> str:
    """Return 'pl' or 'en' from the first ~2000 chars; defaults to 'en'."""
    from lingua import Language

    sample = text[:2000].strip()
    if not sample:
        return "en"
    lang = _get_detector().detect_language_of(sample)
    return "pl" if lang == Language.POLISH else "en"


# --------------------------------------------------------------------------- #
# 3. Chunking
# --------------------------------------------------------------------------- #
# Abbreviations whose trailing period should NOT end a sentence (PL + EN).
_ABBREV = {
    "np", "tj", "tzn", "itd", "itp", "ok", "godz", "ul", "nr", "r", "w", "g",
    "m", "mln", "mld", "tys", "prof", "dr", "hab", "inż", "mgr", "św", "tzw",
    "cd", "pkt", "str", "zob", "por", "wg", "ww", "dot", "gen", "płk", "ks",
    "mr", "mrs", "ms", "st", "vs", "etc", "eg", "ie", "fig", "no", "vol", "pp",
}
_SENT_END_RE = re.compile(r"([.!?…]+)([\"')\]]*)(\s+)")
_LAST_WORD_RE = re.compile(r"(\w+)$", re.UNICODE)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        if m.group(1) == ".":
            prefix = text[start:m.start()]
            lw = _LAST_WORD_RE.search(prefix)
            word = lw.group(1) if lw else ""
            # Skip boundaries that are really abbreviations, initials, decimals,
            # or numbered-list markers ("1. ").
            if word.lower() in _ABBREV or len(word) == 1 or word.isdigit():
                continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _hard_split(s: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for word in s.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur = f"{cur} {word}"
        else:
            out.append(cur)
            cur = word
        if len(cur) > max_chars:  # single oversized token
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _split_long_sentence(s: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for part in _CLAUSE_SPLIT_RE.split(s):
        if len(part) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(part, max_chars))
            continue
        if not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= max_chars:
            cur = f"{cur} {part}"
        else:
            out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_split_long_sentence(s, max_chars))
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def chunk_text(normalized: str, max_chars: int) -> list[Chunk]:
    """Split normalized text into <=max_chars chunks, never mid-sentence unless a
    single sentence is too long. Marks the last chunk of each paragraph."""
    chunks: list[Chunk] = []
    for para in normalized.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        packed = _pack(split_sentences(para), max_chars)
        for i, piece in enumerate(packed):
            chunks.append(Chunk(text=piece, para_end=(i == len(packed) - 1)))
    return chunks


# --------------------------------------------------------------------------- #
# 4. Concatenate
# --------------------------------------------------------------------------- #
def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio
    n_out = int(round(audio.size * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def concat_audio(
    segments: list[np.ndarray], para_flags: list[bool], sample_rate: int
) -> np.ndarray:
    """Join segments with sentence/paragraph silence gaps (no trailing gap)."""
    gap_sentence = np.zeros(int(sample_rate * GAP_SENTENCE_SEC), dtype=np.float32)
    gap_para = np.zeros(int(sample_rate * GAP_PARAGRAPH_SEC), dtype=np.float32)
    out: list[np.ndarray] = []
    last = len(segments) - 1
    for i, seg in enumerate(segments):
        out.append(seg.astype(np.float32, copy=False))
        if i != last:
            out.append(gap_para if para_flags[i] else gap_sentence)
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# 5. Encode to M4A via ffmpeg (raw f32le on stdin → AAC)
# --------------------------------------------------------------------------- #
def encode_to_m4a(audio: np.ndarray, sample_rate: int, out_path: str) -> float:
    """Encode mono float32 → AAC .m4a. Returns duration in seconds."""
    audio = np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype="<f4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, input=audio.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed: " + proc.stderr.decode("utf-8", "replace")[:500]
        )
    return float(audio.size) / float(sample_rate)
