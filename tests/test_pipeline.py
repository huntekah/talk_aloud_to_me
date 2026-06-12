"""Tests for the platform-independent text pipeline (no mlx/torch/models needed)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from server import pipeline as P


# --------------------------------------------------------------------------- #
# normalize / title
# --------------------------------------------------------------------------- #
def test_normalize_strips_markdown_and_urls():
    md = "# Heading\n\nThis is **bold**, *italic* and `code`. See https://example.com/x now."
    out = P.normalize(md)
    assert "#" not in out
    assert "*" not in out
    assert "`" not in out
    assert "http" not in out
    assert "link" in out  # URL replaced
    assert "bold" in out and "italic" in out


def test_normalize_preserves_paragraphs_and_diacritics():
    out = P.normalize("Pierwszy akapit: zażółć gęślą jaźń.\n\nDrugi akapit.")
    assert out.count("\n\n") == 1  # two paragraphs preserved as one blank line
    assert "zażółć gęślą jaźń" in out  # Polish diacritics untouched


def test_make_title_truncates_and_flattens():
    title = P.make_title("Line one\nline two", limit=10)
    assert "\n" not in title
    assert len(title) <= 10


# --------------------------------------------------------------------------- #
# sentence splitting
# --------------------------------------------------------------------------- #
def test_split_sentences_guards_abbreviations_and_decimals():
    s = P.split_sentences("Dr. Smith met Mr. Jones. The value was 3.14 and ok. it worked.")
    assert len(s) == 2
    assert s[0] == "Dr. Smith met Mr. Jones."


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_chunk_respects_limit():
    text = P.normalize("Aaa bbb ccc. " * 40)
    chunks = P.chunk_text(text, 60)
    assert chunks
    assert all(len(c.text) <= 60 for c in chunks)


def test_chunk_marks_last_chunk_of_each_paragraph():
    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph here."
    chunks = P.chunk_text(P.normalize(text), 230)
    para_ends = [c.para_end for c in chunks]
    assert para_ends.count(True) == 2  # one per paragraph
    assert chunks[-1].para_end is True


def test_chunk_splits_overlong_single_sentence():
    long_sentence = "word, " * 50 + "end."  # one sentence, ~300 chars, no '.' until end
    chunks = P.chunk_text(P.normalize(long_sentence), 40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 40 for c in chunks)
    assert chunks[-1].para_end is True


# --------------------------------------------------------------------------- #
# language detection (lingua)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is clearly an English sentence written for testing purposes.", "en"),
        ("To jest wyraźnie polskie zdanie napisane w celu testowania, zażółć gęślą jaźń.", "pl"),
    ],
)
def test_detect_language(text, expected):
    assert P.detect_language(text) == expected


def test_detect_language_defaults_to_en_on_empty():
    assert P.detect_language("   ") == "en"


# --------------------------------------------------------------------------- #
# concatenation
# --------------------------------------------------------------------------- #
def test_concat_audio_inserts_sentence_and_paragraph_gaps():
    sr = 24000
    seg = np.ones(12000, dtype=np.float32)
    segments = [seg, seg, seg]
    flags = [False, True, False]  # gap after seg0 = sentence, after seg1 = paragraph
    out = P.concat_audio(segments, flags, sr)
    expected = 3 * 12000 + int(sr * P.GAP_SENTENCE_SEC) + int(sr * P.GAP_PARAGRAPH_SEC)
    assert out.shape[0] == expected
    assert out.dtype == np.float32


def test_concat_audio_empty():
    out = P.concat_audio([], [], 24000)
    assert out.shape[0] == 0


# --------------------------------------------------------------------------- #
# encoding (needs ffmpeg)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_to_m4a(tmp_path):
    sr = 24000
    audio = (0.2 * np.sin(np.linspace(0, 6.28 * 220, sr))).astype(np.float32)  # 1s tone
    out = tmp_path / "out.m4a"
    duration = P.encode_to_m4a(audio, sr, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert duration == pytest.approx(1.0, abs=0.05)
