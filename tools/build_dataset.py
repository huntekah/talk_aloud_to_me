#!/usr/bin/env python
"""Filter the meeting extraction into XTTS fine-tune CSVs (coqui format).

Keeps clips with a real transcript, sane duration and chars/sec, and writes one
train + eval CSV per language. Audio paths stay relative to the dataset root, so
the trainer's dataloader resamples the 16 kHz clips to 22.05 kHz on the fly.

  python tools/build_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# XTTS GPTArgs limits: max_wav_length 255995 @22050 ≈ 11.6 s; max_text_length 200.
# Loosened to recover the limited transcribed English (most EN clips have no text).
MIN_DUR, MAX_DUR = 1.0, 11.5
CPS_MIN, CPS_MAX = 5.0, 26.0
MIN_TEXT = 5
MAX_TEXT = 190
EVAL_PER_LANG = 25


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="path to the extracted voice dataset (with metadata.csv)")
    ap.add_argument("--out", default="data/finetune/my-voice")
    args = ap.parse_args()

    ds = Path(os.path.expanduser(args.dataset))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_lang: dict[str, list[dict]] = {"en": [], "pl": []}
    for r in csv.DictReader(open(ds / "metadata.csv"), delimiter="|"):
        text = (r.get("text") or "").strip().replace("|", " ").replace("\n", " ")
        dur = _f(r.get("duration"))
        lang = r.get("lang", "")
        if lang not in by_lang:
            continue
        if not (MIN_TEXT <= len(text) <= MAX_TEXT):
            continue
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue
        cps = len(text) / dur if dur else 0
        if not (CPS_MIN <= cps <= CPS_MAX):
            continue
        if not (ds / r["path"]).exists():
            continue
        by_lang[lang].append({"audio_file": r["path"], "text": text})

    def write(path: Path, items: list[dict]):
        with open(path, "w") as f:
            f.write("audio_file|text|speaker_name\n")
            for it in items:
                f.write(f"{it['audio_file']}|{it['text']}|myvoice\n")

    summary = []
    for lang, items in by_lang.items():
        # Deterministic shuffle: sort by filename hash-ish (filename is time-based).
        items.sort(key=lambda d: d["audio_file"])
        items = items[::-1]  # reverse so eval isn't all from the earliest session
        eval_items = items[:EVAL_PER_LANG]
        train_items = items[EVAL_PER_LANG:]
        write(out / f"train_{lang}.csv", train_items)
        write(out / f"eval_{lang}.csv", eval_items)
        mins = sum(0 for _ in items)  # placeholder; durations not retained
        summary.append((lang, len(train_items), len(eval_items)))

    print(f"dataset root: {ds}")
    for lang, ntr, nev in summary:
        print(f"  {lang}: train={ntr}  eval={nev}")
    total = sum(ntr for _, ntr, _ in summary)
    print(f"total train clips: {total}")
    print(f"CSVs → {out}/(train|eval)_(en|pl).csv")


if __name__ == "__main__":
    main()
