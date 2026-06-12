#!/usr/bin/env python
"""Curate the cleanest reference clips for an XTTS few-shot voice clone.

Reads the meeting-extracted dataset (metadata.csv + meetings_report.csv), keeps
clips that look like clean, well-transcribed single-speaker speech, prefers
clips from the cleanest meetings, balances PL/EN and limits per-meeting picks
for variety, then resamples the winners to 24 kHz mono into an output voice dir.

Usage:
  python tools/curate_voice.py \
      --dataset /path/to/voice-dataset \
      --out voices/my-voice --n-en 10 --n-pl 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

CPS_MIN, CPS_MAX = 9.0, 20.0  # chars/sec window for natural, well-aligned speech
PER_MEETING_CAP = 3            # variety: don't take everything from one meeting


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_meeting_cleanliness(report_csv: Path) -> dict[str, float]:
    clean = {}
    if report_csv.exists():
        for r in csv.DictReader(open(report_csv)):
            clean[r.get("meeting", "")] = _f(r.get("clean_min"), 0.0)
    return clean


def candidates(meta_csv: Path, clean: dict, min_dur: float, max_dur: float):
    out = []
    for r in csv.DictReader(open(meta_csv), delimiter="|"):
        text = (r.get("text") or "").strip()
        dur = _f(r.get("duration"))
        if not text or len(text) < 30:
            continue
        if not (min_dur <= dur <= max_dur):
            continue
        cps = len(text) / dur if dur else 0
        if not (CPS_MIN <= cps <= CPS_MAX):
            continue
        out.append({
            "path": r["path"],
            "text": text,
            "lang": r.get("lang", ""),
            "duration": dur,
            "meeting": r.get("meeting", ""),
            "clean": clean.get(r.get("meeting", ""), 0.0),
        })
    return out


def select(cands, n_en, n_pl):
    # Cleanest meeting first, then longest clip.
    cands.sort(key=lambda c: (c["clean"], c["duration"]), reverse=True)
    picked, per_meeting = [], {}
    want = {"en": n_en, "pl": n_pl}
    for c in cands:
        lang = c["lang"]
        if want.get(lang, 0) <= 0:
            continue
        if per_meeting.get(c["meeting"], 0) >= PER_MEETING_CAP:
            continue
        picked.append(c)
        per_meeting[c["meeting"]] = per_meeting.get(c["meeting"], 0) + 1
        want[lang] -= 1
        if want["en"] <= 0 and want["pl"] <= 0:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="path to the extracted voice dataset (with metadata.csv)")
    ap.add_argument("--out", default="voices/my-voice")
    ap.add_argument("--n-en", type=int, default=10)
    ap.add_argument("--n-pl", type=int, default=8)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    args = ap.parse_args()

    ds = Path(os.path.expanduser(args.dataset))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    clean = load_meeting_cleanliness(ds / "meetings_report.csv")
    cands = candidates(ds / "metadata.csv", clean, args.min_dur, args.max_dur)
    picked = select(cands, args.n_en, args.n_pl)

    manifest = []
    print(f"Selected {len(picked)} clips "
          f"(en={sum(c['lang']=='en' for c in picked)}, "
          f"pl={sum(c['lang']=='pl' for c in picked)}):")
    for c in picked:
        src = ds / c["path"]
        dst = out / Path(c["path"]).name
        # Resample to 24 kHz mono (XTTS reference SR); 16 kHz source upsamples.
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), "-ar", "24000", "-ac", "1", str(dst)],
            check=True,
        )
        manifest.append({"file": dst.name, "lang": c["lang"],
                         "duration": round(c["duration"], 2), "text": c["text"]})
        print(f"  [{c['lang']}|{c['duration']:4.1f}s|clean={c['clean']:.0f}] "
              f"{dst.name}  {c['text'][:54]}")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    total = sum(c["duration"] for c in picked)
    print(f"\nWrote {len(picked)} clips ({total:.0f}s of reference) → {out}")
    print(f"Manifest: {out/'manifest.json'}")


if __name__ == "__main__":
    main()
