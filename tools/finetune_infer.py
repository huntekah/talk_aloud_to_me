#!/usr/bin/env python
"""Synthesize EN + PL samples from a fine-tuned XTTS checkpoint. Runs under the xtts_engine env.

Loads the original XTTS config/vocab/speakers but the *fine-tuned* weights, then
conditions on the curated reference clips (XTTS is always speaker-conditioned).

  uv run --project xtts_engine python tools/finetune_infer.py --run <run_dir> [--ckpt best_model.pth]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

PROJ = Path(__file__).resolve().parent.parent
CACHE = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"


def pick_ckpt(run_dir: Path, name: str | None) -> str:
    if name:
        p = run_dir / name
        if p.exists():
            return str(p)
    if (run_dir / "best_model.pth").exists():
        return str(run_dir / "best_model.pth")
    cks = sorted(glob.glob(str(run_dir / "checkpoint_*.pth")),
                 key=lambda s: int(s.split("checkpoint_")[1].split(".")[0]))
    if not cks:
        raise SystemExit(f"no checkpoint found in {run_dir}")
    return cks[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trainer output dir (xtts_finetune-...)")
    ap.add_argument("--ckpt", default=None, help="checkpoint filename (default latest/best)")
    ap.add_argument("--ref-dir", default=str(PROJ / "voices" / "my-voice"))
    ap.add_argument("--out", default=str(PROJ / "preview"))
    ap.add_argument("--tag", default="ft")
    args = ap.parse_args()

    ckpt = pick_ckpt(Path(args.run), args.ckpt)
    print(f"checkpoint: {ckpt}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    config = XttsConfig()
    config.load_json(str(CACHE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=ckpt, vocab_path=str(CACHE / "vocab.json"),
        speaker_file_path=str(CACHE / "speakers_xtts.pth"), use_deepspeed=False,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"loaded fine-tuned XTTS on {device}", flush=True)

    clips = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=clips)

    samples = {
        "en": "Hey, this is my own voice, fine-tuned locally on my Mac. If this sounds like me, we're in business.",
        "pl": "Cześć, to jest mój własny głos, dotrenowany lokalnie na moim Macu. Jeśli to brzmi jak ja, to mamy sukces.",
    }
    sr = int(getattr(config.audio, "output_sample_rate", 24000))
    for lang, text in samples.items():
        out = model.inference(text, lang, gpt_cond_latent, speaker_embedding, temperature=0.7)
        wav = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
        path = os.path.join(args.out, f"my-voice-{args.tag}-{lang}.wav")
        sf.write(path, wav, sr)
        print(f"  {lang}: {len(wav)/sr:.1f}s peak={float(abs(wav).max()):.3f} -> {path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
