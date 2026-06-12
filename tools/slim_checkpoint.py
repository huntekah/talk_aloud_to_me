#!/usr/bin/env python
"""Strip a trainer checkpoint to inference-only weights and verify it still works.

A GPTTrainer checkpoint (~5.2 GB) carries optimizer/scaler state needed only for
resuming. Inference needs just the model weights (~1.8 GB). This drops the rest,
saves a slim file, then loads it and synthesizes a sentence to prove it's valid
BEFORE you delete the originals. Run under the xtts_engine env:

  uv run --project xtts_engine python tools/slim_checkpoint.py --ckpt <full.pth> --out <slim.pth>
"""
import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

PROJ = Path(__file__).resolve().parent.parent
CACHE = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="full trainer checkpoint (.pth)")
    ap.add_argument("--out", required=True, help="slim inference-only output (.pth)")
    ap.add_argument("--ref-dir", default=str(PROJ / "voices" / "my-voice"))
    args = ap.parse_args()

    # weights_only=False is required (trainer checkpoint holds config objects);
    # the checkpoint is one we produced locally, not untrusted input.
    full = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec B614
    keep = {"model": full["model"]} if "model" in full else {"model": full}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, args.out)
    in_gb = os.path.getsize(args.ckpt) / 2**30
    out_gb = os.path.getsize(args.out) / 2**30
    print(f"slimmed {in_gb:.1f} GB -> {out_gb:.1f} GB  ({args.out})", flush=True)

    # ---- verify the slim file actually loads + synthesizes -----------------
    config = XttsConfig()
    config.load_json(str(CACHE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=args.out, vocab_path=str(CACHE / "vocab.json"),
        speaker_file_path=str(CACHE / "speakers_xtts.pth"), use_deepspeed=False,
    )
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    clips = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=clips)
    out = model.inference("Quick check that the slimmed model still works.", "en",
                          gpt_cond_latent, speaker_embedding, temperature=0.7)
    wav = np.asarray(out["wav"], dtype=np.float32)
    ok = wav.size > 1000 and float(np.abs(wav).max()) > 0.01
    print(f"VERIFY {'OK' if ok else 'FAILED'}: {wav.size/24000:.1f}s peak={float(np.abs(wav).max()):.3f}",
          flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
