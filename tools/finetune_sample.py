#!/usr/bin/env python
"""Generate a longer (~1 min) audition sample from a fine-tuned XTTS checkpoint.

Chunks multi-paragraph text exactly like the app (<=230 chars for XTTS, with
paragraph-aware silences) and stitches the pieces, so it represents real app
output rather than a single short utterance. Run under .venv-xtts:

  PYTHONPATH=. .venv-xtts/bin/python tools/finetune_sample.py --run <run_dir>
"""
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

from server import pipeline as P  # reuse the app's chunking + gap logic

PROJ = Path(__file__).resolve().parent.parent
CACHE = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"

EN_TEXT = """There's something quietly satisfying about hearing your own voice read back to you, especially when a machine is doing the reading. For years this kind of thing sounded flat and mechanical, like the voice in an old phone menu. Now it can carry a little warmth, a little rhythm, and just enough imperfection to feel human.

I built this so I could listen to long articles while walking or cooking, instead of staring at a screen the whole time. The idea is simple: paste some text, pick a voice, and let it turn reading into listening. And if the voice happens to sound like me, even better.

Of course, it isn't perfect yet. Some sentences land naturally, while others feel a touch stiff, as if the words are being placed rather than spoken. But it's close, closer than I expected, and that's usually how these things go right before they get good."""

PL_TEXT = """Jest coś dziwnie przyjemnego w słuchaniu własnego głosu czytającego na głos, zwłaszcza gdy robi to maszyna. Przez wiele lat takie głosy brzmiały płasko i mechanicznie, jak nagranie z automatycznej infolinii. Teraz potrafią nieść odrobinę ciepła, trochę rytmu i akurat tyle niedoskonałości, żeby wydawały się ludzkie.

Zbudowałem to po to, żeby słuchać długich artykułów podczas spaceru albo gotowania, zamiast wpatrywać się w ekran przez cały czas. Pomysł jest prosty: wklejasz tekst, wybierasz głos i zamieniasz czytanie w słuchanie. A jeśli ten głos brzmi trochę jak ja, tym lepiej.

Oczywiście nie jest jeszcze idealnie. Niektóre zdania brzmią naturalnie, a inne wydają się odrobinę sztywne, jakby słowa były układane, a nie wypowiadane. Ale jest blisko, bliżej niż się spodziewałem, i zwykle właśnie tak to wygląda tuż przed tym, jak zaczyna działać naprawdę dobrze."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--ref-dir", default=str(PROJ / "voices" / "my-voice"))
    ap.add_argument("--out", default=str(PROJ / "preview"))
    ap.add_argument("--tag", default="long")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = XttsConfig()
    config.load_json(str(CACHE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=str(Path(args.run) / args.ckpt),
        vocab_path=str(CACHE / "vocab.json"),
        speaker_file_path=str(CACHE / "speakers_xtts.pth"), use_deepspeed=False,
    )
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    sr = int(getattr(config.audio, "output_sample_rate", 24000))

    clips = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=clips)
    print(f"loaded {args.ckpt} on {dev}; {len(clips)} reference clips; temp={args.temperature}", flush=True)

    for lang, text in (("en", EN_TEXT), ("pl", PL_TEXT)):
        chunks = P.chunk_text(P.normalize(text), 230)
        segs, flags = [], []
        for c in chunks:
            out = model.inference(c.text, lang, gpt_cond_latent, speaker_embedding,
                                  temperature=args.temperature)
            segs.append(np.asarray(out["wav"], dtype=np.float32).reshape(-1))
            flags.append(c.para_end)
        audio = P.concat_audio(segs, flags, sr)
        path = os.path.join(args.out, f"my-voice-{args.tag}-{lang}.wav")
        sf.write(path, audio, sr)
        print(f"  {lang}: {len(chunks)} chunks -> {len(audio)/sr:.0f}s -> {path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
