#!/usr/bin/env python
"""Quick XTTS few-shot clone preview. Runs under the xtts_engine env.

Computes XTTS conditioning latents from a folder of reference clips, then
synthesizes one English + one Polish sample so you can judge the timbre match
before any app integration.

  uv run --project xtts_engine python tools/clone_preview.py voices/my-voice preview
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

voice_dir = sys.argv[1] if len(sys.argv) > 1 else "voices/my-voice"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "preview"
os.makedirs(out_dir, exist_ok=True)

clips = sorted(glob.glob(os.path.join(voice_dir, "*.wav")))
print(f"reference clips: {len(clips)}", flush=True)

import torch  # noqa: E402
from TTS.api import TTS  # noqa: E402

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading XTTS on {device}…", flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)
model = tts.synthesizer.tts_model
sr = int(getattr(getattr(model, "config", None).audio, "output_sample_rate", 24000))

print("computing speaker latents from references…", flush=True)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=clips)

samples = {
    "en": "Hey, this is my own voice, cloned locally on my Mac. If this sounds like me, we're in business.",
    "pl": "Cześć, to jest mój własny głos, sklonowany lokalnie na moim Macu. Jeśli to brzmi jak ja, to mamy sukces.",
}
for lang, text in samples.items():
    out = model.inference(text, lang, gpt_cond_latent, speaker_embedding, temperature=0.7)
    wav = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
    path = os.path.join(out_dir, f"my-voice-{lang}.wav")
    sf.write(path, wav, sr)
    print(f"  {lang}: {len(wav)/sr:.1f}s  peak={float(abs(wav).max()):.3f}  -> {path}", flush=True)

print("done", flush=True)
