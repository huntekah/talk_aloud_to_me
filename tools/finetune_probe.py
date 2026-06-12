#!/usr/bin/env python
"""Feasibility probe: how fast does XTTS-v2 GPT fine-tuning step on this Mac?

Inlines coqui's xtts_ft_demo training setup (so we can force num_loader_workers=0
and print_step=1) and times the first few real train_steps, then aborts. Base
weights are linked from the existing inference cache; only DVAE/mel_stats are
fetched. Run under .venv-xtts:

    .venv-xtts/bin/python tools/finetune_probe.py [mps|cpu]
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WANT_DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
N_TIME = 6  # stop after this many timed steps

import json

import torch
from huggingface_hub import hf_hub_download
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

PROJ = Path(__file__).resolve().parent.parent
VOICE = PROJ / "voices" / "my-voice"
WORK = PROJ / "data" / "finetune" / "probe"
WORK.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"

# ---- base weights -----------------------------------------------------------
XTTS_CKPT = str(CACHE / "model.pth")
TOKENIZER = str(CACHE / "vocab.json")
print("fetching dvae.pth / mel_stats.pth (small)…", flush=True)
DVAE = hf_hub_download("coqui/XTTS-v2", "dvae.pth")
MEL = hf_hub_download("coqui/XTTS-v2", "mel_stats.pth")

# ---- tiny dataset from the curated English clips ----------------------------
man = json.loads((VOICE / "manifest.json").read_text())
rows = [m for m in man if m["lang"] == "en"]
def clean(t): return t.replace("|", " ").replace("\n", " ").strip()
def write_csv(path, items):
    with open(path, "w") as f:
        f.write("audio_file|text|speaker_name\n")
        for m in items:
            f.write(f"{(VOICE / m['file'])}|{clean(m['text'])}|myvoice\n")
write_csv(WORK / "train.csv", rows[:-2])
write_csv(WORK / "eval.csv", rows[-2:])
print(f"train={len(rows)-2} eval=2 english clips", flush=True)

# ---- config (mirrors xtts_ft_demo, but loader_workers=0 / print_step=1) ------
model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150,
    debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
    mel_norm_file=MEL, dvae_checkpoint=DVAE, xtts_checkpoint=XTTS_CKPT,
    tokenizer_file=TOKENIZER, gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
)
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(
    epochs=1, output_path=str(WORK), model_args=model_args, run_name="probe",
    audio=audio_config, batch_size=2, batch_group_size=0, eval_batch_size=2,
    num_loader_workers=0, print_step=1, plot_step=100000, log_model_step=100000,
    save_step=100000, save_checkpoints=False, dashboard_logger="tensorboard",
    optimizer="AdamW", optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06, lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [50000], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[],
)
model = GPTTrainer.init_from_config(config)
train_samples, eval_samples = load_tts_samples([BaseDatasetConfig(
    formatter="coqui", dataset_name="probe", path=str(VOICE),
    meta_file_train=str(WORK / "train.csv"), meta_file_val=str(WORK / "eval.csv"),
    language="en")], eval_split=True)

# ---- force device (trainer is cuda/cpu-only; patch get_cuda for mps) ---------
if WANT_DEVICE == "mps" and torch.backends.mps.is_available():
    import trainer.generic_utils as gu
    gu.get_cuda = lambda: (False, torch.device("mps"))   # type: ignore
    import trainer.trainer as tt
    if hasattr(tt, "get_cuda"):
        tt.get_cuda = gu.get_cuda
    print("PATCHED device -> mps", flush=True)

# ---- time the first few train_steps, then abort -----------------------------
times = []
_orig = GPTTrainer.train_step
class _Stop(Exception): ...
def timed(self, *a, **k):
    t0 = time.time(); out = _orig(self, *a, **k); dt = time.time() - t0
    times.append(dt); print(f"[probe] step {len(times)}: {dt:.2f}s  (device={next(self.parameters()).device})", flush=True)
    if len(times) >= N_TIME: raise _Stop()
    return out
GPTTrainer.train_step = timed

trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=1),
    config, output_path=str(WORK), model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
print(f"trainer device: {trainer.use_cuda=}  (target {WANT_DEVICE})", flush=True)
try:
    trainer.fit()
except _Stop:
    pass
if times:
    med = sorted(times)[len(times) // 2]
    print(f"\nRESULT device={WANT_DEVICE} steps={len(times)} median={med:.2f}s/step "
          f"min={min(times):.2f}s", flush=True)
else:
    print("\nRESULT: no steps completed", flush=True)
