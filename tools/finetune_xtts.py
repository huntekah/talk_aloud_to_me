#!/usr/bin/env python
"""Bilingual XTTS-v2 GPT fine-tune on the curated meeting data (CPU).

Inlines coqui's xtts_ft_demo setup but feeds TWO dataset configs (en + pl) so a
single model learns the speaker across both languages, runs on CPU (the trainer
has no MPS path), and checkpoints frequently so intermediate models can be
auditioned. Run under the xtts_engine env:

    uv run --project xtts_engine python tools/finetune_xtts.py --dataset /path/to/voice-dataset

Checkpoints land in data/finetune/my-voice/run/<run>-<timestamp>/.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from huggingface_hub import hf_hub_download
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

PROJ = Path(__file__).resolve().parent.parent
WORK = PROJ / "data" / "finetune" / "my-voice"
OUT = WORK / "run"
CACHE = Path.home() / "Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--save-step", type=int, default=300)
    ap.add_argument("--dataset", required=True,
                    help="path to the extracted voice dataset (clips + metadata.csv)")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="resume the latest run from its last checkpoint")
    args = ap.parse_args()
    ds_root = Path(os.path.expanduser(args.dataset))
    OUT.mkdir(parents=True, exist_ok=True)

    cont_path = ""
    if args.cont:
        runs = sorted(glob.glob(str(OUT / "xtts_finetune-*")))
        cont_path = runs[-1] if runs else ""
        print(f"resuming from: {cont_path or '(no prior run found)'}", flush=True)

    xtts_ckpt = str(CACHE / "model.pth")
    tokenizer = str(CACHE / "vocab.json")
    dvae = hf_hub_download("coqui/XTTS-v2", "dvae.pth")
    mel = hf_hub_download("coqui/XTTS-v2", "mel_stats.pth")

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
        mel_norm_file=mel, dvae_checkpoint=dvae, xtts_checkpoint=xtts_ckpt,
        tokenizer_file=tokenizer, gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig(
        epochs=args.epochs, output_path=str(OUT), model_args=model_args,
        run_name="xtts_finetune", project_name="xtts_finetune", audio=audio_config,
        batch_size=args.batch, batch_group_size=0, eval_batch_size=args.batch,
        num_loader_workers=0, print_step=25, plot_step=100000, log_model_step=100000,
        save_step=args.save_step, save_n_checkpoints=4, save_checkpoints=True,
        dashboard_logger="tensorboard", print_eval=False,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18],
                             "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)

    datasets = [
        BaseDatasetConfig(formatter="coqui", dataset_name="voice_en", path=str(ds_root),
                          meta_file_train=str(WORK / "train_en.csv"),
                          meta_file_val=str(WORK / "eval_en.csv"), language="en"),
        BaseDatasetConfig(formatter="coqui", dataset_name="voice_pl", path=str(ds_root),
                          meta_file_train=str(WORK / "train_pl.csv"),
                          meta_file_val=str(WORK / "eval_pl.csv"), language="pl"),
    ]
    train_samples, eval_samples = load_tts_samples(datasets, eval_split=True)
    print(f"train={len(train_samples)} eval={len(eval_samples)} "
          f"epochs={args.epochs} batch={args.batch} grad_accum={args.grad_accum}", flush=True)

    # coqui Trainer re-parses sys.argv (esp. when resuming); hide our CLI flags.
    sys.argv = sys.argv[:1]
    trainer = Trainer(
        TrainerArgs(restore_path=None, continue_path=cont_path,
                    skip_train_epoch=False, start_with_eval=False,
                    grad_accum_steps=args.grad_accum),
        config, output_path=str(OUT), model=model,
        train_samples=train_samples, eval_samples=eval_samples,
    )
    print(f"output dir: {trainer.output_path}", flush=True)
    trainer.fit()
    print("TRAINING_DONE", trainer.output_path, flush=True)


if __name__ == "__main__":
    main()
