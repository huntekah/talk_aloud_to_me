#!/usr/bin/env python
"""Standalone XTTS-v2 worker process.

Runs under the isolated ``xtts_engine`` uv project (transformers < 5, which the
main app venv cannot have because mlx-audio needs transformers >= 5.5). The
parent ``XTTSPolishEngine`` spawns this once and keeps it alive so the ~1.8 GB
model loads a single time.

Protocol (one request per line on stdin, framed responses on the real stdout):
  request : {"cmd": "synthesize"|"load"|"ping"|"shutdown", ...}\n
  response: a JSON header line, then for audio `samples` float32 LE bytes.
            header = {"status":"ok"|"error","samples":N,"sample_rate":SR,"error":?}

stdout is reserved for this binary protocol; every library/print byte is
redirected to stderr so it can never corrupt the channel.
"""
import json
import os
import sys

# Reserve a clean binary channel for the protocol, then push fd 1 to stderr so
# coqui-tts' chatty stdout output lands in the parent's logs instead.
_proto = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
FALLBACK_SPEAKER = "Ana Florence"

_tts = None
_sample_rate = 24000


def _log(msg: str) -> None:
    print(f"[xtts-worker] {msg}", file=sys.stderr, flush=True)


def _select_device() -> str:
    forced = os.environ.get("TTS_XTTS_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load() -> None:
    global _tts, _sample_rate
    if _tts is not None:
        return
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from TTS.api import TTS

    device = _select_device()
    _log(f"loading XTTS-v2 on {device} (first run downloads ~1.8 GB)…")
    tts = TTS(XTTS_MODEL)
    try:
        tts.to(device)
    except Exception as exc:  # hardware dependent
        _log(f"could not use {device} ({exc}); falling back to cpu")
        tts.to("cpu")
    _tts = tts
    _sample_rate = int(getattr(tts.synthesizer, "output_sample_rate", 24000) or 24000)
    _log(f"ready (sample_rate={_sample_rate})")


def _synthesize(text: str, voice_kind: str, voice_ref):
    _load()
    kwargs = {"text": text, "language": "pl"}
    if voice_kind == "wav":
        kwargs["speaker_wav"] = voice_ref
    else:
        kwargs["speaker"] = voice_ref or FALLBACK_SPEAKER
    wav = _tts.tts(**kwargs)
    return np.asarray(wav, dtype="<f4").reshape(-1)


def _send_header(obj: dict) -> None:
    _proto.write((json.dumps(obj) + "\n").encode("utf-8"))
    _proto.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _send_header({"status": "error", "error": f"bad request: {exc}", "samples": 0})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "synthesize":
                arr = _synthesize(
                    req.get("text", ""), req.get("voice_kind"), req.get("voice_ref")
                )
                _send_header(
                    {"status": "ok", "samples": int(arr.size), "sample_rate": _sample_rate}
                )
                _proto.write(arr.tobytes())
                _proto.flush()
            elif cmd == "load":
                _load()
                _send_header({"status": "ok", "samples": 0, "sample_rate": _sample_rate})
            elif cmd == "ping":
                _send_header({"status": "ok", "samples": 0, "sample_rate": _sample_rate})
            elif cmd == "shutdown":
                _send_header({"status": "ok", "samples": 0})
                break
            else:
                _send_header(
                    {"status": "error", "error": f"unknown cmd {cmd!r}", "samples": 0}
                )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            _send_header({"status": "error", "error": str(exc), "samples": 0})


if __name__ == "__main__":
    main()
