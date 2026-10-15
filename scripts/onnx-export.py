#!/usr/bin/env python3
"""Run the pinned upstream exporter with local paths and explicit CUDA checks."""
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/onnx/export"
STAGE.mkdir(parents=True, exist_ok=True)
for source in (ROOT / "engines/irodori-onnx/export").glob("*.py"):
    shutil.copy2(source, STAGE / source.name)
sys.path.insert(0, str(STAGE))

import torch
import common

torch.set_num_threads(8)
common.CKPT_GLOB = str(ROOT / "models/mf/model.safetensors")
common.VOICES = str(ROOT / "artifacts/voices")
common.REF_NAMES = ("ref1",)
common.REF_DIR = str(STAGE / "ref")
Path(common.REF_DIR).mkdir(exist_ok=True)

def load_runtime():
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
    return InferenceRuntime.from_key(RuntimeKey(
        checkpoint=common.CKPT_GLOB, model_device="cpu", model_precision="fp32",
        codec_repo=str(ROOT / "models/codec/weights.pth"),
        codec_device="cpu", codec_precision="fp32"))

common.load_runtime = load_runtime

# Upstream probes catch provider failures. Fail explicitly before a CPU fallback
# can be recorded as CUDA success, and avoid exhaustive per-shape cuDNN searches.
import onnxruntime as ort
original_session = ort.InferenceSession

def checked_session(*args, **kwargs):
    providers = kwargs.get("providers", [])
    wants_cuda = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers)
    if wants_cuda:
        kwargs["providers"] = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC",
                                       "arena_extend_strategy": "kSameAsRequested"})
            if (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" else p
            for p in providers]
    session = original_session(*args, **kwargs)
    if wants_cuda and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("Requested CUDA EP is unavailable; CPU fallback is not a CUDA validation")
    return session

ort.InferenceSession = checked_session
stage = sys.argv[1]
sys.argv = [str(STAGE / stage), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
