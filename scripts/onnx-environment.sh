#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
export PYTHONPATH="$WORKSPACE_ROOT/engines/Irodori-TTS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
site="$WORKSPACE_ROOT/.venvs/onnx/lib/python3.10/site-packages"
nvidia_libs="$(find "$site/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
export LD_LIBRARY_PATH="$nvidia_libs:$site/onnxruntime/capi${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IRODORI_ORT_DLL="$site/onnxruntime/capi/libonnxruntime.so.1.23.2"
export IRODORI_MODEL_DIR="$WORKSPACE_ROOT/artifacts/onnx/export/ref"
export IRODORI_TOKENIZER_DIR="$WORKSPACE_ROOT/models/mf/tokenizer"
export IRODORI_CUDA=0
export IRODORI_FP16=1
# TF32 in the FP32 codec encoder fails the upstream voice-registration parity
# check on this GPU. Keep that encoder's computation in full FP32.
export NVIDIA_TF32_OVERRIDE=0
