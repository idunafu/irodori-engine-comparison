#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/onnx-environment.sh"
cd "$WORKSPACE_ROOT"
mkdir -p artifacts/codec
exec 9>artifacts/gpu.lock
flock 9
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
    .venvs/Irodori-TTS/bin/python engines/Irodori-TTS-burn/scripts/convert_dacvae_weights.py \
    --pth models/codec/weights.pth --output artifacts/codec/dacvae.safetensors
.venvs/onnx/bin/python scripts/prepare-reference.py
for stage in 01_make_step_reference.py 02_export_dit_step.py 03_export_rest.py 04_export_codec_encoder.py; do
    .venvs/onnx/bin/python -u scripts/onnx-export.py "$stage" > "logs/onnx-$stage.log" 2>&1
done
.venvs/onnx/bin/python scripts/onnx-export.py 05_e2e_onnx.py cpu > logs/onnx-parity-fp32.log 2>&1
.venvs/onnx/bin/python scripts/onnx-export.py 06_fp16.py \
    dit_step2 speaker_encoder text_encoder caption_encoder duration codec_decoder > logs/onnx-fp16.log 2>&1
ONNX_SUFFIX=_fp16 .venvs/onnx/bin/python scripts/onnx-export.py 05_e2e_onnx.py cuda > logs/onnx-parity-fp16.log 2>&1
