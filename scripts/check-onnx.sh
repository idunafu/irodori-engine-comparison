#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/onnx-environment.sh"
cd "$WORKSPACE_ROOT"
exec 9>artifacts/gpu.lock
flock 9
export IRODORI_FP16=0
unset IRODORI_CUDA IRODORI_TOL
cargo test --manifest-path engines/irodori-onnx/Cargo.toml --release --locked \
    --target-dir artifacts/build/irodori-onnx -p irodori-tts \
    -- --test-threads=1 --nocapture > logs/onnx-rust-tests-fp32.log 2>&1
export IRODORI_FP16=1 IRODORI_CUDA=0 IRODORI_TOL=0.05
cargo test --manifest-path engines/irodori-onnx/Cargo.toml --release --locked \
    --target-dir artifacts/build/irodori-onnx -p irodori-tts \
    -- --test-threads=1 --nocapture > logs/onnx-rust-tests-fp16-cuda.log 2>&1
echo 'ONNX golden tests passed: FP32 CPU and FP16 CUDA (TF32 disabled).'
