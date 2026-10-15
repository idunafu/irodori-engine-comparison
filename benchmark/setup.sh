#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../scripts/environment.sh"
cd "$WORKSPACE_ROOT"
uv sync --project benchmark --python 3.10.20 --frozen --no-install-project
uv pip install --python .venvs/Irodori-TTS/bin/python --no-deps gguf==0.17.1
benchmark/.venv/bin/python benchmark/prepare.py
cmake --build artifacts/build/audio.cpp/cuda --target audiocpp_gguf --parallel 8
.venvs/Irodori-TTS/bin/python benchmark/convert.py
for engine in audio.cpp speech.cpp; do
    for backend in cuda vulkan; do
        benchmark/.venv/bin/python benchmark/build.py "$engine" --backend "$backend"
    done
done
for engine in sttts-gui Irodori-TTS-burn irodori-onnx; do
    benchmark/.venv/bin/python benchmark/build.py "$engine"
done
