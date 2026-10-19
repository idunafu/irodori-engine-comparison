#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../../scripts/environment.sh"
cd "$WORKSPACE_ROOT"
if [[ ! -x .venvs/quality/bin/python ]]; then
    uv venv --python .venvs/Irodori-TTS/bin/python .venvs/quality
fi
uv pip install --python .venvs/quality/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128' 'torchaudio==2.10.0+cu128'
uv pip install --python .venvs/quality/bin/python 'transformers==4.57.6' 'huggingface-hub==0.36.2' 'numpy==2.2.6' 'soundfile==0.13.1' 'sherpa-onnx==1.13.8' 'rapidfuzz==3.14.5' 'jaconv==0.5.0' 'soxr==1.0.0' 'accelerate==1.12.0' 'gdown==5.2.1' 'pytest==8.4.2' 'scipy==1.15.3' 'safetensors==0.7.0'
mkdir -p artifacts/quality/assets
uv pip freeze --python .venvs/quality/bin/python > artifacts/quality/assets/requirements-resolved.txt
.venvs/quality/bin/python benchmark/quality/prepare.py
