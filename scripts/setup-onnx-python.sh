#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
if [[ ! -x .venvs/onnx/bin/python ]]; then
    uv venv --python 3.10.20 .venvs/onnx
fi
args=(--project environments/onnx)
[[ ! -f environments/onnx/uv.lock ]] || args+=(--locked)
UV_PROJECT_ENVIRONMENT="$WORKSPACE_ROOT/.venvs/onnx" uv sync "${args[@]}"
uv pip freeze --python .venvs/onnx/bin/python > logs/onnx-packages.txt
