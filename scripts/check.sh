#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
.venvs/Irodori-TTS/bin/python scripts/check.py "$@"
if [[ $# == 0 || " $* " == *" irodori-onnx "* ]]; then
    bash scripts/check-onnx.sh
fi
