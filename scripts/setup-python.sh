#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
for engine in Irodori-TTS Irodori-TTS-Optimiz; do
    if [[ ! -x ".venvs/$engine/bin/python" ]]; then
        uv venv --python 3.10.20 ".venvs/$engine"
    fi
    UV_PROJECT_ENVIRONMENT="$WORKSPACE_ROOT/.venvs/$engine" \
        uv sync --project "engines/$engine" --frozen --extra cu128
    uv pip freeze --python ".venvs/$engine/bin/python" > "logs/$engine-packages.txt"
done
