#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/environment.sh"
cd "$WORKSPACE_ROOT"
exec .venvs/Irodori-TTS/bin/python scripts/run.py "$@"
