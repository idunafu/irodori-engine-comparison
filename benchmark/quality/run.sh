#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../../scripts/environment.sh"
cd "$WORKSPACE_ROOT"
exec .venvs/quality/bin/python "$WORKSPACE_ROOT/benchmark/quality/run.py" "$@"
