#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../scripts/environment.sh"
cd "$WORKSPACE_ROOT"
benchmark/.venv/bin/python benchmark/telemetry.py &
BENCH_TELEMETRY_PID=$!
trap 'kill "$BENCH_TELEMETRY_PID" 2>/dev/null || true; wait "$BENCH_TELEMETRY_PID" 2>/dev/null || true' EXIT
benchmark/.venv/bin/python benchmark/run.py "$@"
