#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  export PYTHONPATH="${PYTHONPATH:-src}"
  RUNNER=(python3)
fi

"${RUNNER[@]}" examples/run_task_nctc_proxy.py \
  --sample preset \
  --max-examples 3 \
  --device cpu \
  --mock-scorer
