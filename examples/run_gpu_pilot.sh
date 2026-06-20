#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python)
else
  export PYTHONPATH="${PYTHONPATH:-src}"
  RUNNER=(python3)
fi

"${RUNNER[@]}" examples/run_task_nctc_proxy.py \
  --model "${MODEL}" \
  --device cuda \
  --sample preset \
  --max-examples "${MAX_EXAMPLES}"
