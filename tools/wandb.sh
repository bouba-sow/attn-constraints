#!/bin/bash
# W&B dashboard for the runs, from a Jean Zay LOGIN node (see tools/wandb_runs.py).
# wandb is NOT a training dependency: it runs from a pinned, cached uv environment, so the
# training venv and uv.lock are untouched. API key: ~/.netrc (machine api.wandb.ai, mode 600).
#   tools/wandb.sh [--watch 600] [--runs-root DIR] [--project NAME] [--entity NAME]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCRATCH:?SCRATCH must be set}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}" TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb" WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"
mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
exec uv run --no-project --python 3.12 --with "wandb==0.31.1" python "$REPO_DIR/tools/wandb_runs.py" "$@"
