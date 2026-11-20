#!/bin/bash
# Trackio dashboard for the runs, on a Jean Zay LOGIN node (see tools/trackio_runs.py).
# trackio is NOT a training dependency: it runs from a pinned, cached uv environment, so the
# training venv and uv.lock are untouched.
#   tools/trackio.sh import [--watch 600] [--runs-root DIR] [--project NAME]
#   tools/trackio.sh show   [--port 7860] [--project NAME]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCRATCH:?SCRATCH must be set}"
# Small (one SQLite file per project); kept on $WORK so the $SCRATCH purge cannot remove it.
export TRACKIO_DIR="${TRACKIO_DIR:-$REPO_DIR/trackio}"
# Rollback journal instead of trackio's default WAL: safe when import and show run on
# different login nodes (Lustre is mounted with flock; WAL needs a single host).
export TRACKIO_SQLITE_JOURNAL_MODE=delete
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}" TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$TRACKIO_DIR" "$TMPDIR"
exec uv run --no-project --python 3.12 --with "trackio==0.45.1" \
  python "$REPO_DIR/tools/trackio_runs.py" "$@"
