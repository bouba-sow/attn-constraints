# Common environment for the job scripts (sourced, not run).
# Torch login shells carry conda entries that break the venv (old libcrypto -> hashlib), so
# scrub them before using .venv/bin/python.

unset LD_LIBRARY_PATH PYTHONPATH
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i conda | paste -sd:)"
export PATH

export SCRATCH="${SCRATCH:-/scratch/bs5813}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

export DATA_ROOT="${DATA_ROOT:-$SCRATCH/attn_constraints/data}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/attn_constraints/runs}"
export PILOT_ROOT="${PILOT_ROOT:-$SCRATCH/attn_constraints/pilot}"

# Live Trackio dashboard (private HF Space, created on first use; needs HF_TOKEN with write
# access). Set TRACKIO_SPACE_ID="" to turn it off; TensorBoard logs are written regardless.
export TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID-boubasow/attn-constraints-dashboard}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-attn-constraints}"

# Job scripts run with the submit directory as cwd; submit from the repository root.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
if [[ ! -f "$REPO_DIR/train.py" ]]; then
  echo "env.sh: $REPO_DIR is not the attn-constraints repository (submit from its root)" >&2
  exit 1
fi
PY="$REPO_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "env.sh: $PY missing; run 'uv sync' in $REPO_DIR on a login node first" >&2
  exit 1
fi
cd "$REPO_DIR"

# The 80 stage-1 runs, ordered so that consecutive runs share size and seed: with
# RUNS_PER_GPU=8 (or 4), each GPU trains one matched set (or half of one): all attention types
# for one size x seed. windowN = N-gram attention with window N.
GRID_ATTNS=(vanilla vanilla_pos alibi forget stickbreaking window6 window10 window14)
GRID_SIZES=(2l 4l)
GRID_SEEDS=(1 2 3 4 5)
grid_runs() {
  local size seed attn
  for size in "${GRID_SIZES[@]}"; do
    for seed in "${GRID_SEEDS[@]}"; do
      for attn in "${GRID_ATTNS[@]}"; do
        echo "$attn $size $seed"
      done
    done
  done
}

# Launch "attn size seed" specs as background processes sharing this job's GPU (MPS), wait for
# all, and forward SIGUSR1/SIGTERM so each run saves resume.pt before the job ends.
# Usage: run_parallel <runs_root> <spec>... ; extra train.py arguments come from $TRAIN_ARGS.
run_parallel() {
  local runs_root="$1"; shift
  local pids=() names=() spec attn size seed name status failed=0 i attn_args
  for spec in "$@"; do
    read -r attn size seed <<<"$spec"
    name="${attn}_${size}_s${seed}"
    if [[ "$attn" =~ ^window([0-9]+)$ ]]; then
      attn_args=(--attn window --window-size "${BASH_REMATCH[1]}")
    else
      attn_args=(--attn "$attn")
    fi
    mkdir -p "$runs_root/$name"
    # shellcheck disable=SC2086
    "$PY" train.py "${attn_args[@]}" --size "$size" --seed "$seed" --runs-root "$runs_root" \
      --data-root "$DATA_ROOT" ${TRAIN_ARGS:-} >>"$runs_root/$name/stdout.log" 2>&1 &
    pids+=($!)
    names+=("$name")
    echo "started $name (pid $!)"
  done
  trap 'echo "forwarding stop signal"; kill -USR1 "${pids[@]}" 2>/dev/null' USR1 TERM
  for i in "${!pids[@]}"; do
    # `wait` returns early when a trapped signal arrives; keep waiting until the child exits.
    while true; do
      wait "${pids[$i]}"; status=$?
      kill -0 "${pids[$i]}" 2>/dev/null || break
    done
    echo "${names[$i]} exited with status $status"
    [[ $status -ne 0 ]] && failed=1
  done
  trap - USR1 TERM
  return $failed
}
