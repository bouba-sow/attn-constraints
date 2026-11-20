# Common environment for the job scripts (sourced, not run), on NYU Torch and Jean Zay.
# Torch login shells carry conda entries that break the venv (old libcrypto -> hashlib), so
# scrub them before using .venv/bin/python.

unset LD_LIBRARY_PATH PYTHONPATH
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i conda | paste -sd:)"
export PATH

# Cluster: Jean Zay sets IDRPROJ in every shell; anything else is treated as NYU Torch.
if [[ -n "${IDRPROJ:-}" ]]; then
  export CLUSTER=jeanzay
  # SCRATCH is set by IDRIS (/lustre/fsn1/...). It is purged after 30 days without access:
  # copy finished runs to $STORE (see slurm/jz/archive.sh). Do not write runs to $WORK
  # (500k-inode quota).
  : "${SCRATCH:?SCRATCH must be set on Jean Zay}"
  # Compute nodes have no internet: everything must already be on disk.
  if [[ "${SLURM_JOB_PARTITION:-}" != prepost && -n "${SLURM_JOB_ID:-}" ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
  # Per-job local scratch when available, else a shared tmp on SCRATCH.
  export TMPDIR="${JOBSCRATCH:-$SCRATCH/tmp}"
else
  export CLUSTER=torch
  export SCRATCH="${SCRATCH:-/scratch/bs5813}"
  export TMPDIR="$SCRATCH/tmp"
fi
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
mkdir -p "$TMPDIR"

export DATA_ROOT="${DATA_ROOT:-$SCRATCH/attn_constraints/data}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/attn_constraints/runs}"
export PILOT_ROOT="${PILOT_ROOT:-$SCRATCH/attn_constraints/pilot}"

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

# The 50 stage-1 runs, ordered so that consecutive runs share size and seed: with
# RUNS_PER_GPU=5, each GPU trains one matched set (all attention types for one size x seed).
# Defaults: stage 1 on Torch (2l, 4l; 5 types), stage 2 on Jean Zay (125m; 5 types plus the
# hard N-gram window models N = 3, 4, 5). Override with space-separated strings, e.g.
# GRID_SIZES=125m GRID_SEEDS="1 2 3" GRID_ATTNS="vanilla window3".
if [[ "$CLUSTER" == jeanzay ]]; then
  _attns="vanilla vanilla_pos alibi forget stickbreaking window3 window4 window5" _sizes="125m"
else
  _attns="vanilla vanilla_pos alibi forget stickbreaking" _sizes="2l 4l"
fi
read -r -a GRID_ATTNS <<<"${GRID_ATTNS:-$_attns}"
read -r -a GRID_SIZES <<<"${GRID_SIZES:-$_sizes}"
read -r -a GRID_SEEDS <<<"${GRID_SEEDS:-1 2 3 4 5}"
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
  local pids=() names=() spec attn size seed name status failed=0 i
  for spec in "$@"; do
    read -r attn size seed <<<"$spec"
    name="${attn}_${size}_s${seed}"
    mkdir -p "$runs_root/$name"
    # shellcheck disable=SC2086
    "$PY" train.py --attn "$attn" --size "$size" --seed "$seed" --runs-root "$runs_root" \
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
