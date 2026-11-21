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
# Several runs share a GPU under MPS: less fragmentation in each process's cached memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Live wandb dashboard. The API key comes from ~/.netrc (`wandb login`), never from this repo.
# Set WANDB_PROJECT="" to turn it off; TensorBoard logs are written regardless.
export WANDB_PROJECT="${WANDB_PROJECT-attn-constraints}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/.cache/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$SCRATCH/.cache/wandb-data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$SCRATCH/.config/wandb}"

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

# Grid, ordered so that consecutive runs share size and seed (with RUNS_PER_GPU = number of
# attention types, each GPU trains one matched set: all attention types for one size x seed).
# Defaults: on Torch the 80 stage-1 runs (2l, 4l; 5 types plus N-gram windows N = 6, 10, 14:
# RUNS_PER_GPU=8 or 4); on Jean Zay the 125m stage 2 (5 types plus windows N = 3, 4, 5).
# Override with space-separated strings, e.g. the noisy-attention grid on Torch:
# GRID_ATTNS="vanilla noisy_l0 noisy_l0.001 noisy_l0.01 noisy_l0.1 noisy_l1" GRID_SIZES=125m GRID_SEEDS="1 2 3".
if [[ "$CLUSTER" == jeanzay ]]; then
  _attns="vanilla vanilla_pos alibi forget stickbreaking window3 window4 window5" _sizes="125m"
else
  _attns="vanilla vanilla_pos alibi forget stickbreaking window6 window10 window14" _sizes="2l 4l"
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
# all, and forward SIGUSR1/SIGTERM so each run saves resume.pt before the job ends. A run that
# crashes (exit status other than 0, 98 = trained elsewhere, 99 = stopped by a signal) is
# relaunched from its last resume.pt, up to MAX_RETRIES times.
# Usage: run_parallel <runs_root> <spec>... ; extra train.py arguments come from $TRAIN_ARGS.
run_one() {  # supervisor for one run; runs in a background subshell
  local runs_root="$1" name="$2"; shift 2
  local child status attempt=0 stop=0
  trap 'stop=1; kill -USR1 "$child" 2>/dev/null' USR1 TERM
  while true; do
    # shellcheck disable=SC2086
    "$PY" train.py "$@" --runs-root "$runs_root" --data-root "$DATA_ROOT" ${TRAIN_ARGS:-} \
      >>"$runs_root/$name/stdout.log" 2>&1 &
    child=$!
    while true; do
      wait "$child"; status=$?
      kill -0 "$child" 2>/dev/null || break
    done
    if [[ $status -eq 0 || $status -eq 98 || $status -eq 99 || $stop -eq 1 ]] \
        || (( attempt >= ${MAX_RETRIES:-2} )); then
      exit "$status"
    fi
    attempt=$((attempt + 1))
    echo "$name crashed with status $status; relaunching from its last save (retry $attempt)"
    sleep 30
  done
}

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
    run_one "$runs_root" "$name" "${attn_args[@]}" --size "$size" --seed "$seed" &
    pids+=($!)
    names+=("$name")
    echo "started $name (supervisor pid $!)"
  done
  trap 'echo "forwarding stop signal"; kill -USR1 "${pids[@]}" 2>/dev/null' USR1 TERM
  for i in "${!pids[@]}"; do
    # `wait` returns early when a trapped signal arrives; keep waiting until the child exits.
    while true; do
      wait "${pids[$i]}"; status=$?
      kill -0 "${pids[$i]}" 2>/dev/null || break
    done
    echo "${names[$i]} exited with status $status"
    [[ $status -ne 0 && $status -ne 98 ]] && failed=1
  done
  trap - USR1 TERM
  return $failed
}
