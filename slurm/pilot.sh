#!/bin/bash
# Pilot runs before the grid (README "Pilot"). Submit from the repository root.
#
# MODE=single (default): every grid attention type x size, seed 1, 190 steps (50M tokens), one at a
#   time on one GPU: loss goes down, no NaNs, tokens/s and peak memory per configuration.
#     sbatch slurm/pilot.sh
# MODE=mps: the slowest configuration (forget 4l) as N = 1, 2, 4, 8 parallel processes on one
#   GPU under MPS; total tokens/s for each N picks RUNS_PER_GPU.
#     MODE=mps sbatch slurm/pilot.sh
#
#SBATCH --job-name=attnc_pilot
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --comment=gpu_mps=yes
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
MODE="${MODE:-single}"
# Keep pilot runs out of the main dashboard project.
export TRACKIO_PROJECT="${TRACKIO_PROJECT_PILOT:-attn-constraints-pilot}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} mode=$MODE"
echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset: MPS not active>}"
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader

summarize() {  # print run name, steady tokens/s, peak memory, final val loss
  "$PY" - "$@" <<'EOF'
import json, sys
from pathlib import Path
for d in sys.argv[1:]:
    s = Path(d, "summary.json")
    if not s.exists():
        print(f"{Path(d).name:40s} NO SUMMARY (see {d}/stdout.log)"); continue
    j = json.loads(s.read_text())
    print(f"{j['run_name']:40s} tok/s={j['tokens_per_s_steady']} peak_gb={j['peak_mem_gb']} val_loss={j['final_val_loss']:.4f}")
EOF
}

if [[ "$MODE" == "single" ]]; then
  root="$PILOT_ROOT/single_${SLURM_JOB_ID:-local}"
  export TRAIN_ARGS="--max-steps 190 --eval-every 1000 --eval-max-windows 1024 --ckpt-every 100000 --no-save-init --log-every 10 ${TRAIN_ARGS:-}"
  dirs=()
  for size in 2l 4l; do
    for attn in "${GRID_ATTNS[@]}"; do
      run_parallel "$root" "$attn $size 1"
      dirs+=("$root/${attn}_${size}_s1")
    done
  done
  summarize "${dirs[@]}"
elif [[ "$MODE" == "mps" ]]; then
  for n in 1 2 4 8; do
    root="$PILOT_ROOT/mps_${SLURM_JOB_ID:-local}/n$n"
    # Distinct seeds only so the run names differ; throughput does not depend on the seed.
    specs=()
    for ((k = 1; k <= n; k++)); do specs+=("forget 4l $k"); done
    export TRAIN_ARGS="--max-steps 40 --eval-every 1000 --eval-max-windows 64 --ckpt-every 100000 --no-save-init --log-every 5"
    run_parallel "$root" "${specs[@]}"
    echo "== N=$n"
    summarize "$root"/*
    "$PY" -c "import json,glob; print('total tok/s:', sum(json.load(open(f))['tokens_per_s_steady'] or 0 for f in glob.glob('$root/*/summary.json')))"
  done
else
  echo "unknown MODE=$MODE"; exit 2
fi
