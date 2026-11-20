#!/bin/bash
# One GPU per array task; the task trains RUNS_PER_GPU runs of the grid as parallel processes
# under the job's MPS server (--comment gpu_mps=yes). Relaunching is safe: finished runs are
# skipped (DONE) and unfinished ones resume from resume.pt.
#
# Submit from the repository root (RUNS_PER_GPU from the pilot; 80 runs -> 10 tasks at 8, 20 at 4):
#   RUNS_PER_GPU=8 sbatch --array=0-9 slurm/train_gpu.sh
# Preemptible pool (more GPUs, jobs may be killed and requeued; runs resume):
#   RUNS_PER_GPU=8 sbatch --array=0-9 \
#     --comment="preemption=yes;preemption_partitions_only=yes;requeue=true;gpu_mps=yes" slurm/train_gpu.sh
# Explicit runs instead of the grid:
#   RUNS="forget 4l 1,window10 4l 1" sbatch slurm/train_gpu.sh
#
#SBATCH --job-name=attnc_train
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --comment=gpu_mps=yes
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

if [[ -n "${RUNS:-}" ]]; then
  IFS=',' read -r -a specs <<<"$RUNS"
else
  RUNS_PER_GPU="${RUNS_PER_GPU:?set RUNS_PER_GPU (from the pilot) or RUNS}"
  mapfile -t all < <(grid_runs)
  task="${SLURM_ARRAY_TASK_ID:?submit as an array, or set RUNS}"
  specs=("${all[@]:$((task * RUNS_PER_GPU)):$RUNS_PER_GPU}")
fi
if [[ ${#specs[@]} -eq 0 ]]; then
  echo "no runs for this task"; exit 0
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} restarts=${SLURM_RESTART_COUNT:-0}"
echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset: MPS not active>}"
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
run_parallel "$RUNS_ROOT" "${specs[@]}"
