#!/bin/bash
# Jean Zay pilot (same body as slurm/pilot.sh): every grid attention type (8) at 125m, seed 1, 190 steps
# (50M tokens), one run at a time on one H100: loss goes down, no NaNs, tokens/s, peak memory.
# No MPS on Jean Zay, so only MODE=single applies. Submit from the repository root:
#   sbatch slurm/jz/pilot.sh
#   TRAIN_ARGS="--micro-batch 32" sbatch slurm/jz/pilot.sh
#
#SBATCH --job-name=attnc_pilot
#SBATCH --account=ldh@h100
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
export MODE=single
exec bash "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm/pilot.sh"
