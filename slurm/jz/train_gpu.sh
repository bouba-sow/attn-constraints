#!/bin/bash
# Jean Zay stage-2 training (same body as slurm/train_gpu.sh): 125m with the "opt" recipe in
# train.py (OPT-125M as published: context 2048, 0.5M-token batches, lr 6e-4, 375M-token
# warmup, linear decay to 6e-5, init 0.006 with 1/sqrt(2L) output scaling, tied embeddings),
# 9,540 steps = 5.0B tokens, an HF checkpoint every 954 steps (~500M tokens) plus step 0 and
# final. No MPS: one run per H100, one array task per run. Grid (slurm/env.sh, Jean Zay
# defaults): 8 attention types (vanilla vanilla_pos alibi forget stickbreaking window3 window4
# window5) x seeds 1-5 = 40 tasks, ordered seed then type: tasks 0-7 are the seed-1 matched set,
# 8-15 seed 2, ... windowN = attention over the last N tokens, current one included (the hard
# N-gram models; same convention as Bu et al.'s window_size).
# Relaunching is safe: finished runs are skipped (DONE), unfinished ones resume from resume.pt
# (resume refuses if any recipe setting changed). 20 h is the qos_gpu_h100-t3 limit; 10 min
# before it, SIGUSR1 makes each run save resume.pt; resubmit the same command.
# Submit from the repository root:
#   sbatch --array=0-39 slurm/jz/train_gpu.sh
#   sbatch --array=0-7 slurm/jz/train_gpu.sh                                 # seed 1 only
#   GRID_SEEDS="1 2 3" sbatch --array=0-23 slurm/jz/train_gpu.sh
#   TRAIN_ARGS="--micro-batch 8" sbatch --array=0-39 slurm/jz/train_gpu.sh   # extra train.py flags
#   RUNS="forget 125m 1" sbatch slurm/jz/train_gpu.sh                        # explicit runs
# Quick test (2 h, dev QoS): add --qos=qos_gpu_h100-dev --time=02:00:00
#
#SBATCH --job-name=attnc_train
#SBATCH --account=ldh@h100
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x_%A_%a.out
export RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
# exec keeps the batch shell's PID, so the B:USR1 signal reaches the trap in run_parallel.
exec bash "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm/train_gpu.sh"
