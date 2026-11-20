#!/bin/bash
# Jean Zay data, step 2 of 2: tokenize on a CPU node, dev QoS (no internet needed). Step 1 is the
# download, run on a login node (prepost is often congested; compute nodes have no internet):
#   DATA_ROOT=$SCRATCH/attn_constraints/data TMPDIR=$SCRATCH/tmp bash data/download.sh
# Stage 2 (125m, opt recipe) needs 9,540 steps x 256 windows x 2,049 tokens = 5.004B tokens, so
# train.bin gets 5.1B (TRAIN_TOKENS); val.bin = first 10M tokens of val. Submit from the repo root:
#   sbatch slurm/jz/tokenize_cpu.sh
#
#SBATCH --job-name=attnc_tokenize
#SBATCH --account=ldh@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# Measured (2026-11-20): 8 cores / 7 workers = 2.5M tokens/s -> 5.1B tokens in ~35 min, inside the
# 2 h dev limit. Fewer cores is much slower (each worker's tokenizer also starts threads).
export TRAIN_TOKENS="${TRAIN_TOKENS:-5.1e9}" SKIP_DOWNLOAD=1 WORKERS="${WORKERS:-7}"
exec bash "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm/data_cpu.sh"
