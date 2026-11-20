#!/bin/bash
# Download the Pile shard + validation file + tokenizer, then tokenize: train.bin (2.1B tokens
# from the start of train/00) and val.bin (first 10M tokens of val). Submit from the repo root:
#   sbatch slurm/data_cpu.sh
#
#SBATCH --job-name=attnc_data
#SBATCH --account=torch_pr_287_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
WORKERS="${WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}"

bash data/download.sh
"$PY" data/tokenize_pile.py --split val --input "$DATA_ROOT/raw/val.jsonl.zst" \
  --data-root "$DATA_ROOT" --max-tokens 1e7 --workers "$WORKERS"
"$PY" data/tokenize_pile.py --split train --input "$DATA_ROOT/raw/train/00.jsonl.zst" \
  --data-root "$DATA_ROOT" --max-tokens 2.1e9 --workers "$WORKERS"
ls -l "$DATA_ROOT"
