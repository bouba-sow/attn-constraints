#!/bin/bash
# Jean Zay: copy runs from $SCRATCH (purged after 30 days without access) to $STORE.
# Copies everything except resume.pt and TensorBoard event files; safe to rerun.
# Submit from the repository root:
#   sbatch slurm/jz/archive.sh
#
#SBATCH --job-name=attnc_archive
#SBATCH --account=ldh@cpu
#SBATCH --partition=archive
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${SCRATCH:?}" "${STORE:?}"
SRC="${RUNS_ROOT:-$SCRATCH/attn_constraints/runs}"
DEST="${ARCHIVE_ROOT:-$STORE/attn_constraints/runs}"
mkdir -p "$DEST"
rsync -a --partial --exclude='resume.pt' --exclude='tb/' "$SRC/" "$DEST/"
mkdir -p "$STORE/attn_constraints/data"
rsync -a --partial "${DATA_ROOT:-$SCRATCH/attn_constraints/data}/" "$STORE/attn_constraints/data/" \
  --exclude='raw/'
du -sh "$DEST"
