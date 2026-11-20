#!/bin/bash
# Download the Pile shard, the Pile validation file and the Pythia tokenizer into $DATA_ROOT.
# Run on a compute node (slurm/data_cpu.sh): Torch login nodes kill large downloads.
# On Jean Zay, run it on the prepost partition (slurm/jz/data_prepost.sh): compute nodes have no internet.
#
#   DATA_ROOT  default: $SCRATCH/attn_constraints/data
#   SHARD      default: 00
#   TOKENIZER_REVISION, PILE_REVISION   default: the current HF commit. Set both to the revisions
#              in another cluster's manifest.json to reproduce its data byte for byte.
set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/attn_constraints/data}"
SHARD="${SHARD:-00}"
PILE_REPO="monology/pile-uncopyrighted"
TOKENIZER_REPO="EleutherAI/pythia-70m"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/tokenizer"
cd "$REPO_DIR"

# Resolve the tokenizer revision once and download exactly that commit, so the manifest
# records what was used.
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$("$REPO_DIR/.venv/bin/python" -c \
  "from huggingface_hub import model_info; print(model_info('$TOKENIZER_REPO').sha)")}"
echo "$TOKENIZER_REPO@$TOKENIZER_REVISION"
"$REPO_DIR/.venv/bin/hf" download "$TOKENIZER_REPO" \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --revision "$TOKENIZER_REVISION" --local-dir "$DATA_ROOT/tokenizer"
printf '%s\n%s\n' "$TOKENIZER_REPO" "$TOKENIZER_REVISION" > "$DATA_ROOT/tokenizer/SOURCE"

PILE_REVISION="${PILE_REVISION:-$("$REPO_DIR/.venv/bin/python" -c \
  "from huggingface_hub import dataset_info; print(dataset_info('$PILE_REPO').sha)")}"
echo "$PILE_REPO@$PILE_REVISION"
"$REPO_DIR/.venv/bin/hf" download "$PILE_REPO" "train/$SHARD.jsonl.zst" val.jsonl.zst \
  --repo-type dataset --revision "$PILE_REVISION" --local-dir "$DATA_ROOT/raw"
printf '%s\n%s\n' "$PILE_REPO" "$PILE_REVISION" > "$DATA_ROOT/raw/SOURCE"

ls -l "$DATA_ROOT/raw" "$DATA_ROOT/raw/train" "$DATA_ROOT/tokenizer"
