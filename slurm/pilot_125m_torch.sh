#!/bin/bash
# 125m noisy-attention pilot on one NYU Torch GPU (README "Noisy attention"). Submit from the
# repository root:
#   DATA_ROOT=$SCRATCH/attn_constraints/data_pile5b sbatch slurm/pilot_125m_torch.sh
# Phase 1: vanilla, noisy_l0, noisy_l0.01, noisy_l1 one at a time (tokens/s, peak memory, theta
#          moving in noisy_log.jsonl, faster for larger lambda).
# Phase 2: two runs at once under MPS (vanilla + noisy_l0.01), to decide 1 or 2 runs per GPU.
#
#SBATCH --job-name=attnc_pilot125m
#SBATCH --account=torch_pr_287_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --comment=gpu_mps=yes
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
export WANDB_PROJECT="${WANDB_PROJECT_PILOT-attn-constraints-pilot}"
STEPS="${PILOT_STEPS:-120}"
export TRAIN_ARGS="--max-steps $STEPS --eval-every 1000 --eval-max-windows 64 --ckpt-every 100000 --no-save-init --log-every 5 --noise-final-windows 16 ${TRAIN_ARGS:-}"
root="$PILOT_ROOT/125m_${SLURM_JOB_ID:-local}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} data=$DATA_ROOT steps=$STEPS"
echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset: MPS not active>}"
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader

summarize() {
  "$PY" - "$@" <<'EOF'
import json, sys
from pathlib import Path
for d in sys.argv[1:]:
    p = Path(d)
    s = p / "summary.json"
    if not s.exists():
        print(f"{p.name:30s} NO SUMMARY (see {p}/stdout.log)"); continue
    j = json.loads(s.read_text())
    theta = ""
    log = p / "noisy_log.jsonl"
    if log.exists():
        rows = [json.loads(r) for r in log.read_text().splitlines() if r]
        if rows:
            first, last = rows[0], rows[-1]
            theta = f" noisy_log rows={len(rows)} first_step={first.get('step')} last_step={last.get('step')}"
    print(f"{j['run_name']:30s} tok/s={j['tokens_per_s_steady']} peak_gb={j['peak_mem_gb']} val_loss={j['final_val_loss']:.4f}{theta}")
EOF
}

echo "== phase 1: one run at a time"
dirs=()
for attn in vanilla noisy_l0 noisy_l0.01 noisy_l1; do
  run_parallel "$root/single" "$attn 125m 1"
  dirs+=("$root/single/${attn}_125m_s1")
done
summarize "${dirs[@]}"

echo "== phase 2: two runs at once (MPS)"
run_parallel "$root/pair" "vanilla 125m 2" "noisy_l0.01 125m 2"
summarize "$root/pair/vanilla_125m_s2" "$root/pair/noisy_l0.01_125m_s2"
"$PY" -c "import json,glob; print('pair total tok/s:', sum(json.load(open(f))['tokens_per_s_steady'] or 0 for f in glob.glob('$root/pair/*/summary.json')))"
