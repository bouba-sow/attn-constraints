#!/bin/bash
# Jean Zay pre-flight check for stage 2, on synthetic tokens (no real data needed), ~30 min on
# one H100 (dev QoS). Run it before the pilot and the grid; everything must pass.
#   1. 125m, opt recipe as the grid runs it (torch.compile, recipe micro-batch unless MICRO is
#      set), each grid attention type: 20 steps with checkpoints at steps 0, 10, final -> no
#      crash, finite loss, peak memory and tokens/s.
#   1b. stop and resume: stickbreaking gets SIGUSR1 mid-run (exit 99, resume.pt), is relaunched
#      and must finish with the same step count and a continuous train_log.csv.
#   2. memory probe: all types at micro-batch 8.
#   3. tools/check_benchmark_load.py with the benchmark's interpreter on every checkpoint.
#   4. evaluate.py on the finals, including the ALiBi slopes-at-zero path.
# Submit from the repository root:
#   sbatch slurm/jz/check_125m.sh
#   MICRO=8 sbatch slurm/jz/check_125m.sh
#
#SBATCH --job-name=attnc_check
#SBATCH --account=ldh@h100
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
MICRO_ARGS=(); [[ -n "${MICRO:-}" ]] && MICRO_ARGS=(--micro-batch "$MICRO")
BENCH="${BENCHMARK_ROOT:-$WORK/benchmark}"
ROOT="$SCRATCH/attn_constraints/check_${SLURM_JOB_ID:-local}"
DATA="$ROOT/data"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} micro=${MICRO:-recipe} root=$ROOT"
export TRITON_CACHE_DIR="${JOBSCRATCH:-$TMPDIR}/triton"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fail=0

# Synthetic data: random token ids plus the real Pythia tokenizer files: those written by the
# data job if present, else TOKENIZER_SRC (fetch on a login node with:
#   .venv/bin/hf download EleutherAI/pythia-70m tokenizer.json tokenizer_config.json \
#     special_tokens_map.json --local-dir $SCRATCH/attn_constraints/tokenizer_pythia70m )
if [[ -f "$DATA_ROOT/tokenizer/tokenizer.json" ]]; then
  TOKENIZER_SRC="$DATA_ROOT/tokenizer"
fi
TOKENIZER_SRC="${TOKENIZER_SRC:-$SCRATCH/attn_constraints/tokenizer_pythia70m}"
mkdir -p "$DATA/tokenizer"
if ! cp -L "$TOKENIZER_SRC"/{tokenizer.json,tokenizer_config.json,special_tokens_map.json} "$DATA/tokenizer/"; then
  echo "== RESULT: FAIL (no tokenizer files in $TOKENIZER_SRC)"; exit 1
fi
echo "tokenizer: $TOKENIZER_SRC"
"$PY" - "$DATA" <<'PYEOF'
import json, sys
import numpy as np
root = sys.argv[1]
rng = np.random.default_rng(0)
rng.integers(2, 50000, 21 * 256 * 2049, dtype=np.uint16).tofile(f"{root}/train.bin")
rng.integers(2, 50000, 64 * 2049, dtype=np.uint16).tofile(f"{root}/val.bin")
json.dump({"train": {"sha256": "synthetic"}, "val": {"sha256": "synthetic"}}, open(f"{root}/manifest.json", "w"))
PYEOF

summ() {
  "$PY" - "$@" <<'PYEOF'
import json, sys
from pathlib import Path
for d in sys.argv[1:]:
    s = Path(d, "summary.json")
    if s.exists():
        j = json.loads(s.read_text())
        print(f"{j['run_name']:28s} tok/s={j['tokens_per_s_steady']} peak_gb={j['peak_mem_gb']} val={j['final_val_loss']:.3f}")
    else:
        print(f"{Path(d).name:28s} FAILED: " + Path(d, 'stdout.log').read_text()[-600:])
PYEOF
}

echo "== 1. 125m, opt recipe (compile), micro-batch ${MICRO:-recipe}"
dirs=()
for attn in "${GRID_ATTNS[@]}"; do
  d="$ROOT/runs/${attn}_125m_s1"; mkdir -p "$d"
  "$PY" train.py --attn "$attn" --size 125m --seed 1 --data-root "$DATA" --runs-root "$ROOT/runs" \
    --max-steps 20 --ckpt-every 10 --eval-every 10 --eval-max-windows 32 --log-every 5 \
    "${MICRO_ARGS[@]}" >"$d/stdout.log" 2>&1 || fail=1
  dirs+=("$d")
done
summ "${dirs[@]}"
grep -h '"micro_batch"\|"compile"' "$ROOT"/runs/vanilla_125m_s1/run_config.json

echo "== 1b. stop (SIGUSR1) and resume, stickbreaking"
d="$ROOT/resume/stickbreaking_125m_s1"; mkdir -p "$d"
rargs=(--attn stickbreaking --size 125m --seed 1 --data-root "$DATA" --runs-root "$ROOT/resume"
       --max-steps 20 --ckpt-every 10 --eval-every 10 --eval-max-windows 32 --log-every 1 "${MICRO_ARGS[@]}")
"$PY" train.py "${rargs[@]}" >"$d/stdout.log" 2>&1 &
pid=$!
until [[ -f "$d/train_log.csv" && $(wc -l <"$d/train_log.csv") -ge 7 ]] || ! kill -0 $pid 2>/dev/null; do sleep 2; done
kill -USR1 $pid; wait $pid; s1=$?
"$PY" train.py "${rargs[@]}" >>"$d/stdout.log" 2>&1; s2=$?
steps="$(cut -d, -f1 "$d/train_log.csv" | tail -n +2 | tr '\n' ' ')"
echo "first exit=$s1 (want 99) second exit=$s2 (want 0) resume.pt left: $([[ -f $d/resume.pt ]] && echo yes || echo no)"
echo "logged steps: $steps"
expected="$(seq 1 20 | tr '\n' ' ')"
if [[ $s1 -eq 99 && $s2 -eq 0 && -f "$d/DONE" && ! -f "$d/resume.pt" && "$steps" == "$expected" ]]; then
  echo "resume: ok"
  # Informational: the same run without interruption is in section 1 (same data order, init and
  # settings). GPU kernels are not bit-deterministic, so expect small, not zero, differences.
  "$PY" - "$d/final/model.safetensors" "$ROOT/runs/stickbreaking_125m_s1/final/model.safetensors" <<'PYEOF'
import sys
from safetensors.torch import load_file
a, b = load_file(sys.argv[1]), load_file(sys.argv[2])
d = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
print(f"resumed vs uninterrupted final weights: max abs diff {d:.2e} over {len(a)} tensors")
PYEOF
else
  echo "resume: FAIL"; tail -20 "$d/stdout.log"; fail=1
fi
grep -h "n_params" "$ROOT"/runs/vanilla_125m_s1/run_config.json "$ROOT"/runs/vanilla_pos_125m_s1/run_config.json

echo "== 2. memory probe at micro-batch 8 (failure here only rules out that size)"
for attn in "${GRID_ATTNS[@]}"; do
  d="$ROOT/probe/${attn}_125m_s1"; mkdir -p "$d"
  "$PY" train.py --attn "$attn" --size 125m --seed 1 --data-root "$DATA" --runs-root "$ROOT/probe" \
    --max-steps 3 --ckpt-every 1000 --no-save-init --eval-every 1000 --eval-max-windows 8 --log-every 1 \
    --micro-batch 8 >"$d/stdout.log" 2>&1
  summ "$d"
done

echo "== 3. benchmark load + surprisal"
ckpts=()
for attn in "${GRID_ATTNS[@]}"; do
  ckpts+=("$ROOT/runs/${attn}_125m_s1/checkpoints/step_00010" "$ROOT/runs/${attn}_125m_s1/final")
done
(
  unset LPP_FEATURE_CACHE_DIR
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd "$BENCH" && "$BENCH/.venv/bin/python" "$REPO_DIR/tools/check_benchmark_load.py" "${ckpts[@]}" --benchmark-root "$BENCH"
) || fail=1

echo "== 4. evaluate.py"
"$PY" evaluate.py "$ROOT"/runs/*_125m_s1/final --val-bin "$DATA/val.bin" --max-windows 32 || fail=1
"$PY" evaluate.py "$ROOT"/runs/alibi_125m_s1/final --val-bin "$DATA/val.bin" --alibi-check --max-windows 32 || fail=1

echo "== RESULT: $([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit $fail
