# attn-constraints

A matched suite of small OPT-style language models that differ **only in the attention
mechanism**: same code, data, data order, tokenizer, size, hyperparameters and initial shared
weights, with several seeds. Their surprisal fit to human reading times is scored afterwards
with the NeuroAI benchmark on Jean Zay.

This is our own comparison, not a reproduction of Bu et al. (arXiv 2608.23818): we use their
sizes and optimizer settings, but our own preprocessing and implementation.

## Design

| Model | Position embeddings | Role |
|---|---|---|
| `vanilla` | none (NoPE) | matched baseline |
| `alibi`, `forget`, `stickbreaking` | none, as originally published | constrained models |
| `vanilla_pos` | learned absolute (OPT standard) | reference: a normal transformer |

Constrained vs `vanilla` differs only in the attention score function. A causal transformer
without position embeddings still learns position from the causal mask (Haviv et al. 2022;
Kazemnejad et al. 2023). Constrained vs `vanilla_pos` is the comparison most papers make
implicitly, and `vanilla` vs `vanilla_pos` shows what removing position embeddings does by itself.

With s_ij = q_i·k_j/√d_head and causal masking (j ≤ i):

| `attn_type` | Weights |
|---|---|
| `vanilla`, `vanilla_pos` | softmax_j(s_ij) |
| `alibi` | softmax_j(s_ij − m_h(i − j)); fixed Press et al. slopes, m_h > 0 (H=4: 1/4…1/256; H=6: 1/4, 1/16, 1/64, 1/256, 1/2, 1/8) |
| `forget` | softmax_j(s_ij + Σ_{j<t≤i} log f_t,h), f_t,h = σ(w_h·x_t + b_h) from the post-LayerNorm input (FoX "full"; gate init normal(0, 0.02), bias 0) |
| `stickbreaking` | A_ij = σ(s_ij) ∏_{j<k≤i}(1 − σ(s_ik)); the current token is included, no softmax, no renormalization (rows sum to ≤ 1) |
| `window` (pilot only) | vanilla with keys restricted to i − 3 < j ≤ i |

Scores, masks, softmax and cumulative sums are computed in fp32 inside bf16 autocast as explicit
T×T tensors: no custom kernels, so any GPU works.

Sizes (vocab: the Pythia/GPT-NeoX tokenizer, padded to 50,304; untied input/output embeddings;
pre-LayerNorm, ReLU, dropout 0, init std 0.02, biases 0):

| size | layers / hidden / heads / FFN | parameters |
|---|---|---|
| `2l` | 2 / 256 / 4 / 1024 | ≈27.3M (+0.26M for `vanilla_pos`) |
| `4l` | 4 / 384 / 6 / 1536 | ≈45.7M (+0.39M for `vanilla_pos`) |
| `125m` (stage 2) | 12 / 768 / 12 / 3072 | ≈162M with untied embeddings |

**Matched initialization.** `init_matched_weights` draws every shared parameter from a generator
seeded with `seed`, in module order, and the constraint-only parameters (forget-gate projection,
position table) from a second generator seeded with `seed + 1000`. Within a seed, the shared
weights are bit-identical across attention types (tested).

**Data and training.**
- Data: the first 2.1B tokens (whole documents) of `monology/pile-uncopyrighted` `train/00.jsonl.zst`,
  in file order, `<|endoftext|>` (id 0) after each document, no BOS. Validation: the first 10M
  tokens of `val.jsonl.zst`.
- Sequences: the train stream is cut into non-overlapping 1,025-token windows (documents are
  packed across window boundaries), and the window order is `np.random.default_rng(seed).permutation`,
  identical for every attention type and size within a seed.
- Optimizer: 256 × 1024 = 262,144 tokens per step × 7,630 steps ≈ 2.0B tokens, one pass.
  AdamW (lr 1e-3, betas 0.9/0.95, weight decay 0.1 on 2-D weight matrices only), gradient
  clipping at 1.0, 1% linear warmup, then cosine decay to 0. bf16 autocast (fp16 with loss
  scaling on GPUs without bf16).

**Stage 1 grid**: {vanilla, vanilla_pos, alibi, forget, stickbreaking} × {2l, 4l} × seeds 1–5 =
50 runs, named `{attn}_{size}_s{seed}` (e.g. `forget_4l_s3`).

## Layout

```
pyproject.toml, uv.lock   dependencies (uv)
data/download.sh          Pile shard + validation file + tokenizer files
data/tokenize_pile.py     jsonl.zst -> train.bin / val.bin (uint16) + manifest.json
model/configuration.py    ConstrainedOPTConfig (OPTConfig + attn_type) and the size table
model/modeling.py         attention variants, ConstrainedOPTForCausalLM, init_matched_weights
model/checkpoint.py       save a self-contained HF checkpoint (weights, config, code, tokenizer)
train.py                  plain PyTorch training loop (one run per process)
evaluate.py               validation loss of saved checkpoints; ALiBi slopes-at-zero check
slurm/env.sh              environment, grid definition, parallel launcher (sourced)
slurm/data_cpu.sh         download + tokenize (CPU job)
slurm/pilot.sh            pilot: per-configuration check, MPS scaling
slurm/train_gpu.sh        one GPU per array task, RUNS_PER_GPU runs under MPS
tests/                    attention, I/O and training tests (CPU)
```

The spec named the tokenizer script `data/tokenize.py`. It is `tokenize_pile.py` here because a
script called `tokenize.py` shadows Python's standard `tokenize` module.

## Setup (NYU Torch)

```bash
cd /scratch/bs5813/Documents/attn-constraints
export TMPDIR=$SCRATCH/tmp     # login-node /tmp is a small, often full tmpfs
uv sync                        # creates .venv (torch 2.6.0+cu124, transformers 5.13.1)
uv run pytest                  # must pass before the pilot
```

The job scripts call `.venv/bin/python` directly, clear `LD_LIBRARY_PATH`, `PYTHONPATH` and conda
entries from `PATH`, and expect to be submitted from the repository root (`logs/` collects job
output). Data goes to `$SCRATCH/attn_constraints/data`, runs to `$SCRATCH/attn_constraints/runs`
(override with `DATA_ROOT`, `RUNS_ROOT`).

## Steps

**1. Data** (CPU job; Torch login nodes kill large downloads):

```bash
sbatch slurm/data_cpu.sh
```

This writes `train.bin`, `val.bin` and `manifest.json`. The manifest holds token, document and
`<|endoftext|>` counts, the sha256 of each `.bin` and of the source file, and the tokenizer and
dataset revisions.

**2. Pilot** (before the grid):

```bash
sbatch slurm/pilot.sh             # each attn type × size, seed 1, 50M tokens: loss, NaNs, tok/s, memory
MODE=mps sbatch slurm/pilot.sh    # forget 4l as 1, 2, 4, 8 processes on one GPU: total tok/s
```

Pick `RUNS_PER_GPU` where total tokens/s levels off, and `--micro-batch` from peak memory.
`RUNS_PER_GPU=5` gives each GPU one matched set (all five attention types for one size × seed).

**3. Grid**:

```bash
RUNS_PER_GPU=5 sbatch --array=0-9 slurm/train_gpu.sh
# extra train.py flags for every run, e.g.:
TRAIN_ARGS="--micro-batch 32" RUNS_PER_GPU=5 sbatch --array=0-9 slurm/train_gpu.sh
# preemptible pool (more GPUs; jobs may be killed and requeued, runs resume):
RUNS_PER_GPU=5 sbatch --array=0-9 \
  --comment="preemption=yes;preemption_partitions_only=yes;requeue=true;gpu_mps=yes" slurm/train_gpu.sh
```

`--comment=gpu_mps=yes` starts an MPS server inside each job, so the job's processes run on its
GPU at the same time. Without it, the GPUs' `Exclusive_Process` mode allows only one process.
Each job still counts as one GPU against the 16-GPU per-user limit.

Relaunching is safe: runs with `DONE` are skipped and unfinished runs resume from `resume.pt`.
On SIGTERM/SIGUSR1 (time limit, preemption) each run saves `resume.pt` after its current step
and exits with code 99.

**Live dashboard (Trackio).** Every run logs train/validation loss, learning rate, gradient
norm and tokens/s to a private Hugging Face Space,
[`boubasow/attn-constraints-dashboard`](https://huggingface.co/spaces/boubasow/attn-constraints-dashboard).
The Space is created on first use and needs `HF_TOKEN` with write access in the submitting shell.
- Runs are grouped by matched set (`{size}_s{seed}`). Grid runs log to project
  `attn-constraints`, pilot runs to `attn-constraints-pilot`.
- Set `TRACKIO_SPACE_ID=""` to turn it off. A Trackio error never stops training: it disables
  the dashboard for that run, and `train_log.csv` and TensorBoard (`tb/`) stay the record.
- TensorBoard without a Space: `tensorboard --logdir $RUNS_ROOT` on a login node, opened through
  `ssh -L 6006:localhost:6006`.

**4. Checks after training**:

```bash
.venv/bin/python evaluate.py $RUNS_ROOT/alibi_*/final --alibi-check   # loss must rise with slopes = 0
```

**5. Hand-off to Jean Zay** (`ProxyJump oberon` in `~/.ssh/config`):

```bash
rsync -avP --partial --include='*/' --include='final/***' --include='checkpoints/***' \
  --include='*.json' --include='*.csv' --exclude='*' \
  $RUNS_ROOT/ jeanzay:<dest>/attn_constraints_runs/
```

## Run directory

```
{runs_root}/{attn}_{size}_s{seed}/
  run_config.json        all arguments, git commit, data manifest hash, versions, GPU
  train_log.csv          step, tokens, train/val loss, lr, grad norm, tokens/s, peak memory
  tb/                    TensorBoard
  checkpoints/step_XXXXX HF checkpoints: step 0 (initial weights) and every 1,000 steps
  final/                 HF checkpoint at the end
  resume.pt              latest resume state (deleted once the run is done)
  summary.json, DONE     written at the end
  stdout.log
```

Every checkpoint loads with
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` and
`AutoTokenizer.from_pretrained(path)`. The parameter names follow HF OPT, so the benchmark finds
`model.decoder.final_layer_norm` and `get_output_embeddings()`, and `output_hidden_states=True`
returns L + 1 states as HF OPT does (the last one after the final LayerNorm). There is no KV
cache: `use_cache` is accepted and ignored. ALiBi slopes can be scaled at load time with
`from_pretrained(..., alibi_slope_scale=0.0)`.

## Tests

`uv run pytest` (CPU, a few minutes):
- `vanilla_pos` matches HF `OPTForCausalLM` with eager attention for the same weights;
  NoPE types have no position parameters.
- Causality for every type.
- ALiBi slopes and bias; slopes set to 0 give vanilla.
- An open forget gate gives vanilla; the forget bias and the stick-breaking weights match explicit loops.
- Shared weights are identical across types for the same seed.
- Save, then load in a fresh process with `trust_remote_code`, gives the same logits (all types).
- Tokenization round trip, with `<|endoftext|>` count = document count.
- Training on synthetic data (`--size test`, hidden 32, for CPU speed): the loss decreases, and a
  killed run resumes to bit-identical weights.

## Stage 2 (later, same code)

`--size 125m` (12 layers, hidden 768, 12 heads, FFN 3072), about 5B tokens (`--max-steps`
≈ 19,000; still within shard 00, so re-tokenize with a larger `--max-tokens`), checkpoints
every ~500M tokens (`--ckpt-every 2000`), only for the constraints that show an effect in stage 1.
