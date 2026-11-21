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
| `windowN` (stage 2: N = 3, 4, 5; plain `window` = 3) | vanilla with keys restricted to i − N < j ≤ i: the hard N-gram models (N tokens visible per layer, current one included; Bu et al.'s `window_size` convention). Over L layers the receptive field is L(N − 1) + 1 tokens: 25 / 37 / 49 at 12 layers |

| `noisy_l<λ>` (λ = 0, 0.001, 0.01, 0.1, 1) | vanilla weights; the head output gets Gaussian noise (Xu, Dillon & Futrell 2026; see "Noisy attention") |

Scores, masks, softmax and cumulative sums are computed in fp32 inside bf16 autocast as explicit
T×T tensors: no custom kernels, so any GPU works.

Sizes (vocab: the Pythia/GPT-NeoX tokenizer, padded to 50,304; untied input/output embeddings;
pre-LayerNorm, ReLU, dropout 0, init std 0.02, biases 0):

| size | layers / hidden / heads / FFN | parameters |
|---|---|---|
| `2l` | 2 / 256 / 4 / 1024 | ≈27.3M (+0.26M for `vanilla_pos`) |
| `4l` | 4 / 384 / 6 / 1536 | ≈45.7M (+0.39M for `vanilla_pos`) |
| `125m` (stage 2) | 12 / 768 / 12 / 3072 | ≈125M (tied embeddings, OPT recipe; see "Jean Zay") |

**Matched initialization.** `init_matched_weights` draws every shared parameter from a generator
seeded with `seed`, in module order, and the constraint-only parameters (forget-gate projection,
position table, noisy decay rates) from a second generator seeded with `seed + 1000`. Within a seed, the shared
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
slurm/jz/                 Jean Zay (stage 2): check_125m, tokenize (CPU dev), pilot, train, archive
tools/check_benchmark_load.py  checkpoint check with the benchmark's loader (benchmark venv)
tools/trackio.sh          trackio dashboard from train_log.csv (login node)
tools/wandb.sh            W&B sync from train_log.csv (login node)
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

## Jean Zay (stage 2: 125m, OPT-125M recipe)

Stage 1 (2l, 4l) runs on Torch; stage 2 (125m) runs on Jean Zay with the same code and
environment (`uv.lock`). The two stages are not compared with each other, so each cluster
downloads and tokenizes its own data; all runs of a size train on one cluster from one `train.bin`.
`slurm/env.sh` detects Jean Zay (`IDRPROJ` is set), and the job scripts in `slurm/jz/` carry the
Jean Zay headers (accounts `ldh@h100` / `ldh@cpu`) before running the shared bodies in `slurm/`.

**Recipe** (`train.py` `RECIPES["opt"]`, the default for `--size 125m`; any flag overrides it).
OPT-125M as published (Zhang et al. 2022, Sec. 2.2 and Table 1; metaseq
`opt_job_constants.py` / `opt_baselines.py`):

| | OPT-125M | here |
|---|---|---|
| layers / hidden / heads / FFN | 12 / 768 / 12 / 3072 | same |
| context | 2048 | same |
| batch | 0.5M tokens | 256 x 2048 = 524,288 |
| optimizer | AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1 | same (decay on 2-D weights, not embeddings) |
| peak lr, schedule | 6e-4; linear warmup over 375M tokens, linear decay to 10% | same: 715 warmup steps, decay to 6e-5 at the last step |
| gradient clipping | 1.0 | same |
| init | normal(0, 0.006), output layers / sqrt(2L), biases 0 | same (`self_attn.out_proj`, `fc2`) |
| embeddings | tied | tied (`tie_word_embeddings`): ~125M parameters |
| training length | 300B tokens | 9,540 steps = 5.0B tokens; schedule spans our 5B |
| dropout | 0.1 (not on embeddings) | **0** (single pass; suite spec) |
| position embeddings | learned | **none**, except `vanilla_pos` (study design) |
| tokenizer | GPT-2 | Pythia / GPT-NeoX (suite spec) |

Checkpoints: step 0, then every 954 steps (~500M tokens; 10 checkpoints, the last = final).

**Grid**: {vanilla, vanilla_pos, alibi, forget, stickbreaking, window3, window4, window5} x seeds
1-5 = 40 runs (`slurm/env.sh` Jean Zay defaults), named e.g. `window4_125m_s2`. Bu et al. used
N = 2, 3, 5 at 2 layers (receptive field 3 / 5 / 9 tokens); here N = 3, 4, 5 at 12 layers.

**Cluster notes.**
- No MPS: one run per H100, one array task per run; the pilot has only the per-configuration mode.
- Time limit 20 h (`qos_gpu_h100-t3`). At the limit, runs save `resume.pt` (SIGUSR1, 10 min
  before); resubmit the same command and they resume. Resume refuses if any recipe setting changed.
- Compute nodes have no internet: data is downloaded on a login node and tokenized on a CPU node (dev QoS, 8 cores); GPU jobs run
  with `HF_HUB_OFFLINE=1`.
- Data and runs go to `$SCRATCH/attn_constraints/`. `$SCRATCH` is purged after 30 days without
  access: copy runs to `$STORE` with `slurm/jz/archive.sh`. Never write runs to `$WORK` (inode quota).

```bash
cd $WORK/attn-constraints
export TMPDIR=$SCRATCH/tmp UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy
uv sync --frozen                                     # login node (has internet)
srun -A ldh@cpu --qos=qos_cpu-dev -c 8 --hint=nomultithread -t 00:45:00 .venv/bin/python -m pytest -q

sbatch slurm/jz/check_125m.sh                        # 0. pre-flight on synthetic tokens: must PASS
DATA_ROOT=$SCRATCH/attn_constraints/data bash data/download.sh   # 1a. download (login node)
sbatch slurm/jz/tokenize_cpu.sh                      # 1b. tokenize (dev QoS, 8 cores, ~35 min): 5.1B tokens
sbatch slurm/jz/pilot.sh                             # 2. pilot on real data
sbatch --array=0-39 slurm/jz/train_gpu.sh            # 3. grid: 8 types x 5 seeds (tasks 0-7 = seed 1, ...)
sbatch slurm/jz/archive.sh                           # 4. copy runs and data to $STORE
```

`check_125m.sh` trains every grid type at 125m for 20 steps (memory, speed, checkpoints), then
`tools/check_benchmark_load.py` loads each checkpoint with the benchmark's own loader and checks
tied weights, identical logits, surprisal against a hand computation, inputs longer than 2048
tokens and the benchmark's sliding-window length. `TRAIN_ARGS` adds train.py flags to the
pilot and grid jobs, e.g. `TRAIN_ARGS="--micro-batch 8"`.

## Monitoring (W&B or trackio, Jean Zay login node)

Training never talks to a tracker (compute nodes have no internet). Both tools below read the
runs' `train_log.csv` from a login node, run from pinned, cached `uv run --with` environments
(training venv and `uv.lock` untouched), and can be used side by side.

**W&B** (`tools/wandb.sh`, wandb 0.31.1; API key in `~/.netrc`, `machine api.wandb.ai`, mode 600):

```bash
tools/wandb.sh                         # push new rows of every run to <entity>/attn-constraints-125m
tools/wandb.sh --watch 600             # ... every 10 min (leave it in tmux/screen)
```

One W&B run per training run (id = run name, grouped by attention type, tagged with seed and
size, config from `run_config.json`). `<run_dir>/.wandb_sync.json` records the last step pushed,
so each sync only appends. If a crash made a run redo steps after its last resume point, the
W&B run is replaced by `{run}-v2` with the whole CSV.

**trackio** (local, no account):

Training does not use trackio; `tools/trackio.sh` copies every run's `train_log.csv` (and its
attention type, size, seed and recipe from `run_config.json`) into one local trackio project and
serves the dashboard. trackio 0.45.1 runs from a cached `uv run --with` environment, so the
training venv and `uv.lock` are unchanged. Run it on a login node:

```bash
tools/trackio.sh import                 # (re)build the project from $SCRATCH/attn_constraints/runs
tools/trackio.sh import --watch 600     # refresh every 10 min (leave it in tmux/screen)
tools/trackio.sh show                   # dashboard on 127.0.0.1:7860 of this login node
# from your machine: ssh -N -L 7860:localhost:7860 <you>@jean-zayN.idris.fr  (the same login node)
```

Each import deletes and re-logs every run, so repeating it is safe and follows resumed runs.
The project lives in `trackio/` (gitignored, on `$WORK`), with SQLite's rollback journal
instead of WAL so `import` and `show` may run on different login nodes.

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

## Noisy attention (Xu, Dillon & Futrell 2026; 125m on NYU Torch)

Xu, Dillon & Futrell (2026, ACL), "Memory efficiency and resource-rational encoding in sentence
processing" ([code](https://github.com/weijiexu-charlie/resource-rational-encoding)), as an extra
attention type on top of `vanilla` (NoPE), trained with the same 125m OPT recipe and 5.0B Pile
tokens as the rest of stage 2:

- Head output z_i = Σ_j a_ij v_j + σ_i ε, ε ~ N(0, I), σ_i² = Σ_j a_ij² / τ_ij (their
  reparameterization of noisy value vectors), τ_ij = τ₀ exp(−θ_h (i − j)) + 10⁻⁴, τ₀ = 10⁴
  (σ = 0.01 at distance 0), θ_h = softplus(·) learned per head, initialized at 1 + 10⁻³ N(0, 1)
  (their code), no weight decay.
- Loss = cross-entropy + λ × mean τ over all (q, k ≤ q) pairs of a fixed **512-token** window,
  heads and layers (their `precision_regularizer_sum`, closed form). The penalty depends only on
  θ, and with a longer window the same θ gives a ~1/T smaller mean: computing it on their 512-token
  window keeps their λ values meaningful while training at context 2048. `train_loss` and
  `val_loss` in `train_log.csv` are the cross-entropy alone.
- Evaluation (`noise_eval_samples` in `config.json`, 16 by default): log of the mean next-token
  probability over 16 noise draws from a generator seeded at every call (the model's predictive
  distribution under its noisy memory; reproducible). `noise_eval_samples = 0` turns the noise
  off. Their released models sample one fresh draw per forward pass, also at inference.
  `hidden_states` come from a pass with the noise off. The periodic `val_loss` uses
  `--noise-val-samples` draws (default 1, as in their training log); `summary.json` adds
  `noisy_final_val` (noise off on all of `val.bin`, and noise off vs K = 16 on the first 512 windows).
- Diagnostics: `noisy_log.jsonl` (every `--eval-every` steps): θ per layer and head, the noise
  std at distances 0 / 10 / 100, the penalty, and the RMS of the value vectors (the noise std only
  matters relative to it).

**Grid**: {vanilla, noisy_l0, noisy_l0.001, noisy_l0.01, noisy_l0.1, noisy_l1} × 125m × seeds 1–3 =
18 runs, all on NYU (every type of a seed on the same machine; `vanilla` s1–s3 are retrained there,
which also measures how much the cluster alone moves the results). λ = 0 has the noise without
the penalty: it separates "adding noise" from "constraining precision".

```bash
# 1. Jean Zay's 125m data (5.1B tokens), then check it against the manifest:
rsync -avP jeanzay:/lustre/fsn1/projects/rech/ldh/una68ug/attn_constraints/data/{train.bin,val.bin,manifest.json,tokenizer} $SCRATCH/attn_constraints/data/
.venv/bin/python tools/check_data.py $SCRATCH/attn_constraints/data
# 2. pilot (one H200, ~15 min per type): tok/s, peak memory, loss stable; in each run's
#    noisy_log.jsonl theta must rise from its init, faster for larger lambda
GRID_ATTNS="vanilla noisy_l0 noisy_l0.01 noisy_l1" GRID_SIZES=125m TRAIN_ARGS="--noise-final-windows 32" sbatch slurm/pilot.sh
# 3. grid: one run per GPU (6 types x 3 seeds = 18 array tasks)
GRID_ATTNS="vanilla noisy_l0 noisy_l0.001 noisy_l0.01 noisy_l0.1 noisy_l1" GRID_SIZES=125m GRID_SEEDS="1 2 3" \
  RUNS_PER_GPU=1 sbatch --array=0-17 slurm/train_gpu.sh
.venv/bin/python evaluate.py $RUNS_ROOT/noisy_l*_125m_s*/final --noise-samples 0 16 --max-windows 512
```

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
- Training on synthetic data: the loss decreases, and a killed run resumes to bit-identical weights.
- Noisy attention: the penalty equals Xu et al.'s `precision_regularizer_sum`; σ_i matches an
  explicit loop; noise off (or negligible noise) gives `vanilla`; evaluation is reproducible and
  equals the mean probability over K draws; a short run raises θ under the penalty and writes
  the diagnostics and both final evaluations.

## Stage 2 (same code)

`--size 125m` uses the OPT-125M recipe (see "Jean Zay"): context 2048, 5.0B tokens,
checkpoints every ~500M tokens. It runs on Jean Zay in parallel with stage 1.
