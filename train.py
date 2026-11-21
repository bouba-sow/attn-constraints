"""Train one model of the matched attention-constraint suite.

One process = one run ``{attn}_{size}_s{seed}``. Within a seed, every attention type and size
sees the same windows in the same order and starts from the same shared weights.

Data: ``train.bin`` is cut into non-overlapping windows of seq_len + 1 tokens (inputs plus the
shifted targets; documents are packed across window boundaries). The window order is
``np.random.default_rng(seed).permutation``; optimizer step s uses windows
``perm[s * batch_seqs : (s + 1) * batch_seqs]``, so no window is seen twice.

Restarts: rerunning the same command resumes from ``resume.pt`` (model, optimizer, loss scaler,
step, RNG states; the data position and learning rate follow from the step). SIGTERM/SIGUSR1
(time limit, preemption) save ``resume.pt`` at the end of the current step and exit with
code 99. A finished run writes ``DONE`` and is skipped when relaunched.

    uv run python train.py --attn forget --size 4l --seed 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from model.checkpoint import save_hf
from model.configuration import ATTN_TYPES, SIZES, make_config
from model.modeling import ConstrainedOPTForCausalLM, init_matched_weights, noisy_precision_penalty

REPO_DIR = Path(__file__).resolve().parent
EXIT_INTERRUPTED = 99
LOG_FIELDS = ["step", "tokens", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_s", "peak_mem_gb"]


# Training recipes. A flag given on the command line overrides the recipe value.
# * stage1 (2l, 4l; NYU Torch): the agreed stage-1 spec.
# * opt (125m; Jean Zay): OPT-125M as published (Zhang et al. 2022, Sec. 2.2 and Table 1;
#   metaseq opt_job_constants.py / opt_baselines.py): context 2048, 0.5M-token batches
#   (256 x 2048), peak lr 6e-4, linear warmup over 375M tokens (715 steps), linear decay to 10%
#   of the peak, AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1, clip 1.0, init std 0.006 with
#   output projections scaled by 1/sqrt(2L), tied input/output embeddings. Our budget: 9,540
#   steps x 524,288 = 5.0B tokens (OPT decays over 300B), a checkpoint every 954 steps
#   (~500M tokens). Kept from the suite spec, not OPT: dropout 0 (OPT: 0.1) and no position
#   embeddings except for vanilla_pos. Implementation: torch.compile (same math, fused; on an
#   H100 at 2048 tokens: ~130k tok/s vanilla/ALiBi, ~118k forget, ~95k stick-breaking, vs
#   25-53k eager) and micro-batch 4 for every type (stick-breaking needs ~32 GB at 4).
RECIPES = {
    "stage1": dict(seq_len=1024, batch_seqs=256, max_steps=7630, lr=1e-3, min_lr=0.0,
                   schedule="cosine", warmup_frac=0.01, warmup_steps=None, adam_eps=1e-8,
                   init_std=0.02, init_output_scale=False, tie_embeddings=False, ckpt_every=1000,
                   micro_batch=16, compile=False),
    "opt": dict(seq_len=2048, batch_seqs=256, max_steps=9540, lr=6e-4, min_lr=6e-5,
                schedule="linear", warmup_frac=None, warmup_steps=715, adam_eps=1e-8,
                init_std=0.006, init_output_scale=True, tie_embeddings=True, ckpt_every=954,
                micro_batch=4, compile=True),
}
DEFAULT_RECIPE = {"2l": "stage1", "4l": "stage1", "125m": "opt"}


NOISY_SPEC = re.compile(r"noisy_l((?:0|[1-9][0-9]*)(?:\.[0-9]+)?)")


def parse_attn_spec(attn: str, parser: argparse.ArgumentParser | None = None) -> tuple[str, int | None, float | None]:
    """--attn -> (attn_type, window_size, noisy_lambda).

    'forget' -> ('forget', None, None); 'window4' -> ('window', 4, None) (plain 'window' keeps the
    config default, 3); 'noisy_l0.01' -> ('noisy', None, 0.01). A noisy model needs its lambda in
    the name (noisy_l0, noisy_l0.001, ..., noisy_l1), so run names say which constraint they used.
    """
    if attn in ATTN_TYPES and attn != "noisy":
        return attn, None, None
    m = re.fullmatch(r"window([1-9][0-9]*)", attn)
    if m is not None:
        return "window", int(m.group(1)), None
    m = NOISY_SPEC.fullmatch(attn)
    if m is not None:
        return "noisy", None, float(m.group(1))
    msg = f"--attn must be one of {tuple(a for a in ATTN_TYPES if a != 'noisy')}, windowN or noisy_l<lambda>, got {attn!r}"
    if parser is not None:
        parser.error(msg)
    raise ValueError(msg)


def parse_attn(attn: str, parser: argparse.ArgumentParser | None = None) -> tuple[str, int | None]:
    """(attn_type, window_size) of ``parse_attn_spec``."""
    return parse_attn_spec(attn, parser)[:2]


def default_root(name: str) -> Path | None:
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) / "attn_constraints" / name if scratch else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", required=True, metavar="ATTN",
                   help=f"one of {tuple(a for a in ATTN_TYPES if a != 'noisy')}, windowN (window attention "
                        "over the last N tokens, current one included: the hard N-gram models, e.g. window3), "
                        "or noisy_l<lambda> (Xu et al. 2026 noisy attention with penalty weight lambda, "
                        "e.g. noisy_l0.01)")
    p.add_argument("--size", required=True, choices=tuple(SIZES))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--data-root", type=Path, default=default_root("data"))
    p.add_argument("--runs-root", type=Path, default=default_root("runs"))
    p.add_argument("--run-name", default=None, help="default: {attn}_{size}_s{seed}")
    p.add_argument("--recipe", choices=tuple(RECIPES), default=None,
                   help="default: stage1 for 2l/4l, opt for 125m (see RECIPES)")
    # Recipe-controlled settings: None = take the recipe value.
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-seqs", type=int, default=None, help="sequences per optimizer step")
    p.add_argument("--micro-batch", type=int, default=None, help="sequences per forward pass")
    p.add_argument("--max-steps", type=int, default=None, help="stage1: 7630 x 256 x 1024 = 2.0B tokens")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--schedule", choices=("cosine", "linear"), default=None, help="decay after warmup")
    p.add_argument("--warmup-frac", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None, help="overrides --warmup-frac")
    p.add_argument("--adam-eps", type=float, default=None)
    p.add_argument("--init-std", type=float, default=None)
    p.add_argument("--init-output-scale", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-max-windows", type=int, default=None, help="cap validation (pilot only)")
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--no-save-init", action="store_true", help="skip checkpoints/step_00000")
    p.add_argument("--resume-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="torch.compile the model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--noise-val-samples", type=int, default=1,
                   help="noisy only: noise draws averaged for the periodic validation loss (1 = one draw, as Xu "
                        "et al.; 0 = noise off). The final evaluation always reports noise off and K = 16.")
    p.add_argument("--noise-final-windows", type=int, default=512,
                   help="noisy only: validation windows for the final K = 16 evaluation (16 passes each)")
    args = p.parse_args(argv)
    args.attn_type, args.window_size, args.noisy_lambda = parse_attn_spec(args.attn, p)
    args.recipe = args.recipe or DEFAULT_RECIPE[args.size]
    for key, value in RECIPES[args.recipe].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.warmup_steps is None and args.warmup_frac is None:
        p.error("set --warmup-steps or --warmup-frac")
    if args.data_root is None or args.runs_root is None:
        p.error("--data-root and --runs-root are required when $SCRATCH is not set")
    if args.batch_seqs % args.micro_batch:
        p.error("--batch-seqs must be a multiple of --micro-batch")
    args.run_name = args.run_name or f"{args.attn}_{args.size}_s{args.seed}"
    return args


class TokenWindows:
    """Non-overlapping windows of ``length`` tokens over a flat uint16 token file."""

    def __init__(self, path: Path, length: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.length = length
        self.n_windows = len(self.tokens) // length

    def get(self, indices: np.ndarray) -> torch.Tensor:
        L = self.length
        batch = np.stack([self.tokens[i * L : (i + 1) * L] for i in indices]).astype(np.int64)
        return torch.from_numpy(batch)


def warmup_steps(args: argparse.Namespace) -> int:
    if args.warmup_steps is not None:
        return max(1, args.warmup_steps)
    return max(1, math.ceil(args.warmup_frac * args.max_steps))


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup to lr, then cosine or linear decay to min_lr at max_steps."""
    warmup = warmup_steps(args)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.max_steps - warmup)
    if args.schedule == "linear":
        return args.lr - (args.lr - args.min_lr) * progress
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Weight decay on 2-D weight matrices only: not on biases, LayerNorm or embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        is_embedding = "embed_tokens" in name or "embed_positions" in name
        (decay if param.ndim == 2 and not is_embedding else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = args.device.startswith("cuda")
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=fused)


def resolve_precision(args: argparse.Namespace) -> torch.dtype | None:
    if args.precision == "fp32" or not args.device.startswith("cuda"):
        return None
    if args.precision == "bf16" or (args.precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def lm_loss(model: torch.nn.Module, windows: torch.Tensor, amp_dtype: torch.dtype | None) -> torch.Tensor:
    inputs, targets = windows[:, :-1], windows[:, 1:]
    with torch.autocast(device_type=inputs.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        logits = model(input_ids=inputs).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


@contextlib.contextmanager
def noise_eval_samples(model: ConstrainedOPTForCausalLM, k: int):
    """Temporarily set how a noisy model is evaluated (K draws averaged; 0 = noise off)."""
    old = model.config.noise_eval_samples
    model.config.noise_eval_samples = k
    try:
        yield
    finally:
        model.config.noise_eval_samples = old


@torch.no_grad()
def evaluate(model, val: TokenWindows, args, amp_dtype, max_windows: int | None = None, batch_size: int | None = None) -> float:
    """Mean next-token loss over (the first eval_max_windows of) the validation windows."""
    model.eval()
    cap = args.eval_max_windows if max_windows is None else max_windows
    n = val.n_windows if cap is None else min(val.n_windows, cap)
    bs = batch_size or args.eval_batch
    total, count = 0.0, 0
    for start in range(0, n, bs):
        idx = np.arange(start, min(n, start + bs))
        batch = val.get(idx).to(args.device, non_blocking=True)
        total += lm_loss(model, batch, amp_dtype).item() * len(idx)
        count += len(idx)
    model.train()
    return total / count


@torch.no_grad()
def noisy_diagnostics(model: ConstrainedOPTForCausalLM, batch: torch.Tensor, amp_dtype) -> dict:
    """Per layer and head: decay rate theta = softplus(.) and the noise std at distances 0/10/100
    (1 / sqrt(tau)); per layer: RMS of the value vectors on ``batch`` (the noise std only means
    something relative to it, i.e. the signal-to-noise ratio). Plus the penalty term."""
    rms = []
    hooks = [a.v_proj.register_forward_hook(lambda m, i, o: rms.append(float(o.float().pow(2).mean().sqrt())))
             for a in model.noisy_attentions()]
    was_training = model.training
    model.eval()
    try:
        with noise_eval_samples(model, 0), torch.autocast(device_type=batch.device.type, dtype=amp_dtype or torch.float32,
                                                          enabled=amp_dtype is not None):
            model(input_ids=batch[:, :-1])
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    cfg = model.config
    theta = [a.decay_rate().tolist() for a in model.noisy_attentions()]
    std_at = {f"noise_std_d{d}": [[(cfg.noisy_tau0 * math.exp(-t * d) + cfg.noisy_tau_floor) ** -0.5 for t in layer]
                                  for layer in theta] for d in (0, 10, 100)}
    return {"penalty": float(noisy_precision_penalty(model)), "theta": theta, "value_rms": rms, **std_at}


def git_commit() -> dict[str, str | bool | None]:
    def run(*cmd: str) -> str | None:
        try:
            return subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


class CsvLog:
    """train_log.csv; on resume, rows from steps after the resume point are dropped."""

    def __init__(self, path: Path, resume_step: int):
        self.path = path
        rows = []
        if path.exists() and resume_step > 0:
            with open(path, newline="") as fh:
                rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= resume_step]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=LOG_FIELDS).writerow({k: row.get(k, "") for k in LOG_FIELDS})


class StopFlag:
    def __init__(self):
        self.stop = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._handler)

    def _handler(self, signum, frame):
        print(f"received signal {signum}; saving resume state after this step", flush=True)
        self.stop = True


def save_resume(path: Path, model, optimizer, scaler, step: int, args) -> None:
    state = {
        "step": step,  # number of optimizer steps completed = index of the next step
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def check_resume_args(saved: dict[str, str], args: argparse.Namespace) -> None:
    keys = ("attn", "size", "seed", "recipe", "seq_len", "batch_seqs", "max_steps", "lr", "min_lr", "schedule",
            "warmup_frac", "warmup_steps", "adam_eps", "init_std", "init_output_scale", "tie_embeddings",
            "weight_decay", "beta1", "beta2", "grad_clip")
    if args.attn_type == "noisy":
        keys = (*keys, "noisy_lambda")
    changed = {k: (saved.get(k), str(getattr(args, k))) for k in keys if saved.get(k) != str(getattr(args, k))}
    if changed:
        raise SystemExit(f"resume.pt was written with different settings: {changed}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.runs_root / args.run_name
    if (run_dir / "DONE").exists():
        print(f"{args.run_name}: already done", flush=True)
        return 0
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.data_root / "tokenizer"
    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    train = TokenWindows(args.data_root / "train.bin", args.seq_len + 1)
    val = TokenWindows(args.data_root / "val.bin", args.seq_len + 1)
    needed = args.max_steps * args.batch_seqs
    if needed > train.n_windows:
        raise SystemExit(f"need {needed:,} windows, train.bin has {train.n_windows:,}")
    perm = np.random.default_rng(args.seed).permutation(train.n_windows)[:needed]

    torch.manual_seed(args.seed)
    window_kw = {} if args.window_size is None else {"window_size": args.window_size}
    if args.noisy_lambda is not None:
        window_kw["noisy_lambda"] = args.noisy_lambda
    config = make_config(
        args.attn_type,
        args.size,
        **window_kw,
        max_position_embeddings=args.seq_len,
        init_std=args.init_std,
        init_output_scale=args.init_output_scale,
        tie_word_embeddings=args.tie_embeddings,
    )
    model = ConstrainedOPTForCausalLM(config)
    init_matched_weights(model, args.seed)
    model.to(args.device)
    model.train()
    amp_dtype = resolve_precision(args)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    resume_path = run_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        # Load on the CPU: torch.set_rng_state / cuda.set_rng_state_all only accept CPU ByteTensors
        # (map_location="cuda" made every GPU resume crash). load_state_dict copies the model
        # weights to their device, and the optimizer moves its state to each parameter's device.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        check_resume_args(state["args"], args)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        start_step = state["step"]
        print(f"{args.run_name}: resuming at step {start_step}", flush=True)

    if args.tie_embeddings and model.lm_head.weight is not model.model.decoder.embed_tokens.weight:
        raise SystemExit("tie_embeddings is set but lm_head is not tied to embed_tokens")
    n_params = sum(p.numel() for p in model.parameters())  # shared tensors counted once
    n_embed = sum(p.numel() for n, p in model.named_parameters() if "embed_" in n or n.startswith("lm_head"))
    tokens_per_step = args.batch_seqs * args.seq_len
    write_json_atomic(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "git": git_commit(),
            "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "train_bin_sha256": manifest.get("train", {}).get("sha256"),
            "val_bin_sha256": manifest.get("val", {}).get("sha256"),
            "tokenizer": manifest.get("train", {}).get("tokenizer"),
            "n_params": n_params,
            "n_params_non_embedding": n_params - n_embed,
            "tokens_per_step": tokens_per_step,
            "total_tokens": tokens_per_step * args.max_steps,
            "train_windows_available": train.n_windows,
            "val_windows": val.n_windows,
            "precision": str(amp_dtype or torch.float32),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
            },
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    if start_step == 0 and not args.no_save_init:
        save_hf(model, run_dir / "checkpoints" / "step_00000", tokenizer_dir)

    log = CsvLog(run_dir / "train_log.csv", start_step)
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(run_dir / "tb", purge_step=start_step)
    except ImportError:
        tb = None
    step_fn_model = torch.compile(model) if args.compile else model
    noisy = args.attn_type == "noisy"
    noisy_lambda = args.noisy_lambda or 0.0
    # Noisy evaluation averages over a seeded generator: run it eagerly (not through torch.compile).
    eval_model = model if noisy else step_fn_model
    noisy_log = run_dir / "noisy_log.jsonl"
    if noisy and start_step > 0 and noisy_log.exists():
        rows = [r for r in noisy_log.read_text().splitlines() if r and json.loads(r)["step"] <= start_step]
        noisy_log.write_text("".join(r + "\n" for r in rows))

    def log_noisy(step: int) -> None:
        diag = noisy_diagnostics(model, val.get(np.arange(min(2, val.n_windows))).to(args.device), amp_dtype)
        with open(noisy_log, "a") as fh:
            fh.write(json.dumps({"step": step, **diag}) + "\n")
        if tb is not None:
            tb.add_scalar("noisy/penalty", diag["penalty"], step)
            tb.add_scalar("noisy/theta_mean", float(np.mean(diag["theta"])), step)

    if noisy and start_step == 0:
        log_noisy(0)
    stop = StopFlag()
    n_micro = args.batch_seqs // args.micro_batch
    window_loss, window_steps = 0.0, 0
    window_t0 = time.time()
    steady_t0, steady_step0 = None, None

    step = start_step
    while step < args.max_steps:
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = train.get(perm[step * args.batch_seqs : (step + 1) * args.batch_seqs])
        batch = batch.pin_memory().to(args.device, non_blocking=True) if args.device.startswith("cuda") else batch
        step_loss = 0.0
        for micro in batch.split(args.micro_batch):
            ce = lm_loss(step_fn_model, micro, amp_dtype)
            loss = ce
            if noisy and noisy_lambda > 0:
                # Xu et al.'s hybrid loss: cross-entropy + lambda * mean precision. The penalty only
                # depends on theta, so adding it to every micro-batch (each scaled by 1/n_micro) adds
                # it once per optimizer step. train_loss logs the cross-entropy alone.
                loss = ce + noisy_lambda * noisy_precision_penalty(model)
            scaler.scale(loss / n_micro).backward()
            step_loss += ce.item() / n_micro
        if not math.isfinite(step_loss):
            raise SystemExit(f"{args.run_name}: non-finite loss at step {step}")
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_loss += step_loss
        window_steps += 1
        if step == start_step + 5:  # steady-state throughput excludes warm-up/compilation steps
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            steady_t0, steady_step0 = time.time(), step

        val_loss = None
        if step % args.eval_every == 0 or step == args.max_steps:
            with noise_eval_samples(model, args.noise_val_samples) if noisy else contextlib.nullcontext():
                val_loss = evaluate(eval_model, val, args, amp_dtype)
            if noisy:
                log_noisy(step)
        if step % args.log_every == 0 or val_loss is not None or step == args.max_steps:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - window_t0
            row = {
                "step": step,
                "tokens": step * tokens_per_step,
                "train_loss": round(window_loss / window_steps, 5),
                "val_loss": "" if val_loss is None else round(val_loss, 5),
                "lr": f"{lr:.3e}",
                "grad_norm": round(grad_norm, 4),
                "tokens_per_s": round(window_steps * tokens_per_step / elapsed),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if args.device.startswith("cuda") else "",
            }
            log.write(row)
            print(" ".join(f"{k}={v}" for k, v in row.items() if v != ""), flush=True)
            if tb is not None:
                tb.add_scalar("train/loss", row["train_loss"], step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/grad_norm", grad_norm, step)
                tb.add_scalar("train/tokens_per_s", row["tokens_per_s"], step)
                if val_loss is not None:
                    tb.add_scalar("val/loss", val_loss, step)
            window_loss, window_steps, window_t0 = 0.0, 0, time.time()

        if step % args.ckpt_every == 0 and step < args.max_steps:
            save_hf(model, run_dir / "checkpoints" / f"step_{step:05d}", tokenizer_dir)
        if step == args.max_steps:
            break
        if step % args.resume_every == 0 or stop.stop:
            save_resume(resume_path, model, optimizer, scaler, step, args)
            if stop.stop:
                print(f"{args.run_name}: stopped at step {step}, resume state saved", flush=True)
                return EXIT_INTERRUPTED

    save_hf(model, run_dir / "final", tokenizer_dir)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    steady = None
    if steady_t0 is not None and step > steady_step0:
        steady = (step - steady_step0) * tokens_per_step / (time.time() - steady_t0)
    extra = {}
    if noisy:
        # final_val_loss above used --noise-val-samples draws; report both evaluation modes too.
        k = model.config.noise_eval_samples
        nw = args.noise_final_windows
        with noise_eval_samples(model, 0):
            off_all = evaluate(model, val, args, amp_dtype)
            off_sub = evaluate(model, val, args, amp_dtype, max_windows=nw)
        with noise_eval_samples(model, k):
            k_sub = evaluate(model, val, args, amp_dtype, max_windows=nw, batch_size=max(1, args.eval_batch // 4))
        extra["noisy_final_val"] = {"noise_off": off_all, f"noise_off_first{nw}": off_sub, f"k{k}_first{nw}": k_sub,
                                    "noise_val_samples_in_log": args.noise_val_samples}
        print(f"{args.run_name}: val noise off {off_all:.4f}; first {nw} windows: off {off_sub:.4f}, K={k} {k_sub:.4f}",
              flush=True)
    write_json_atomic(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "steps": step,
            "tokens": step * tokens_per_step,
            "final_val_loss": val_loss,
            **extra,
            "tokens_per_s_steady": None if steady is None else round(steady),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if args.device.startswith("cuda") else None,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    (run_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
    resume_path.unlink(missing_ok=True)
    if tb is not None:
        tb.close()
    print(f"{args.run_name}: done, final val loss {val_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
