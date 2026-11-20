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
import csv
import hashlib
import json
import math
import os
import platform
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
from model.modeling import ConstrainedOPTForCausalLM, init_matched_weights

REPO_DIR = Path(__file__).resolve().parent
EXIT_INTERRUPTED = 99
LOG_FIELDS = ["step", "tokens", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_s", "peak_mem_gb"]


def default_root(name: str) -> Path | None:
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) / "attn_constraints" / name if scratch else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", required=True, choices=ATTN_TYPES)
    p.add_argument("--size", required=True, choices=tuple(SIZES))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--window-size", type=int, default=None,
                   help="N for --attn window (N-gram attention); the run is named window{N}_...")
    p.add_argument("--data-root", type=Path, default=default_root("data"))
    p.add_argument("--runs-root", type=Path, default=default_root("runs"))
    p.add_argument("--run-name", default=None, help="default: {attn}_{size}_s{seed}")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-seqs", type=int, default=256, help="sequences per optimizer step")
    p.add_argument("--micro-batch", type=int, default=16, help="sequences per forward pass")
    p.add_argument("--max-steps", type=int, default=7630, help="7630 x 256 x 1024 = 2.0B tokens")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-max-windows", type=int, default=None, help="cap validation (pilot only)")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--no-save-init", action="store_true", help="skip checkpoints/step_00000")
    p.add_argument("--resume-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT") or None,
                   help="wandb project for the live dashboard (default: $WANDB_PROJECT; "
                        "unset = no wandb, TensorBoard only)")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    args = p.parse_args(argv)
    if args.data_root is None or args.runs_root is None:
        p.error("--data-root and --runs-root are required when $SCRATCH is not set")
    if args.batch_seqs % args.micro_batch:
        p.error("--batch-seqs must be a multiple of --micro-batch")
    if (args.attn == "window") != (args.window_size is not None):
        p.error("--window-size is required with --attn window, and only allowed with it")
    args.attn_label = f"window{args.window_size}" if args.attn == "window" else args.attn
    args.run_name = args.run_name or f"{args.attn_label}_{args.size}_s{args.seed}"
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


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup over the first warmup_frac of steps, then cosine decay to min_lr."""
    warmup = max(1, math.ceil(args.warmup_frac * args.max_steps))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.max_steps - warmup)
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
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), fused=fused)


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


@torch.no_grad()
def evaluate(model, val: TokenWindows, args, amp_dtype) -> float:
    """Mean next-token loss over (the first eval_max_windows of) the validation windows."""
    model.eval()
    n = val.n_windows if args.eval_max_windows is None else min(val.n_windows, args.eval_max_windows)
    total, count = 0.0, 0
    for start in range(0, n, args.eval_batch):
        idx = np.arange(start, min(n, start + args.eval_batch))
        batch = val.get(idx).to(args.device, non_blocking=True)
        total += lm_loss(model, batch, amp_dtype).item() * len(idx)
        count += len(idx)
    model.train()
    return total / count


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


class Tracker:
    """Optional live dashboard (Weights & Biases).

    The wandb run id is derived from the run directory, so a restarted (resumed) run continues
    the same wandb run. Logging must never stop training: any wandb error disables it for the
    rest of the run and leaves train_log.csv and TensorBoard as the record.
    """

    def __init__(self, args: argparse.Namespace, config: dict, run_dir: Path):
        self.run = None
        if not args.wandb_project:
            return
        try:
            import wandb

            run_id = f"{args.run_name}-{hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:8]}"
            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                group=f"{args.size}_s{args.seed}",  # one matched set per group
                job_type=args.attn_label,
                id=run_id,
                resume="allow",
                config=config,
                dir=str(run_dir),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: wandb disabled ({type(exc).__name__}: {exc})", flush=True)

    def log(self, metrics: dict, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: wandb disabled ({type(exc).__name__}: {exc})", flush=True)
            self.run = None

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: wandb finish failed ({type(exc).__name__}: {exc})", flush=True)


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
    keys = ("attn", "window_size", "size", "seed", "seq_len", "batch_seqs", "max_steps", "lr", "min_lr",
            "warmup_frac", "weight_decay", "beta1", "beta2", "grad_clip")
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
    config = make_config(
        args.attn, args.size, max_position_embeddings=args.seq_len, window_size=args.window_size
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
        state = torch.load(resume_path, map_location=args.device, weights_only=False)
        check_resume_args(state["args"], args)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_cpu"])
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start_step = state["step"]
        print(f"{args.run_name}: resuming at step {start_step}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
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
    tracker = Tracker(
        args,
        config={
            "attn": args.attn_label, "size": args.size, "seed": args.seed, "n_params": n_params,
            "lr": args.lr, "batch_tokens": tokens_per_step, "max_steps": args.max_steps,
            "micro_batch": args.micro_batch, "precision": str(amp_dtype or torch.float32),
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        },
        run_dir=run_dir,
    )
    step_fn_model = torch.compile(model) if args.compile else model
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
            loss = lm_loss(step_fn_model, micro, amp_dtype) / n_micro
            scaler.scale(loss).backward()
            step_loss += loss.item()
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
            val_loss = evaluate(step_fn_model, val, args, amp_dtype)
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
            metrics = {
                "train/loss": row["train_loss"], "train/lr": lr, "train/grad_norm": grad_norm,
                "train/tokens_per_s": row["tokens_per_s"], "train/tokens": row["tokens"],
            }
            if val_loss is not None:
                metrics["val/loss"] = val_loss
            tracker.log(metrics, step)
            window_loss, window_steps, window_t0 = 0.0, 0, time.time()

        if step % args.ckpt_every == 0 and step < args.max_steps:
            save_hf(model, run_dir / "checkpoints" / f"step_{step:05d}", tokenizer_dir)
        if step == args.max_steps:
            break
        if step % args.resume_every == 0 or stop.stop:
            save_resume(resume_path, model, optimizer, scaler, step, args)
            if stop.stop:
                print(f"{args.run_name}: stopped at step {step}, resume state saved", flush=True)
                tracker.finish()
                return EXIT_INTERRUPTED

    save_hf(model, run_dir / "final", tokenizer_dir)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    steady = None
    if steady_t0 is not None and step > steady_step0:
        steady = (step - steady_step0) * tokens_per_step / (time.time() - steady_t0)
    write_json_atomic(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "steps": step,
            "tokens": step * tokens_per_step,
            "final_val_loss": val_loss,
            "tokens_per_s_steady": None if steady is None else round(steady),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if args.device.startswith("cuda") else None,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    (run_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
    tracker.finish()
    resume_path.unlink(missing_ok=True)
    if tb is not None:
        tb.close()
    print(f"{args.run_name}: done, final val loss {val_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
