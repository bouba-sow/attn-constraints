"""Validation loss of saved checkpoints, loaded the way the benchmark loads them.

Loads each checkpoint with ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
and reports the mean next-token loss on ``val.bin`` (same windows as training). ``--alibi-check``
also evaluates ALiBi checkpoints with their slopes set to 0: the loss must get worse, otherwise
the model ignores its bias. Noisy checkpoints are evaluated as saved (K noise draws averaged,
config ``noise_eval_samples``) and, with ``--noise-samples 0 16``, in each listed mode (0 = noise off).

    uv run python evaluate.py $RUNS_ROOT/alibi_4l_s1/final --alibi-check
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@torch.no_grad()
def val_loss(model, tokens: np.ndarray, seq_len: int, batch: int, max_windows: int | None, device: str) -> float:
    length = seq_len + 1
    n = len(tokens) // length
    if max_windows is not None:
        n = min(n, max_windows)
    total, count = 0.0, 0
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        windows = torch.from_numpy(tokens[start * length : stop * length].astype(np.int64)).view(-1, length)
        windows = windows.to(device)
        logits = model(input_ids=windows[:, :-1]).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), windows[:, 1:].reshape(-1))
        total += loss.item() * (stop - start)
        count += stop - start
    return total / count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    default_val = Path(os.environ.get("SCRATCH", ".")) / "attn_constraints" / "data" / "val.bin"
    p.add_argument("--val-bin", type=Path, default=default_val)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--alibi-check", action="store_true", help="also evaluate ALiBi models with slopes = 0")
    p.add_argument("--noise-samples", type=int, nargs="+", default=None,
                   help="noisy models: evaluate with each K (noise draws averaged; 0 = noise off)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tokens = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    results = []
    for ckpt in args.checkpoints:
        scales = [1.0]
        attn_type = json.loads((ckpt / "config.json").read_text())["attn_type"]
        if args.alibi_check and attn_type == "alibi":
            scales.append(0.0)
        row = {"checkpoint": str(ckpt), "attn_type": attn_type}
        for scale in scales:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, trust_remote_code=True, alibi_slope_scale=scale, torch_dtype=torch.float32
            ).to(args.device).eval()
            seq_len = model.config.max_position_embeddings
            loss = val_loss(model, tokens, seq_len, args.batch, args.max_windows, args.device)
            row["val_loss" if scale == 1.0 else "val_loss_slopes0"] = round(loss, 5)
            if attn_type == "noisy":
                row["noise_eval_samples"] = model.config.noise_eval_samples
                for k in args.noise_samples or []:
                    model.config.noise_eval_samples = k
                    row[f"val_loss_noise_k{k}"] = round(
                        val_loss(model, tokens, seq_len, args.batch, args.max_windows, args.device), 5)
            del model
        if "val_loss_slopes0" in row:
            row["alibi_uses_bias"] = row["val_loss_slopes0"] > row["val_loss"]
        row["val_ppl"] = round(math.exp(row["val_loss"]), 3)
        results.append(row)
        print(json.dumps(row), flush=True)
    if any(r.get("alibi_uses_bias") is False for r in results):
        raise SystemExit("ALiBi check failed: a model is not worse with its slopes set to 0")


if __name__ == "__main__":
    main()
