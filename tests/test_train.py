"""Training loop on synthetic data (CPU): it learns, and a killed run resumes exactly."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

import train

COMMON = ["--size", "2l", "--seed", "1", "--seq-len", "32", "--batch-seqs", "4", "--micro-batch", "2",
          "--max-steps", "6", "--eval-every", "3", "--eval-batch", "4", "--ckpt-every", "100",
          "--resume-every", "1000", "--log-every", "1", "--device", "cpu"]


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    (root / "tokenizer").mkdir(parents=True)
    rng = np.random.default_rng(0)
    # A learnable stream: a repeating pattern with some noise.
    pattern = np.tile(np.arange(2, 50, dtype=np.uint16), 400)
    noise = rng.integers(2, 50, pattern.size, dtype=np.uint16)
    stream = np.where(rng.random(pattern.size) < 0.1, noise, pattern).astype(np.uint16)
    stream.tofile(root / "train.bin")
    stream[:2000].tofile(root / "val.bin")
    (root / "manifest.json").write_text(json.dumps({"train": {"sha256": "x"}, "val": {"sha256": "y"}}))
    return root


class StopAfter:
    """Stands in for train.StopFlag: reports a stop request after n steps."""

    def __init__(self, n):
        self.n, self.reads = n, 0

    @property
    def stop(self):
        self.reads += 1
        return self.reads >= self.n


def run(args, data_root, runs_root):
    return train.main(["--data-root", str(data_root), "--runs-root", str(runs_root), *args])


def test_resume_is_exact(data_root, tmp_path, monkeypatch):
    args = ["--attn", "forget", *COMMON]
    assert run(args, data_root, tmp_path / "a") == 0

    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(3))
    assert run(args, data_root, tmp_path / "b") == train.EXIT_INTERRUPTED
    run_b = tmp_path / "b" / "forget_2l_s1"
    assert (run_b / "resume.pt").exists() and not (run_b / "DONE").exists()
    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(10**9))
    assert run(args, data_root, tmp_path / "b") == 0

    wa = load_file(tmp_path / "a" / "forget_2l_s1" / "final" / "model.safetensors")
    wb = load_file(run_b / "final" / "model.safetensors")
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    log = (run_b / "train_log.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in log[1:]] == [1, 2, 3, 4, 5, 6]
    # A finished run is skipped when relaunched.
    assert run(args, data_root, tmp_path / "b") == 0


def test_loss_decreases_and_matched_init(data_root, tmp_path):
    for attn in ("vanilla", "stickbreaking"):
        assert run(["--attn", attn, *COMMON, "--max-steps", "20", "--lr", "3e-3"], data_root, tmp_path) == 0
        rows = (tmp_path / f"{attn}_2l_s1" / "train_log.csv").read_text().splitlines()[1:]
        losses = [float(r.split(",")[2]) for r in rows]
        assert losses[-1] < losses[0] - 0.5, (attn, losses)
    init = [load_file(tmp_path / f"{a}_2l_s1" / "checkpoints" / "step_00000" / "model.safetensors")
            for a in ("vanilla", "stickbreaking")]
    assert all(torch.equal(init[0][k], init[1][k]) for k in init[0])
