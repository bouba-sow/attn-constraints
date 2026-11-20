"""Training loop on synthetic data (CPU): it learns, and a killed run resumes exactly."""

import json
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

import train

COMMON = ["--size", "test", "--seed", "1", "--seq-len", "32", "--batch-seqs", "4", "--micro-batch", "2",
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
    run_b = tmp_path / "b" / "forget_test_s1"
    assert (run_b / "resume.pt").exists() and not (run_b / "DONE").exists()
    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(10**9))
    assert run(args, data_root, tmp_path / "b") == 0

    wa = load_file(tmp_path / "a" / "forget_test_s1" / "final" / "model.safetensors")
    wb = load_file(run_b / "final" / "model.safetensors")
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    log = (run_b / "train_log.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in log[1:]] == [1, 2, 3, 4, 5, 6]
    # A finished run is skipped when relaunched.
    assert run(args, data_root, tmp_path / "b") == 0


class FakeWandb:
    """Stands in for the wandb module. Records calls; ``fail_after`` makes run.log() raise,
    as a network error would."""

    def __init__(self, fail_after=None):
        self.inits, self.logged, self.finished, self.fail_after = [], [], 0, fail_after

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return self

    def log(self, metrics, step=None):
        if self.fail_after is not None and len(self.logged) >= self.fail_after:
            raise ConnectionError("wandb unreachable")
        self.logged.append((step, metrics))

    def finish(self):
        self.finished += 1


@pytest.mark.parametrize("fail_after", [None, 2])
def test_wandb_logging_never_stops_training(data_root, tmp_path, monkeypatch, fail_after):
    fake = FakeWandb(fail_after)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = ["--attn", "alibi", *COMMON, "--wandb-project", "p"]
    assert run(args, data_root, tmp_path) == 0
    init = fake.inits[0]
    assert init["project"] == "p" and init["name"] == "alibi_test_s1" and init["group"] == "test_s1"
    assert init["id"].startswith("alibi_test_s1-") and init["resume"] == "allow"
    if fail_after is None:
        assert [s for s, _ in fake.logged] == [1, 2, 3, 4, 5, 6]
        assert "val/loss" in dict(fake.logged)[3] and fake.finished == 1
    else:
        assert len(fake.logged) == 2 and fake.finished == 0  # disabled after the failure
    assert (tmp_path / "alibi_test_s1" / "DONE").exists()


def test_wandb_run_id_is_stable_across_restarts(data_root, tmp_path, monkeypatch):
    ids = []
    for _ in range(2):
        fake = FakeWandb()
        monkeypatch.setitem(sys.modules, "wandb", fake)
        monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(2))
        run(["--attn", "forget", *COMMON, "--wandb-project", "p"], data_root, tmp_path)
        ids.append(fake.inits[0]["id"])
    assert ids[0] == ids[1]


def test_loss_decreases_and_matched_init(data_root, tmp_path):
    for attn in ("vanilla", "stickbreaking"):
        assert run(["--attn", attn, *COMMON, "--max-steps", "20", "--lr", "3e-3"], data_root, tmp_path) == 0
        rows = (tmp_path / f"{attn}_test_s1" / "train_log.csv").read_text().splitlines()[1:]
        losses = [float(r.split(",")[2]) for r in rows]
        assert losses[-1] < losses[0] - 0.5, (attn, losses)
    init = [load_file(tmp_path / f"{a}_test_s1" / "checkpoints" / "step_00000" / "model.safetensors")
            for a in ("vanilla", "stickbreaking")]
    assert all(torch.equal(init[0][k], init[1][k]) for k in init[0])


def test_window_run_names_and_validation():
    base = ["--size", "4l", "--seed", "2", "--data-root", "d", "--runs-root", "r"]
    args = train.parse_args(["--attn", "window", "--window-size", "10", *base])
    assert args.run_name == "window10_4l_s2" and args.attn_label == "window10"
    assert train.parse_args(["--attn", "forget", *base]).run_name == "forget_4l_s2"
    for bad in (["--attn", "window", *base], ["--attn", "alibi", "--window-size", "6", *base]):
        with pytest.raises(SystemExit):
            train.parse_args(bad)
    from model.configuration import make_config

    assert make_config("window", "2l", window_size=14).window_size == 14
    with pytest.raises(ValueError):
        make_config("window", "2l")
