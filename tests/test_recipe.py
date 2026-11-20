"""The OPT-125M recipe (Jean Zay stage 2): recipe resolution, LR schedule, tied embeddings,
scaled output init, matched init, save/load and exact resume with every OPT setting on."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import REPO_DIR, build
from safetensors.torch import load_file

import train
from model.checkpoint import save_hf
from model.modeling import OUTPUT_PROJ_SUFFIXES, is_constraint_param

GRID = ("vanilla", "vanilla_pos", "alibi", "forget", "stickbreaking")
OPT = dict(init_std=0.006, init_output_scale=True, tie_word_embeddings=True)


def args_for(*argv):
    return train.parse_args(["--attn", "vanilla", "--seed", "1", "--data-root", "d", "--runs-root", "r", *argv])


# --- recipe resolution -------------------------------------------------------------------


def test_125m_defaults_to_opt_recipe():
    a = args_for("--size", "125m")
    assert a.recipe == "opt"
    assert (a.seq_len, a.batch_seqs, a.max_steps) == (2048, 256, 9540)
    assert a.batch_seqs * a.seq_len == 524_288  # OPT-125M: 0.5M tokens per batch
    assert a.batch_seqs * a.seq_len * a.max_steps == pytest.approx(5.0e9, rel=1e-3)
    assert (a.lr, a.min_lr, a.schedule) == (6e-4, 6e-5, "linear")
    assert train.warmup_steps(a) == 715  # 375M tokens / 524,288
    assert (a.beta1, a.beta2, a.adam_eps, a.weight_decay, a.grad_clip) == (0.9, 0.95, 1e-8, 0.1, 1.0)
    assert (a.init_std, a.init_output_scale, a.tie_embeddings) == (0.006, True, True)
    assert (a.micro_batch, a.compile) == (4, True)
    # A checkpoint every ~500M tokens, the last one landing on the final step.
    assert a.ckpt_every == 954 and a.max_steps % a.ckpt_every == 0
    assert a.ckpt_every * a.batch_seqs * a.seq_len == pytest.approx(5.0e8, rel=1e-3)


@pytest.mark.parametrize("size", ["2l", "4l"])
def test_stage1_defaults_unchanged(size):
    a = args_for("--size", size)
    assert a.recipe == "stage1"
    assert (a.seq_len, a.batch_seqs, a.max_steps, a.lr, a.min_lr) == (1024, 256, 7630, 1e-3, 0.0)
    assert (a.schedule, a.warmup_frac, a.warmup_steps, train.warmup_steps(a)) == ("cosine", 0.01, None, 77)
    assert (a.adam_eps, a.init_std, a.init_output_scale, a.tie_embeddings, a.ckpt_every) == (
        1e-8, 0.02, False, False, 1000)
    assert (a.micro_batch, a.compile) == (16, False)


def test_flags_override_recipe():
    a = args_for("--size", "125m", "--max-steps", "10", "--lr", "1e-4", "--no-tie-embeddings")
    assert (a.max_steps, a.lr, a.tie_embeddings, a.seq_len) == (10, 1e-4, False, 2048)


# --- learning-rate schedule --------------------------------------------------------------


def test_opt_lr_schedule():
    a = args_for("--size", "125m")
    lrs = np.array([train.lr_at(s, a) for s in range(a.max_steps)])
    assert lrs[0] == pytest.approx(6e-4 / 715)
    assert lrs[714] == pytest.approx(6e-4)  # end of warmup = peak
    assert lrs[715] == pytest.approx(6e-4)
    assert np.all(np.diff(lrs[:715]) > 0) and np.all(np.diff(lrs[715:]) < 0)
    # Linear: constant decrement after warmup, reaching 10% of the peak at max_steps.
    assert np.allclose(np.diff(lrs[715:]), -(6e-4 - 6e-5) / (a.max_steps - 715))
    assert train.lr_at(a.max_steps, a) == pytest.approx(6e-5)
    assert lrs[715:].min() >= 6e-5 - 1e-12  # never below 10% of the peak after warmup


def test_stage1_lr_schedule_is_cosine():
    a = args_for("--size", "2l")
    assert train.lr_at(0, a) == pytest.approx(1e-3 / 77)
    mid = 77 + (a.max_steps - 77) // 2
    assert train.lr_at(mid, a) == pytest.approx(5e-4, rel=1e-3)
    assert train.lr_at(a.max_steps, a) == pytest.approx(0.0, abs=1e-12)


# --- model: tying, init ------------------------------------------------------------------


@pytest.mark.parametrize("attn", GRID)
def test_tied_embeddings(attn, ids):
    model = build(attn, seed=1, **OPT)
    assert model.lm_head.weight is model.model.decoder.embed_tokens.weight
    names = [n for n, _ in model.named_parameters()]
    assert "lm_head.weight" not in names  # counted and initialized once
    # Gradients from the output layer reach the shared matrix.
    model.train()
    model(input_ids=ids, labels=ids).loss.backward()
    assert model.model.decoder.embed_tokens.weight.grad is not None


def test_output_projection_init_is_scaled():
    layers = 6
    model = build("vanilla", seed=1, num_hidden_layers=layers, hidden_size=128, ffn_dim=512, **OPT)
    scaled = 0.006 / math.sqrt(2 * layers)
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            assert torch.all(p == 0)
        elif "layer_norm" in name:
            assert torch.all(p == 1)
        else:
            expected = scaled if name.endswith(OUTPUT_PROJ_SUFFIXES) else 0.006
            assert p.std().item() == pytest.approx(expected, rel=0.1), name
    n_scaled = sum(n.endswith(OUTPUT_PROJ_SUFFIXES) for n, _ in model.named_parameters())
    assert n_scaled == 2 * layers


def test_shared_init_matched_with_opt_settings():
    ref = dict(build("vanilla", seed=4, **OPT).named_parameters())
    for attn in GRID[1:]:
        for name, p in build(attn, seed=4, **OPT).named_parameters():
            if is_constraint_param(name):
                continue
            assert torch.equal(p, ref[name]), (attn, name)


LOAD_SCRIPT = r"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM
ckpts, ids = json.loads(sys.argv[1]), torch.tensor(json.loads(sys.argv[2]))
for path in ckpts:
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32).eval()
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr(), "not tied"
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    diff = float(np.abs(out.logits.numpy() - np.load(path + "/expected_logits.npy")).max())
    assert diff < 1e-6, (path, diff)
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1
    print("ok", path)
"""


def test_tied_save_and_load_in_fresh_process(tmp_path, ids):
    paths = []
    for attn in GRID:
        model = build(attn, seed=2, **OPT)
        out = save_hf(model, tmp_path / attn, tokenizer_dir=None)
        with torch.no_grad():
            np.save(out / "expected_logits.npy", model(input_ids=ids).logits.numpy())
        config = json.loads((out / "config.json").read_text())
        assert config["tie_word_embeddings"] is True and config["init_output_scale"] is True
        paths.append(str(out))
    result = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, json.dumps(paths), json.dumps(ids.tolist())],
        cwd=tmp_path, capture_output=True, text=True,
        env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin", "HF_HOME": str(tmp_path / "hf")},
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.count("ok ") == len(GRID)


# --- training with the OPT recipe ----------------------------------------------------------

# The opt recipe on a small size, shrunk to CPU scale: tied embeddings, scaled init, linear
# schedule with warmup steps, and min_lr all stay on.
OPT_SMALL = ["--size", "2l", "--recipe", "opt", "--seed", "1", "--seq-len", "32", "--batch-seqs", "4",
             "--micro-batch", "2", "--max-steps", "8", "--warmup-steps", "3", "--eval-every", "4",
             "--eval-batch", "4", "--ckpt-every", "4", "--resume-every", "1000", "--log-every", "1",
             "--no-compile", "--device", "cpu"]


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    (root / "tokenizer").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pattern = np.tile(np.arange(2, 50, dtype=np.uint16), 400)
    noise = rng.integers(2, 50, pattern.size, dtype=np.uint16)
    stream = np.where(rng.random(pattern.size) < 0.1, noise, pattern).astype(np.uint16)
    stream.tofile(root / "train.bin")
    stream[:2000].tofile(root / "val.bin")
    (root / "manifest.json").write_text(json.dumps({"train": {"sha256": "x"}, "val": {"sha256": "y"}}))
    return root


class StopAfter:
    def __init__(self, n):
        self.n, self.reads = n, 0

    @property
    def stop(self):
        self.reads += 1
        return self.reads >= self.n


def run(args, data_root, runs_root):
    return train.main(["--data-root", str(data_root), "--runs-root", str(runs_root), *args])


@pytest.mark.parametrize("attn", ["forget", "vanilla_pos"])
def test_opt_recipe_trains_saves_and_resumes_exactly(attn, data_root, tmp_path, monkeypatch):
    args = ["--attn", attn, *OPT_SMALL]
    assert run(args, data_root, tmp_path / "a") == 0
    run_a = tmp_path / "a" / f"{attn}_2l_s1"
    cfg = json.loads((run_a / "run_config.json").read_text())
    assert cfg["args"]["recipe"] == "opt" and cfg["args"]["tie_embeddings"] is True
    final_cfg = json.loads((run_a / "final" / "config.json").read_text())
    assert final_cfg["tie_word_embeddings"] and final_cfg["max_position_embeddings"] == 32
    assert sorted(p.name for p in (run_a / "checkpoints").iterdir()) == ["step_00000", "step_00004"]
    log = (run_a / "train_log.csv").read_text().splitlines()
    lrs = [float(r.split(",")[4]) for r in log[1:]]
    assert lrs[2] == pytest.approx(6e-4, rel=1e-3) and lrs[-1] == pytest.approx(6e-4 - (6e-4 - 6e-5) * 4 / 5, rel=1e-3)

    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(3))
    assert run(args, data_root, tmp_path / "b") == train.EXIT_INTERRUPTED
    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(10**9))
    assert run(args, data_root, tmp_path / "b") == 0
    wa = load_file(run_a / "final" / "model.safetensors")
    wb = load_file(tmp_path / "b" / f"{attn}_2l_s1" / "final" / "model.safetensors")
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


def test_resume_refuses_changed_recipe(data_root, tmp_path, monkeypatch):
    args = ["--attn", "vanilla", *OPT_SMALL]
    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(2))
    assert run(args, data_root, tmp_path) == train.EXIT_INTERRUPTED
    monkeypatch.setattr(train, "StopFlag", lambda: StopAfter(10**9))
    with pytest.raises(SystemExit, match="different settings"):
        run([*args, "--no-tie-embeddings"], data_root, tmp_path)


# --- hard N-gram window models (window3, window4, window5) ---------------------------------


@pytest.mark.parametrize("spec, expected", [("forget", ("forget", None)), ("window", ("window", None)),
                                            ("window3", ("window", 3)), ("window12", ("window", 12))])
def test_parse_attn(spec, expected):
    assert train.parse_attn(spec) == expected


@pytest.mark.parametrize("bad", ["window0", "windowx", "window-3", "ngram3", "Window3"])
def test_parse_attn_rejects(bad):
    with pytest.raises(ValueError):
        train.parse_attn(bad)
    with pytest.raises(SystemExit):
        train.parse_args(["--attn", bad, "--size", "125m", "--seed", "1", "--data-root", "d", "--runs-root", "r"])


def test_window_run_name_and_args():
    a = train.parse_args(["--attn", "window4", "--size", "125m", "--seed", "2", "--data-root", "d", "--runs-root", "r"])
    assert (a.run_name, a.attn_type, a.window_size, a.recipe) == ("window4_125m_s2", "window", 4, "opt")


@pytest.mark.parametrize("n", [3, 4, 5])
def test_window_n_mask(n):
    """Query i attends exactly to keys i-n+1..i (n tokens, current included), weights sum to 1."""
    attn = build("window", window_size=n).model.decoder.layers[0].self_attn
    t = 12
    w = attn.attention_weights(torch.randn(2, 4, t, t), torch.ones(2, t))
    i, j = torch.meshgrid(torch.arange(t), torch.arange(t), indexing="ij")
    inside = (j <= i) & (i - j < n)
    assert (w[..., ~inside] == 0).all() and (w[..., inside] > 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, t))
    assert torch.equal(inside.sum(-1), torch.clamp(torch.arange(t) + 1, max=n))


def test_window_receptive_field_grows_with_depth():
    """A token's logits depend on exactly the last L(n-1)+1 tokens (L layers, window n)."""
    n, layers, t = 3, 3, 16
    model = build("window", seed=1, window_size=n, num_hidden_layers=layers, max_position_embeddings=t)
    x = torch.randint(0, 97, (1, t), generator=torch.Generator().manual_seed(0))
    reach = layers * (n - 1) + 1
    base = model(input_ids=x).logits[0, -1]
    for pos in range(t - 1):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 1) % 97
        changed = not torch.allclose(model(input_ids=y).logits[0, -1], base)
        assert changed == (pos >= t - reach), pos


def test_window_shares_every_weight_with_vanilla():
    ref = build("vanilla", seed=5, **OPT).state_dict()
    for n in (3, 4, 5):
        sd = build("window", seed=5, window_size=n, **OPT).state_dict()
        assert sd.keys() == ref.keys() and all(torch.equal(sd[k], ref[k]) for k in ref)


def test_window_save_load_keeps_n(tmp_path, ids):
    paths = []
    for n in (3, 4, 5):
        model = build("window", seed=2, window_size=n, **OPT)
        out = save_hf(model, tmp_path / f"window{n}", tokenizer_dir=None)
        with torch.no_grad():
            np.save(out / "expected_logits.npy", model(input_ids=ids).logits.numpy())
        assert json.loads((out / "config.json").read_text())["window_size"] == n
        paths.append(str(out))
    result = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, json.dumps(paths), json.dumps(ids.tolist())],
        cwd=tmp_path, capture_output=True, text=True,
        env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin", "HF_HOME": str(tmp_path / "hf")},
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_window_trains_with_opt_recipe(data_root, tmp_path):
    assert run(["--attn", "window4", *OPT_SMALL], data_root, tmp_path) == 0
    d = tmp_path / "window4_2l_s1"
    cfg = json.loads((d / "final" / "config.json").read_text())
    assert (cfg["attn_type"], cfg["window_size"], cfg["tie_word_embeddings"]) == ("window", 4, True)
    assert json.loads((d / "run_config.json").read_text())["args"]["window_size"] == 4
