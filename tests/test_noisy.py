"""Noisy attention (Xu, Dillon & Futrell 2026): noise model, penalty, evaluation modes (CPU, fp32)."""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import build

import train
from model.modeling import (
    noisy_output_std,
    noisy_precision_penalty,
    noisy_variance,
    precision_penalty_terms,
)

BIG_STD = 0.5


def xu_precision_regularizer_sum(thetas_raw: list[torch.Tensor], T: int, tau0: float) -> torch.Tensor:
    """Their train_noisy.py::precision_regularizer_sum, transcribed (one theta tensor per layer)."""
    reg_sum, total_terms = 0.0, 0.0
    for raw in thetas_raw:
        theta = F.softplus(raw)
        pos = torch.arange(T, dtype=theta.dtype)
        d = (pos.view(1, T, 1) - pos.view(1, 1, T)).abs()
        tau = tau0 * torch.exp(-theta.view(-1, 1, 1) * d)
        mask = torch.tril(torch.ones(T, T, dtype=theta.dtype)).view(1, T, T)
        tau = tau * mask
        reg_sum += tau.sum()
        total_terms += float(theta.numel()) * float(mask.sum().item())
    return reg_sum / total_terms


def test_penalty_matches_xu_code():
    model = build("noisy", seed=3, num_hidden_layers=3)
    with torch.no_grad():  # spread the decay rates so every term matters
        for i, attn in enumerate(model.noisy_attentions()):
            attn.noisy_theta.copy_(torch.tensor([-3.0, -0.5, 0.2 + i, 2.0]))
    ours = noisy_precision_penalty(model)
    theirs = xu_precision_regularizer_sum([a.noisy_theta.detach() for a in model.noisy_attentions()], 512, 1e4)
    assert torch.allclose(ours, theirs, rtol=1e-5)
    s, n = precision_penalty_terms(torch.tensor([0.0]), 4, 2.0)  # theta = 0: every pair has tau = tau0
    assert n == 10 and torch.isclose(s, torch.tensor(20.0))


def test_penalty_uses_fixed_reference_window_and_has_gradient():
    model = build("noisy", seed=1, max_position_embeddings=32)
    p512 = noisy_precision_penalty(model)
    model.config.noisy_ref_len = 2048
    assert noisy_precision_penalty(model) < p512  # ~1/T: why lambda needs a fixed window
    model.config.noisy_ref_len = 512
    noisy_precision_penalty(model).backward()
    grads = [a.noisy_theta.grad for a in model.noisy_attentions()]
    assert all(g is not None and (g < 0).all() for g in grads)  # larger theta -> lower penalty


def test_variance_and_std_match_loop():
    torch.manual_seed(0)
    b, h, t = 2, 3, 6
    theta = torch.tensor([0.1, 1.0, 3.0])
    pos = torch.arange(t)
    distance = (pos[:, None] - pos[None, :]).clamp(min=0)
    weights = torch.softmax(torch.randn(b, h, t, t).masked_fill(pos[None, :] > pos[:, None], float("-inf")), -1)
    var = noisy_variance(theta, distance, 1e4, 1e-4)
    std = noisy_output_std(weights, var)
    for bi in range(b):
        for hi in range(h):
            for i in range(t):
                v = sum(weights[bi, hi, i, j] ** 2 / (1e4 * math.exp(-theta[hi].item() * (i - j)) + 1e-4)
                        for j in range(i + 1))
                assert math.isclose(std[bi, hi, i, 0].item(), math.sqrt(v), rel_tol=1e-5)


def test_noise_off_equals_vanilla(ids):
    a = build("noisy", seed=4, init_std=BIG_STD, noise_eval_samples=0)
    v = build("vanilla", seed=4, init_std=BIG_STD)
    with torch.no_grad():
        assert torch.allclose(a(input_ids=ids).logits, v(input_ids=ids).logits, atol=1e-5)


def test_negligible_noise_equals_vanilla_with_noise_on(ids):
    """theta ~ 0 and a huge tau0: every pair is almost noise-free, so K draws ~ the vanilla output."""
    a = build("noisy", seed=4, init_std=BIG_STD, noise_eval_samples=4, noisy_tau0=1e14, noisy_tau_floor=0.0)
    with torch.no_grad():
        for attn in a.noisy_attentions():
            attn.noisy_theta.fill_(-60.0)
    v = build("vanilla", seed=4, init_std=BIG_STD)
    with torch.no_grad():
        ref = F.log_softmax(v(input_ids=ids).logits, -1)
        assert torch.allclose(a(input_ids=ids).logits, ref, atol=1e-4)


def test_eval_is_reproducible_and_averages_k_draws(ids):
    model = build("noisy", seed=5, init_std=BIG_STD, noise_eval_samples=3)
    with torch.no_grad():
        out1, out2 = model(input_ids=ids).logits, model(input_ids=ids).logits
        assert torch.equal(out1, out2)  # seeded generator at every call
        # Manual: three passes with the same seeded generator, mean of the probabilities.
        gen = torch.Generator().manual_seed(model.config.noise_eval_seed)
        attns = model.noisy_attentions()
        for attn in attns:
            attn.noise_eval_active, attn.noise_generator = True, gen
        logps = [F.log_softmax(model.lm_head(model.model.decoder(ids, None, False)[0]), -1) for _ in range(3)]
        for attn in attns:
            attn.noise_eval_active, attn.noise_generator = False, None
        manual = torch.logsumexp(torch.stack(logps), 0) - math.log(3)
        assert torch.allclose(out1, manual, atol=1e-6)
        assert torch.allclose(out1.exp().sum(-1), torch.ones(ids.shape))  # a distribution
        model.config.noise_eval_samples = 0
        assert not torch.allclose(model(input_ids=ids).logits, out1, atol=1e-4)  # the noise does something


def test_training_mode_samples_noise_and_eval_leaves_no_state(ids):
    model = build("noisy", seed=6, init_std=BIG_STD).train()
    with torch.no_grad():
        assert not torch.allclose(model(input_ids=ids).logits, model(input_ids=ids).logits)
    model.eval()
    model(input_ids=ids)
    assert all(not a.noise_eval_active and a.noise_generator is None for a in model.noisy_attentions())
    model.config.noise_eval_samples = 0
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, model(input_ids=ids).logits)


def test_hidden_states_from_noise_off_pass(ids):
    model = build("noisy", seed=2, init_std=BIG_STD, noise_eval_samples=2)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        model.config.noise_eval_samples = 0
        ref = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert len(hs) == model.config.num_hidden_layers + 1
    assert all(torch.equal(x, y) for x, y in zip(hs, ref))


def test_theta_init_as_xu():
    model = build("noisy", seed=8, num_hidden_layers=3)
    raw = torch.stack([a.noisy_theta.detach() for a in model.noisy_attentions()])
    assert torch.allclose(raw, torch.ones_like(raw), atol=5e-3) and raw.std() > 0
    again = torch.stack([a.noisy_theta.detach() for a in build("noisy", seed=8, num_hidden_layers=3).noisy_attentions()])
    assert torch.equal(raw, again)


@pytest.mark.parametrize("spec, lam", [("noisy_l0", 0.0), ("noisy_l0.001", 0.001), ("noisy_l0.01", 0.01),
                                       ("noisy_l0.1", 0.1), ("noisy_l1", 1.0), ("noisy_l1.5", 1.5)])
def test_parse_noisy_spec(spec, lam):
    assert train.parse_attn_spec(spec) == ("noisy", None, lam)
    a = train.parse_args(["--attn", spec, "--size", "125m", "--seed", "2", "--data-root", "d", "--runs-root", "r"])
    assert (a.run_name, a.attn_type, a.noisy_lambda, a.window_size) == (f"{spec}_125m_s2", "noisy", lam, None)


@pytest.mark.parametrize("bad", ["noisy", "noisy_l", "noisy_l-1", "noisy_lx", "noisy_l01", "noisy0.01", "noisy_l.5"])
def test_parse_noisy_rejects(bad):
    with pytest.raises(ValueError):
        train.parse_attn_spec(bad)


def test_noisy_training_run(tmp_path):
    """Short CPU run: the penalty pushes theta up, diagnostics and both final evaluations are written."""
    root = tmp_path / "data"
    (root / "tokenizer").mkdir(parents=True)
    rng = np.random.default_rng(0)
    stream = np.tile(np.arange(2, 50, dtype=np.uint16), 400)
    stream = np.where(rng.random(stream.size) < 0.1, rng.integers(2, 50, stream.size), stream).astype(np.uint16)
    stream.tofile(root / "train.bin")
    stream[:2000].tofile(root / "val.bin")
    (root / "manifest.json").write_text(json.dumps({"train": {"sha256": "x"}, "val": {"sha256": "y"}}))
    common = ["--size", "2l", "--seed", "1", "--seq-len", "32", "--batch-seqs", "4", "--micro-batch", "2",
              "--max-steps", "6", "--eval-every", "3", "--eval-batch", "4", "--ckpt-every", "100", "--resume-every",
              "1000", "--log-every", "1", "--device", "cpu", "--lr", "0.05", "--noise-final-windows", "8",
              "--data-root", str(root), "--runs-root", str(tmp_path / "runs")]
    for lam in ("0.1", "0"):
        assert train.main(["--attn", f"noisy_l{lam}", *common]) == 0
    runs = tmp_path / "runs"
    logs = {lam: [json.loads(r) for r in (runs / f"noisy_l{lam}_2l_s1" / "noisy_log.jsonl").read_text().splitlines()]
            for lam in ("0.1", "0")}
    assert [r["step"] for r in logs["0.1"]] == [0, 3, 6]
    theta = {lam: np.mean(rows[-1]["theta"]) for lam, rows in logs.items()}
    assert theta["0.1"] > np.mean(logs["0.1"][0]["theta"])  # the penalty raises the decay rate
    assert theta["0.1"] > theta["0"]
    assert len(logs["0.1"][-1]["value_rms"]) == 2 and logs["0.1"][-1]["penalty"] < logs["0.1"][0]["penalty"]
    summary = json.loads((runs / "noisy_l0.1_2l_s1" / "summary.json").read_text())
    final = summary["noisy_final_val"]
    assert set(final) == {"noise_off", "noise_off_first8", "k16_first8", "noise_val_samples_in_log"}
    cfg = json.loads((runs / "noisy_l0.1_2l_s1" / "final" / "config.json").read_text())
    assert (cfg["attn_type"], cfg["noisy_lambda"], cfg["noise_eval_samples"]) == ("noisy", 0.1, 16)
