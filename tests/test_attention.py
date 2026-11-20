"""Attention variants: equivalences, causality, and vectorized-vs-loop checks (CPU, fp32)."""

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import ALL_TYPES, build
from transformers import OPTConfig, OPTForCausalLM

from model.modeling import alibi_slopes, is_constraint_param, stickbreaking_weights

# Larger init than training (0.02) so that differences between variants are not hidden by
# near-zero activations.
BIG_STD = 0.5


def test_vanilla_pos_matches_hf_opt_eager(ids):
    ours = build("vanilla_pos", init_std=BIG_STD)
    cfg = ours.config
    hf_cfg = OPTConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        ffn_dim=cfg.ffn_dim,
        num_attention_heads=cfg.num_attention_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        word_embed_proj_dim=cfg.hidden_size,
        do_layer_norm_before=True,
        activation_function="relu",
        enable_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=1,
    )
    hf_cfg._attn_implementation = "eager"
    hf = OPTForCausalLM(hf_cfg).eval()
    hf.load_state_dict(ours.state_dict(), strict=True)
    mask = torch.ones_like(ids)
    mask[1, 12:] = 0  # right padding on the second sequence
    with torch.no_grad():
        a = ours(input_ids=ids, attention_mask=mask).logits
        b = hf(input_ids=ids, attention_mask=mask).logits
    valid = mask.bool()
    assert (a[valid] - b[valid]).abs().max().item() < 1e-4


def test_nope_types_have_no_position_parameters():
    for attn in ALL_TYPES:
        names = [n for n, _ in build(attn).named_parameters()]
        has_pos = any("embed_positions" in n for n in names)
        assert has_pos == (attn == "vanilla_pos"), attn


@pytest.mark.parametrize("attn", ALL_TYPES)
def test_causality(attn, ids):
    model = build(attn, init_std=BIG_STD)
    t = 7
    changed = ids.clone()
    changed[:, t + 1 :] = (ids[:, t + 1 :] + 13) % 97
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=changed).logits
    assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-5)
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :], atol=1e-5)


def test_alibi_slopes():
    assert alibi_slopes(4) == [1 / 4, 1 / 16, 1 / 64, 1 / 256]
    assert alibi_slopes(6) == [1 / 4, 1 / 16, 1 / 64, 1 / 256, 1 / 2, 1 / 8]
    s12 = alibi_slopes(12)
    assert len(s12) == 12 and all(s > 0 for s in s12)
    attn = build("alibi").model.decoder.layers[0].self_attn
    assert attn.alibi_slopes == alibi_slopes(4)
    assert not any("alibi" in n for n, _ in attn.named_parameters())
    assert not any("alibi" in n for n, _ in attn.named_buffers())


def test_alibi_bias_is_minus_slope_times_distance():
    attn = build("alibi").model.decoder.layers[0].self_attn
    t, h = 6, 4
    scores = torch.randn(1, h, t, t)
    weights = attn.attention_weights(scores, torch.ones(1, t))
    slopes = torch.tensor(alibi_slopes(h)).view(1, h, 1, 1)
    i, j = torch.meshgrid(torch.arange(t), torch.arange(t), indexing="ij")
    expected = scores - slopes * (i - j).float()
    expected = expected.masked_fill(j > i, float("-inf")).softmax(-1)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_alibi_slope_scale_zero_is_vanilla(ids):
    alibi0 = build("alibi", init_std=BIG_STD, alibi_slope_scale=0.0)
    vanilla = build("vanilla", init_std=BIG_STD)
    with torch.no_grad():
        assert torch.allclose(alibi0(input_ids=ids).logits, vanilla(input_ids=ids).logits, atol=1e-5)


def test_forget_gate_open_equals_vanilla(ids):
    forget = build("forget", init_std=BIG_STD)
    vanilla = build("vanilla", init_std=BIG_STD)
    with torch.no_grad():
        for layer in forget.model.decoder.layers:
            layer.self_attn.fgate_proj.weight.zero_()
            layer.self_attn.fgate_proj.bias.fill_(50.0)  # f = sigmoid(50) ~ 1, log f ~ 0
        assert torch.allclose(forget(input_ids=ids).logits, vanilla(input_ids=ids).logits, atol=1e-5)


def test_forget_bias_matches_loop():
    attn = build("forget").model.decoder.layers[0].self_attn
    t, h = 8, 4
    scores = torch.randn(1, h, t, t)
    fgate = torch.randn(1, t, h) * 2
    weights = attn.attention_weights(scores, torch.ones(1, t), fgate)
    log_f = F.logsigmoid(fgate)[0]  # (t, h)
    expected = torch.full((1, h, t, t), float("-inf"))
    for hh in range(h):
        for i in range(t):
            for j in range(i + 1):
                bias = sum(log_f[k, hh].item() for k in range(j + 1, i + 1))
                expected[0, hh, i, j] = scores[0, hh, i, j] + bias
    assert torch.allclose(weights, expected.softmax(-1), atol=1e-6)


def test_stickbreaking_matches_loop():
    t, h = 8, 3
    scores = torch.randn(2, h, t, t) * 2
    allowed = torch.ones(t, t).tril().bool()[None, None].expand(2, h, t, t)
    weights = stickbreaking_weights(scores, allowed)
    beta = torch.sigmoid(scores)
    expected = torch.zeros_like(scores)
    for b in range(2):
        for hh in range(h):
            for i in range(t):
                for j in range(i + 1):
                    keep = math.prod(1 - beta[b, hh, i, k].item() for k in range(j + 1, i + 1))
                    expected[b, hh, i, j] = beta[b, hh, i, j] * keep
    assert torch.allclose(weights, expected, atol=1e-6)
    assert (weights.sum(-1) <= 1 + 1e-6).all()
    # The current token is included: the diagonal is sigmoid(s_ii), never 0.
    assert torch.allclose(weights.diagonal(dim1=-2, dim2=-1), beta.diagonal(dim1=-2, dim2=-1))


def test_window_restricts_keys():
    attn = build("window", window_size=3).model.decoder.layers[0].self_attn
    t = 8
    weights = attn.attention_weights(torch.randn(1, 4, t, t), torch.ones(1, t))
    i, j = torch.meshgrid(torch.arange(t), torch.arange(t), indexing="ij")
    outside = (j > i) | (i - j >= 3)
    assert (weights[..., outside] == 0).all()
    assert torch.allclose(weights.sum(-1), torch.ones(1, 4, t))


def test_shared_weights_identical_across_types():
    seed = 7
    states = {attn: build(attn, seed=seed).state_dict() for attn in ALL_TYPES}
    ref = states["vanilla"]
    for attn, state in states.items():
        shared = {k for k in state if not is_constraint_param(k)}
        assert shared == set(ref), attn
        for k in shared:
            assert torch.equal(state[k], ref[k]), (attn, k)
    other_seed = build("vanilla", seed=seed + 1).state_dict()
    assert not torch.equal(other_seed["lm_head.weight"], ref["lm_head.weight"])
    # Constraint-only parameters are deterministic too.
    a, b = build("forget", seed=seed).state_dict(), build("forget", seed=seed).state_dict()
    assert torch.equal(a["model.decoder.layers.0.self_attn.fgate_proj.weight"],
                       b["model.decoder.layers.0.self_attn.fgate_proj.weight"])


def test_bf16_autocast_runs_and_stays_finite(ids):
    for attn in ALL_TYPES:
        model = build(attn)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits
        assert torch.isfinite(logits.float()).all(), attn
