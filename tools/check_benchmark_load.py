"""Check that checkpoints work in the NeuroAI benchmark before (and after) a long training run.

Run with the BENCHMARK's interpreter, from anywhere:

    $WORK/benchmark/.venv/bin/python tools/check_benchmark_load.py CKPT [CKPT ...] \
        --benchmark-root $WORK/benchmark

For each checkpoint it compares a plain ``AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)`` load with the benchmark's own loader (``_get_cached_causal_lm``,
which has lm_head repair paths for trust_remote_code models), on CPU in fp32:

1. config: model_type, attn_type, max_position_embeddings, tie_word_embeddings;
   tokenizer: loads from the checkpoint, EOS id 0, no BOS added.
2. benchmark load: lm_head still tied to the input embedding, every tensor identical to the
   plain load, logits identical.
3. hidden states: L + 1 states, the benchmark's final-norm lookup finds final_layer_norm.
4. surprisal: ``batch_surprisal_one_model`` equals a hand computation from the plain model.
5. long input (> max_position_embeddings tokens): the benchmark truncates from the left and
   returns finite surprisal; the plain model also runs on exactly max_position_embeddings tokens.
6. the context length the benchmark's sliding-window code would use for this checkpoint.

Exits non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

PAIRS = [
    ("The cat sat on the", "mat"),
    ("After the long meeting, the committee finally reached a", "decision"),
    ("She opened the door and saw that the", "garden"),
    ("In 1969, astronauts landed on the", "moon"),
]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print(f"  ok  {msg}", flush=True)


def hand_surprisal(model, tokenizer, context: str, target: str) -> float:
    """Mean -log p over the target's tokens (prefix form, as the benchmark's _surprisal_one_pair)."""
    full = context.strip() + " " + target.strip()
    n_ctx = len(tokenizer(context.strip(), add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    tgt = ids[0, n_ctx:]
    return float(np.mean([-logp[n_ctx + j - 1, tgt[j]].item() for j in range(len(tgt))]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--benchmark-root", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.benchmark_root))
    torch.set_grad_enabled(False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neuroai.features.llm import _cap_to_training_length, _final_norm_module, _get_cached_causal_lm, clear_hf_model_cache
    from neuroai.features.surprisal import batch_surprisal_one_model

    for ckpt in args.checkpoints:
        path = str(ckpt.resolve())
        print(f"== {path}", flush=True)
        ref = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.float32).eval()
        cfg = ref.config
        L, T = cfg.num_hidden_layers, cfg.max_position_embeddings
        check(cfg.model_type == "constrained_opt", f"model_type={cfg.model_type} attn_type={cfg.attn_type}")
        print(f"      layers={L} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
              f"max_position_embeddings={T} tie={cfg.tie_word_embeddings} init_std={cfg.init_std} "
              f"init_output_scale={getattr(cfg, 'init_output_scale', None)}")
        if cfg.tie_word_embeddings:
            check(ref.lm_head.weight.data_ptr() == ref.model.decoder.embed_tokens.weight.data_ptr(),
                  "plain load: lm_head tied to embed_tokens")
        tok = AutoTokenizer.from_pretrained(path)
        check(tok.eos_token_id == 0, f"tokenizer eos_token_id={tok.eos_token_id}")
        ids = tok("Hello world", add_special_tokens=True)["input_ids"]
        check(ids == tok("Hello world", add_special_tokens=False)["input_ids"], "tokenizer adds no BOS/EOS")

        # 2. the benchmark's loader
        btok, bmodel = _get_cached_causal_lm(path, revision=None, device="cpu", load_kw={"trust_remote_code": True})
        bmodel = bmodel.float().eval()
        if cfg.tie_word_embeddings:
            check(bmodel.lm_head.weight.data_ptr() == bmodel.get_input_embeddings().weight.data_ptr(),
                  "benchmark load: lm_head still tied")
        ref_sd, b_sd = ref.state_dict(), bmodel.state_dict()
        check(ref_sd.keys() == b_sd.keys(), f"benchmark load: same {len(ref_sd)} state-dict keys")
        bad = [k for k in ref_sd if not torch.equal(ref_sd[k], b_sd[k].to(ref_sd[k].dtype))]
        check(not bad, f"benchmark load: all tensors identical to the plain load {bad[:3]}")
        x = torch.tensor([tok("The quick brown fox jumps over the lazy dog.")["input_ids"]])
        out_ref = ref(input_ids=x, output_hidden_states=True)
        out_b = bmodel(input_ids=x, output_hidden_states=True, use_cache=False)
        diff = (out_ref.logits - out_b.logits.float()).abs().max().item()
        check(diff < 1e-5, f"benchmark load: logits match (max diff {diff:.2e})")
        check(btok("Hello world")["input_ids"] == ids, "benchmark tokenizer = checkpoint tokenizer")

        # 3. hidden states / final norm
        check(len(out_b.hidden_states) == L + 1, f"{len(out_b.hidden_states)} hidden states = L + 1")
        check(_final_norm_module(bmodel) is bmodel.model.decoder.final_layer_norm, "final-norm module found")
        check(bmodel.get_output_embeddings() is bmodel.lm_head, "get_output_embeddings() is lm_head")
        clear_hf_model_cache()

        # 4. surprisal through the benchmark vs by hand
        ctxs, tgts = [c for c, _ in PAIRS], [t for _, t in PAIRS]
        bench = batch_surprisal_one_model(path, ctxs, tgts, device="cpu", trust_remote_code=True)[:, 0]
        hand = np.array([hand_surprisal(ref, tok, c, t) for c, t in PAIRS])
        check(np.all(np.isfinite(bench)), f"benchmark surprisal finite: {np.round(bench, 3).tolist()}")
        check(np.allclose(bench, hand, atol=1e-3), f"benchmark surprisal = hand computation (max diff "
              f"{np.abs(bench - hand).max():.2e})")

        # 5. inputs longer than the training context
        long_ctx = " ".join(["The committee met again on the following day to discuss the report."] * 400)
        n_long = len(tok(long_ctx)["input_ids"])
        check(n_long > T, f"long context has {n_long} tokens > {T}")
        s_long = batch_surprisal_one_model(path, [long_ctx], ["report"], device="cpu", trust_remote_code=True)[0, 0]
        check(math.isfinite(float(s_long)), f"benchmark surprisal on a {n_long}-token context is finite ({s_long:.3f})")
        full = ref(input_ids=torch.tensor([tok(long_ctx)["input_ids"][-T:]])).logits
        check(bool(torch.isfinite(full).all()), f"plain forward on exactly {T} tokens is finite")

        # 6. sliding-window context length the benchmark would use
        try:
            from transformers import AutoConfig

            pre = getattr(AutoConfig.from_pretrained(path), "max_position_embeddings", None)
        except Exception as exc:  # custom model_type without trust_remote_code
            pre = None
            print(f"      note: AutoConfig without trust_remote_code fails ({type(exc).__name__}); "
                  f"the benchmark then falls back to 2048")
        eff = _cap_to_training_length(pre or 2048, path)
        check(eff == T, f"benchmark sliding-window length {eff} = training context {T}")
        clear_hf_model_cache()
        del ref, bmodel
    print("ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
