"""Checkpoint round trip through a fresh process, and the tokenization stream."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zstandard
from conftest import ALL_TYPES, REPO_DIR, build

from model.checkpoint import save_hf

LOAD_SCRIPT = r"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM
ckpts, ids = json.loads(sys.argv[1]), torch.tensor(json.loads(sys.argv[2]))
for path in ckpts:
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
    expected = np.load(path + "/expected_logits.npy")
    diff = float(np.abs(out.logits.numpy() - expected).max())
    assert diff < 1e-6, (path, diff)
    # What the benchmark relies on: L + 1 hidden states, the final-norm path, the unembedding.
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1
    assert isinstance(model.model.decoder.final_layer_norm, torch.nn.LayerNorm)
    assert model.get_output_embeddings() is model.lm_head
    print("ok", path)
"""


def test_save_and_load_in_fresh_process(tmp_path, ids):
    paths = []
    for attn in ALL_TYPES:
        model = build(attn, seed=3)
        out = save_hf(model, tmp_path / attn, tokenizer_dir=None)
        with torch.no_grad():
            np.save(out / "expected_logits.npy", model(input_ids=ids).logits.numpy())
        assert (out / "modeling.py").exists() and (out / "configuration.py").exists()
        config = json.loads((out / "config.json").read_text())
        assert config["model_type"] == "constrained_opt" and config["attn_type"] == attn
        assert config["auto_map"]["AutoModelForCausalLM"] == "modeling.ConstrainedOPTForCausalLM"
        paths.append(str(out))
    # Run outside the repository so the copied code is what gets imported.
    result = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, json.dumps(paths), json.dumps(ids.tolist())],
        cwd=tmp_path, capture_output=True, text=True,
        env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin", "HF_HOME": str(tmp_path / "hf")},
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.count("ok ") == len(ALL_TYPES)


# --- tokenization -------------------------------------------------------------------------


def find_tokenizer() -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    for repo in ("EleutherAI/pythia-70m", "EleutherAI/pythia-70m-deduped"):
        path = try_to_load_from_cache(repo, "tokenizer.json")
        if isinstance(path, str):
            return Path(path)
    return None


DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "",  # empty documents are skipped
    "A document that mentions <|endoftext|> in its text.",
    "Unicode: café, naïve, 東京.\nSecond line.",
    "Short.",
    "Last document, which should be cut off by max_tokens.",
]


def write_jsonl_zst(path: Path, docs: list[str]) -> None:
    raw = "".join(json.dumps({"text": d, "meta": {}}) + "\n" for d in docs).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))


def test_tokenize_round_trip(tmp_path):
    tokenizer_path = find_tokenizer()
    if tokenizer_path is None:
        pytest.skip("Pythia tokenizer not in the HF cache")
    sys.path.insert(0, str(REPO_DIR / "data"))
    from tokenizers import Tokenizer
    from tokenize_pile import tokenize_split

    src = tmp_path / "docs.jsonl.zst"
    write_jsonl_zst(src, DOCS)
    kept = [d for d in DOCS if d]
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.encode_special_tokens = True
    lengths = [len(tok.encode(d, add_special_tokens=False).ids) + 1 for d in kept]

    # Full file: every document, EOS count == document count, exact decode.
    stats = tokenize_split(src, tmp_path / "all.bin", tokenizer_path, 10**9, workers=2, chunk_docs=2)
    tokens = np.fromfile(tmp_path / "all.bin", dtype=np.uint16)
    assert stats["n_docs"] == len(kept) and stats["n_eos"] == len(kept)
    assert stats["n_empty_docs_skipped"] == 1
    assert len(tokens) == sum(lengths) == stats["n_tokens"]
    docs_out = np.split(tokens, np.flatnonzero(tokens == 0) + 1)[:-1]
    assert [tok.decode(d[:-1].tolist(), skip_special_tokens=False) for d in docs_out] == kept

    # max_tokens inside document 3: stop at the end of document 3.
    limit = sum(lengths[:2]) + 1
    stats = tokenize_split(src, tmp_path / "cut.bin", tokenizer_path, limit, workers=2, chunk_docs=2)
    assert stats["n_docs"] == 3 and stats["n_tokens"] == sum(lengths[:3])
    assert np.fromfile(tmp_path / "cut.bin", dtype=np.uint16)[-1] == 0
