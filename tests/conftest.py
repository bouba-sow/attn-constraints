import sys
from pathlib import Path

import pytest
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from model.configuration import ATTN_TYPES, ConstrainedOPTConfig  # noqa: E402
from model.modeling import ConstrainedOPTForCausalLM, init_matched_weights  # noqa: E402

# The five stage-1 grid types plus window (the hard N-gram models; window3-5 in the stage-2 grid).
ALL_TYPES = ATTN_TYPES


def tiny_config(attn_type: str, **overrides) -> ConstrainedOPTConfig:
    kw = dict(
        attn_type=attn_type,
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=64,
        max_position_embeddings=32,
    )
    kw.update(overrides)
    return ConstrainedOPTConfig(**kw)


def build(attn_type: str, seed: int = 0, **overrides) -> ConstrainedOPTForCausalLM:
    torch.manual_seed(seed)
    model = ConstrainedOPTForCausalLM(tiny_config(attn_type, **overrides))
    init_matched_weights(model, seed)
    return model.eval()


@pytest.fixture
def ids() -> torch.Tensor:
    return torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(123))
