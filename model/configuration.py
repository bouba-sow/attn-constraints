"""Config for the matched attention-constraint suite.

``ConstrainedOPTConfig`` is an ``OPTConfig`` plus ``attn_type``. All other settings are the
same for every attention type. Position embeddings exist only for ``vanilla_pos``; every
other type is NoPE (the matched comparison).
"""

from __future__ import annotations

from transformers import OPTConfig

ATTN_TYPES = ("vanilla", "vanilla_pos", "alibi", "forget", "stickbreaking", "window")

# Stage 1 (2l, 4l) and stage 2 (125m). Vocab is the Pythia/GPT-NeoX tokenizer padded to 50,304.
SIZES = {
    "2l": dict(num_hidden_layers=2, hidden_size=256, num_attention_heads=4, ffn_dim=1024),
    "4l": dict(num_hidden_layers=4, hidden_size=384, num_attention_heads=6, ffn_dim=1536),
    "125m": dict(num_hidden_layers=12, hidden_size=768, num_attention_heads=12, ffn_dim=3072),
}


class ConstrainedOPTConfig(OPTConfig):
    model_type = "constrained_opt"

    def __init__(
        self,
        attn_type: str = "vanilla",
        window_size: int = 3,
        alibi_slope_scale: float = 1.0,
        init_output_scale: bool = False,
        **kwargs,
    ):
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"attn_type must be one of {ATTN_TYPES}, got {attn_type!r}")
        self.attn_type = attn_type
        # Only used by attn_type="window": keys restricted to i - n < j <= i.
        self.window_size = window_size
        # Only used by attn_type="alibi": multiplies the fixed slopes. 1.0 for training;
        # set to 0.0 at evaluation time to check that a trained model relies on its bias.
        self.alibi_slope_scale = alibi_slope_scale
        # OPT / Megatron-LM init: the residual-branch output projections (self_attn.out_proj,
        # fc2) use init_std / sqrt(2 * num_hidden_layers). Off for stage 1, on for the OPT recipe.
        self.init_output_scale = init_output_scale
        kwargs.setdefault("vocab_size", 50304)
        kwargs.setdefault("max_position_embeddings", 1024)
        kwargs.setdefault("do_layer_norm_before", True)
        kwargs.setdefault("activation_function", "relu")
        kwargs.setdefault("enable_bias", True)
        kwargs.setdefault("layer_norm_elementwise_affine", True)
        kwargs.setdefault("dropout", 0.0)
        kwargs.setdefault("attention_dropout", 0.0)
        kwargs.setdefault("activation_dropout", 0.0)
        kwargs.setdefault("layerdrop", 0.0)
        kwargs.setdefault("init_std", 0.02)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 0)
        kwargs.setdefault("pad_token_id", 1)
        super().__init__(**kwargs)
        # No project_in/project_out: embeddings live in the hidden dimension.
        self.word_embed_proj_dim = self.hidden_size

    @property
    def use_positions(self) -> bool:
        return self.attn_type == "vanilla_pos"


def make_config(attn_type: str, size: str, **overrides) -> ConstrainedOPTConfig:
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}, got {size!r}")
    return ConstrainedOPTConfig(attn_type=attn_type, **SIZES[size], **overrides)
