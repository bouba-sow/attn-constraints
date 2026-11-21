"""OPT-style decoder whose attention score function is the only thing that varies.

Parameter names follow HF ``OPTForCausalLM`` exactly (``model.decoder.layers.N.self_attn.q_proj``,
``model.decoder.final_layer_norm``, ``lm_head``...), so a ``vanilla_pos`` state dict loads
into HF's OPT unchanged (tests/test_attention.py checks the logits match). The modules
are written out here rather than subclassed from ``transformers.models.opt`` because OPT's
internal attention API differs between transformers 4.x and 5.x, and checkpoints must load
on both Torch and Jean Zay.

Attention types (s_ij = q_i . k_j / sqrt(d_head), causal j <= i):

* ``vanilla`` / ``vanilla_pos``: softmax(s). ``vanilla_pos`` alone has learned absolute
  position embeddings (OPT's, offset 2); every other type has none (NoPE).
* ``alibi``: softmax(s_ij - m_h (i - j)), fixed Press et al. slopes m_h > 0.
* ``forget``: softmax(s_ij + F_i - F_j), F_i = sum_{t<=i} log sigmoid(w_h . x_t + b_h) (FoX).
* ``stickbreaking``: A_ij = sigmoid(s_ij) prod_{j<k<=i} (1 - sigmoid(s_ik)), no softmax and
  no renormalization (rows sum to <= 1).
* ``window``: softmax(s) with keys restricted to i - n < j <= i (pilot sanity check only).
* ``noisy`` (Xu, Dillon & Futrell 2026): vanilla attention whose head output gets Gaussian
  noise, z_i = sum_j a_ij v_j + sigma_i * eps, sigma_i^2 = sum_j a_ij^2 / tau_ij,
  tau_ij = tau0 exp(-theta_h (i - j)) + tau_floor (their reparameterization of noisy value
  vectors), with a learned decay rate theta_h = softplus(.) per head and a precision penalty
  (``noisy_precision_penalty``) added to the training loss. Noise is sampled in training; at
  evaluation see ``ConstrainedOPTConfig.noise_eval_samples``.

Scores, masks, softmax and cumulative sums run in fp32 even under bf16 autocast, as explicit
T x T tensors (no custom kernels, no flash attention), so any GPU works.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration import ConstrainedOPTConfig

# OPT's learned position embeddings are indexed with an offset of 2 (HF compatibility).
POSITION_OFFSET = 2

# Parameters that exist only for some attention types. They are initialized from a separate
# generator so the shared weights are bit-identical across attention types within a seed.
CONSTRAINT_PARAM_KEYS = ("fgate_proj", "embed_positions", "noisy_theta")
# Xu et al.'s code initializes theta (before softplus) as init + 1e-3 * N(0, 1).
NOISY_THETA_INIT_STD = 1e-3


def alibi_slopes(n_heads: int) -> list[float]:
    """Press et al. (2022) slopes, including their recipe for a non-power-of-2 head count."""

    def power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return power_of_2(closest) + alibi_slopes(2 * closest)[0::2][: n_heads - closest]


def stickbreaking_weights(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """A_ij = sigmoid(s_ij) * prod_{j<k<=i} (1 - sigmoid(s_ik)) over allowed (i, j), else 0.

    Computed in log space: log A_ij = logsigmoid(s_ij) + sum_{k>j, allowed} logsigmoid(-s_ik).
    The sum over k > j is an exclusive reversed cumulative sum along the key axis.
    """
    log_beta = F.logsigmoid(scores)
    log_keep = F.logsigmoid(-scores).masked_fill(~allowed, 0.0)
    inclusive = log_keep.flip(-1).cumsum(-1).flip(-1)  # sum over k >= j
    exclusive = torch.cat([inclusive[..., 1:], torch.zeros_like(inclusive[..., :1])], dim=-1)
    return torch.exp(log_beta + exclusive).masked_fill(~allowed, 0.0)


def noisy_variance(theta: torch.Tensor, distance: torch.Tensor, tau0: float, tau_floor: float) -> torch.Tensor:
    """Per-pair noise variance 1 / tau_ij, tau_ij = tau0 * exp(-theta_h * d_ij) + tau_floor.

    ``theta`` is the positive decay rate (h,), ``distance`` the (t, t) matrix of i - j clamped at 0
    (entries above the diagonal get attention weight 0, so their value does not matter). Returns
    (1, h, t, t). The floor is Xu et al.'s ("added to every tau_qk to avoid numerical underflow").
    """
    tau = tau0 * torch.exp(-theta.view(-1, 1, 1) * distance.float()) + tau_floor
    return (1.0 / tau)[None]


def noisy_output_std(weights: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """sigma_i = sqrt(sum_j a_ij^2 / tau_ij): std of the noise on each head output, (b, h, t, 1)."""
    return torch.sqrt((weights * weights * variance).sum(-1, keepdim=True))


def precision_penalty_terms(theta: torch.Tensor, ref_len: int, tau0: float) -> tuple[torch.Tensor, int]:
    """Sum and count of tau0 * exp(-theta_h * (q - k)) over 0 <= k <= q < ref_len and heads.

    Closed form of Xu et al.'s ``precision_regularizer_sum`` (train_noisy.py): a distance d occurs
    ref_len - d times; the count is n_heads * ref_len (ref_len + 1) / 2 (their code counts the
    diagonal, unlike the T (T - 1) / 2 of their Eq. 9). No floor, as in their code.
    """
    d = torch.arange(ref_len, device=theta.device, dtype=torch.float32)
    tau = tau0 * torch.exp(-theta.float().view(-1, 1) * d)  # (h, ref_len)
    total = (tau * (ref_len - d)).sum()
    return total, theta.numel() * ref_len * (ref_len + 1) // 2


class ConstrainedOPTAttention(nn.Module):
    def __init__(self, config: ConstrainedOPTConfig):
        super().__init__()
        self.attn_type = config.attn_type
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scaling = self.head_dim**-0.5
        self.window_size = config.window_size
        bias = config.enable_bias
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if self.attn_type == "alibi":
            # Fixed, not trained, and deliberately not a buffer: transformers 5 builds models on
            # the meta device and only fills in saved tensors, so a non-persistent buffer would
            # come back uninitialized from from_pretrained.
            self.alibi_slopes = [m * config.alibi_slope_scale for m in alibi_slopes(self.num_heads)]
        if self.attn_type == "forget":
            # FoX "full" forget gate: one scalar gate per head from the (post-LayerNorm) input.
            self.fgate_proj = nn.Linear(self.embed_dim, self.num_heads, bias=True)
        if self.attn_type == "noisy":
            # Decay rate before softplus, one per head (init_matched_weights sets init + 1e-3 N(0,1)).
            self.noisy_theta = nn.Parameter(torch.full((self.num_heads,), float(config.noisy_theta_init)))
            self.noisy_tau0 = float(config.noisy_tau0)
            self.noisy_tau_floor = float(config.noisy_tau_floor)
            # Noise is on in training mode. At evaluation it is on only while the causal-LM wrapper
            # averages over noise draws (noise_eval_active), drawn from noise_generator.
            self.noise_eval_active = False
            self.noise_generator: torch.Generator | None = None

    def decay_rate(self) -> torch.Tensor:
        """theta_h = softplus(noisy_theta) > 0, (h,)."""
        return F.softplus(self.noisy_theta.float())

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        q = self._split(self.q_proj(hidden_states))
        k = self._split(self.k_proj(hidden_states))
        v = self._split(self.v_proj(hidden_states))
        out_dtype = v.dtype
        fgate = self.fgate_proj(hidden_states) if self.attn_type == "forget" else None

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # (b, h, t, t)
            weights = self.attention_weights(scores, key_padding_mask, fgate)
            out = torch.matmul(weights, v)
            if self.attn_type == "noisy" and (self.training or self.noise_eval_active):
                pos = torch.arange(t, device=scores.device)
                distance = (pos[:, None] - pos[None, :]).clamp(min=0)
                variance = noisy_variance(self.decay_rate(), distance, self.noisy_tau0, self.noisy_tau_floor)
                eps = torch.randn(out.shape, dtype=out.dtype, device=out.device, generator=self.noise_generator)
                out = out + noisy_output_std(weights, variance) * eps

        out = out.to(out_dtype).transpose(1, 2).reshape(b, t, self.embed_dim)
        return self.out_proj(out)

    def attention_weights(
        self,
        scores: torch.Tensor,
        key_padding_mask: torch.Tensor,
        fgate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw scores (b, h, t, t) -> attention weights for this layer's attn_type, in fp32.

        ``fgate`` is the forget-gate pre-activation (b, t, h); only used by ``forget``.
        """
        scores = scores.float()
        t = scores.shape[-1]
        pos = torch.arange(t, device=scores.device)
        distance = pos[:, None] - pos[None, :]  # i - j
        allowed = (distance >= 0)[None, None] & key_padding_mask.bool()[:, None, None, :]

        if self.attn_type == "stickbreaking":
            return stickbreaking_weights(scores, allowed)
        if self.attn_type == "alibi":
            slopes = torch.tensor(self.alibi_slopes, dtype=scores.dtype, device=scores.device)
            scores = scores - slopes.view(1, -1, 1, 1) * distance.clamp(min=0).float()
        elif self.attn_type == "forget":
            cum = F.logsigmoid(fgate.float()).transpose(1, 2).cumsum(-1)  # (b, h, t): F_i
            scores = scores + cum[..., :, None] - cum[..., None, :]
        elif self.attn_type == "window":
            allowed = allowed & (distance < self.window_size)[None, None]
        # finfo.min rather than -inf: a row with no allowed key (a left-padded query) then gets
        # finite, ignorable weights instead of NaN.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)


class ConstrainedOPTDecoderLayer(nn.Module):
    def __init__(self, config: ConstrainedOPTConfig):
        super().__init__()
        affine = config.layer_norm_elementwise_affine
        self.self_attn = ConstrainedOPTAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=affine)
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=affine)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states), key_padding_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.fc2(F.relu(self.fc1(self.final_layer_norm(hidden_states))))
        return residual + hidden_states


class ConstrainedOPTDecoder(nn.Module):
    def __init__(self, config: ConstrainedOPTConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.use_positions:
            self.embed_positions = nn.Embedding(
                config.max_position_embeddings + POSITION_OFFSET, config.hidden_size
            )
        else:
            self.embed_positions = None
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.layers = nn.ModuleList(
            [ConstrainedOPTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.embed_positions is not None:
            # Same position ids as HF OPT: count only non-padding tokens.
            position_ids = (torch.cumsum(attention_mask, dim=1) * attention_mask - 1).long()
            hidden_states = hidden_states + self.embed_positions(position_ids + POSITION_OFFSET)
        key_padding_mask = attention_mask.bool()

        # HF OPT convention: (embeddings, output of layers 1..L-1, final-LayerNorm(output of layer L)).
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, key_padding_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
            return hidden_states, tuple(all_hidden_states)
        return hidden_states, None


class ConstrainedOPTPreTrainedModel(PreTrainedModel):
    config_class = ConstrainedOPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["ConstrainedOPTDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class ConstrainedOPTModel(ConstrainedOPTPreTrainedModel):
    def __init__(self, config: ConstrainedOPTConfig):
        super().__init__(config)
        self.decoder = ConstrainedOPTDecoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.decoder.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        last, hidden = self.decoder(input_ids, attention_mask, bool(output_hidden_states))
        return BaseModelOutputWithPast(last_hidden_state=last, hidden_states=hidden)


class ConstrainedOPTForCausalLM(ConstrainedOPTPreTrainedModel):
    # Same as HF OPT: lm_head shares the input embedding when config.tie_word_embeddings is True.
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: ConstrainedOPTConfig):
        super().__init__(config)
        self.model = ConstrainedOPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,  # use_cache, past_key_values, ...: accepted and ignored (no KV cache)
    ) -> CausalLMOutputWithPast:
        n_samples = self.config.noise_eval_samples if self.config.attn_type == "noisy" else 0
        if self.training or n_samples <= 0:
            # Training (noisy: one noise draw per pass), every non-noisy type, and noisy with the
            # noise off at evaluation.
            last, hidden = self.model.decoder(input_ids, attention_mask, bool(output_hidden_states))
            logits = self.lm_head(last)
        else:
            logits, hidden = self._noisy_eval_logits(input_ids, attention_mask, n_samples, bool(output_hidden_states))
        loss = None
        if labels is not None:
            # HF convention: labels are the input ids; predict token t+1 from positions <= t.
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=hidden)

    def noisy_attentions(self) -> list[ConstrainedOPTAttention]:
        return [layer.self_attn for layer in self.model.decoder.layers if layer.self_attn.attn_type == "noisy"]

    @torch.no_grad()
    def _noisy_eval_logits(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None, n_samples: int, want_hidden: bool
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """log((1/K) sum_k softmax(logits_k)) over K noise draws from a generator seeded with
        ``noise_eval_seed`` at every call, so the same input gives the same output. Hidden states,
        if requested, come from an extra pass with the noise off (the benchmark reads the logits).
        """
        attns = self.noisy_attentions()
        generator = torch.Generator(device=input_ids.device).manual_seed(int(self.config.noise_eval_seed))
        acc = None
        try:
            for attn in attns:
                attn.noise_eval_active, attn.noise_generator = True, generator
            for _ in range(n_samples):
                last, _ = self.model.decoder(input_ids, attention_mask, False)
                logp = F.log_softmax(self.lm_head(last).float(), dim=-1)
                acc = logp if acc is None else torch.logaddexp(acc, logp)
        finally:
            for attn in attns:
                attn.noise_eval_active, attn.noise_generator = False, None
        hidden = self.model.decoder(input_ids, attention_mask, True)[1] if want_hidden else None
        return acc - math.log(n_samples), hidden


def noisy_precision_penalty(model: ConstrainedOPTForCausalLM) -> torch.Tensor:
    """Mean precision over all (q, k <= q) pairs of a ``noisy_ref_len`` window, heads and layers:
    the memory term of Xu et al.'s loss, which is multiplied by ``noisy_lambda``. It depends on
    the decay rates only, not on the data. Gradients flow into ``noisy_theta``.
    """
    cfg = model.config
    total, count = 0.0, 0
    for attn in model.noisy_attentions():
        s, n = precision_penalty_terms(attn.decay_rate(), cfg.noisy_ref_len, cfg.noisy_tau0)
        total, count = total + s, count + n
    if count == 0:
        raise ValueError("noisy_precision_penalty needs a model with attn_type='noisy'")
    return total / count


def is_constraint_param(name: str) -> bool:
    return any(key in name for key in CONSTRAINT_PARAM_KEYS)


OUTPUT_PROJ_SUFFIXES = ("self_attn.out_proj.weight", "fc2.weight")


def init_std_for(name: str, config: ConstrainedOPTConfig) -> float:
    """init_std, divided by sqrt(2L) for residual output projections when init_output_scale."""
    if config.init_output_scale and name.endswith(OUTPUT_PROJ_SUFFIXES):
        return config.init_std / math.sqrt(2 * config.num_hidden_layers)
    return config.init_std


@torch.no_grad()
def init_matched_weights(model: ConstrainedOPTForCausalLM, seed: int) -> None:
    """Deterministic init: normal(0, init_std) weights, zero biases, unit LayerNorm.

    With ``config.init_output_scale`` the residual output projections use init_std / sqrt(2L)
    (OPT / Megatron-LM). With tied embeddings, ``lm_head.weight`` is the input embedding and is
    initialized once (``named_parameters`` yields shared tensors once).

    Shared parameters draw from a CPU generator seeded with ``seed``, in module order;
    constraint-only parameters (``CONSTRAINT_PARAM_KEYS``) draw from a second generator
    seeded with ``seed + 1000``. Shared weights are therefore bit-identical across attention
    types for the same seed, whatever device the model ends up on.
    """
    shared = torch.Generator().manual_seed(seed)
    extra = torch.Generator().manual_seed(seed + 1000)
    for name, param in model.named_parameters():
        if name.endswith("noisy_theta"):
            # Xu et al.: theta (before softplus) = init + 1e-3 * N(0, 1).
            values = torch.empty(param.shape, dtype=torch.float32).normal_(0.0, NOISY_THETA_INIT_STD, generator=extra)
            param.copy_(values + model.config.noisy_theta_init)
        elif name.endswith(".bias"):
            param.zero_()
        elif "layer_norm" in name:
            param.fill_(1.0)
        else:
            gen = extra if is_constraint_param(name) else shared
            std = init_std_for(name, model.config)
            values = torch.empty(param.shape, dtype=torch.float32).normal_(0.0, std, generator=gen)
            param.copy_(values)
