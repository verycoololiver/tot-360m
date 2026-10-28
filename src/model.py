import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tot_360m.config import Tot360MConfig


class TotRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x_norm).to(input_dtype)


class TotRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with base theta = 10,000."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        init_device = torch.device("meta") if inv_freq.device.type == "meta" else torch.device("cpu")
        self._set_cos_sin_cache(max_position_embeddings, device=init_device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        if device.type == "meta" or self.inv_freq.device.type == "meta":
            self.register_buffer("cos_cached", torch.empty((seq_len, self.dim), device="meta"), persistent=False)
            self.register_buffer("sin_cached", torch.empty((seq_len, self.dim), device="meta"), persistent=False)
            return

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            not hasattr(self, "cos_cached")
            or seq_len > self.max_seq_len_cached
            or self.cos_cached.device != x.device
        ):
            self._set_cos_sin_cache(max(seq_len, self.max_position_embeddings), device=x.device)
        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX-style split-half rotation matching HF and stock llama.cpp qwen3."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # RoPE caches follow the embedding dtype (FP32 master weights), while
    # autocast projections may be BF16/FP16. Preserve the projection dtypes
    # after rotation so SDPA receives matching Q/K/V on XLA as well as CPU/CUDA.
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Combines causality (lower-triangular) with optional padding/custom masks for SDPA.
    
    Returns:
    - None if attention_mask is None (letting SDPA use native is_causal=True).
    - 4D boolean or additive float tensor [batch_size, 1, seq_len, seq_len] with causal constraint enforced.
    """
    if attention_mask is None:
        return None

    causal_bool = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

    if attention_mask.dim() == 2:
        # [batch, seq_len]
        if torch.is_floating_point(attention_mask):
            # Float padding mask: 0.0 means valid, negative/inf means masked
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            causal_additive = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
            causal_additive.masked_fill_(~causal_bool, torch.finfo(dtype).min)
            return causal_additive.unsqueeze(0).unsqueeze(1) + key_mask
        else:
            # Boolean or integer binary mask: True/1 means attend, False/0 means mask
            key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            return causal_bool.unsqueeze(0).unsqueeze(1) & key_mask

    elif attention_mask.dim() == 4:
        # [batch, 1 or heads, q_len, kv_len]
        if torch.is_floating_point(attention_mask):
            causal_additive = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
            causal_additive.masked_fill_(~causal_bool, torch.finfo(dtype).min)
            return attention_mask + causal_additive.unsqueeze(0).unsqueeze(1)
        else:
            return attention_mask.bool() & causal_bool.unsqueeze(0).unsqueeze(1)

    else:
        raise ValueError(
            f"Unsupported attention_mask dimension: {attention_mask.dim()}. Expected 2D [batch, seq_len] or 4D [batch, 1, seq_len, seq_len]."
        )


class TotMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron (8/3 hidden expansion ratio)."""
    def __init__(self, config: Tot360MConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TotAttention(nn.Module):
    """Grouped-Query Attention (GQA) with shared head_dim QK-Norm (llama.cpp qwen3.cpp layout)."""
    def __init__(self, config: Tot360MConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Stock llama.cpp qwen3.cpp layout: two 64-element vectors shared across heads
        if config.use_qk_norm:
            self.q_norm = TotRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 64 params
            self.k_norm = TotRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 64 params
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Per-head QK-Norm: normalizes over the last dimension (head_dim = 64)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand key and value for GQA groups
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Causal Attention via PyTorch SDPA
        # If attention_mask is None, rely on is_causal fast path. Otherwise mask has causality baked in.
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None and seq_len > 1),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class TotDecoderLayer(nn.Module):
    """Transformer Decoder Layer with Pre-RMSNorm residual connections."""
    def __init__(self, config: Tot360MConfig):
        super().__init__()
        self.input_layernorm = TotRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TotAttention(config)
        self.post_attention_layernorm = TotRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TotMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_emb=rotary_emb, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TotModel(nn.Module):
    """Core Transformer Stack for Tot-360M."""
    def __init__(self, config: Tot360MConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TotDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TotRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TotRotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)

        prepared_mask = prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_emb=(cos, sin), attention_mask=prepared_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class TotForCausalLM(nn.Module):
    """Full Causal LM with Tied Embeddings, Masked Loss, and FP32 Auxiliary Z-Loss."""
    def __init__(self, config: Tot360MConfig):
        super().__init__()
        self.config = config
        self.model = TotModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight Tying: lm_head shares identical tensor storage with embed_tokens
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.device.type != "meta":
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                if module.weight.device.type != "meta":
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if logits.shape[1] <= 1:
                # Graph-connected zero loss for 1-token sequence
                loss = logits.float().sum() * 0.0
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Static-shape FP32 masked reduction (XLA / TPU graph compilation safe)
                fp32_logits = shift_logits.float()
                valid_mask = (shift_labels != -100).float()
                safe_labels = torch.where(shift_labels != -100, shift_labels, 0)

                # Per-token cross entropy without dynamic shape reduction
                per_token_ce = F.cross_entropy(
                    fp32_logits.reshape(-1, fp32_logits.shape[-1]),
                    safe_labels.reshape(-1),
                    reduction="none",
                ).reshape_as(shift_labels)

                valid_count = valid_mask.sum().clamp(min=1.0)
                ce_loss = (per_token_ce * valid_mask).sum() / valid_count

                if self.config.z_loss_weight > 0.0:
                    log_z = torch.logsumexp(fp32_logits, dim=-1)
                    per_token_z = log_z.square()
                    z_loss = self.config.z_loss_weight * ((per_token_z * valid_mask).sum() / valid_count)
                    total_loss = ce_loss + z_loss
                else:
                    total_loss = ce_loss

                # Static gate for all-ignored batches without Python device synchronization
                has_valid = (valid_mask.sum() > 0).float()
                loss = total_loss * has_valid

        return logits, loss
