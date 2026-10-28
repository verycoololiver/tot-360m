from dataclasses import dataclass

@dataclass
class Tot360MConfig:
    """Frozen configuration for Tot-360M (Baseline 960-dim Qwen 3 architecture).
    
    Exact parameters: 345,359,296.
    Uses stock llama.cpp qwen3.cpp compatible layout:
    - Shared 64-element QK-Norm vectors per layer (q_norm: 64, k_norm: 64).
    - Tied 32k word embeddings.
    - SwiGLU 8/3 expansion ratio (2560).
    - 3:1 GQA ratio (15 query heads, 5 key-value heads, head dimension 64).
    - RoPE base theta = 10,000 for 2,048 context.
    """
    vocab_size: int = 32000
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5   # Exact 3:1 GQA ratio
    head_dim: int = 64             # 960 / 15 = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    z_loss_weight: float = 1e-4


@dataclass
class Tot1024Config(Tot360MConfig):
    """Aligned 1024-dim candidate for 256-block quantization probe comparison.
    
    Exact parameters: 393,548,800.
    100% of matrix rows are divisible by 256 (ne[0] % 256 == 0):
    - hidden_size = 1024 (1024 % 256 == 0)
    - intermediate_size = 2816 (2816 % 256 == 0)
    - 16 query heads, 4 key-value heads (exact 4:1 GQA ratio).
    """
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64


@dataclass
class TotMiniConfig(Tot360MConfig):
    """Miniature configuration for memory-safe local CPU tests (< 5 MB RAM).
    
    Preserves exact architectural properties:
    - 3:1 GQA ratio (6 query heads, 2 key-value heads).
    - SwiGLU FFN.
    - Shared head_dim QK-Norm vectors.
    - Pre-RMSNorm residual layers.
    """
    hidden_size: int = 192
    intermediate_size: int = 512
    num_hidden_layers: int = 2
    num_attention_heads: int = 6    # 6 query heads
    num_key_value_heads: int = 2    # 2 KV heads (exact 3:1 GQA ratio)
    head_dim: int = 32
    vocab_size: int = 1000
    max_position_embeddings: int = 512
