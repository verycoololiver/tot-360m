"""GGUF Exporter for Tot-360M (stock llama.cpp qwen3 architecture).

Pinned Reference:
- llama.cpp commit 3d65c90d04d337e88f2b1f7f0061f40a5324e662 (Release b11515)
- Architecture tag: qwen3
- Layout: shared 64-element QK-Norm, SwiGLU, tied embeddings, NeoX split-half RoPE
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import gguf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tot_360m.config import Tot360MConfig, TotMiniConfig
from tot_360m.src.model import TotForCausalLM


def export_model_to_gguf(
    model: TotForCausalLM,
    config: Tot360MConfig,
    output_path: str,
    tokenizer_json_path: str = "tot_360m/tokenizer/production/tokenizer.json",
    dtype: str = "f32",
    materialize_output: bool = False,
    allow_vocab_mismatch: bool = False,
):
    """Exports a Tot PyTorch model to standard GGUF format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = gguf.GGUFWriter(output_path, "qwen3")

    # Only fall back to the dev tokenizer when the caller kept the default path. An explicitly
    # requested path must never be silently swapped for a different tokenizer.
    default_tok = "tot_360m/tokenizer/production/tokenizer.json"
    dev_tok = "tot_360m/tokenizer/output/tokenizer.json"
    if tokenizer_json_path == default_tok and not os.path.exists(default_tok) and os.path.exists(dev_tok):
        tokenizer_json_path = dev_tok

    # 1. Architecture Hyperparameters
    writer.add_block_count(config.num_hidden_layers)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_key_length(config.head_dim)
    writer.add_value_length(config.head_dim)

    # 2. Tokenizer Metadata & Merges
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)

        vocab = tok_data["model"]["vocab"]
        raw_merges = tok_data["model"].get("merges", [])
        merges = []
        for m in raw_merges:
            if isinstance(m, (list, tuple)):
                merges.append(f"{m[0]} {m[1]}")
            elif isinstance(m, str):
                merges.append(m)

        # Strict vocabulary size validation
        target_vocab_size = config.vocab_size
        if len(vocab) != target_vocab_size and not allow_vocab_mismatch:
            raise ValueError(
                f"Vocabulary size mismatch! Tokenizer has {len(vocab)} tokens, "
                f"but model config expects {target_vocab_size}. "
                f"Silent padding with dummy tokens is rejected. Use allow_vocab_mismatch=True only for test fixtures."
            )

        tokens = [None] * target_vocab_size
        for tok_str, tok_id in vocab.items():
            if tok_id < target_vocab_size:
                tokens[tok_id] = tok_str

        # Fill any unassigned slots with dummy tokens (only if allow_vocab_mismatch is explicitly True)
        for i in range(target_vocab_size):
            if tokens[i] is None:
                tokens[i] = f"<extra_tok_{i}>"

        scores = [0.0] * target_vocab_size
        tok_types = [gguf.TokenType.NORMAL] * target_vocab_size

        special_map = {
            "<|endoftext|>": (0, gguf.TokenType.CONTROL),
            "<|im_start|>": (1, gguf.TokenType.CONTROL),
            "<|im_end|>": (2, gguf.TokenType.CONTROL),
            "<unk>": (3, gguf.TokenType.UNKNOWN),
            "<pad>": (4, gguf.TokenType.CONTROL),
        }
        for s_token, (s_id, s_type) in special_map.items():
            if s_id < target_vocab_size and tokens[s_id] == s_token:
                tok_types[s_id] = s_type

        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(tok_types)
        if merges:
            writer.add_token_merges(merges)
        writer.add_bos_token_id(0)
        writer.add_eos_token_id(0)
        writer.add_unk_token_id(3)
        writer.add_pad_token_id(4)
    else:
        if config.vocab_size == 32000 and not allow_vocab_mismatch:
            raise FileNotFoundError(
                f"Production tokenizer not found at {tokenizer_json_path!r}. "
                f"Refusing to export baseline 32k model with dummy tokens."
            )
        # Minimal dummy BPE with dummy merge for test
        dummy_tokens = [f"<tok_{i}>" for i in range(config.vocab_size)]
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(dummy_tokens)
        writer.add_token_scores([0.0] * config.vocab_size)
        writer.add_token_types([gguf.TokenType.NORMAL] * config.vocab_size)
        writer.add_token_merges(["<tok_0> <tok_1>"])
        writer.add_bos_token_id(0)
        writer.add_eos_token_id(0)
        writer.add_unk_token_id(0)

    # 3. Tensor Mappings
    state_dict = model.state_dict()
    np_dtype = np.float32 if dtype == "f32" else np.float16

    # Token embeddings
    embed_weights = state_dict["model.embed_tokens.weight"].detach().cpu().to(torch.float32).numpy()
    writer.add_tensor("token_embd.weight", embed_weights.astype(np_dtype))

    # Layers
    for i in range(config.num_hidden_layers):
        prefix_pt = f"model.layers.{i}."
        prefix_gguf = f"blk.{i}."

        # Attention Projections
        writer.add_tensor(
            f"{prefix_gguf}attn_q.weight",
            state_dict[f"{prefix_pt}self_attn.q_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}attn_k.weight",
            state_dict[f"{prefix_pt}self_attn.k_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}attn_v.weight",
            state_dict[f"{prefix_pt}self_attn.v_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}attn_output.weight",
            state_dict[f"{prefix_pt}self_attn.o_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )

        # Norms (Always FP32 in GGUF)
        writer.add_tensor(
            f"{prefix_gguf}attn_norm.weight",
            state_dict[f"{prefix_pt}input_layernorm.weight"].detach().cpu().to(torch.float32).numpy()
        )
        if config.use_qk_norm:
            writer.add_tensor(
                f"{prefix_gguf}attn_q_norm.weight",
                state_dict[f"{prefix_pt}self_attn.q_norm.weight"].detach().cpu().to(torch.float32).numpy()
            )
            writer.add_tensor(
                f"{prefix_gguf}attn_k_norm.weight",
                state_dict[f"{prefix_pt}self_attn.k_norm.weight"].detach().cpu().to(torch.float32).numpy()
            )

        # MLP Block
        writer.add_tensor(
            f"{prefix_gguf}ffn_gate.weight",
            state_dict[f"{prefix_pt}mlp.gate_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}ffn_up.weight",
            state_dict[f"{prefix_pt}mlp.up_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}ffn_down.weight",
            state_dict[f"{prefix_pt}mlp.down_proj.weight"].detach().cpu().to(torch.float32).numpy().astype(np_dtype)
        )
        writer.add_tensor(
            f"{prefix_gguf}ffn_norm.weight",
            state_dict[f"{prefix_pt}post_attention_layernorm.weight"].detach().cpu().to(torch.float32).numpy()
        )

    # Final Output Norm
    writer.add_tensor(
        "output_norm.weight",
        state_dict["model.norm.weight"].detach().cpu().to(torch.float32).numpy()
    )

    # Output head (Tied embedding handling)
    if materialize_output:
        # Compatibility flag: write output.weight explicitly if loader does not support tied output
        writer.add_tensor("output.weight", embed_weights.astype(np_dtype))
    # Otherwise omitted (stock qwen3 uses token_embd.weight when output.weight is absent)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully exported GGUF model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Tot-360M PyTorch model to GGUF format.")
    parser.add_argument("--config", choices=["mini", "baseline"], default="mini")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to trained TPU/GPU checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tot_360m/tokenizer/production/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--out", type=str, default="tot_360m/export/tot_mini.gguf")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--materialize-output", action="store_true")
    parser.add_argument("--allow-vocab-mismatch", action="store_true", help="Permit vocab size mismatch for test fixtures only")
    args = parser.parse_args()

    cfg = TotMiniConfig() if args.config == "mini" else Tot360MConfig()
    m = TotForCausalLM(cfg)
    if args.checkpoint:
        import gc
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.pop("model_state") if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        del ckpt
        gc.collect()
        m.load_state_dict(state)
        del state
        gc.collect()
        if cfg.tie_word_embeddings:
            m.lm_head.weight = m.model.embed_tokens.weight
    m.eval()
    export_model_to_gguf(
        m,
        cfg,
        args.out,
        tokenizer_json_path=args.tokenizer,
        dtype=args.dtype,
        materialize_output=args.materialize_output,
        allow_vocab_mismatch=args.allow_vocab_mismatch,
    )
