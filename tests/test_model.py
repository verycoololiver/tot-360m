import unittest
import math
import torch
import torch.nn.functional as F
from unittest.mock import patch

from tot_360m.config import Tot360MConfig, TotMiniConfig
from tot_360m.src.model import TotForCausalLM, TotAttention, apply_rotary_pos_emb


class TestTot360MArchitecture(unittest.TestCase):
    def test_rotary_preserves_reduced_projection_dtype(self):
        """FP32 rotary caches must not promote BF16/FP16 Q/K before SDPA."""
        torch.manual_seed(123)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                q = torch.randn(1, 6, 8, 32).to(dtype)
                k = torch.randn(1, 2, 8, 32).to(dtype)
                angles = torch.randn(8, 32)
                cos, sin = angles.cos(), angles.sin()
                actual_q, actual_k = apply_rotary_pos_emb(q, k, cos, sin)
                self.assertEqual(actual_q.dtype, dtype)
                self.assertEqual(actual_k.dtype, dtype)
                # Independent split-half rotation in FP32, rounded only at output.
                for original, actual in ((q, actual_q), (k, actual_k)):
                    x = original.float()
                    rotated = torch.cat((-x[..., 16:], x[..., :16]), dim=-1)
                    expected = (x * cos[None, None] + rotated * sin[None, None]).to(dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_autocast_attention_inputs_and_backward(self):
        """Exercise the real model and inspect SDPA inputs before autocast can hide a mismatch."""
        torch.manual_seed(123)
        model = TotForCausalLM(TotMiniConfig())
        tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        original_sdpa = F.scaled_dot_product_attention
        seen = []

        def checked_sdpa(q, k, v, **kwargs):
            seen.append((q.dtype, k.dtype, v.dtype))
            self.assertEqual((q.dtype, k.dtype, v.dtype), (torch.bfloat16,) * 3)
            return original_sdpa(q, k, v, **kwargs)

        with patch("tot_360m.src.model.F.scaled_dot_product_attention", side_effect=checked_sdpa):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits, loss = model(input_ids=tokens, labels=tokens)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        self.assertEqual(len(seen), model.config.num_hidden_layers)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_full_model_meta_device_parameters(self):
        """Verify full 32-layer architecture structurally on meta device (0 bytes weight allocation)."""
        config = Tot360MConfig()
        
        with torch.device("meta"):
            model = TotForCausalLM(config)

        total_params = sum(p.numel() for p in model.parameters())
        unique_params = sum(p.numel() for p in set(model.parameters()))

        print(f"\n==========================================")
        print(f"Tot-360M Full Model Meta-Device Audit")
        print(f"==========================================")
        print(f"Total Parameters (accounting for tied head): {total_params:,}")
        print(f"Unique Parameters in memory:                 {unique_params:,}")

        # Exact target: 345,359,296
        self.assertEqual(
            unique_params,
            345359296,
            f"Expected exactly 345,359,296 parameters, but got {unique_params:,}"
        )
        self.assertTrue(
            model.lm_head.weight is model.model.embed_tokens.weight,
            "LM head and word embeddings must share identical tensor identity."
        )

    def test_mini_numerical_forward_and_backward(self):
        """Verify numerical sanity on lightweight CPU mini configuration (< 5 MB RAM)."""
        config = TotMiniConfig()
        model = TotForCausalLM(config)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Forward pass
        logits, loss = model(input_ids=input_ids, labels=labels)

        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertFalse(torch.isnan(logits).any(), "Logits contain NaN values!")
        self.assertFalse(torch.isinf(logits).any(), "Logits contain Inf values!")
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0, "Loss must be positive.")
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN.")

        # Backward pass
        loss.backward()

        # Verify finite, non-zero gradient norm across every unique parameter tensor
        zero_grad_tensors = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient has non-finite values in {name}")
                grad_norm = param.grad.abs().sum().item()
                if grad_norm == 0.0:
                    zero_grad_tensors.append(name)

        self.assertEqual(
            len(zero_grad_tensors),
            0,
            f"The following parameters have zero gradient norm: {zero_grad_tensors}"
        )

    def test_causal_mask_invariance_no_mask(self):
        """Verify causality with attention_mask=None (native SDPA is_causal path)."""
        torch.manual_seed(42)
        config = TotMiniConfig()
        model = TotForCausalLM(config)
        model.eval()

        input_ids_1 = torch.tensor([[10, 20, 30, 40]])
        input_ids_2 = torch.tensor([[10, 20, 30, 99]])  # Perturbed position 3

        with torch.no_grad():
            logits_1, _ = model(input_ids=input_ids_1)
            logits_2, _ = model(input_ids=input_ids_2)

        diff = (logits_1[0, :3, :] - logits_2[0, :3, :]).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            f"Causal mask leakage with no mask! Max diff: {diff}"
        )

    def test_causal_mask_invariance_all_valid_boolean_mask(self):
        """Codex regression test: verify causality when an all-valid boolean mask is supplied."""
        torch.manual_seed(42)
        config = TotMiniConfig()
        model = TotForCausalLM(config)
        model.eval()

        input_ids_1 = torch.tensor([[10, 20, 30, 40]])
        input_ids_2 = torch.tensor([[10, 20, 30, 99]])  # Perturbed position 3
        all_valid_mask = torch.ones(1, 4, dtype=torch.bool)

        with torch.no_grad():
            logits_1, _ = model(input_ids=input_ids_1, attention_mask=all_valid_mask)
            logits_2, _ = model(input_ids=input_ids_2, attention_mask=all_valid_mask)

        diff = (logits_1[0, :3, :] - logits_2[0, :3, :]).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            f"Causal mask leakage with all-valid mask! Max diff: {diff} (must be < 1e-5)"
        )

    def test_causal_mask_invariance_batched_padding(self):
        """Verify causality and key-padding isolation in a batch-of-2 with padding."""
        torch.manual_seed(42)
        config = TotMiniConfig()
        model = TotForCausalLM(config)
        model.eval()

        # Batch 2, seq_len 4. Sequence 0 has 4 tokens. Sequence 1 has 2 tokens + 2 pad tokens.
        input_ids_a = torch.tensor([
            [10, 20, 30, 40],
            [15, 25, 0, 0]
        ])
        input_ids_b = torch.tensor([
            [10, 20, 30, 99],  # Perturbed pos 3 in seq 0
            [15, 25, 0, 0]
        ])
        pad_mask = torch.tensor([
            [True, True, True, True],
            [True, True, False, False]
        ])

        with torch.no_grad():
            logits_a, _ = model(input_ids=input_ids_a, attention_mask=pad_mask)
            logits_b, _ = model(input_ids=input_ids_b, attention_mask=pad_mask)

        # 1. Seq 0 positions 0..2 must not change
        diff_seq0 = (logits_a[0, :3, :] - logits_b[0, :3, :]).abs().max().item()
        self.assertLess(diff_seq0, 1e-5, f"Seq 0 leakage across time! Diff: {diff_seq0}")

        # 2. Seq 1 must be 100% bitwise identical despite batch mate changes
        diff_seq1 = (logits_a[1, :2, :] - logits_b[1, :2, :]).abs().max().item()
        self.assertLess(diff_seq1, 1e-5, f"Cross-sequence batch leakage! Diff: {diff_seq1}")

    def test_loss_masking_and_numerical_safety(self):
        """Verify loss behavior on ignored labels (-100), all-ignored batches, and 1-token inputs."""
        torch.manual_seed(42)
        config = TotMiniConfig()
        model = TotForCausalLM(config)

        # Case 1: Sequence length = 1 token
        input_1tok = torch.tensor([[42]])
        labels_1tok = torch.tensor([[42]])
        logits, loss_1tok = model(input_ids=input_1tok, labels=labels_1tok)
        self.assertEqual(loss_1tok.item(), 0.0)
        loss_1tok.backward()  # Graph connected: does not crash

        # Case 2: All shifted labels are -100
        input_ids = torch.tensor([[10, 20, 30, 40]])
        labels_all_ignored = torch.full_like(input_ids, -100)
        model.zero_grad()
        logits, loss_all_ignored = model(input_ids=input_ids, labels=labels_all_ignored)
        self.assertEqual(loss_all_ignored.item(), 0.0)
        self.assertFalse(torch.isnan(loss_all_ignored))
        loss_all_ignored.backward()  # Graph connected: does not crash

        # Case 3: Mixed labels (positions 0, 1 ignored, pos 2 valid)
        labels_mixed = torch.tensor([[-100, -100, -100, 50]])  # shift_labels has pos 2 valid
        model.zero_grad()
        logits, loss_mixed = model(input_ids=input_ids, labels=labels_mixed)
        self.assertGreater(loss_mixed.item(), 0.0)
        self.assertFalse(torch.isnan(loss_mixed))
        loss_mixed.backward()
        # Verify gradients are non-zero for active parameters
        self.assertGreater(model.lm_head.weight.grad.abs().sum().item(), 0.0)

    def test_gqa_numerical_equivalence_against_manual_reference(self):
        """Verify TotAttention against an independent manual reference implementation."""
        torch.manual_seed(42)
        config = TotMiniConfig()  # 6 Q heads, 2 KV heads, head_dim 32, hidden 192
        attn = TotAttention(config)
        attn.eval()

        batch_size = 2
        seq_len = 8
        hidden = torch.randn(batch_size, seq_len, config.hidden_size)

        # Mock cos/sin
        cos = torch.ones(seq_len, config.head_dim)
        sin = torch.zeros(seq_len, config.head_dim)

        with torch.no_grad():
            output_model = attn(hidden, rotary_emb=(cos, sin))

            # Manual Reference:
            q = attn.q_proj(hidden).view(batch_size, seq_len, 6, 32).transpose(1, 2)
            k = attn.k_proj(hidden).view(batch_size, seq_len, 2, 32).transpose(1, 2)
            v = attn.v_proj(hidden).view(batch_size, seq_len, 2, 32).transpose(1, 2)

            if attn.q_norm is not None:
                q = attn.q_norm(q)
            if attn.k_norm is not None:
                k = attn.k_norm(k)

            # Expand KV for 3:1 ratio
            k_exp = k.repeat_interleave(3, dim=1)  # [batch, 6, seq_len, 32]
            v_exp = v.repeat_interleave(3, dim=1)  # [batch, 6, seq_len, 32]

            # Manual attention scores
            scale = 1.0 / math.sqrt(config.head_dim)
            scores = torch.matmul(q, k_exp.transpose(-1, -2)) * scale

            # Manual causal mask: upper triangle is -inf
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            scores = scores + causal_mask

            probs = F.softmax(scores, dim=-1)
            attn_manual = torch.matmul(probs, v_exp)
            attn_manual = attn_manual.transpose(1, 2).contiguous().view(batch_size, seq_len, config.hidden_size)
            output_manual = attn.o_proj(attn_manual)

            max_diff = (output_model - output_manual).abs().max().item()
            self.assertLess(
                max_diff,
                1e-5,
                f"GQA model output deviates from manual reference! Max diff: {max_diff}"
            )


if __name__ == "__main__":
    unittest.main()
