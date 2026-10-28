"""Unit and Regression Test Suite for Production 32k Tokenizer."""

import os
import json
import unittest
from pathlib import Path
from tokenizers import Tokenizer

from tot_360m.tokenizer.build_production_tokenizer import SPECIAL_TOKENS, TARGET_VOCAB_SIZE


class TestProductionTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok_path = Path("tot_360m/tokenizer/production/tokenizer.json")
        if not cls.tok_path.exists():
            raise unittest.SkipTest("Production tokenizer not built yet. Run build_production_tokenizer.py first.")
        cls.tokenizer = Tokenizer.from_file(str(cls.tok_path))

    def test_exact_32k_vocabulary(self):
        """Assert vocabulary is exactly 32,000 tokens."""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, TARGET_VOCAB_SIZE)

    def test_special_tokens_atomic_and_reserved(self):
        """Assert special tokens have exact fixed IDs and are not fragmented."""
        expected = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<unk>": 3,
            "<pad>": 4,
        }
        for token_str, expected_id in expected.items():
            encoded = self.tokenizer.encode(token_str)
            self.assertEqual(
                len(encoded.ids),
                1,
                f"Special token '{token_str}' was fragmented into {encoded.tokens}"
            )
            self.assertEqual(encoded.ids[0], expected_id)
            self.assertEqual(self.tokenizer.token_to_id(token_str), expected_id)

    def test_lossless_roundtrip_multilingual_and_code(self):
        """Assert 100% loss-free roundtrip on diverse scripts and code."""
        samples = [
            "The quick brown fox jumps over the lazy dog.",
            "    def compute_attention(q, k, v, mask=None):\n\t\tscores = torch.matmul(q, k.transpose(-2, -1))\n\t\treturn scores",
            "한국어 인공지능 모델 토트 360M 프로젝트. 자연어 처리 테스트 문장입니다.",
            "수학 공식: \\int_{-\\infty}^\\infty e^{-x^2} dx = \\sqrt{\\pi}, \\sum_{i=1}^n i = \\frac{n(n+1)}{2}",
            "이모지 테스트: 🚀 🤖 ✨ 🌟 💡 🎯",
            "White   spaces  \t\t tabs and \r\n multiple newlines \n\n end.",
        ]
        for s in samples:
            encoded = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(encoded.ids)
            self.assertEqual(decoded, s, f"Roundtrip failed for: {repr(s)}")

    def test_chatml_full_prompt_roundtrip(self):
        """Assert ChatML prompts with special tokens roundtrip losslessly when skip_special_tokens=False."""
        prompt = (
            "<|im_start|>system\nYou are Iris, the companion in Tot.<|im_end|>\n"
            "<|im_start|>user\nHello Iris, can you help me?<|im_end|>\n"
            "<|im_start|>assistant\nI am here. What is the first thing we need to solve?<|im_end|>"
        )
        encoded = self.tokenizer.encode(prompt)
        decoded = self.tokenizer.decode(encoded.ids, skip_special_tokens=False)
        self.assertEqual(decoded, prompt)

    def test_byte_fallback_safety(self):
        """Assert raw arbitrary byte strings encode and decode without crashing or unk tokens."""
        raw_bytes = "\x00\x01\x02\x7f\x80\x99\xff\xfe"
        encoded = self.tokenizer.encode(raw_bytes)
        self.assertGreater(len(encoded.ids), 0)
        self.assertNotIn(3, encoded.ids)  # Must not use <unk>
        decoded = self.tokenizer.decode(encoded.ids)
        self.assertEqual(decoded, raw_bytes)

    def test_fertility_rates(self):
        """Verify reasonable token fertility (chars per token) across domains."""
        en_text = "Educational language modeling requires high token efficiency and clean vocabulary merges."
        en_encoded = self.tokenizer.encode(en_text)
        en_cpt = len(en_text) / len(en_encoded.ids)
        self.assertGreater(en_cpt, 3.0, f"English chars/token too low: {en_cpt:.2f}")

        ko_text = "한국어 자연어 처리 모델의 토큰 압축 효율성을 측정합니다."
        ko_encoded = self.tokenizer.encode(ko_text)
        ko_cpt = len(ko_text) / len(ko_encoded.ids)
        self.assertGreater(ko_cpt, 1.0, f"Korean chars/token too low: {ko_cpt:.2f}")


if __name__ == "__main__":
    unittest.main()
