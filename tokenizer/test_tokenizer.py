import os
import unittest
from tokenizers import Tokenizer
from tot_360m.tokenizer.build_tokenizer import train_tokenizer, get_default_training_corpus, SPECIAL_TOKENS


class TestTotTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_output"))
        cls.tokenizer = train_tokenizer(
            get_default_training_corpus(),
            vocab_size=1000,  # lightweight for unit tests
            output_dir=cls.test_dir,
        )

    def test_special_tokens_atomic(self):
        """Verify that ChatML special tokens are parsed as single atomic tokens."""
        for token_str in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            encoded = self.tokenizer.encode(token_str)
            self.assertEqual(
                len(encoded.ids),
                1,
                f"Special token '{token_str}' was split into multiple IDs: {encoded.tokens}"
            )
            self.assertEqual(encoded.tokens[0], token_str)

    def test_lossless_roundtrip_multilingual_and_code(self):
        """Verify 100% loss-free roundtrip on English, code, Korean, Japanese, and symbols."""
        test_strings = [
            "Hello, world! 12345",
            "    def indent_test():\n\t\treturn True",
            "한국어 자연어 처리 모델 테스트. 토트 360M 프로젝트.",
            "日本語のテスト。こんにちは世界！",
            "Special math: ∫ e^(-x^2) dx = √π",
            "Emojis: 🚀 ✨ 🤖 ⚡",
            "Multiple   spaces   and\nnewlines\r\nand tabs\t\tend.",
        ]
        for text in test_strings:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded.ids)
            self.assertEqual(
                decoded,
                text,
                f"Roundtrip failed for text: {repr(text)}\nGot: {repr(decoded)}"
            )

    def test_chatml_roundtrip(self):
        """Verify ChatML full prompt roundtrip with skip_special_tokens=False."""
        chatml_text = "<|im_start|>system\nYou are Iris.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"
        encoded = self.tokenizer.encode(chatml_text)
        decoded = self.tokenizer.decode(encoded.ids, skip_special_tokens=False)
        self.assertEqual(decoded, chatml_text)

    def test_arbitrary_binary_byte_safety(self):
        """Verify that arbitrary binary byte sequences do not crash and decode properly."""
        raw_bytes_str = "\x00\x01\x02\x7f\x80\xff\xfe"
        encoded = self.tokenizer.encode(raw_bytes_str)
        self.assertGreater(len(encoded.ids), 0)
        decoded = self.tokenizer.decode(encoded.ids)
        self.assertEqual(decoded, raw_bytes_str)


if __name__ == "__main__":
    unittest.main()
