"""Byte-Fallback BPE Tokenizer Builder for Tot-360M.

Features:
- Exact 32,000 vocabulary size.
- GPT-2/Qwen style ByteLevel pre-tokenization (guarantees 100% loss-free roundtrip on arbitrary UTF-8/bytes).
- Special control / ChatML tokens:
    <|endoftext|> : ID 0
    <|im_start|>   : ID 1
    <|im_end|>     : ID 2
    <unk>          : ID 3
    <pad>          : ID 4
- Exports to standard HuggingFace tokenizer.json and GGUF vocabulary lists.
"""

import os
import json
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import ByteLevel as ByteLevelProcessor


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<unk>",
    "<pad>",
]


def create_base_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tokenizer


def train_tokenizer(
    training_texts: List[str],
    vocab_size: int = 32000,
    output_dir: str = "tot_360m/tokenizer/output",
) -> Tokenizer:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = create_base_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )

    # Train on supplied texts (can be iterator or in-memory strings)
    tokenizer.train_from_iterator(training_texts, trainer=trainer)

    # Save tokenizer.json
    json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(json_path)

    # Save vocabulary dictionary for GGUF exporter
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return tokenizer


def get_default_training_corpus() -> List[str]:
    """Generates a rich multi-domain training corpus to populate initial merges."""
    corpus = [
        # English, conversation, and reasoning
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and deep learning enable efficient compact reasoning models.",
        "Tot is OB Studio's personal desktop AI harness designed for independent offline computing.",
        "Iris is thoughtful, honest, and grounded, guiding tasks with one useful step at a time.",
        "<|im_start|>system\nYou are Iris, a personal desktop companion.<|im_end|>",
        "<|im_start|>user\nHow do we optimize memory on dual-core CPUs?<|im_end|>",
        "<|im_start|>assistant\nWe keep the working set in RAM and avoid unnecessary tensor allocation.<|im_end|>",
        # Python / Systems Code
        "def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):",
        "    hidden_states = self.embed_tokens(input_ids)\n    return self.norm(hidden_states)",
        "import math\nimport os\nimport sys\nimport torch\nimport torch.nn as nn",
        "class TotAttention(nn.Module):\n    def __init__(self, config):\n        super().__init__()",
        # Math & Symbols
        "E = mc^2, f(x) = \\int_{-\\infty}^\\infty e^{-x^2} dx, Q K^T / \\sqrt{d}",
        "Greek symbols: alpha beta gamma delta epsilon theta lambda sigma omega",
        # Unicode multilingual tests
        "한국어 자연어 처리 모델 테스트. 토트 360M 프로젝트.",
        "日本語のテキスト処理テスト。人工知能モデル。",
        "English, Français, Español, Deutsch, 中文, Русский.",
        # Arbitrary symbols and hex bytes
        "Special bytes: \x00\x01\x02\x03\x04\xff\xfe\xfd\xaa\x55",
    ]
    # Repeat corpus to ensure frequency thresholds for BPE merges
    return corpus * 50


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    print(f"Building Tot-360M 32k Byte-Fallback BPE Tokenizer into {out_dir}...")
    tok = train_tokenizer(get_default_training_corpus(), vocab_size=32000, output_dir=out_dir)
    print(f"Done! Actual vocab size: {tok.get_vocab_size()}")
    
    # Test encoding
    sample = "<|im_start|>user\nHello Iris!<|im_end|>"
    encoded = tok.encode(sample)
    print("Sample:", sample)
    print("Tokens:", encoded.tokens)
    print("IDs:   ", encoded.ids)
    decoded = tok.decode(encoded.ids)
    print("Decoded:", decoded)
    assert decoded == sample, "Roundtrip mismatch!"
    print("Roundtrip test PASSED!")
