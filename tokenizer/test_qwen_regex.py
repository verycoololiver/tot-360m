from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence

pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
pre_tok = Sequence([
    Split(Regex(pattern), behavior="isolated"),
    ByteLevel(add_prefix_space=False, use_regex=False)
])
tok = Tokenizer(BPE())
tok.pre_tokenizer = pre_tok
enc = tok.encode("Hello, world! 12345")
print("Pre-tokenized splits:", enc.tokens)
