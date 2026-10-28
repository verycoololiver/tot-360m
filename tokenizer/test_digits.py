from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Digits, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

seq = Sequence([
    Digits(individual_digits=True),
    ByteLevel(add_prefix_space=False, use_regex=True),
])

tok = Tokenizer(BPE())
tok.pre_tokenizer = seq
tok.decoder = ByteLevelDecoder()

# Use pre_tokenize_str
pre_tokenized = seq.pre_tokenize_str("Hello, world! 12345")
print("Pre-tokenized splits:", [s[0] for s in pre_tokenized])
