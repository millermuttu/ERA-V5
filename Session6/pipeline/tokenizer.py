"""Frozen, stdlib-only whitespace/punctuation tokenizer with a content hash.

Toy scale is explicitly allowed by the assignment - no BPE/tiktoken needed
to demonstrate a frozen tokenizer with a verifiable hash.
"""
import hashlib
import json
import re

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _split(text):
    return _TOKEN_RE.findall(text.lower())


def compute_hash(vocab):
    return hashlib.sha256(json.dumps(vocab, sort_keys=False).encode("utf-8")).hexdigest()


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.tokenizer_hash = compute_hash(vocab)

    @property
    def pad_id(self):
        return self.token_to_id["<pad>"]

    @property
    def eos_id(self):
        return self.token_to_id["<eos>"]

    def encode(self, text):
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(tok, unk) for tok in _split(text)]

    def to_dict(self):
        return {"vocab": self.vocab, "tokenizer_hash": self.tokenizer_hash}


def build_tokenizer(all_texts):
    tokens = set()
    for text in all_texts:
        tokens.update(_split(text))
    vocab = SPECIAL_TOKENS + sorted(tokens - set(SPECIAL_TOKENS))
    return Tokenizer(vocab)


def save_tokenizer(tokenizer, path):
    with open(path, "w") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)


def load_tokenizer(path):
    with open(path) as f:
        data = json.load(f)
    return Tokenizer(data["vocab"])
