"""
Real vocabularies, turned into the byte strings the codec would actually see.

The released module keeps a byte buffer per token id, filled from the token's *text*. So the
measurement has to recover that text faithfully, and every tokenizer family mangles it
differently:

    SentencePiece   U+2581 marks a word start; the real byte buffer holds a space
    byte-level BPE  token chars are a printable re-encoding of raw bytes; invert it
    WordPiece       '##' marks a continuation; it is KEPT, see below

Getting this wrong inflates byte lengths (U+2581 is 3 bytes, ' ' is 1) and would quietly
change the answer, so each family is handled explicitly rather than by str.encode().

The '##' decision is load-bearing and was got wrong first. Stripping it makes the
continuation token '##ka' and the standalone token 'ka' the same byte string, so they
collide -- and that collision is an artifact of the stripping, not of the codec. It is a
large artifact: mBERT read 25.7% of its vocabulary as colliding at R=32 with '##' stripped,
and the number did not fall as R grew, which is the tell that truncation was not the cause.
Two distinct vocabulary entries have to stay distinct byte strings, so the marker stays.
"""

from __future__ import annotations

import functools
import json
import unicodedata
from pathlib import Path

SESSION2_TOKENIZER = Path(__file__).resolve().parents[2] / "Session2/src/tokenizer_10k.json"

MODELS = {
    "xlm-roberta-base": "sp",       # 250k, SentencePiece, the most multilingual thing on HF
    "Qwen/Qwen2.5-0.5B-Instruct": "bbpe",
    "bert-base-multilingual-cased": "wordpiece",
}


@functools.lru_cache(maxsize=1)
def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode: printable char -> raw byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok: str, family: str) -> bytes:
    if family == "sp":
        return tok.replace("▁", " ").encode("utf-8")
    if family == "wordpiece":
        return tok.encode("utf-8")                  # '##' kept: it distinguishes two entries
    if family == "bbpe":
        dec = _byte_decoder()
        try:
            return bytes(dec[c] for c in tok)
        except KeyError:                            # added/special tokens are plain text
            return tok.encode("utf-8")
    return tok.encode("utf-8")


def script_of(tok: str) -> str:
    """Script of the token's first cased/letter character, via the Unicode character name.

    Attribution is by first letter, not majority: a token is written in one script in
    practice, and the mixed ones are punctuation-plus-word, where the letter is what matters.
    """
    for ch in tok:
        if ch in "▁# ":
            continue
        cat = unicodedata.category(ch)
        if cat[0] not in ("L", "M", "N"):
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return "OTHER"
        head = name.split()[0]
        if head in ("CJK", "HIRAGANA", "KATAKANA"):
            return "CJK"
        if head in ("DIGIT", "LATIN"):
            return "LATIN"
        return head
    return "SYMBOL"


def load_hf(name: str):
    """-> list of (token text, byte string). Special tokens dropped: they never reach the
    codec as text in the released path."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    family = MODELS[name]
    special = set(tk.all_special_tokens)
    out = []
    for tok in tk.get_vocab():
        if tok in special:
            continue
        b = token_bytes(tok, family)
        if b:
            out.append((tok, b))
    return out


def load_session2():
    """This course's own Session 2 BPE tokenizer.

    Included because the lesson asks for the collision count on *our* vocabulary, and this is
    the only vocabulary in this repo that we actually trained.

    Uses the tokenizer's own id_to_str rather than re-deriving it from base_chars + merges.
    Re-deriving it is off by one -- id 0 is the UNK slot, so every merge index refers to a
    table that already has UNK at the front -- and the resulting tokens are plausible-looking
    concatenations of the wrong pieces, which is exactly the kind of wrong that does not
    announce itself."""
    import sys
    sys.path.insert(0, str(SESSION2_TOKENIZER.parent))
    from bpe_tokenizer import BalancedBPETokenizer

    tk = BalancedBPETokenizer.load(str(SESSION2_TOKENIZER))
    return [(t, t.encode("utf-8")) for t in tk.id_to_str]


def load(name: str):
    return load_session2() if name == "session2-bpe" else load_hf(name)


ALL = ["session2-bpe", *MODELS]


if __name__ == "__main__":
    for n in ALL:
        v = load(n)
        lens = sorted(len(b) for _, b in v)
        over32 = sum(l > 32 for l in lens)
        print(f"{n:35s} {len(v):>7,} tokens  max {lens[-1]:>3}B  "
              f">32B: {over32:>5} ({100*over32/len(v):.2f}%)")
