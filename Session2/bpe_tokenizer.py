"""Balanced multilingual BPE tokenizer (codepoint-level).

Trained on several corpora at once: each merge is chosen from the language
whose fertility (tokens per word) is currently worst, keeping the
per-language compression ratios close together.
"""
from __future__ import annotations

import re

UNK_ID = 0
UNK_TOKEN = "<unk>"
UNK_RENDER = "�"

_UNIT_RE = re.compile(r"\s*\S+")


def pretokenize(text: str) -> list[str]:
    """Split text into units of optional leading whitespace + one word.

    Merges never cross unit boundaries.  ``''.join(pretokenize(t)) == t``.
    """
    units = _UNIT_RE.findall(text)
    consumed = sum(len(u) for u in units)
    if consumed < len(text):
        units.append(text[consumed:])
    return units


def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair in seq with new_id."""
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BalancedBPETokenizer:
    def __init__(self, base_chars: list[str], merges: list[tuple[int, int]]):
        self.base_chars = list(base_chars)
        self.merges = [tuple(m) for m in merges]
        self._build_tables()

    def _build_tables(self) -> None:
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.base_chars)}
        self.id_to_str = [UNK_RENDER] + list(self.base_chars)
        self.merge_rank: dict[tuple[int, int], int] = {}
        for a, b in self.merges:
            new_id = len(self.id_to_str)
            self.merge_rank[(a, b)] = new_id
            self.id_to_str.append(self.id_to_str[a] + self.id_to_str[b])
        self._encode_cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_str)

    def _encode_unit(self, unit: str) -> list[int]:
        cached = self._encode_cache.get(unit)
        if cached is not None:
            return cached
        seq = [self.char_to_id.get(ch, UNK_ID) for ch in unit]
        while len(seq) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(seq, seq[1:]):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            seq = _apply_merge(seq, best_pair, best_rank)
        self._encode_cache[unit] = seq
        return seq

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for unit in pretokenize(text):
            out.extend(self._encode_unit(unit))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_str[i] for i in ids)
