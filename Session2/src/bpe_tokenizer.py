"""Balanced multilingual BPE tokenizer (codepoint-level).

Trained on several corpora at once: each merge is chosen from the language
whose fertility (tokens per word) is currently worst, keeping the
per-language compression ratios close together.
"""
from __future__ import annotations

import json
import re
from collections import Counter

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


def _merge_in_state(st: dict, pair: tuple[int, int], new_id: int) -> None:
    """Apply a merge to one language's training state, updating pair counts
    and the running token total incrementally."""
    if st["pairs"].get(pair, 0) == 0:
        return
    a, b = pair
    seqs, counts, pairs = st["seqs"], st["counts"], st["pairs"]
    for idx, seq in enumerate(seqs):
        if a not in seq:
            continue
        has_pair = any(seq[i] == a and seq[i + 1] == b
                       for i in range(len(seq) - 1))
        if not has_pair:
            continue
        n = counts[idx]
        for p in zip(seq, seq[1:]):
            pairs[p] -= n
            if pairs[p] <= 0:
                del pairs[p]
        new_seq = _apply_merge(seq, pair, new_id)
        for p in zip(new_seq, new_seq[1:]):
            pairs[p] += n
        st["tokens"] += (len(new_seq) - len(seq)) * n
        seqs[idx] = new_seq


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

    def fertility(self, text: str) -> float:
        """Tokens per whitespace-separated word."""
        return len(self.encode(text)) / max(len(text.split()), 1)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"base_chars": self.base_chars,
                       "merges": [list(m) for m in self.merges]},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "BalancedBPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["base_chars"], [tuple(m) for m in d["merges"]])

    @classmethod
    def train(cls, corpora: dict[str, str], vocab_size: int = 10000,
              verbose: bool = False) -> "BalancedBPETokenizer":
        base_chars = sorted({ch for text in corpora.values() for ch in text})
        target_merges = vocab_size - 1 - len(base_chars)
        if target_merges <= 0:
            raise ValueError(
                f"vocab_size={vocab_size} too small: need > {1 + len(base_chars)} "
                f"(<unk> + {len(base_chars)} base codepoints)")
        char_to_id = {c: i + 1 for i, c in enumerate(base_chars)}

        state: dict[str, dict] = {}
        for lang, text in corpora.items():
            unit_counts = Counter(pretokenize(text))
            seqs, counts = [], []
            for unit, n in unit_counts.items():
                seqs.append([char_to_id[ch] for ch in unit])
                counts.append(n)
            pairs: Counter = Counter()
            for seq, n in zip(seqs, counts):
                for pair in zip(seq, seq[1:]):
                    pairs[pair] += n
            state[lang] = {
                "seqs": seqs,
                "counts": counts,
                "pairs": pairs,
                "tokens": sum(len(s) * n for s, n in zip(seqs, counts)),
                "words": max(len(text.split()), 1),
            }

        merges: list[tuple[int, int]] = []
        next_id = 1 + len(base_chars)
        for step in range(target_merges):
            candidates = [l for l in state if state[l]["pairs"]]
            if not candidates:
                print(f"warning: ran out of pairs after {len(merges)} merges")
                break
            worst = max(candidates,
                        key=lambda l: state[l]["tokens"] / state[l]["words"])
            pair, _ = max(state[worst]["pairs"].items(),
                          key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            merges.append(pair)
            for st in state.values():
                _merge_in_state(st, pair, next_id)
            next_id += 1
            if verbose and (step + 1) % 1000 == 0:
                ferts = {l: round(st["tokens"] / st["words"], 4)
                         for l, st in state.items()}
                print(f"merge {step + 1}/{target_merges}: fertility={ferts}")
        return cls(base_chars, merges)
