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
