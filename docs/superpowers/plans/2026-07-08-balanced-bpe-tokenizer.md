# Balanced Multilingual BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a single 10,000-token BPE tokenizer on the Wikipedia "India" article in English, Hindi, Telugu, and Kannada with per-language fertility ratios X_i ≤ 1.2 kept as close together as possible (score = 1000/(X_max − X_min)).

**Architecture:** Codepoint-level BPE (not byte-level) with word-boundary pre-splitting where leading whitespace attaches to the following word (GPT-style, so spaces don't inflate token counts). The four corpora stay separate during training; each merge step picks the language with the worst current fertility and takes its most frequent pair — a direct greedy optimizer of the score.

**Tech Stack:** Pure Python stdlib (`re`, `json`, `collections`, `urllib`) for tokenizer + download; `pytest` for tests; `nbformat`/`nbconvert` to build and execute the deliverable notebook. All confirmed installed (Python 3.13.9, pytest 9.0.3, nbconvert 7.16.6).

## Global Constraints

- Tokenizer implementation is **from scratch, pure Python stdlib** — no HuggingFace/SentencePiece.
- Total vocabulary is **exactly 10,000**: id 0 = `<unk>`, ids 1..C = base codepoints (sorted), remaining = merges in creation order.
- Fertility `X_i = len(encode(text_i)) / len(text_i.split())` must be **≤ 1.2 for all four languages**.
- Score reported as `1000 / (X_max − X_min)`.
- Round-trip `decode(encode(text)) == text` must hold exactly on every full corpus.
- Downloaded corpora are committed under `Session2/data/` for offline reproducibility.
- Spec: `docs/superpowers/specs/2026-07-08-balanced-bpe-tokenizer-design.md`.
- Working directory for all commands: repo root `/home/muttu/Desktop/ERA`.

---

### Task 1: Pre-tokenizer

**Files:**
- Create: `Session2/bpe_tokenizer.py`
- Test: `Session2/test_bpe_tokenizer.py`

**Interfaces:**
- Produces: `pretokenize(text: str) -> list[str]` — splits text into units of optional leading whitespace + one word; trailing pure-whitespace (if any) is its own final unit; `''.join(pretokenize(t)) == t` always. Also module constants `UNK_ID = 0`, `UNK_TOKEN = "<unk>"`.

- [ ] **Step 1: Write the failing tests**

Create `Session2/test_bpe_tokenizer.py`:

```python
import pytest

from bpe_tokenizer import UNK_ID, pretokenize


@pytest.mark.parametrize("text", [
    "hello world",
    "  leading and trailing  ",
    "line one\n\nline two\n",
    "नमस्ते   दुनिया",
    "",
    "   ",
])
def test_pretokenize_round_trip(text):
    assert "".join(pretokenize(text)) == text


def test_pretokenize_attaches_leading_whitespace():
    assert pretokenize("a b  c") == ["a", " b", "  c"]


def test_pretokenize_trailing_whitespace_is_own_unit():
    assert pretokenize("ab \n") == ["ab", " \n"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bpe_tokenizer'` (pytest adds the test file's dir to `sys.path`, so once the module exists it will import).

- [ ] **Step 3: Write the implementation**

Create `Session2/bpe_tokenizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Session2/bpe_tokenizer.py Session2/test_bpe_tokenizer.py
git commit -m "feat(session2): add pre-tokenizer for BPE units"
```

---

### Task 2: Tokenizer core — vocab tables, encode, decode

**Files:**
- Modify: `Session2/bpe_tokenizer.py`
- Test: `Session2/test_bpe_tokenizer.py`

**Interfaces:**
- Consumes: `pretokenize` from Task 1.
- Produces: `class BalancedBPETokenizer` with:
  - `__init__(self, base_chars: list[str], merges: list[tuple[int, int]])` — id 0 is `<unk>`; ids 1..len(base_chars) map to `base_chars` in given order; each merge `(a, b)` appends a new id whose string is `id_to_str[a] + id_to_str[b]`.
  - `vocab_size: int` property (= 1 + len(base_chars) + len(merges)).
  - `encode(text: str) -> list[int]` — applies merges lowest-rank-first per unit; unknown chars → `UNK_ID`.
  - `decode(ids: list[int]) -> str` — `<unk>` renders as `"�"`.
  - Module helper `_apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

Append to `Session2/test_bpe_tokenizer.py` (and add `BalancedBPETokenizer` to the import):

```python
from bpe_tokenizer import BalancedBPETokenizer


def test_encode_decode_with_explicit_merges():
    # base ids: ' '=1, 'a'=2, 'b'=3, 'c'=4; merge (2,3) -> id 5 = "ab"
    tok = BalancedBPETokenizer([" ", "a", "b", "c"], [(2, 3)])
    assert tok.vocab_size == 6
    assert tok.encode("ab c") == [5, 1, 4]
    assert tok.decode([5, 1, 4]) == "ab c"


def test_chained_merges():
    # merge (2,3) -> 5 = "ab", merge (5,4) -> 6 = "abc"
    tok = BalancedBPETokenizer([" ", "a", "b", "c"], [(2, 3), (5, 4)])
    assert tok.encode("abc") == [6]
    assert tok.decode([6]) == "abc"


def test_unknown_char_maps_to_unk():
    tok = BalancedBPETokenizer([" ", "a"], [])
    ids = tok.encode("a ω")
    assert ids == [2, 1, UNK_ID]
    assert tok.decode(ids) == "a �"


def test_round_trip_no_merges():
    tok = BalancedBPETokenizer(sorted(set("hello world")), [])
    assert tok.decode(tok.encode("hello world")) == "hello world"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'BalancedBPETokenizer'`; Task 1 tests still PASS.

- [ ] **Step 3: Write the implementation**

Append to `Session2/bpe_tokenizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Session2/bpe_tokenizer.py Session2/test_bpe_tokenizer.py
git commit -m "feat(session2): BPE encode/decode with merge ranks and <unk>"
```

---

### Task 3: Balanced training loop

**Files:**
- Modify: `Session2/bpe_tokenizer.py`
- Test: `Session2/test_bpe_tokenizer.py`

**Interfaces:**
- Consumes: `pretokenize`, `_apply_merge`, `BalancedBPETokenizer.__init__` from Tasks 1–2.
- Produces:
  - `BalancedBPETokenizer.train(corpora: dict[str, str], vocab_size: int = 10000, verbose: bool = False) -> BalancedBPETokenizer` (classmethod). Raises `ValueError` if `vocab_size <= 1 + number of unique codepoints`. Stops early (with a warning print) if all languages run out of pairs.
  - Module helper `_merge_in_state(st: dict, pair: tuple[int, int], new_id: int) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `Session2/test_bpe_tokenizer.py`:

```python
TINY = {
    "en": "the cat sat on the mat the cat sat " * 20,
    "xx": "aba abb aba abb aba " * 20,
}
TINY_BASE = len(set(TINY["en"] + TINY["xx"]))  # unique codepoints across corpora


def test_train_reaches_exact_vocab_size():
    target = 1 + TINY_BASE + 10
    tok = BalancedBPETokenizer.train(TINY, vocab_size=target)
    assert tok.vocab_size == target


def test_train_round_trip_on_training_text():
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    for text in TINY.values():
        assert tok.decode(tok.encode(text)) == text


def test_merges_reduce_token_count():
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    base = BalancedBPETokenizer(tok.base_chars, [])
    text = TINY["en"]
    assert len(tok.encode(text)) < len(base.encode(text))


def test_vocab_size_must_exceed_base_charset():
    with pytest.raises(ValueError):
        BalancedBPETokenizer.train(TINY, vocab_size=5)


def test_merge_loop_helps_worst_language_first():
    corpora = {
        "short": "a b c d " * 50,               # fertility ~2 (space+char units)
        "long": "qqqqqqqq rrrrrrrr " * 50,      # fertility ~9, clearly worst
    }
    n_base = len(set(corpora["short"] + corpora["long"]))
    tok = BalancedBPETokenizer.train(corpora, vocab_size=1 + n_base + 4)
    base = BalancedBPETokenizer(tok.base_chars, [])
    # all 4 merges must have gone to the worst language ("long")
    assert tok.fertility(corpora["long"]) < base.fertility(corpora["long"])
    assert tok.fertility(corpora["short"]) == base.fertility(corpora["short"])
```

Note: `fertility` is implemented in Task 4, but defining it now keeps this
test file stable; implement a minimal `fertility` in this task's Step 3 (one
line, spec'd in Task 4) so this test can pass here.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'train'`.

- [ ] **Step 3: Write the implementation**

Add `from collections import Counter` to the imports of `Session2/bpe_tokenizer.py`, then append inside the class:

```python
    def fertility(self, text: str) -> float:
        """Tokens per whitespace-separated word."""
        return len(self.encode(text)) / max(len(text.split()), 1)

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
```

And append at module level:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Session2/bpe_tokenizer.py Session2/test_bpe_tokenizer.py
git commit -m "feat(session2): balanced BPE training loop (worst-fertility-first)"
```

---

### Task 4: fertility metric + save/load

**Files:**
- Modify: `Session2/bpe_tokenizer.py`
- Test: `Session2/test_bpe_tokenizer.py`

**Interfaces:**
- Consumes: `BalancedBPETokenizer` from Tasks 2–3 (`fertility` already added in Task 3).
- Produces:
  - `save(self, path)` — writes JSON `{"base_chars": [...], "merges": [[a, b], ...]}` (UTF-8, `ensure_ascii=False`).
  - `load(cls, path) -> BalancedBPETokenizer` (classmethod).

- [ ] **Step 1: Write the failing tests**

Append to `Session2/test_bpe_tokenizer.py`:

```python
def test_fertility_counts_tokens_per_word():
    tok = BalancedBPETokenizer([" ", "a", "b"], [(2, 3)])  # "ab" -> one token
    assert tok.fertility("ab ab") == 1.0          # 2 tokens / 2 words
    assert tok.fertility("ba ba") == 2.5          # units "ba", " ba" -> 2 + 3 tokens


def test_save_load_identity(tmp_path):
    tok = BalancedBPETokenizer.train(TINY, vocab_size=1 + TINY_BASE + 10)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BalancedBPETokenizer.load(path)
    sample = "the cat sat aba"
    assert tok2.encode(sample) == tok.encode(sample)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.merges == tok.merges
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: `test_save_load_identity` FAILS with `AttributeError: ... no attribute 'save'`; `test_fertility_counts_tokens_per_word` PASSES already (fertility was added in Task 3 — that's fine, it pins the behavior).

- [ ] **Step 3: Write the implementation**

Add `import json` to `Session2/bpe_tokenizer.py` imports, then append inside the class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Session2/test_bpe_tokenizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Session2/bpe_tokenizer.py Session2/test_bpe_tokenizer.py
git commit -m "feat(session2): tokenizer save/load and fertility metric"
```

---

### Task 5: Data download script + fetch the four corpora

**Files:**
- Create: `Session2/download_data.py`
- Create (by running it): `Session2/data/en_india.txt`, `Session2/data/hi_india.txt`, `Session2/data/te_india.txt`, `Session2/data/kn_india.txt`

**Interfaces:**
- Produces: running `python3 Session2/download_data.py` writes the four corpus files. Later tasks read them via `Session2/data/{lang}_india.txt` for `lang` in `["en", "hi", "te", "kn"]`.

- [ ] **Step 1: Write the script**

Create `Session2/download_data.py`:

```python
"""Download the Wikipedia 'India' article in en, hi, te, kn as plain text.

Resolves the article title on each wiki by following the English article's
language links, then fetches plain-text extracts via the MediaWiki API.
Fails loudly on any missing link or empty extract.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "ERA-Session2-BPE/1.0 (educational assignment)"
DATA_DIR = Path(__file__).parent / "data"
TARGET_LANGS = ["hi", "te", "kn"]


def api_get(host: str, params: dict) -> dict:
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def get_titles() -> dict:
    """Resolve each wiki's title for the India article via language links."""
    data = api_get("en.wikipedia.org", {
        "action": "query", "titles": "India", "prop": "langlinks",
        "lllimit": "500", "format": "json", "redirects": "1",
    })
    page = next(iter(data["query"]["pages"].values()))
    titles = {"en": "India"}
    for link in page.get("langlinks", []):
        if link["lang"] in TARGET_LANGS:
            titles[link["lang"]] = link["*"]
    missing = set(["en"] + TARGET_LANGS) - set(titles)
    if missing:
        sys.exit(f"No language link found for: {sorted(missing)}")
    return titles


def fetch_extract(lang: str, title: str) -> str:
    data = api_get(f"{lang}.wikipedia.org", {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "titles": title, "format": "json", "redirects": "1",
    })
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text.strip():
        sys.exit(f"Empty extract for {lang}:{title}")
    return text


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for lang, title in get_titles().items():
        text = fetch_extract(lang, title)
        out = DATA_DIR / f"{lang}_india.txt"
        out.write_text(text, encoding="utf-8")
        print(f"{lang}: {title!r} -> {out.name}: "
              f"{len(text):,} chars, {len(text.split()):,} words")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python3 Session2/download_data.py`
Expected output (numbers approximate): four lines, one per language, each reporting thousands of words, e.g.

```
en: 'India' -> en_india.txt: ~100,000 chars, ~15,000 words
hi: 'भारत' -> hi_india.txt: ...
te: 'భారతదేశం' -> te_india.txt: ...
kn: 'ಭಾರತ' -> kn_india.txt: ...
```

If any language exits with "Empty extract" or "No language link", stop and investigate (do not stub data).

- [ ] **Step 3: Sanity-check the files contain the right scripts**

Run:

```bash
python3 - <<'EOF'
from pathlib import Path
ranges = {"en": (0x0041, 0x007A), "hi": (0x0900, 0x097F),
          "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF)}
for lang, (lo, hi) in ranges.items():
    text = Path(f"Session2/data/{lang}_india.txt").read_text(encoding="utf-8")
    in_script = sum(lo <= ord(c) <= hi for c in text) / len(text)
    print(f"{lang}: {len(text):,} chars, {in_script:.0%} in native script")
    assert len(text) > 10_000, f"{lang} corpus suspiciously small"
    assert in_script > 0.3, f"{lang} corpus not mostly in expected script"
print("OK")
EOF
```

Expected: `OK` with each language > 10k chars and a healthy native-script share.

- [ ] **Step 4: Commit script and data**

```bash
git add Session2/download_data.py Session2/data/
git commit -m "feat(session2): download India article corpora (en/hi/te/kn)"
```

(If `.gitignore` excludes data files, force-add with `git add -f Session2/data/` — the spec requires the corpora committed.)

---

### Task 6: Full training run + evaluation script

**Files:**
- Create: `Session2/train_and_evaluate.py`
- Create (by running it): `Session2/tokenizer_10k.json`, `Session2/results.json`

**Interfaces:**
- Consumes: `BalancedBPETokenizer` (Tasks 2–4), corpus files (Task 5).
- Produces: `Session2/tokenizer_10k.json` (tokenizer artifact loadable via `BalancedBPETokenizer.load`) and `Session2/results.json` with shape `{"per_language": {lang: {"words": int, "tokens": int, "X": float}}, "score": float}`.

- [ ] **Step 1: Write the script**

Create `Session2/train_and_evaluate.py`:

```python
"""Train the balanced 10k-vocab BPE tokenizer on the four India corpora
and report per-language fertility X_i and the assignment score."""
import json
import time
from pathlib import Path

from bpe_tokenizer import BalancedBPETokenizer

HERE = Path(__file__).parent
LANGS = ["en", "hi", "te", "kn"]
VOCAB_SIZE = 10_000


def load_corpora() -> dict:
    return {lang: (HERE / "data" / f"{lang}_india.txt").read_text(encoding="utf-8")
            for lang in LANGS}


def main() -> None:
    corpora = load_corpora()
    t0 = time.time()
    tok = BalancedBPETokenizer.train(corpora, vocab_size=VOCAB_SIZE, verbose=True)
    print(f"\ntrained in {time.time() - t0:.1f}s, vocab_size={tok.vocab_size}")
    assert tok.vocab_size == VOCAB_SIZE

    results = {}
    for lang, text in corpora.items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round-trip failed for {lang}"
        words = len(text.split())
        results[lang] = {"words": words, "tokens": len(ids),
                         "X": len(ids) / words}

    xs = [r["X"] for r in results.values()]
    spread = max(xs) - min(xs)
    score = 1000 / spread if spread > 0 else float("inf")

    print(f"\n{'lang':<6}{'words':>10}{'tokens':>10}{'X':>10}")
    for lang, r in results.items():
        print(f"{lang:<6}{r['words']:>10,}{r['tokens']:>10,}{r['X']:>10.4f}")
        assert r["X"] <= 1.2, f"X for {lang} = {r['X']:.4f} exceeds 1.2"
    print(f"\nX_max - X_min = {spread:.6f}")
    print(f"score = 1000 / (X_max - X_min) = {score:,.1f}")

    tok.save(HERE / "tokenizer_10k.json")
    (HERE / "results.json").write_text(
        json.dumps({"per_language": results, "score": score}, indent=2))
    print(f"\nsaved {HERE / 'tokenizer_10k.json'} and {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full training**

Run: `cd Session2 && python3 train_and_evaluate.py` (expect a few minutes; verbose prints fertility every 1000 merges — the four values should visibly converge toward each other and end ≤ 1.2).
Expected: table of four rows, all `X ≤ 1.2`, all four X values within ~0.01 of each other, a large score, no assertion failures.

If any `X > 1.2` after 10k vocab: the corpora are too small/diverse for the budget — investigate corpus sizes before changing any code (this is not expected with full Wikipedia articles).

- [ ] **Step 3: Verify the saved artifact reloads**

Run:

```bash
cd Session2 && python3 - <<'EOF'
from bpe_tokenizer import BalancedBPETokenizer
tok = BalancedBPETokenizer.load("tokenizer_10k.json")
assert tok.vocab_size == 10_000
sample = "India is a country in South Asia."
print(tok.decode(tok.encode(sample)) == sample, len(tok.encode(sample)), "tokens")
EOF
```

Expected: `True <n> tokens` with n well under the word count × 1.2 + a few.

- [ ] **Step 4: Commit**

```bash
git add Session2/train_and_evaluate.py Session2/tokenizer_10k.json Session2/results.json
git commit -m "feat(session2): train balanced 10k BPE, report fertility and score"
```

---

### Task 7: Deliverable notebook

**Files:**
- Create: `Session2/make_notebook.py`
- Create (by running it): `Session2/Session2_BPE_Tokenizer.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 1–6 (module, data files, `tokenizer_10k.json`, `results.json`).
- Produces: an executed notebook telling the full story; re-runnable top-to-bottom inside `Session2/`.

- [ ] **Step 1: Write the notebook builder**

Create `Session2/make_notebook.py`:

```python
"""Build Session2_BPE_Tokenizer.ipynb. Run once, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md("# Session 2 — Balanced Multilingual BPE Tokenizer\n"
       "\n"
       "Train a single **10,000-token** BPE tokenizer on the Wikipedia *India*\n"
       "article in **English, Hindi, Telugu and Kannada** such that each\n"
       "language's fertility ratio\n"
       "\n"
       "$$X_i = \\frac{\\text{tokens produced on language } i}{\\text{whitespace-separated words}}$$\n"
       "\n"
       "is **≤ 1.2** and the four ratios are as close as possible.\n"
       "\n"
       "**Score** $= 1000 / (X_{max} - X_{min})$\n"
       "\n"
       "Design: codepoint-level BPE, whitespace attached to the following word,\n"
       "and a *balanced merge loop* — every merge is taken from whichever\n"
       "language currently has the worst fertility, which greedily minimizes\n"
       "$X_{max} - X_{min}$ at every step."),
    code("import json, subprocess, sys\n"
         "from pathlib import Path\n"
         "from bpe_tokenizer import BalancedBPETokenizer\n"
         "\n"
         "LANGS = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'kn': 'Kannada'}\n"
         "DATA = Path('data')\n"
         "\n"
         "# Download the four corpora if not already present\n"
         "if not all((DATA / f'{l}_india.txt').exists() for l in LANGS):\n"
         "    subprocess.run([sys.executable, 'download_data.py'], check=True)\n"
         "\n"
         "corpora = {l: (DATA / f'{l}_india.txt').read_text(encoding='utf-8')\n"
         "           for l in LANGS}\n"
         "print(f\"{'lang':<10}{'chars':>12}{'words':>10}{'unique chars':>15}\")\n"
         "for l, text in corpora.items():\n"
         "    print(f'{LANGS[l]:<10}{len(text):>12,}{len(text.split()):>10,}'\n"
         "          f'{len(set(text)):>15,}')"),
    md("## Train (or load) the tokenizer\n"
       "\n"
       "Training takes a few minutes, so the committed artifact\n"
       "`tokenizer_10k.json` is loaded by default. Set `RETRAIN = True` to\n"
       "reproduce it from scratch (equivalent to `python3 train_and_evaluate.py`)."),
    code("RETRAIN = False\n"
         "if RETRAIN or not Path('tokenizer_10k.json').exists():\n"
         "    tok = BalancedBPETokenizer.train(corpora, vocab_size=10_000,\n"
         "                                     verbose=True)\n"
         "    tok.save('tokenizer_10k.json')\n"
         "else:\n"
         "    tok = BalancedBPETokenizer.load('tokenizer_10k.json')\n"
         "print('vocab size:', tok.vocab_size)\n"
         "print('base codepoints:', len(tok.base_chars))\n"
         "print('learned merges:', len(tok.merges))"),
    md("## Per-language fertility and score"),
    code("results = {}\n"
         "for l, text in corpora.items():\n"
         "    ids = tok.encode(text)\n"
         "    assert tok.decode(ids) == text, f'round-trip failed for {l}'\n"
         "    words = len(text.split())\n"
         "    results[l] = (words, len(ids), len(ids) / words)\n"
         "\n"
         "print(f\"{'lang':<10}{'words':>10}{'tokens':>10}{'X (tok/word)':>15}\")\n"
         "for l, (w, t, x) in results.items():\n"
         "    flag = 'OK' if x <= 1.2 else 'FAIL'\n"
         "    print(f'{LANGS[l]:<10}{w:>10,}{t:>10,}{x:>15.4f}  {flag}')\n"
         "\n"
         "xs = [x for _, _, x in results.values()]\n"
         "spread = max(xs) - min(xs)\n"
         "print(f'\\nX_max - X_min = {spread:.6f}')\n"
         "print(f'score = 1000 / (X_max - X_min) = '\n"
         "      f\"{1000 / spread:,.1f}\" if spread > 0 else 'score = inf')"),
    md("## Encode / decode demo"),
    code("samples = {\n"
         "    'en': 'India is the seventh-largest country in the world.',\n"
         "    'hi': 'भारत दक्षिण एशिया में स्थित एक देश है।',\n"
         "    'te': 'భారతదేశం ప్రపంచంలో ఏడవ పెద్ద దేశం.',\n"
         "    'kn': 'ಭಾರತವು ದಕ್ಷಿಣ ಏಷ್ಯಾದಲ್ಲಿರುವ ಒಂದು ದೇಶ.',\n"
         "}\n"
         "for l, s in samples.items():\n"
         "    ids = tok.encode(s)\n"
         "    pieces = [tok.id_to_str[i] for i in ids]\n"
         "    assert tok.decode(ids) == s\n"
         "    print(f'{LANGS[l]}: {len(s.split())} words -> {len(ids)} tokens')\n"
         "    print('  ', pieces, '\\n')"),
]

nbf.write(nb, "Session2_BPE_Tokenizer.ipynb")
print("wrote Session2_BPE_Tokenizer.ipynb")
```

- [ ] **Step 2: Build and execute the notebook**

Run:

```bash
cd Session2 && python3 make_notebook.py && \
python3 -m jupyter nbconvert --to notebook --execute --inplace Session2_BPE_Tokenizer.ipynb
```

Expected: nbconvert exits 0.

- [ ] **Step 3: Verify executed outputs**

Run: `cd Session2 && python3 -c "
import json
nb = json.load(open('Session2_BPE_Tokenizer.ipynb'))
outs = [o for c in nb['cells'] for o in c.get('outputs', [])]
text = ''.join(''.join(o.get('text', '')) for o in outs)
assert 'score' in text and 'OK' in text and 'FAIL' not in text.replace('round-trip failed', '')
print(text)
"`
Expected: printed cell outputs show the four corpora stats, four `OK` fertility rows, the spread, the score, and the token demos; no errors.

- [ ] **Step 4: Run the whole test suite one last time**

Run: `python3 -m pytest Session2/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add Session2/make_notebook.py Session2/Session2_BPE_Tokenizer.ipynb
git commit -m "feat(session2): deliverable notebook with training story and score"
```

---

## Self-review notes

- Spec coverage: pre-split/codepoint BPE (Tasks 1–2), balanced merge loop (Task 3), save/load + fertility (Tasks 3–4), download with langlink resolution + loud failures (Task 5), 10k training + X ≤ 1.2 asserts + score + artifact (Task 6), notebook narrative (Task 7). `train_and_evaluate.py` is one file beyond the spec's component list, kept for headless reproducibility (noted to user).
- Type consistency: `BalancedBPETokenizer(base_chars, merges)` constructor, `train`/`load` classmethods, `encode`/`decode`/`fertility`/`save` used identically across Tasks 2–7.
- `fertility` is introduced in Task 3 (needed by the balance test) and pinned in Task 4 — intentional, called out in both tasks.
