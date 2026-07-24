# Session 4 — Kannada Corpus Cleaning + Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Session 4's cleaning strategies for real to a 50M-token Kannada slice of Sangraha, then build one self-contained widget (deployable to Netlify) that reports the strategies, dataset, per-stage cleaning, and final statistics.

**Architecture:** Seven independent, pure-function pipeline stages (normalize → langid → quality → dedup → pii → decontam → manifest) each with unit tests, orchestrated by `run_pipeline.py` which emits `cleaned_corpus.parquet`, `manifest.json`, and `stats.json`. A build step injects `stats.json` into a static HTML template to produce the widget.

**Tech Stack:** Python 3.13, pyarrow/pandas, tiktoken (o200k_base), numpy (from-scratch MinHash/LSH), transformers+torch (ai4bharat/IndicNER on CUDA), vanilla HTML/CSS/JS, pytest.

## Global Constraints

- Corpus slice: `ai4bharat/sangraha` config `unverified` language `kan`, shard `data-0.parquet`, License **CC-BY-4.0**. Slice = first docs until 50M o200k tokens; already saved at `Session4/data/raw/kan_slice.parquet` (59,603 docs).
- Token counting uses **`tiktoken` o200k_base** everywhere. Never estimate tokens with a word×ratio.
- Normalization MUST **keep** U+200C (ZWNJ) and U+200D (ZWJ); MUST remove U+200B (ZWSP), U+FEFF (BOM), U+202A–U+202E (bidi), U+FFFD (replacement).
- Cleaning happens **before** the content hash (manifest reflects cleaned text).
- The deliverable widget must be **self-contained** (no external JS/CSS; Google Fonts link is the only permitted external resource) and answer **"8 strategies"** (7 applied + ghost-tags demo; extraction inherited from Session 3 as a 9th stage).
- Raw/intermediate data stays under `Session4/data/` (gitignored). Never commit corpus data.
- Tests run from repo root: `python -m pytest Session4/pipeline/tests/<file> -v` (a `conftest.py` puts the pipeline dir on `sys.path`; tests use flat imports like `from normalize import normalize_text`).
- Commit after each task with the message shown. Work stays on the `master` branch (matches repo history).

---

### Task 1: Scaffold + Text Normalization

**Files:**
- Create: `Session4/pipeline/conftest.py`
- Create: `Session4/pipeline/normalize.py`
- Test: `Session4/pipeline/tests/test_normalize.py`

**Interfaces:**
- Produces: `normalize_text(text: str) -> tuple[str, list[str]]` — returns `(clean_text, ops_applied)` where ops is a subset of `["html_unescape","nfc","strip_noise","strip_control","collapse_ws"]`.

- [ ] **Step 1: Create the conftest so flat imports work**

```python
# Session4/pipeline/conftest.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 2: Write the failing test**

```python
# Session4/pipeline/tests/test_normalize.py
from normalize import normalize_text


def test_keeps_brahmic_joiners_strips_noise_and_entities():
    # ZWSP, BOM, RLO, replacement char = noise; ZWNJ + ZWJ = keep
    dirty = "ಅ​﻿‮ &amp; ಬ‌ಸ‍   ಕ\x07"
    clean, ops = normalize_text(dirty)
    assert "​" not in clean          # ZWSP removed
    assert "﻿" not in clean          # BOM removed
    assert "‮" not in clean          # RLO removed
    assert "�" not in clean          # replacement never introduced
    assert "\x07" not in clean            # control char removed
    assert "‌" in clean              # ZWNJ preserved
    assert "‍" in clean              # ZWJ preserved
    assert "&amp;" not in clean and "&" in clean   # entity unescaped
    assert "   " not in clean             # whitespace collapsed
    assert {"html_unescape", "strip_noise", "strip_control", "collapse_ws"} <= set(ops)


def test_idempotent_on_clean_text():
    clean_once, _ = normalize_text("ಕನ್ನಡ ಪಠ್ಯ")
    clean_twice, ops = normalize_text(clean_once)
    assert clean_once == clean_twice and ops == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'normalize'`

- [ ] **Step 4: Write minimal implementation**

```python
# Session4/pipeline/normalize.py
import html
import re
import unicodedata

# Invisible characters that are noise and must be removed.
# NOTE: U+200C ZWNJ and U+200D ZWJ are intentionally absent — they are
# legitimate Brahmic joiners and must survive.
_NOISE = ["​", "﻿", "‎", "‏",
          "‪", "‫", "‬", "‭", "‮", "�"]

# Control chars except tab/newline/carriage-return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def normalize_text(text):
    ops = []

    unescaped = html.unescape(text)
    if unescaped != text:
        ops.append("html_unescape")
    text = unescaped

    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        ops.append("nfc")
    text = nfc

    before = text
    for ch in _NOISE:
        text = text.replace(ch, "")
    if text != before:
        ops.append("strip_noise")

    before = text
    text = _CONTROL.sub("", text)
    if text != before:
        ops.append("strip_control")

    before = text
    text = _SPACES.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    if text != before:
        ops.append("collapse_ws")

    return text, ops
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_normalize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add Session4/pipeline/conftest.py Session4/pipeline/normalize.py Session4/pipeline/tests/test_normalize.py
git commit -m "feat(session4): text normalization stage (keeps Brahmic joiners)"
```

---

### Task 2: Language-ID Validation

**Files:**
- Create: `Session4/pipeline/langid.py`
- Test: `Session4/pipeline/tests/test_langid.py`

**Interfaces:**
- Produces: `kannada_ratio(text: str) -> float` (fraction of letters in the Kannada block); `is_kannada(text: str, threshold: float = 0.5) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_langid.py
from langid import kannada_ratio, is_kannada


def test_pure_kannada_passes_english_fails():
    assert is_kannada("ಕನ್ನಡ ಭಾಷೆ ಒಂದು ಸುಂದರ ಭಾಷೆ")
    assert not is_kannada("this is plain english text")


def test_ratio_between_zero_and_one_for_codeswitch():
    r = kannada_ratio("ಕನ್ನಡ hello world")
    assert 0.0 < r < 1.0


def test_empty_is_zero():
    assert kannada_ratio("12345 !!! ") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_langid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langid'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/langid.py
_KANNADA_LO, _KANNADA_HI = 0x0C80, 0x0CFF


def kannada_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    kn = sum(1 for c in letters if _KANNADA_LO <= ord(c) <= _KANNADA_HI)
    return kn / len(letters)


def is_kannada(text, threshold=0.5):
    return kannada_ratio(text) >= threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_langid.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/langid.py Session4/pipeline/tests/test_langid.py
git commit -m "feat(session4): dependency-free Kannada language-ID validation"
```

---

### Task 3: Quality Filtering (with Indic-bias demonstration)

**Files:**
- Create: `Session4/pipeline/quality.py`
- Test: `Session4/pipeline/tests/test_quality.py`

**Interfaces:**
- Produces: `DEFAULT_THRESHOLDS: dict`; `quality_check(text: str, thresholds: dict | None = None, indic_aware: bool = True) -> tuple[bool, list[str]]` — returns `(passed, reasons)`.

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_quality.py
from quality import quality_check, DEFAULT_THRESHOLDS

GOOD = ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಒಂದು ದೊಡ್ಡ ನಗರವಾಗಿದೆ. "
        "ಇಲ್ಲಿ ಅನೇಕ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿಗಳು ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತವೆ. "
        "ಈ ನಗರವು ಶಿಕ್ಷಣ ಮತ್ತು ಸಂಶೋಧನೆಗೆ ಹೆಸರುವಾಸಿಯಾಗಿದೆ. ") * 3


def test_good_doc_passes_tiny_doc_fails():
    ok, reasons = quality_check(GOOD)
    assert ok and reasons == []
    ok2, reasons2 = quality_check("ಸಣ್ಣ ಪಠ್ಯ")
    assert not ok2 and "too_short" in reasons2


def test_boilerplate_nav_is_rejected():
    nav = "\n".join(["ಮುಖಪುಟ | ಸುದ್ದಿ | ಕ್ರೀಡೆ | ಸಂಪರ್ಕ"] * 20)
    ok, reasons = quality_check(nav)
    assert not ok and "boilerplate" in reasons


def test_indic_bias_english_tuned_filter_wrongly_fails_good_kannada():
    passed_aware, _ = quality_check(GOOD, indic_aware=True)
    passed_naive, reasons = quality_check(GOOD, indic_aware=False)
    assert passed_aware is True
    assert passed_naive is False and "too_many_symbols" in reasons
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/quality.py
import re

DEFAULT_THRESHOLDS = {
    "min_chars": 200,
    "max_symbol_ratio": 0.25,
    "max_boiler_line_ratio": 0.5,
    "max_dup_line_ratio": 0.3,
    "min_mean_word_len": 1.2,
    "max_mean_word_len": 40.0,
}

# A line with two or more pipe separators reads like a nav/boilerplate strip.
_NAV_LINE = re.compile(r".+\|.+\|.+")
# Indic combining marks and letters live in these blocks (Devanagari..Malayalam
# covers Kannada 0C80-0CFF). Marks are category Mn/Mc so `isalnum()` is False;
# treating them as letters is what makes the filter Indic-aware.
_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F


def _is_letterlike(ch, indic_aware):
    if ch.isalnum():
        return True
    if indic_aware and _INDIC_LO <= ord(ch) <= _INDIC_HI:
        return True
    return False


def quality_check(text, thresholds=None, indic_aware=True):
    t = thresholds or DEFAULT_THRESHOLDS
    reasons = []
    s = text.strip()

    if len(s) < t["min_chars"]:
        reasons.append("too_short")

    total = len(s) or 1
    symbols = sum(1 for c in s if not _is_letterlike(c, indic_aware) and not c.isspace())
    if symbols / total > t["max_symbol_ratio"]:
        reasons.append("too_many_symbols")

    lines = [ln for ln in s.split("\n") if ln.strip()]
    if lines:
        nav = sum(1 for ln in lines if _NAV_LINE.match(ln))
        if nav / len(lines) > t["max_boiler_line_ratio"]:
            reasons.append("boilerplate")
        dup = len(lines) - len(set(lines))
        if dup / len(lines) > t["max_dup_line_ratio"]:
            reasons.append("dup_lines")

    words = s.split()
    if words:
        mean_wl = sum(len(w) for w in words) / len(words)
        if not (t["min_mean_word_len"] <= mean_wl <= t["max_mean_word_len"]):
            reasons.append("word_len")

    return (len(reasons) == 0, reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_quality.py -v`
Expected: PASS (3 passed). If `test_indic_bias...` fails because GOOD's naive symbol ratio is below 0.25, lengthen GOOD or lower the assertion's threshold by passing `dict(DEFAULT_THRESHOLDS, max_symbol_ratio=0.15)` to the `indic_aware=False` call — but first confirm real Kannada trips it; combining marks typically push it above 0.25.

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/quality.py Session4/pipeline/tests/test_quality.py
git commit -m "feat(session4): quality-filter cascade + Indic filter-bias demo"
```

---

### Task 4: Deduplication (from-scratch MinHash + LSH)

**Files:**
- Create: `Session4/pipeline/dedup.py`
- Test: `Session4/pipeline/tests/test_dedup.py`

**Interfaces:**
- Produces: `shingles(text: str, k: int = 5) -> set[str]`; `minhash_signature(shingle_set: set[str]) -> np.ndarray` (length 128, int64); `find_duplicates(docs: Iterable[tuple[str, str]], threshold: float = 0.8) -> set[str]` (keys to drop; keeps the first-seen of each near-dup group).

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_dedup.py
from dedup import shingles, minhash_signature, find_duplicates
import numpy as np


def test_shingles_and_signature_shape():
    sh = shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು", k=3)
    assert len(sh) == 4
    sig = minhash_signature(sh)
    assert sig.shape == (128,) and sig.dtype == np.int64


def test_identical_docs_have_identical_signatures():
    a = minhash_signature(shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"))
    b = minhash_signature(shingles("ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"))
    assert np.array_equal(a, b)


def test_finds_near_duplicate_keeps_distinct():
    base = ("ಕರ್ನಾಟಕ ಸರ್ಕಾರವು ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ ಇದು "
            "ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಅನುಕೂಲಕರವಾಗಿದೆ ಎಂದು ಅಧಿಕಾರಿಗಳು ತಿಳಿಸಿದ್ದಾರೆ")
    near = base + " ಹೆಚ್ಚಿನ ವಿವರ ಶೀಘ್ರದಲ್ಲೇ"       # ~95% overlap
    distinct = ("ಮುಂಬೈನಲ್ಲಿ ಇಂದು ಭಾರೀ ಮಳೆ ಸುರಿಯಿತು ಸಂಚಾರ "
                "ಸಂಪೂರ್ಣವಾಗಿ ಅಸ್ತವ್ಯಸ್ತಗೊಂಡಿತು ಜನರು ಪರದಾಡಿದರು")
    dups = find_duplicates([("a", base), ("b", near), ("c", distinct)], threshold=0.7)
    assert "b" in dups
    assert "a" not in dups and "c" not in dups
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dedup'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/dedup.py
import hashlib
import re
import numpy as np

_WORD = re.compile(r"\S+")
_NUM_PERM = 128
_BANDS = 32
_ROWS = 4                       # _BANDS * _ROWS == _NUM_PERM
_P = (1 << 31) - 1              # Mersenne prime; keeps a*h+b within int64

# Fixed random permutation parameters (a, b) shared by every document.
_rng = np.random.RandomState(42)
_A = _rng.randint(1, _P, size=_NUM_PERM).astype(np.int64)
_B = _rng.randint(0, _P, size=_NUM_PERM).astype(np.int64)


def shingles(text, k=5):
    words = _WORD.findall(text)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _shingle_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big") % _P


def minhash_signature(shingle_set):
    if not shingle_set:
        return np.full(_NUM_PERM, _P, dtype=np.int64)
    hs = np.fromiter((_shingle_hash(s) for s in shingle_set), dtype=np.int64)
    # (A[:,None] * hs[None,:] + B[:,None]) mod P, then min over the shingle axis.
    perm = (np.outer(_A, hs) + _B[:, None]) % _P
    return perm.min(axis=1)


def _band_keys(sig):
    return [sig[b * _ROWS:(b + 1) * _ROWS].tobytes() for b in range(_BANDS)]


def _jaccard(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))


def find_duplicates(docs, threshold=0.8):
    buckets = [dict() for _ in range(_BANDS)]
    sigs = {}
    duplicates = set()
    for key, text in docs:
        sig = minhash_signature(shingles(text))
        candidates = set()
        bkeys = _band_keys(sig)
        for b, bk in enumerate(bkeys):
            bucket = buckets[b].setdefault(bk, [])
            candidates.update(bucket)
            bucket.append(key)
        sigs[key] = sig
        for c in candidates:
            if c in duplicates:
                continue
            if _jaccard(sigs[c], sig) >= threshold:
                duplicates.add(key)         # drop later doc, keep first-seen
                break
    return duplicates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_dedup.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/dedup.py Session4/pipeline/tests/test_dedup.py
git commit -m "feat(session4): from-scratch MinHash+LSH near-duplicate detection"
```

---

### Task 5: PII Removal (regex + IndicNER names)

**Files:**
- Create: `Session4/pipeline/pii.py`
- Test: `Session4/pipeline/tests/test_pii.py`

**Interfaces:**
- Produces: `mask_structured(text: str) -> tuple[str, dict]` (counts keyed `email/phone/url/ip`); `mask_names(text: str, pipe=None) -> tuple[str, int]` (uses an injected `pipe` callable in tests, lazy-loads IndicNER otherwise); `load_ner()` returns a callable pipeline.

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_pii.py
from pii import mask_structured, mask_names


def test_mask_structured_identifiers():
    text = "ಸಂಪರ್ಕಿಸಿ ram.k@example.com ಅಥವಾ +91 9876543210 ಭೇಟಿ https://foo.example/x IP 10.0.0.1"
    out, counts = mask_structured(text)
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out and "[IP]" in out
    assert counts["email"] == 1 and counts["phone"] == 1
    assert counts["url"] == 1 and counts["ip"] == 1
    assert "ram.k@example.com" not in out and "9876543210" not in out


def test_mask_names_with_injected_pipeline():
    name = "ರಾಮಪ್ಪ"
    text = f"ಹೆಸರು {name} ಅವರು ಬಂದರು"
    s = text.index(name)
    e = s + len(name)

    class StubPipe:
        def __call__(self, t):
            return [{"entity_group": "PER", "start": s, "end": e, "word": name}]

    out, n = mask_names(text, pipe=StubPipe())
    assert out == "ಹೆಸರು [NAME] ಅವರು ಬಂದರು" and n == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_pii.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pii'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/pii.py
import re
from functools import lru_cache

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def mask_structured(text):
    counts = {"email": 0, "phone": 0, "url": 0, "ip": 0}
    for pat, tag, key in [(_URL, "URL", "url"), (_EMAIL, "EMAIL", "email"),
                          (_IP, "IP", "ip"), (_PHONE, "PHONE", "phone")]:
        text, n = pat.subn(f"[{tag}]", text)
        counts[key] += n
    return text, counts


@lru_cache(maxsize=1)
def load_ner():
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)
    import torch
    tok = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("token-classification", model=model, tokenizer=tok,
                    aggregation_strategy="simple", device=device)


def mask_names(text, pipe=None):
    pipe = pipe or load_ner()
    ents = pipe(text)
    spans = sorted(((e["start"], e["end"]) for e in ents
                    if e.get("entity_group", "").upper() in ("PER", "PERSON")),
                   reverse=True)
    for s, e in spans:
        text = text[:s] + "[NAME]" + text[e:]
    return text, len(spans)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_pii.py -v`
Expected: PASS (2 passed). (The name test uses a stub, so no model download happens in unit tests.)

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/pii.py Session4/pipeline/tests/test_pii.py
git commit -m "feat(session4): PII removal (regex identifiers + IndicNER names)"
```

---

### Task 6: Decontamination

**Files:**
- Create: `Session4/pipeline/decontam.py`
- Test: `Session4/pipeline/tests/test_decontam.py`

**Interfaces:**
- Produces: `ngram_fingerprints(text: str, n: int = 8) -> set[str]`; `build_eval_fingerprint_set(eval_texts: Iterable[str], n: int = 8) -> set[str]`; `is_contaminated(text: str, eval_fps: set[str], n: int = 8, min_overlap: int = 1) -> bool`; `CANARY: str`; `has_canary(text: str, canary: str = CANARY) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_decontam.py
from decontam import (build_eval_fingerprint_set, is_contaminated,
                      has_canary, CANARY)


def test_flags_overlap_only():
    eval_texts = ["ಇದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು"]
    fps = build_eval_fingerprint_set(eval_texts, n=6)
    leak = "ಪೀಠಿಕೆ ಇದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು ಕೊನೆ"
    clean = "ಸಂಪೂರ್ಣ ಬೇರೆ ವಿಷಯ ಇಲ್ಲಿದೆ ಯಾವುದೇ ಹೋಲಿಕೆ ಇಲ್ಲ ಎಂದು ಹೇಳಬಹುದು"
    assert is_contaminated(leak, fps, n=6)
    assert not is_contaminated(clean, fps, n=6)


def test_canary_detection():
    assert has_canary("ರಾಂಡಮ್ " + CANARY + " ಪಠ್ಯ")
    assert not has_canary("ಶುದ್ಧ ಪಠ್ಯ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_decontam.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decontam'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/decontam.py
import hashlib
import re

CANARY = "CANARY-9f3a1b7c-session4-do-not-train"
_WS = re.compile(r"\s+")


def _norm(text):
    return _WS.sub(" ", text.strip().lower())


def _fp(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def ngram_fingerprints(text, n=8):
    words = _norm(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {_fp(" ".join(words))}
    return {_fp(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def build_eval_fingerprint_set(eval_texts, n=8):
    fps = set()
    for t in eval_texts:
        fps |= ngram_fingerprints(t, n)
    return fps


def is_contaminated(text, eval_fps, n=8, min_overlap=1):
    return len(ngram_fingerprints(text, n) & eval_fps) >= min_overlap


def has_canary(text, canary=CANARY):
    return canary in text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_decontam.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/decontam.py Session4/pipeline/tests/test_decontam.py
git commit -m "feat(session4): decontamination via n-gram fingerprints + canary"
```

---

### Task 7: Manifest + Reproducibility

**Files:**
- Create: `Session4/pipeline/manifest.py`
- Test: `Session4/pipeline/tests/test_manifest.py`

**Interfaces:**
- Produces: `content_hash(texts: Iterable[str]) -> str`; `script_hash(path: str) -> str`; `build_manifest(source, license, cleaning_script_hash, texts, token_count, language, extra=None) -> dict`; `validate_manifest(m: dict) -> dict` (`{"blocked": bool, "missing": list, "unsafe_license": bool}`).

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_manifest.py
from manifest import content_hash, build_manifest, validate_manifest


def test_content_hash_deterministic_and_order_sensitive():
    texts = ["ಒಂದು", "ಎರಡು", "ಮೂರು"]
    assert content_hash(texts) == content_hash(list(texts))
    assert content_hash(texts) != content_hash(["ಎರಡು", "ಒಂದು", "ಮೂರು"])


def test_valid_manifest_not_blocked():
    m = build_manifest("ai4bharat/sangraha::unverified/kan", "CC-BY-4.0",
                       "abc123", ["ಒಂದು", "ಎರಡು"], 42, "kan")
    v = validate_manifest(m)
    assert v["blocked"] is False and v["missing"] == []


def test_unknown_license_and_missing_field_block():
    m = build_manifest("s", "CC-BY-4.0", "h", ["ಒಂದು"], 1, "kan")
    bad_license = dict(m, license="unknown")
    assert validate_manifest(bad_license)["blocked"] is True
    missing = dict(m)
    del missing["content_hash"]
    assert validate_manifest(missing)["blocked"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# Session4/pipeline/manifest.py
import hashlib

REQUIRED = ["source", "license", "cleaning_script_hash", "content_hash",
            "token_count", "doc_count", "language"]
_UNSAFE = {"", "unknown", "unsafe", "none", "proprietary"}


def content_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def script_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_manifest(source, license, cleaning_script_hash, texts, token_count,
                   language, extra=None):
    m = {
        "source": source,
        "license": license,
        "cleaning_script_hash": cleaning_script_hash,
        "content_hash": content_hash(texts),
        "token_count": token_count,
        "doc_count": len(texts) if hasattr(texts, "__len__") else None,
        "language": language,
    }
    if extra:
        m.update(extra)
    return m


def validate_manifest(m):
    missing = [k for k in REQUIRED if m.get(k) in (None, "", 0)]
    unsafe = str(m.get("license", "")).strip().lower() in _UNSAFE
    return {"blocked": bool(missing) or unsafe,
            "missing": missing, "unsafe_license": unsafe}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_manifest.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add Session4/pipeline/manifest.py Session4/pipeline/tests/test_manifest.py
git commit -m "feat(session4): provenance manifest + deterministic content hash"
```

---

### Task 8: Pipeline Orchestration + Real Run

**Files:**
- Create: `Session4/pipeline/run_pipeline.py`
- Create: `Session4/pipeline/tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: all stage modules from Tasks 1–7.
- Produces: `run(input_parquet: str, outdir: str, use_ner: bool = True, ner_pipe=None, sample_diffs: int = 3) -> dict` — executes the seven stages in order, writes `cleaned_corpus.parquet`, `manifest.json`, `stats.json` into `outdir`, and returns the stats dict. `count_tokens(text) -> int` (o200k).

- [ ] **Step 1: Write the failing test (tiny synthetic corpus, NER stubbed)**

```python
# Session4/pipeline/tests/test_run_pipeline.py
import json
import pyarrow as pa
import pyarrow.parquet as pq
from run_pipeline import run


def _make_corpus(path):
    good = ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ತಂತ್ರಜ್ಞಾನ ನಗರವಾಗಿದೆ ಇಲ್ಲಿ ಹಲವು "
            "ಕಂಪನಿಗಳು ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತವೆ ಶಿಕ್ಷಣಕ್ಕೂ ಹೆಸರುವಾಸಿ") * 3
    rows = [
        ("d1", good),
        ("d2", good + " ಹೆಚ್ಚಿನ ವಿವರ ಬರಲಿದೆ"),          # near-dup of d1 -> dropped
        ("d3", "this document is english only and should be dropped by langid " * 4),
        ("d4", "ಸಣ್ಣ"),                                     # too short -> dropped
        ("d5", "ಸಂಪರ್ಕ ram@example.com ಮತ್ತು ಮುಂಬೈ ಮಳೆ ಸುದ್ದಿ ವರದಿ ಇಲ್ಲಿದೆ "
               "ಸಂಚಾರ ಅಸ್ತವ್ಯಸ್ತ ಜನ ಪರದಾಡಿದರು ಎಂದು ವರದಿ ಹೇಳಿದೆ" * 2),  # PII masked, kept
    ]
    pq.write_table(pa.table({"doc_id": [r[0] for r in rows],
                             "text": [r[1] for r in rows]}), path)


class StubNER:
    def __call__(self, text):
        return []          # no names in the synthetic set


def test_run_produces_valid_stats_and_reduces(tmp_path):
    inp = tmp_path / "mini.parquet"
    _make_corpus(str(inp))
    out = tmp_path / "out"
    stats = run(str(inp), str(out), use_ner=True, ner_pipe=StubNER())

    # schema
    for key in ("baseline", "stages", "pii", "final", "manifest", "meta"):
        assert key in stats
    assert len(stats["stages"]) == 6            # normalize..decontam (manifest is separate)

    # monotonic reduction: final docs < baseline docs
    assert stats["final"]["docs"] < stats["baseline"]["docs"]

    # d2 (near-dup) and d3 (english) and d4 (tiny) gone; d1 and d5 survive
    cleaned = pq.read_table(str(out / "cleaned_corpus.parquet")).to_pydict()
    assert "d1" in cleaned["doc_id"] and "d5" in cleaned["doc_id"]
    assert "d3" not in cleaned["doc_id"] and "d4" not in cleaned["doc_id"]

    # PII masked in d5
    d5 = cleaned["text"][cleaned["doc_id"].index("d5")]
    assert "[EMAIL]" in d5 and "ram@example.com" not in d5

    # files written and manifest valid
    assert (out / "manifest.json").exists()
    saved = json.loads((out / "stats.json").read_text())
    assert saved["final"]["tokens"] <= saved["baseline"]["tokens"]
    assert stats["manifest"]["license"] == "CC-BY-4.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_run_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# Session4/pipeline/run_pipeline.py
import json
import os
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

import normalize as _normalize
import langid as _langid
import quality as _quality
import dedup as _dedup
import pii as _pii
import decontam as _decontam
import manifest as _manifest

_ENC = tiktoken.get_encoding("o200k_base")
SOURCE = "ai4bharat/sangraha::unverified/kan/data-0.parquet"
LICENSE = "CC-BY-4.0"
LANG = "kan"

# Held-out probe used for the decontamination stage. If FLORES-Kannada is
# available it is loaded in main(); this fallback keeps the stage meaningful.
_EVAL_FALLBACK = [
    "ಈ ವಾಕ್ಯವು ಮೌಲ್ಯಮಾಪನ ಸೆಟ್‌ನ ಭಾಗವಾಗಿದೆ ಮತ್ತು ತರಬೇತಿಯಲ್ಲಿ ಬರಬಾರದು",
]


def count_tokens(text):
    return len(_ENC.encode(text))


def _diff_samples(pairs, n):
    out = []
    for before, after in pairs:
        if before != after:
            out.append({"before": before[:400], "after": after[:400]})
        if len(out) >= n:
            break
    return out


def run(input_parquet, outdir, use_ner=True, ner_pipe=None, sample_diffs=3,
        eval_texts=None):
    os.makedirs(outdir, exist_ok=True)
    tbl = pq.read_table(input_parquet, columns=["doc_id", "text"]).to_pydict()
    docs = list(zip(tbl["doc_id"], tbl["text"]))

    # token cache keyed by doc_id (recomputed only when text changes)
    tok = {d: count_tokens(t) for d, t in docs}

    def totals(items):
        return {"docs": len(items), "tokens": sum(tok[d] for d, _ in items)}

    baseline = totals(docs)
    stages = []

    def record(name, before_items, after_items, indic, diffs):
        b, a = totals(before_items), totals(after_items)
        stages.append({
            "name": name,
            "docs_in": b["docs"], "docs_out": a["docs"],
            "tokens_in": b["tokens"], "tokens_out": a["tokens"],
            "removed_docs": b["docs"] - a["docs"],
            "removed_tokens": b["tokens"] - a["tokens"],
            "removed_pct": round(100 * (b["tokens"] - a["tokens"]) / b["tokens"], 2)
                           if b["tokens"] else 0.0,
            "indic_concern": indic,
            "example_diffs": diffs,
        })

    # 1. NORMALIZE (mutates text)
    before = docs
    norm_pairs, out = [], []
    for d, t in docs:
        clean, _ops = _normalize.normalize_text(t)
        norm_pairs.append((t, clean))
        tok[d] = count_tokens(clean)
        out.append((d, clean))
    record("normalize", before, out,
           "Keeps Brahmic joiners (ZWNJ/ZWJ) while stripping zero-width noise.",
           _diff_samples(norm_pairs, sample_diffs))
    docs = out

    # 2. LANGUAGE-ID
    before = docs
    out = [(d, t) for d, t in docs if _langid.is_kannada(t, 0.5)]
    dropped = [(d, t) for d, t in docs if not _langid.is_kannada(t, 0.5)]
    record("langid", before, out,
           "Detects real Kannada vs code-switched/mislabelled docs.",
           _diff_samples([(t, "[DROPPED: not Kannada]") for _, t in dropped], sample_diffs))
    docs = out

    # 3. QUALITY
    before = docs
    kept, drops = [], []
    for d, t in docs:
        ok, reasons = _quality.quality_check(t, indic_aware=True)
        (kept if ok else drops).append((d, t, reasons if not ok else None))
    out = [(d, t) for d, t, _ in kept]
    record("quality", before, out,
           "Indic-aware thresholds so combining marks are not counted as symbols.",
           _diff_samples([(t, f"[DROPPED: {r}]") for _, t, r in drops], sample_diffs))
    docs = out

    # 4. DEDUP
    before = docs
    dup_keys = _dedup.find_duplicates(docs, threshold=0.8)
    out = [(d, t) for d, t in docs if d not in dup_keys]
    record("dedup", before, out,
           "Near-duplicate removal the Indic crawl never had (exact dupes were 0).",
           _diff_samples([(t, "[DROPPED: near-duplicate]")
                          for d, t in docs if d in dup_keys], sample_diffs))
    docs = out

    # 5. PII (mutates text)
    before = docs
    pii_counts = {"email": 0, "phone": 0, "url": 0, "ip": 0, "name": 0}
    pipe = ner_pipe if ner_pipe is not None else (_pii.load_ner() if use_ner else None)
    pii_pairs, out = [], []
    for d, t in docs:
        masked, c = _pii.mask_structured(t)
        for k in ("email", "phone", "url", "ip"):
            pii_counts[k] += c[k]
        if pipe is not None:
            masked, n = _pii.mask_names(masked, pipe=pipe)
            pii_counts["name"] += n
        pii_pairs.append((t, masked))
        if masked != t:
            tok[d] = count_tokens(masked)
        out.append((d, masked))
    record("pii", before, out,
           "Regex identifiers + IndicNER PER names (precision/recall tension for Indic names).",
           _diff_samples(pii_pairs, sample_diffs))
    docs = out

    # 6. DECONTAM
    before = docs
    eval_fps = _decontam.build_eval_fingerprint_set(eval_texts or _EVAL_FALLBACK, n=8)
    kept, dropped = [], []
    for d, t in docs:
        if _decontam.is_contaminated(t, eval_fps, n=8) or _decontam.has_canary(t):
            dropped.append((d, t))
        else:
            kept.append((d, t))
    out = kept
    record("decontam", before, out,
           "Fingerprint vs held-out eval + canary strings to keep scores honest.",
           _diff_samples([(t, "[DROPPED: eval overlap]") for _, t in dropped], sample_diffs))
    docs = out

    # 7. MANIFEST
    texts = [t for _, t in docs]
    final_tokens = sum(tok[d] for d, _ in docs)
    script_h = _manifest.script_hash(os.path.abspath(__file__))
    man = _manifest.build_manifest(
        SOURCE, LICENSE, script_h, texts, final_tokens, LANG,
        extra={"pipeline_stages": [s["name"] for s in stages],
               "generated_at": datetime.now(timezone.utc).isoformat()})
    validation = _manifest.validate_manifest(man)

    final = {
        "docs": len(docs),
        "tokens": final_tokens,
        "fertility": round(final_tokens / max(1, sum(len(t.split()) for t in texts)), 3),
        "total_reduction_pct": round(100 * (baseline["tokens"] - final_tokens)
                                     / baseline["tokens"], 2) if baseline["tokens"] else 0.0,
    }

    stats = {
        "baseline": {"docs": baseline["docs"], "tokens": baseline["tokens"]},
        "stages": stages,
        "pii": pii_counts,
        "final": final,
        "manifest": {**man, "validation": validation},
        "meta": {"source": SOURCE, "license": LICENSE, "language": LANG,
                 "generated_at": man["generated_at"], "script_hash": script_h},
    }

    pq.write_table(pa.table({"doc_id": [d for d, _ in docs], "text": texts}),
                   os.path.join(outdir, "cleaned_corpus.parquet"))
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({**man, "validation": validation}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    t0 = time.time()
    stats = run("Session4/data/raw/kan_slice.parquet",
                "Session4/data/cleaned", use_ner=True)
    print(json.dumps({"baseline": stats["baseline"], "final": stats["final"],
                      "pii": stats["pii"],
                      "stages": [(s["name"], s["docs_in"], s["docs_out"])
                                 for s in stats["stages"]]},
                     ensure_ascii=False, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_run_pipeline.py -v`
Expected: PASS (1 passed). If d2 is not detected as a near-dup at this tiny size, lower the dedup threshold in the near-dup assertion by editing `run(...)`'s dedup call is NOT allowed (keep 0.8 for the real run); instead make d2 share more text with d1 in `_make_corpus` so the near-dup is unambiguous.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest Session4/pipeline/tests/ -v`
Expected: PASS (all tests from Tasks 1–8)

- [ ] **Step 6: Execute the real pipeline on the 50M-token slice**

Run: `python Session4/pipeline/run_pipeline.py`
Expected: prints baseline/final/stage table; writes `Session4/data/cleaned/{cleaned_corpus.parquet,manifest.json,stats.json}`. First run downloads IndicNER (~240MB). If GPU OOM occurs, edit `mask_names` call path to truncate: in `run`, replace `masked, n = _pii.mask_names(masked, pipe=pipe)` with a truncated copy `masked_head = masked[:1500]` NER pass — but keep masking applied to the full doc by re-inserting; simplest safe fallback is to run NER on `masked[:1500]` and accept head-of-doc name coverage. Document whichever path was taken in the commit message.

- [ ] **Step 7: Commit**

```bash
git add Session4/pipeline/run_pipeline.py Session4/pipeline/tests/test_run_pipeline.py
git commit -m "feat(session4): pipeline orchestration + real 50M-token Kannada run"
```

---

### Task 9: Widget (template + strict data injection)

**Files:**
- Create: `Session4/widget/template.html`
- Create: `Session4/widget/build_widget.py`
- Create: `Session4/pipeline/tests/test_build_widget.py`

**Interfaces:**
- Consumes: `Session4/data/cleaned/stats.json` from Task 8.
- Produces: `build(stats_path: str, template_path: str, out_path: str) -> None` — injects the stats JSON into the template's `<script id="stats" type="application/json">…</script>` block and writes a self-contained `index.html`. Exits with `SystemExit` (non-zero) on malformed JSON (matches the repo's strict-load pattern from Session 2).

- [ ] **Step 1: Write the failing test**

```python
# Session4/pipeline/tests/test_build_widget.py
import json
import pytest
from build_widget import build


TEMPLATE = ('<!doctype html><html><body><h1>S4</h1>'
            '<script id="stats" type="application/json">__STATS__</script>'
            '<script>const S=JSON.parse(document.getElementById("stats").textContent);</script>'
            '</body></html>')


def test_injects_stats_and_strips_placeholder(tmp_path):
    stats = {"final": {"docs": 3, "tokens": 10}, "stages": []}
    sp = tmp_path / "stats.json"; sp.write_text(json.dumps(stats), encoding="utf-8")
    tp = tmp_path / "template.html"; tp.write_text(TEMPLATE, encoding="utf-8")
    out = tmp_path / "index.html"
    build(str(sp), str(tp), str(out))
    html = out.read_text(encoding="utf-8")
    assert "__STATS__" not in html
    assert '"tokens": 10' in html or '"tokens":10' in html
    # closing-script-safe: no raw </script> inside injected JSON
    body = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(body)["final"]["docs"] == 3


def test_malformed_stats_json_exits_nonzero(tmp_path):
    sp = tmp_path / "bad.json"; sp.write_text("{not json", encoding="utf-8")
    tp = tmp_path / "template.html"; tp.write_text(TEMPLATE, encoding="utf-8")
    with pytest.raises(SystemExit):
        build(str(sp), str(tp), str(tmp_path / "index.html"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Session4/pipeline/tests/test_build_widget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_widget'`

- [ ] **Step 3: Write the builder**

```python
# Session4/widget/build_widget.py
import json
import sys


def build(stats_path, template_path, out_path):
    try:
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        sys.exit(f"build_widget: cannot load stats json: {e}")

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    if "__STATS__" not in template:
        sys.exit("build_widget: template missing __STATS__ placeholder")

    # Re-serialize and escape any </script> so the inline JSON can't break out.
    blob = json.dumps(stats, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__STATS__", blob)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    build("Session4/data/cleaned/stats.json",
          "Session4/widget/template.html",
          "Session4/widget/index.html")
    print("wrote Session4/widget/index.html")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Session4/pipeline/tests/test_build_widget.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Author `template.html` (Session 3 aesthetic, self-contained)**

Create `Session4/widget/template.html` as a full HTML page in the Session 3 dark
jade/ochre style (reuse the CSS variables from `Session3/index.html`: `--ink`,
`--jade`, `--ochre`, fonts Archivo / Hanken Grotesk / IBM Plex Mono). It MUST
contain exactly one `<script id="stats" type="application/json">__STATS__</script>`
and vanilla JS that reads it and renders these sections, each fed by the injected
data (no hard-coded numbers):

1. **Header + dataset card** — Sangraha `unverified/kan`, 50M tokens, why chosen.
2. **The 8 strategies** — list with one-line description + the V4 defect each fixes
   (static copy is fine; the *count* 8 is asserted in Step 6).
3. **Pipeline walkthrough** — iterate `stats.stages`; per stage show name,
   `docs_in→docs_out`, `tokens_in→tokens_out`, `removed_pct`, `indic_concern`,
   and one `example_diffs` before/after.
4. **Surviving-token bar** — a horizontal bar that shrinks stage by stage using
   `tokens_out`, ending at `stats.final.tokens`.
5. **Ghost-tags demo** — a small static illustration of the four conversation
   formats collapsing to one canonical special-token format (strategy 2, demo).
6. **Concerns panel** — fertility (from `stats.final.fertility` and baseline),
   32.8% code-switch, near-dup-vs-exact-dup, keep-the-joiner nuance.
7. **Final statistics + manifest** — `stats.final` totals and a `<pre>` manifest
   viewer from `stats.manifest` with a "Download manifest.json" button (Blob).

Verify the rendered widget in a browser (or via the `run` skill) and confirm the
numbers match `stats.json`.

- [ ] **Step 6: Build the real widget and sanity-check it**

Run: `python Session4/widget/build_widget.py`
Then verify:

```bash
python - <<'PY'
import re
html = open("Session4/widget/index.html", encoding="utf-8").read()
assert "__STATS__" not in html, "placeholder not replaced"
assert html.count('id="stats"') == 1
# the widget claims 8 strategies
assert re.search(r"\b8\b\s*(strateg|cleaning)", html, re.I), "8-strategies claim missing"
print("widget OK:", len(html), "bytes")
PY
```

- [ ] **Step 7: Commit**

```bash
git add Session4/widget/template.html Session4/widget/build_widget.py \
        Session4/pipeline/tests/test_build_widget.py Session4/widget/index.html
git commit -m "feat(session4): tokenizer-style cleaning widget + strict data injection"
```

---

### Task 10: Finalize deliverable + Netlify deploy

**Files:**
- Create: `Session4/index.html` (copy of the built widget — matches the repo's `SessionN/index.html` deliverable convention)
- Create: `Session4/README.md`

- [ ] **Step 1: Publish the widget as the Session 4 deliverable**

```bash
cp Session4/widget/index.html Session4/index.html
```

- [ ] **Step 2: Write `Session4/README.md`**

Short doc: the "8 strategies" answer, the dataset (Sangraha `unverified/kan`, 50M
tokens, CC-BY-4.0), how to reproduce (`slice_and_profile.py` → `run_pipeline.py`
→ `build_widget.py`), and the final headline stats pulled from `stats.json`.

- [ ] **Step 3: Commit**

```bash
git add Session4/index.html Session4/README.md
git commit -m "feat(session4): publish cleaning widget as Session 4 deliverable"
```

- [ ] **Step 4: Deploy to Netlify (user-driven)**

The widget folder is a static site. Deploy either way:
- **Drag-and-drop:** zip/drag `Session4/widget/` (which now contains `index.html`)
  onto app.netlify.com/drop.
- **CLI:** `npx netlify-cli deploy --dir=Session4/widget --prod` (requires a
  Netlify login/token; this is the user's account, so confirm before running).

Capture the deployed URL and share it. Optionally add it to `Session4/README.md`.

---

## Notes for the executor

- Run every `pytest` command from the repo root; `Session4/pipeline/conftest.py`
  puts the stage modules on `sys.path`.
- The only network steps are the IndicNER download (Task 8) and Netlify (Task 10);
  the corpus slice is already local.
- If FLORES-Kannada is wanted for decontam, load it in `main()` and pass as
  `eval_texts=`; otherwise the built-in fallback probe + canary keeps the stage real.
- Keep dedup threshold at 0.8 and the o200k tokenizer fixed for reproducible stats.
