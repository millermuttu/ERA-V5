"""Decontamination, PII screening and near-duplicate detection.

Vendored from Session 4's cleaning pipeline (`Session4/pipeline/decontam.py`,
`pii.py`, `dedup.py`), which is submitted and frozen. Copied rather than
imported because both sessions ship a package named `pipeline` - importing
across them needs `importlib` path loading and would stop Session 6 being a
self-contained submission. Everything here is stdlib-only.

Two deliberate deviations from the Session 4 originals:

- `near_duplicates` uses exact Jaccard over shingles instead of Session 4's
  MinHash/LSH. MinHash exists to avoid the O(n^2) comparison at corpus scale;
  at 67 toy documents that is 2,211 comparisons and the approximation only
  costs accuracy. `shingles()` itself is reused verbatim.
  # ponytail: exact O(n^2) Jaccard; swap in Session4/pipeline/dedup.find_duplicates
  # (MinHash + LSH banding, needs numpy) if the corpus outgrows a few thousand docs.
- Session 4's `pii.mask_names` NER pass is dropped: it needs transformers +
  torch and a multilingual model download. Only `mask_structured` (regex
  identifiers) is carried over, so the reported PII counts cover structured
  identifiers and are honestly labelled as such.
"""
import hashlib
import re

# --- decontamination (Session4/pipeline/decontam.py) -----------------------

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
    for text in eval_texts:
        fps |= ngram_fingerprints(text, n)
    return fps


def has_canary(text, canary=CANARY):
    return canary in text


def overlap_pct(text, eval_fps, n=8):
    """Share of this document's n-grams that also appear in the eval set.

    This is the number the eval firewall thresholds on. Session 4 only needed
    a contaminated/clean boolean (`is_contaminated`); Session 6's manifest
    records the percentage, so the gate is auditable rather than binary."""
    fps = ngram_fingerprints(text, n)
    if not fps:
        return 0.0
    return round(100.0 * len(fps & eval_fps) / len(fps), 2)


# --- PII screening (Session4/pipeline/pii.py, structured identifiers only) --

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# US/international style (area code 2-9) plus Indian 10-digit mobiles.
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    r"|(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def mask_structured(text):
    """Returns (masked_text, counts). Verbatim from Session 4."""
    counts = {"email": 0, "phone": 0, "url": 0, "ip": 0}
    for pat, tag, key in [(_URL, "URL", "url"), (_EMAIL, "EMAIL", "email"),
                          (_IP, "IP", "ip"), (_PHONE, "PHONE", "phone")]:
        text, n = pat.subn(f"[{tag}]", text)
        counts[key] += n
    return text, counts


# --- near-duplicate detection (Session4/pipeline/dedup.py) ------------------

_WORD = re.compile(r"\S+")


def shingles(text, k=5):
    """Verbatim from Session 4."""
    words = _WORD.findall(text)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicates(docs, threshold=0.8):
    """docs: [(key, text)]. Returns {duplicate_key: kept_key} for later
    documents that duplicate an earlier one - first-seen wins, matching
    Session 4's drop-later-doc semantics."""
    sigs = {}
    duplicates = {}
    for key, text in docs:
        sig = shingles(text)
        for seen_key, seen_sig in sigs.items():
            if seen_key in duplicates:
                continue
            if jaccard(seen_sig, sig) >= threshold:
                duplicates[key] = seen_key
                break
        sigs[key] = sig
    return duplicates
