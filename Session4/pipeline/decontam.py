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
