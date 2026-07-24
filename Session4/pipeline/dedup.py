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
