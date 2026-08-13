"""
The Kronecker codec with a pluggable ROLE assignment.

The released codec binds byte value (filler) to byte position (role) and keeps the first
`R` bytes:

    L = min(len(byte_seq), pos_dim)        # everything past R is dropped, silently

Only the second factor is at issue here. The filler stays the byte value in every scheme
below, `D = 256 * R` stays identical, so the projection is the same size for all of them and
the comparison is like-for-like. What changes is the map

    role: (byte index i, token length L) -> slot in [0, R)

A scheme may send two bytes to the same slot; the cells superpose (scatter-add), exactly as
two bytes at the same position would. That is the whole mechanism.

Design rule everything here obeys: **tokens that already fit must not move.** The paper's own
number is that R=32 covers >=99.82% of tokens, so any scheme that reshuffles the roles of
short tokens is trading a certain regression on 99.82% for a fix on 0.18%. The `fit_*` schemes
are the identity map whenever `L <= R` and differ only on overflow.
"""

from __future__ import annotations

import torch

CHAR_DIM = 256


# ---------------------------------------------------------------- role schemes
# each takes (L, R) and returns a LongTensor of L slots, one per byte index.

def role_trunc(L: int, R: int) -> list[int]:
    """The released codec: slot = position, drop the tail. The baseline."""
    return list(range(min(L, R))) + [-1] * max(0, L - R)


def role_fit_wrap(L: int, R: int) -> list[int]:
    """Fits -> unchanged. Overflows -> wrap around, superposing byte i and byte i+R."""
    return [i % R for i in range(L)]


def role_fit_ends(L: int, R: int) -> list[int]:
    """Fits -> unchanged. Overflows -> first half of the slots from the head, second half
    from the tail, and the middle is dropped. Aimed straight at the Indic failure, where the
    words that collide share a long prefix and differ in the suffix."""
    if L <= R:
        return list(range(L))
    h = R // 2
    slot = [-1] * L                                # -1 marks a dropped byte
    slot[:h] = range(h)
    slot[L - (R - h):] = range(h, R)
    return slot


def role_fit_relative(L: int, R: int) -> list[int]:
    """Fits -> unchanged. Overflows -> squeeze proportionally, so every byte lands somewhere
    and the whole token is represented at reduced resolution."""
    if L <= R:
        return list(range(L))
    return [(i * R) // L for i in range(L)]


def role_relative_always(L: int, R: int) -> list[int]:
    """Squeeze proportionally for *every* token. Included to show why the fit-first rule
    exists: this one breaks the prefix similarity that makes the codec work at all."""
    return [(i * R) // L for i in range(L)] if L else []


SCHEMES = {
    "trunc": role_trunc,
    "fit-wrap": role_fit_wrap,
    "fit-ends": role_fit_ends,
    "fit-rel": role_fit_relative,
    "rel-always": role_relative_always,
}


# ---------------------------------------------------------------- codec

def encode(byte_seq: bytes, R: int = 32, scheme: str = "trunc",
           normalize: bool = True) -> torch.Tensor:
    """kappa(b) = (1/sqrt(n)) * sum_i  c[byte_i] (x) p[role(i)], then z-normalised.

    Returns a flat [256*R] vector. Cell index is byte_value * R + slot, the same layout the
    released codec uses, so `decode` below is the released argmax."""
    grid = torch.zeros(CHAR_DIM, R)
    if byte_seq:
        pairs = [(v, s) for v, s in zip(byte_seq, SCHEMES[scheme](len(byte_seq), R)) if s >= 0]
        n = max(len(pairs), 1)
        vals = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        slots = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        grid.index_put_((vals, slots), torch.full((len(pairs),), 1 / n ** 0.5),
                        accumulate=True)
    k = grid.reshape(-1)
    if normalize:
        k = (k - k.mean()) / (k.std() + 1e-8)
    return k


def decode(kappa: torch.Tensor, R: int = 32) -> bytes:
    """Released inverse: reshape to (256, R), argmax down the byte axis per occupied slot.

    Exact for `trunc` (one nonzero per column, up to the crop). For schemes that superpose,
    a shared slot keeps only the argmax, so this is lossy by construction -- which is the
    trade each scheme is measured on."""
    grid = kappa.reshape(CHAR_DIM, R)
    vals, idx = grid.max(dim=0)
    active = vals > grid.mean() + 1e-6
    out, seen_gap = [], False
    for s in range(R):
        if active[s]:
            if seen_gap:
                break                              # trailing slots after a gap are not bytes
            out.append(int(idx[s]))
        else:
            seen_gap = True
    return bytes(out)


def _demo():
    R = 8
    # 1. schemes agree with the released codec on everything that fits
    for w in [b"a", b"at", b"train", b"abcdefgh"]:
        base = encode(w, R, "trunc")
        for s in ("fit-wrap", "fit-ends", "fit-rel"):
            assert torch.equal(base, encode(w, R, s)), (w, s)
        assert decode(base, R) == w, (w, decode(base, R))

    # 2. the failure the lesson names: same 32-byte prefix, different tail
    a, b = b"abcdefgh" + b"XX", b"abcdefgh" + b"YY"
    assert torch.equal(encode(a, R, "trunc"), encode(b, R, "trunc")), "baseline must collide"
    for s in ("fit-wrap", "fit-ends", "fit-rel"):
        assert not torch.equal(encode(a, R, s), encode(b, R, s)), f"{s} still collides"

    # 3. rel-always breaks short tokens, which is why fit-first exists
    assert not torch.equal(encode(b"train", R, "rel-always"),
                           encode(b"train", R, "trunc"))
    print("roles: ok  (fit-* match the baseline where it fits, separate where it crops)")


if __name__ == "__main__":
    _demo()
