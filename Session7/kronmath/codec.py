"""
KronMath codec — a fixed, parameter-free number channel appended to the Kronecker byte grid.

The Kronecker codec (Shravan 2026) encodes a token as (byte value) x (byte position),
a 256 x pos_dim grid, and learns nothing about it. That grid says what a token *spells*.
It says nothing about what a number *is*: "9" and "8" are one byte apart and otherwise
unrelated, so 9 + 9 = 18 has to be learned from data as an arbitrary association.

This module appends a second fixed block that carries the arithmetic. For each prime p we
store two characters of the two cyclic groups that live on Z/p:

    A-channel   [cos(2*pi*n/p), sin(2*pi*n/p)]              character of (Z/p, +)
    M-channel   [cos(2*pi*ind(n)/(p-1)), sin(...)]          character of (Z/p)^x

where ind() is the discrete logarithm to a primitive root g mod p. Both groups are cyclic,
so both operations become *the same* operation on the embedding: adding phases, i.e.
multiplying the (cos, sin) pair as a complex number.

    add:  A-channels multiply     ->  decode gives a + b
    mul:  M-channels multiply     ->  decode gives a * b
    sub:  A-channels conjugate-multiply
    div:  M-channels conjugate-multiply
    pow:  M-channel angle scales by k

n == 0 (mod p) has no discrete logarithm. It encodes as the zero vector, which is absorbing
under complex multiplication -- exactly the behaviour 0 * x = 0 requires. The degenerate
case is handled by construction rather than by a special case.

Exactness holds on Z/M for M = product of the primes; see `demo()`, which asserts it over
the full operand grid. This bound is real and is stated up front: a vector space has one
addition, so no finite-dimensional embedding is a faithful ring homomorphism for unbounded
integers. Fixing a modulus is the standard price (cf. Prime Fourier Embeddings, which pays
it for addition alone).

Prior art this builds on: FoNE (arXiv:2502.09741) and Prime Fourier Embeddings
(arXiv:2606.23044) both encode integers as prime/period-indexed (cos, sin) pairs, and both
carry *additive* characters only. The multiplicative character channel is the delta here,
and it is what makes multiplication a rotation rather than something the model must learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

# Primes whose product bounds the exactly-representable range.
# 3*5*7*11*13 = 15015, which covers a*b for two-digit operands with room to spare.
DEFAULT_PRIMES: tuple[int, ...] = (3, 5, 7, 11, 13)


def _primitive_root(p: int) -> int:
    """Smallest primitive root mod p. Brute force; p is small and this runs once."""
    if p == 2:
        return 1
    factors = set()
    m = p - 1
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors.add(d)
            m //= d
        d += 1
    if m > 1:
        factors.add(m)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root found for {p}")


def _crt(residues: Sequence[int], moduli: Sequence[int]) -> int:
    """Chinese Remainder Theorem for pairwise-coprime moduli."""
    total, prod = 0, 1
    for m in moduli:
        prod *= m
    for r, m in zip(residues, moduli):
        q = prod // m
        total += r * q * pow(q, -1, m)
    return total % prod


@dataclass(frozen=True)
class MathChannels:
    """
    Fixed number encoder. No parameters, no training, no state that changes.

    Layout of the produced vector, all blocks concatenated:
        [A_p1(2), A_p2(2), ...] [M_p1(2), M_p2(2), ...] [linear(1), log(1)]
    """

    primes: tuple[int, ...] = DEFAULT_PRIMES
    magnitude: bool = True  # append linear + log channels for ordering / comparison
    families: str = "AM"    # "A" = additive only (the PFE/FoNE baseline), "AM" = both

    @property
    def modulus(self) -> int:
        m = 1
        for p in self.primes:
            m *= p
        return m

    @property
    def dim(self) -> int:
        return 2 * len(self.primes) * len(self.families) + (2 if self.magnitude else 0)

    @property
    def _roots(self) -> tuple[int, ...]:
        # ponytail: recomputed per access, and decode() touches it per row. Fine at these
        # prime sizes (microseconds); memoize with functools.cache if a hot path needs it.
        return tuple(_primitive_root(p) for p in self.primes)

    def _ind_table(self, p: int, g: int) -> dict[int, int]:
        """value -> discrete log, for the p-1 nonzero residues."""
        table, x = {}, 1
        for e in range(p - 1):
            table[x] = e
            x = (x * g) % p
        return table

    # ---------------------------------------------------------------- encode

    def encode(self, n: int | Tensor) -> Tensor:
        """Encode integer(s) to the fixed math block. Accepts a python int or a 1-D tensor."""
        ns = torch.as_tensor(n, dtype=torch.long).reshape(-1)
        blocks_a, blocks_m = [], []

        for p, g in zip(self.primes, self._roots):
            res = (ns % p).tolist()
            ang_a = torch.tensor([2 * math.pi * r / p for r in res], dtype=torch.float32)
            blocks_a.append(torch.stack([ang_a.cos(), ang_a.sin()], dim=1))

            ind = self._ind_table(p, g)
            # residue 0 -> zero vector: absorbing under complex multiply, matching 0 * x = 0
            rows = []
            for r in res:
                if r == 0:
                    rows.append(torch.zeros(2))
                else:
                    a = 2 * math.pi * ind[r] / (p - 1)
                    rows.append(torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float32))
            blocks_m.append(torch.stack(rows))

        keep = {"A": blocks_a, "M": blocks_m}
        out = torch.cat([b for f in self.families for b in keep[f]], dim=1)
        if self.magnitude:
            f = ns.to(torch.float32)
            mag = torch.stack([f / self.modulus, torch.log1p(f.clamp_min(0)) / math.log(self.modulus)], dim=1)
            out = torch.cat([out, mag], dim=1)
        return out if out.size(0) > 1 else out.squeeze(0)

    # ------------------------------------------------------------- operators
    # Every operator below is a fixed function of its inputs. Nothing is learned,
    # and no operator consults the integers -- they act on the vectors only.

    def _idx(self, family: str, i: int) -> slice:
        if family not in self.families:
            raise ValueError(f"family {family!r} not carried by this codec ({self.families})")
        base = self.families.index(family) * 2 * len(self.primes)
        return slice(base + 2 * i, base + 2 * i + 2)

    def _slice(self, v: Tensor, family: str, i: int) -> Tensor:
        return v[..., self._idx(family, i)]

    @staticmethod
    def _cmul(u: Tensor, w: Tensor, conj: bool = False) -> Tensor:
        """Complex multiply of (cos, sin) pairs == adding the phases."""
        a, b = u[..., 0], u[..., 1]
        c, d = (w[..., 0], -w[..., 1]) if conj else (w[..., 0], w[..., 1])
        return torch.stack([a * c - b * d, a * d + b * c], dim=-1)

    def _combine(self, x: Tensor, y: Tensor, family: str, conj: bool) -> Tensor:
        out = x.clone()
        for i in range(len(self.primes)):
            out[..., self._idx(family, i)] = self._cmul(
                self._slice(x, family, i), self._slice(y, family, i), conj
            )
        return out

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Rotate every additive character. Read the result with decode(..., family='A')."""
        return self._combine(x, y, "A", conj=False)

    def sub(self, x: Tensor, y: Tensor) -> Tensor:
        return self._combine(x, y, "A", conj=True)

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        """Rotate every multiplicative character. Read with decode(..., family='M')."""
        return self._combine(x, y, "M", conj=False)

    def div(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Exact iff the divisor is a unit mod M, i.e. coprime to every carried prime.

        A divisor sharing a prime with the modulus has a zero M-channel and no inverse, and
        conjugate-multiplying by it silently returns a wrong answer (14/7 decodes to 8582).
        That is detectable from the vector alone, so it raises instead of lying.
        """
        for i in range(len(self.primes)):
            if (self._slice(y, "M", i).abs().sum(-1) < 1e-6).any():
                raise ValueError(
                    f"divisor is not invertible mod {self.primes[i]} (zero multiplicative "
                    f"channel); division is only exact for divisors coprime to {self.modulus}"
                )
        return self._combine(x, y, "M", conj=True)

    def pow(self, x: Tensor, k: int) -> Tensor:
        """a**k: scale each multiplicative phase by k."""
        out = x.clone()
        for i in range(len(self.primes)):
            blk = self._slice(x, "M", i)
            ang = torch.atan2(blk[..., 1], blk[..., 0]) * k
            new = torch.stack([ang.cos(), ang.sin()], dim=-1)
            if k > 0:
                # 0**k stays 0 for k > 0; at k == 0 every base gives 1, zero divisors included
                new[blk.abs().sum(-1) < 1e-6] = 0.0
            out[..., self._idx("M", i)] = new
        return out

    # ---------------------------------------------------------------- decode

    def decode(self, v: Tensor, family: str = "A") -> int | list[int]:
        """
        Recover the integer from one character family, by CRT over the per-prime residues.

        'A' is valid for a freshly encoded value and after add/sub.
        'M' is valid for a freshly encoded value and after mul/div/pow.
        """
        single = v.dim() == 1
        vv = v.reshape(1, -1) if single else v
        out = []
        for row in vv:
            residues = []
            for i, (p, g) in enumerate(zip(self.primes, self._roots)):
                blk = self._slice(row, family, i)
                if family == "A":
                    ang = math.atan2(float(blk[1]), float(blk[0]))
                    residues.append(round(ang * p / (2 * math.pi)) % p)
                else:
                    if float(blk.abs().sum()) < 1e-6:
                        residues.append(0)
                    else:
                        ang = math.atan2(float(blk[1]), float(blk[0]))
                        e = round(ang * (p - 1) / (2 * math.pi)) % (p - 1)
                        residues.append(pow(g, e, p))
            out.append(_crt(residues, self.primes))
        return out[0] if single else out

    def canonicalize(self, v: Tensor, family: str = "A") -> Tensor:
        """
        Re-encode from whichever family is currently valid, restoring a fully consistent
        vector. A fixed change of basis between the two character families (the discrete
        log is the isomorphism); no parameters involved.
        """
        return self.encode(torch.tensor(self.decode(v, family)))


def demo() -> None:
    """Exactness over the full operand grid. Fails loudly if the algebra is wrong."""
    ch = MathChannels()
    M = ch.modulus
    print(f"primes={ch.primes}  modulus={M}  math dims={ch.dim}")

    # round trip
    for n in [0, 1, 2, 9, 42, 100, M - 1]:
        assert ch.decode(ch.encode(n)) == n, n
        assert ch.decode(ch.encode(n), "M") == n, ("M", n)

    # the assignment's headline cases, read straight out of the vector
    e9 = ch.encode(9)
    assert ch.decode(ch.add(e9, e9)) == 18, "9 + 9"
    assert ch.decode(ch.mul(e9, e9), "M") == 81, "9 * 9"
    print("9 + 9 ->", ch.decode(ch.add(e9, e9)), "   9 * 9 ->", ch.decode(ch.mul(e9, e9), "M"))

    # exhaustive over a grid, both operations
    bad = 0
    for a in range(0, 120):
        ea = ch.encode(a)
        for b in range(0, 120):
            eb = ch.encode(b)
            bad += ch.decode(ch.add(ea, eb)) != (a + b) % M
            bad += ch.decode(ch.mul(ea, eb), "M") != (a * b) % M
            bad += ch.decode(ch.sub(ea, eb)) != (a - b) % M
    assert bad == 0, f"{bad} mismatches"
    print(f"add/sub/mul exact on {120 * 120 * 3:,} cases over Z/{M}")

    # powers
    for a in range(2, 40):
        for k in range(0, 5):
            assert ch.decode(ch.pow(ch.encode(a), k), "M") == pow(a, k, M), (a, k)
    print("pow exact")

    # 0 is absorbing without a special case
    assert ch.decode(ch.mul(ch.encode(0), ch.encode(7)), "M") == 0
    print("0 * x = 0 by construction")

    # chained expressions: canonicalize between operations and the whole expression tree
    # is evaluated in embedding space, never touching an integer until the final decode
    expr = ch.mul(ch.canonicalize(ch.add(ch.encode(9), ch.encode(9)), "A"), ch.encode(3))
    assert ch.decode(expr, "M") == 54, "(9+9)*3"
    expr2 = ch.add(ch.canonicalize(ch.mul(ch.encode(12), ch.encode(12)), "M"), ch.encode(6))
    assert ch.decode(expr2, "A") == 150, "12*12+6"
    print("chained: (9+9)*3 ->", ch.decode(expr, "M"), "  12*12+6 ->", ch.decode(expr2, "A"))

    # negatives are free: -k is the residue M-k, and sub is a conjugate rotation
    assert ch.decode(ch.sub(ch.encode(5), ch.encode(9))) == M - 4

    # division is exact exactly when the divisor is a unit mod M, and refuses otherwise
    n_div = 0
    for a in range(1, 200):
        for b in range(1, 60):
            if all(b % p for p in ch.primes):
                assert ch.decode(ch.div(ch.encode(a * b % M), ch.encode(b)), "M") == a % M
                n_div += 1
    try:
        ch.div(ch.encode(14), ch.encode(7))   # 7 is a zero divisor here
        raise SystemExit("div should refuse a non-invertible divisor")
    except ValueError:
        pass
    print(f"sub/negatives exact; division exact on {n_div:,} unit divisors, refuses non-units")

    # the block is small next to the 8192-dim byte grid
    print(f"appended cost: {ch.dim} dims on top of 256*32=8192  ({ch.dim / 8192 * 100:.2f}%)")

    # the additive-only variant is the published baseline (PFE/FoNE): addition still works,
    # multiplication has no channel to live in, which is exactly the gap this module fills
    a_only = MathChannels(families="A")
    assert a_only.decode(a_only.add(a_only.encode(9), a_only.encode(9))) == 18
    try:
        a_only.mul(a_only.encode(9), a_only.encode(9))
        raise SystemExit("additive-only codec should not support mul")
    except ValueError:
        print("additive-only variant: add works, mul has no channel  <- the delta")

    # single-prime codec, used by the controlled training comparison
    p97 = MathChannels(primes=(97,), magnitude=False)
    assert p97.decode(p97.add(p97.encode(60), p97.encode(50))) == 110 % 97
    assert p97.decode(p97.mul(p97.encode(60), p97.encode(50)), "M") == (60 * 50) % 97
    print(f"single-prime p=97 codec: {p97.dim} dims, add and mul exact")


if __name__ == "__main__":
    demo()
