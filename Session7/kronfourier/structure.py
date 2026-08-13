"""
Head to head against the released byte codec, at matched D, on real vocabularies.

Three questions, in order of how much they matter:

    identification   at equal D and under noise, which code lets you tell tokens apart?
                     The byte codec's ceiling is fixed by its collisions (a collided token is
                     unidentifiable at any SNR); FHRR's is set by superposition noise instead.
    locality         does "similar spellings start out similar" survive? Split by whether the
                     token exceeds 32 bytes, since that is where the byte codec stops looking.
    families         the suffix test -- the limitation the paper calls structural.

Nothing is trained. Both codecs are fixed by construction.
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fhrr import aligned_cosine, as_real, cosine, encode, tables

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronbudget"))
import vocab                                                    # noqa: E402
from fidelity import edit_distance                              # noqa: E402
from roles import encode as byte_encode                         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
POS_DIM = 32                                    # the released budget -> D = 8192
N_COMPLEX = 4096                                # -> D = 8192, matched exactly
NOISE = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)

PREFIX_FAMILY = [("compute", "commute"), ("train", "training"), ("train", "trainer"),
                 ("nation", "national")]
SUFFIX_FAMILY = [("nation", "creation"), ("separate", "operate"), ("training", "running"),
                 ("nation", "information")]


ARMS = ("byte", "byte-raw", "fhrr")


def codes(items, kind, phi=None, theta=None):
    """-> [n, 8192] real, unit-normalised rows.

    `byte-raw` is the released codec with its z-normalisation switched off. It is here because
    z-norm subtracts a per-token mean, which leaves every code sharing a large length-dependent
    component -- and that component, not the outer product, turns out to be carrying most of
    the byte codec's short-token locality."""
    if kind == "fhrr":
        rows = [as_real(encode(b, phi, theta)) for _, b in items]
    else:
        rows = [byte_encode(b, POS_DIM, "trunc", normalize=(kind == "byte")) for _, b in items]
    M = torch.stack(rows).float()
    return M / (M.norm(dim=1, keepdim=True) + 1e-12)


def identify(M, noise, seed=0, chunk=512):
    """Top-1 retrieval of each row against the whole set, after relative Gaussian noise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = M + noise * torch.randn(M.shape, generator=g) if noise else M.clone()
    Q = (Q / (Q.norm(dim=1, keepdim=True) + 1e-12)).to(DEV)
    Md = M.to(DEV)
    hit = 0
    for i in range(0, len(M), chunk):
        nn = (Q[i:i + chunk] @ Md.T).argmax(1).cpu()
        hit += int((nn == torch.arange(i, min(i + chunk, len(M)))).sum())
    return hit / len(M)


def locality(pick, kind, phi=None, theta=None):
    M = codes(pick, kind, phi, theta)
    cos = (M @ M.T).numpy()
    ed = np.array([[-edit_distance(a, b) for _, b in pick] for _, a in pick], dtype=np.float32)
    iu = np.triu_indices(len(pick), k=1)
    return spearmanr(cos[iu], ed[iu]).statistic


def families(phi, theta):
    print("\nword families: cosine between the two codes")
    print(f"{'pair':<28}{'byte codec':>12}{'FHRR':>10}{'FHRR aligned':>16}")
    print("-" * 66)
    for label, fam in (("shared prefix", PREFIX_FAMILY), ("shared suffix", SUFFIX_FAMILY)):
        print(f"-- {label}")
        for u, v in fam:
            bu, bv = u.encode(), v.encode()
            ku, kv = byte_encode(bu, POS_DIM, "trunc"), byte_encode(bv, POS_DIM, "trunc")
            byte_cos = float(ku @ kv / (ku.norm() * kv.norm()))
            zu, zv = encode(bu, phi, theta), encode(bv, phi, theta)
            al, k = aligned_cosine(zu, zv, theta)
            # the recovered shift must be the real offset between the shared substrings
            m = SequenceMatcher(None, u, v).find_longest_match(0, len(u), 0, len(v))
            assert k == m.a - m.b, (u, v, k, m.a - m.b)
            print(f"  {u + '/' + v:<26}{byte_cos:>12.4f}{cosine(zu, zv):>10.4f}"
                  f"{al:>12.4f} (k={k:+d})")
    print("  every k above equals the true offset between the shared substrings (asserted)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="session2-bpe")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--n-local", type=int, default=140)
    a = ap.parse_args()

    phi, theta = tables(N_COMPLEX)
    items = vocab.load(a.vocab)
    rng = np.random.default_rng(0)
    if len(items) > a.sample:
        items = [items[i] for i in rng.choice(len(items), a.sample, replace=False)]
    print(f"{a.vocab}: {len(items):,} tokens   byte codec D={256*POS_DIM}   "
          f"FHRR D={2*N_COMPLEX}\n")

    print("identification: top-1 retrieval against the whole set, relative Gaussian noise")
    head = f"{'codec':<12}" + "".join(f"{str(int(100*n)) + '%':>10}" for n in NOISE)
    print(head); print("-" * len(head))
    for kind in ARMS:
        M = codes(items, kind, phi, theta)
        print(f"{kind:<12}" + "".join(f"{identify(M, n):>10.4f}" for n in NOISE))

    print("\nlocality: Spearman(cosine, -byte edit distance)")
    head = f"{'codec':<12}{'L <= 32':>12}{'L > 32':>12}"
    print(head); print("-" * len(head))
    pops = {}
    for name, pop in (("L <= 32", [x for x in items if len(x[1]) <= POS_DIM]),
                      ("L > 32", [x for x in items if len(x[1]) > POS_DIM])):
        k = min(a.n_local, len(pop))
        pops[name] = [pop[i] for i in rng.choice(len(pop), k, replace=False)] if k else []
    for kind in ARMS:
        row = f"{kind:<12}"
        for name in ("L <= 32", "L > 32"):
            row += f"{locality(pops[name], kind, phi, theta):>12.3f}" if pops[name] \
                else f"{'n/a':>12}"
        print(row)

    families(phi, theta)


if __name__ == "__main__":
    main()
