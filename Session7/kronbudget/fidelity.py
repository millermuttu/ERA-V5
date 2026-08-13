"""
What each scheme costs, on the tokens that were never broken.

Removing collisions is easy if you are allowed to wreck everything else. Two properties the
released codec has, that a replacement has to keep:

    invertibility    reshape to (256, R), argmax per slot, get the bytes back
    locality         "train", "training", "trainer" start out near each other

Both are measured here. `fit-*` schemes are the identity map for tokens that fit, so any
damage they do is confined to the overflow tail -- this file is what turns that claim from an
argument into a number.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

import vocab
from roles import SCHEMES, decode, encode

# the pair the lesson uses to demonstrate the failure
LESSON_PAIR = ("अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता")


def edit_distance(a: bytes, b: bytes) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def roundtrip(items, R, scheme):
    """Exact-decode rate, split by whether the token fits the budget."""
    fit = [0, 0]
    over = [0, 0]
    for _, b in items:
        bucket = fit if len(b) <= R else over
        bucket[1] += 1
        bucket[0] += decode(encode(b, R, scheme), R) == b
    return fit, over


def locality(pick, R, scheme):
    """Spearman between codec cosine and negative byte edit distance, on a fixed sample.

    Split by population, because that is where the schemes can differ at all: on tokens that
    fit the budget every fit-* scheme IS the baseline, so the only interesting number is the
    one on the overflow tail."""
    n = len(pick)
    codes = torch.stack([encode(b, R, scheme) for _, b in pick])
    codes = codes / codes.norm(dim=1, keepdim=True)
    cos = (codes @ codes.T).numpy()
    ed = np.array([[-edit_distance(a, b) for _, b in pick] for _, a in pick], dtype=np.float32)
    iu = np.triu_indices(n, k=1)
    return spearmanr(cos[iu], ed[iu]).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="session2-bpe")
    ap.add_argument("--schemes", nargs="+",
                    default=["trunc", "fit-ends", "fit-wrap", "fit-rel", "rel-always"])
    ap.add_argument("--R", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--n", type=int, default=140)
    a = ap.parse_args()

    items = vocab.load(a.vocab)
    print(f"{a.vocab}: {len(items):,} tokens\n")

    print("exact round-trip decode      fits budget        overflows")
    print("-" * 62)
    for R in a.R:
        for s in a.schemes:
            fit, over = roundtrip(items, R, s)
            o = f"{100*over[0]/over[1]:6.1f}% of {over[1]:>5,}" if over[1] else "        n/a"
            print(f"R={R:<3} {s:<12}  {100*fit[0]/fit[1]:6.1f}% of {fit[1]:>5,}   {o}")
        print()

    print("locality: Spearman(codec cosine, -byte edit distance)")
    rng = np.random.default_rng(0)
    head = f"{'scheme':<12}" + "".join(f"{f'R={r} {p}':>18}" for r in a.R
                                       for p in ("fits", "overflow"))
    print(head); print("-" * len(head))
    samples = {}
    for R in a.R:
        for name, pop in (("fits", [x for x in items if len(x[1]) <= R]),
                          ("overflow", [x for x in items if len(x[1]) > R])):
            k = min(a.n, len(pop))
            samples[(R, name)] = [pop[i] for i in rng.choice(len(pop), k, replace=False)]
    for s in a.schemes:
        row = f"{s:<12}"
        for R in a.R:
            for name in ("fits", "overflow"):
                row += f"{locality(samples[(R, name)], R, s):>18.3f}"
        print(row)
    for R in a.R:                                  # the identity claim, asserted not asserted-at
        base = locality(samples[(R, "fits")], R, "trunc")
        for s in a.schemes:
            if s.startswith("fit-"):
                assert abs(locality(samples[(R, "fits")], R, s) - base) < 1e-9
    print("  fit-* are bit-identical to trunc on tokens that fit, so those columns must match")

    print(f"\nthe lesson's pair, cosine between the two codes")
    ba, bb = (w.encode() for w in LESSON_PAIR)
    print(f"  {LESSON_PAIR[0]}  {len(ba)}B   vs   {LESSON_PAIR[1]}  {len(bb)}B")
    head = f"{'scheme':<12}" + "".join(f"{'R=' + str(r):>10}" for r in a.R)
    print(head); print("-" * len(head))
    for s in a.schemes:
        row = f"{s:<12}"
        for R in a.R:
            ca, cb = encode(ba, R, s), encode(bb, R, s)
            cos = float(ca @ cb / (ca.norm() * cb.norm()))
            row += f"{cos:>10.4f}" + ("*" if torch.equal(ca, cb) else " ")
        print(row)
    print("  * = bit-identical, i.e. the model can never tell them apart")


if __name__ == "__main__":
    main()
