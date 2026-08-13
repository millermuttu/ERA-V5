"""
The number the lesson asks for: silent collisions per script, over real vocabularies.

    "take the V5 vocabulary, encode every token, and count the collisions per script...
     If the collision count says buy it, buy it. The assignment asks you to produce that
     count."                                             -- session7-lesson.md, section 8

A collision is not "these codes are close". It is *bit-identical codes*, which no amount of
training can separate, because the codec is frozen and the projection is shared. So it is
detected exactly rather than by a cosine threshold.

Two tokens collide iff their (byte, slot) count grids are equal up to a positive scalar --
z-normalisation is an affine map, so a proportional grid normalises to the same vector. The
key below is therefore the grid's sparse content divided through by its gcd, which makes the
test exact and O(L) per token instead of materialising 250k x 8192 floats.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import gcd
from functools import reduce

import vocab
from roles import SCHEMES

R_GRID = (16, 32, 48, 64)


def key(b: bytes, R: int, scheme: str):
    """Canonical, scale-free content of the code grid for one token."""
    cells = Counter((v, s) for v, s in zip(b, SCHEMES[scheme](len(b), R)) if s >= 0)
    if not cells:
        return ()
    g = reduce(gcd, cells.values())
    return tuple(sorted((c, n // g) for c, n in cells.items()))


def dropped(b: bytes, R: int, scheme: str) -> int:
    """Bytes the scheme never shows the model at all."""
    return sum(s < 0 for s in SCHEMES[scheme](len(b), R))


def survey(items, R: int, scheme: str):
    """items: list of (token, bytes, script). -> per-script stats + the colliding groups."""
    base_needed = scheme != "trunc"
    keys = []
    for _, b, _ in items:
        if base_needed and len(b) <= R:
            keys.append(key(b, R, "trunc"))          # fit-* are the identity here, by design
        else:
            keys.append(key(b, R, scheme))

    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    per = defaultdict(lambda: {"n": 0, "collided": 0, "cropped": 0, "bytes_lost": 0})
    examples = []
    for k, idx in groups.items():
        if len(idx) > 1:
            for i in idx:
                per[items[i][2]]["collided"] += 1
            if len(examples) < 40:
                examples.append([items[i][0] for i in idx[:4]])
    for tok, b, sc in items:
        d = dropped(b, R, scheme)
        per[sc]["n"] += 1
        if d:
            per[sc]["cropped"] += 1
            per[sc]["bytes_lost"] += d
    return dict(per), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", nargs="+", default=vocab.ALL)
    ap.add_argument("--schemes", nargs="+", default=["trunc", "fit-ends", "fit-wrap", "fit-rel"])
    ap.add_argument("--min-script", type=int, default=200,
                    help="only print scripts with at least this many tokens")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    out = {}
    for name in a.vocabs:
        items = [(t, b, vocab.script_of(t)) for t, b in vocab.load(name)]
        big = {s for s, c in Counter(sc for _, _, sc in items).items() if c >= a.min_script}
        print(f"\n=== {name}   {len(items):,} tokens ===")

        # 1. the baseline count, per script, across the budget grid
        print("\ncollided tokens, scheme=trunc (the released codec)")
        head = f"{'script':<12}{'tokens':>9}" + "".join(f"{'R=' + str(r):>12}" for r in R_GRID)
        print(head); print("-" * len(head))
        rows = {}
        for R in R_GRID:
            rows[R], _ = survey(items, R, "trunc")
        for sc in sorted(big, key=lambda s: -rows[32].get(s, {}).get("n", 0)):
            n = rows[32][sc]["n"]
            line = f"{sc:<12}{n:>9,}"
            for R in R_GRID:
                c = rows[R][sc]["collided"]
                line += f"{c:>7,} ({100*c/n:4.1f}%)" if c else f"{'0':>12}"
            print(line)
        tot = {R: sum(v["collided"] for v in rows[R].values()) for R in R_GRID}
        print(f"{'TOTAL':<12}{len(items):>9,}" +
              "".join(f"{tot[R]:>7,} ({100*tot[R]/len(items):4.2f}%)" for R in R_GRID))

        # 2. schemes against each other at a fixed budget
        print(f"\nschemes at fixed R (same D = 256*R, same projection size)")
        head = f"{'scheme':<12}" + "".join(f"{'R=' + str(r):>16}" for r in R_GRID)
        print(head); print("-" * len(head))
        table = {}
        for s in a.schemes:
            line, table[s] = f"{s:<12}", {}
            for R in R_GRID:
                per, ex = survey(items, R, s)
                c = sum(v["collided"] for v in per.values())
                lost = sum(v["bytes_lost"] for v in per.values())
                table[s][R] = {"collided": c, "bytes_lost": lost,
                               "per_script": {k: v for k, v in per.items() if k in big}}
                line += f"{c:>9,} /{lost:>6,}B"
            print(line)
        print("                 (collided tokens / bytes never shown to the model)")
        out[name] = {"n": len(items), "table": table}

        worst = max(big, key=lambda s: rows[32].get(s, {}).get("collided", 0))
        if rows[32][worst]["collided"]:
            _, ex = survey(items, 32, "trunc")
            print(f"\nworst script at R=32: {worst}  "
                  f"({rows[32][worst]['collided']:,} collided)")
            for g in ex[:5]:
                print("   ", "  ==  ".join(repr(t) for t in g))

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
