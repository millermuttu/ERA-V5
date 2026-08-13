"""
When does the multiplicative channel actually matter?

The additive channels alone can support multiplication *if* the model can memorise the
per-channel multiplication table. That table has p^2 entries for prime p, and the training
data covers it only when the held-out operands still expose their residues.

Two regimes fall out:

  CRT, small primes   an operand held out at the integer level (say 613 of 0..1000) still
                      has residues 613 mod 7, mod 11, mod 13 that appear in hundreds of
                      other training operands. Each channel's table (49, 121, 169 entries)
                      is fully covered, so the additive baseline learns multiplication as
                      three small lookups and the M-channel is redundant.

  single large prime  held-out operand == held-out residue. The table row for that residue
                      is never seen, and p^2 is far too large to interpolate. The additive
                      baseline has nothing to fall back on; the M-channel turns the same
                      operation into one rotation.

This sweep varies p and reports test multiplication accuracy for both, so the crossover is
measured rather than asserted. Accuracy is normalised by chance (1/p) since p varies.
"""

from __future__ import annotations

import argparse

from train import Task, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[7, 13, 31, 61, 97])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=["math+A", "math+AM"])
    a = ap.parse_args()

    print(f"{'p':>4s} {'table':>7s} {'arm':8s} {'test +':>8s} {'test *':>8s} {'* / chance':>11s}")
    for p in a.primes:
        task = Task((p,))
        for arm in a.arms:
            rs = [run(task, arm, s, steps=a.steps, quiet=True) for s in range(a.seeds)]
            add = sum(r["add"] for r in rs) / len(rs)
            mul = sum(r["mul"] for r in rs) / len(rs)
            print(f"{p:4d} {p*p:7d} {arm:8s} {add:8.3f} {mul:8.3f} {mul*p:11.1f}x", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
