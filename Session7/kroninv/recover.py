"""
How small can d_model get before a token stops coming back?

`W_proj` maps D=8192 to d_model and is not injective, so it has no inverse. That does not
settle the question, because kappa is sparse: at most `pos_dim` non-zeros out of 8192, one per
byte position. Recovering it is compressed sensing, and the requirement scales with the
SPARSITY -- the token's byte length -- not with D and not with the vocabulary.

So the sweep below is over (d_model, token length), and the shape of the answer is the claim.
Everything is exact arithmetic on real vocabularies; nothing is trained.

Two accuracies are reported throughout, and the gap between them is the honest part:

    length known   are the BYTES recoverable? the compressed-sensing question
    blind          plus finding the token's length from the code, which is what a deployed
                   head-free decoder would actually have to do
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from invert import POS_DIM, D, backward, forward, projection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronbudget"))
import vocab                                                        # noqa: E402
from roles import CHAR_DIM, encode                                  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_GRID = (64, 96, 128, 192, 256, 384, 512, 1024, 2048)
L_BANDS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 32))


def recover_batch(items, W, noise=0.0, seed=0):
    """-> (bytes_ok_known_length, bytes_ok_blind) as boolean arrays over the batch."""
    K = torch.stack([encode(b, POS_DIM, "trunc") for _, b in items]).to(DEV)
    H = forward(K, W)
    if noise:
        g = torch.Generator(device="cpu").manual_seed(seed)
        H = H + noise * H.norm(dim=1, keepdim=True) / H.shape[1] ** 0.5 * \
            torch.randn(H.shape, generator=g).to(DEV)
    grid = backward(H, W).reshape(len(items), CHAR_DIM, POS_DIM)
    best = grid.argmax(dim=1).cpu()                                 # [B, POS_DIM]
    peak = grid.max(dim=1).values                                   # [B, POS_DIM]
    lens = (peak[:, :-1] - peak[:, 1:]).argmax(dim=1).cpu() + 1     # inferred length

    known, blind = [], []
    for i, (_, b) in enumerate(items):
        L = min(len(b), POS_DIM)
        got = bytes(best[i, :L].tolist())
        known.append(got == b[:L])
        blind.append(int(lens[i]) == L and got == b[:L])
    return np.array(known), np.array(blind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="session2-bpe")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--kind", default="orth", choices=["orth", "random"])
    a = ap.parse_args()

    items = vocab.load(a.vocab)
    rng = np.random.default_rng(0)
    items = [items[i] for i in rng.choice(len(items), min(a.sample, len(items)),
                                          replace=False)]
    items = [x for x in items if x[1]]
    print(f"{a.vocab}: {len(items):,} tokens, D={D}, projection={a.kind}, device={DEV}\n")

    print("exact token recovery through W_proj, by token byte length (sparsity)")
    print("length known / blind")
    bands = [(lo, hi, [x for x in items if lo <= len(x[1]) <= hi]) for lo, hi in L_BANDS]
    head = f"{'d_model':<9}" + "".join(f"{f'L {lo}-{hi}':>14}" for lo, hi, _ in bands)
    print(head); print("-" * len(head))
    for d in D_GRID:
        W = projection(d, a.kind).to(DEV)
        row = f"{d:<9}"
        for lo, hi, pool in bands:
            if not pool:
                row += f"{'n/a':>14}"; continue
            k, bl = recover_batch(pool, W)
            row += f"{100*k.mean():>7.1f}/{100*bl.mean():<6.1f}"
        print(row)
    print(f"  (all tokens fit pos_dim={POS_DIM}; the released d_model would be far right)")

    print("\nrobustness: recovery vs relative noise on the hidden vector "
          "(all tokens, length known)")
    noises = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
    head = f"{'d_model':<9}" + "".join(f"{str(int(100*n)) + '%':>9}" for n in noises)
    print(head); print("-" * len(head))
    for d in (256, 512, 1024, 2048):
        W = projection(d, a.kind).to(DEV)
        print(f"{d:<9}" + "".join(f"{100*recover_batch(items, W, n)[0].mean():>9.1f}"
                                  for n in noises))

    print("\nwhat the output side costs, d_model=512")
    print(f"{'vocabulary':<14}{'tied head d*V':>16}{'head-free d*D':>16}{'ratio':>9}")
    print("-" * 55)
    for name, V in (("Session 2", 10_000), ("GPT-2", 50_257), ("V5", 131_072),
                    ("XLM-R", 250_002), ("1M", 1_000_000)):
        tied, free = 512 * V, 512 * D
        print(f"{name:<14}{tied/1e6:>14.1f}M{free/1e6:>14.1f}M{tied/free:>8.1f}x")
    print("  head-free does not depend on V at all -- that is the whole claim")


if __name__ == "__main__":
    main()
