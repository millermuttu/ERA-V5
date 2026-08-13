"""
How much can one vector hold before the waves drown each other?

A superposition code trades the byte codec's exact sparse recovery for approximate dense
recall. That trade is the whole risk of the Fourier answer, so it is measured rather than
argued: byte recovery as a function of N (the number of complex dims) and L (token length),
then the same thing at reduced precision, where a dense sum is far more exposed than a sparse
scatter.

The byte codec's numbers to beat, already measured in ../kronbudget: exact recovery for every
token up to pos_dim bytes, and 0% past it -- because the tail was never stored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fhrr import CHAR_DIM, decode, encode, psi, tables

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronbudget"))
import vocab                                                        # noqa: E402
from roles import decode as byte_decode, encode as byte_encode      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_GRID = (64, 128, 256, 512, 1024, 2048, 4096)
L_GRID = (8, 32, 64, 128, 256, 512, 1024)


def recover_batch(byte_rows: torch.Tensor, phi, theta, dtype=None):
    """byte_rows [B, L] -> recovered [B, L]. Encode, then unbind every position.

    `dtype` casts the CODE only, not the tables: the code is the activation that would travel
    through a model in reduced precision, while phi and theta are constants."""
    B, L = byte_rows.shape
    pos = psi(theta, torch.arange(L, dtype=theta.dtype, device=theta.device))   # [L, N]
    z = (phi[byte_rows] * pos).sum(1) / L ** 0.5                                # [B, N]
    if dtype is not None:
        z = torch.complex(z.real.to(dtype).float(), z.imag.to(dtype).float())
    y = z[:, None, :] * pos.conj()[None, :, :]                                  # [B, L, N]
    scores = (y.reshape(B * L, -1) @ phi.conj().T).real                         # [B*L, 256]
    return scores.argmax(-1).reshape(B, L)


def rate(N, L, trials, seed=0, dtype=None, chunk=16):
    """Per-byte recovery rate on random byte strings."""
    phi, theta = tables(N, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ok = tot = 0
    for i in range(0, trials, chunk):
        b = torch.randint(0, CHAR_DIM, (min(chunk, trials - i), L), generator=g).to(DEV)
        got = recover_batch(b, phi, theta, dtype)
        ok += int((got == b).sum()); tot += b.numel()
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--vocab", default="session2-bpe")
    a = ap.parse_args()
    print(f"device={DEV}  phasors and positions fixed by seed, nothing trained\n")

    print(f"per-byte recovery rate, random tokens, {a.trials} trials")
    head = f"{'N (D=2N)':<12}" + "".join(f"{'L=' + str(l):>9}" for l in L_GRID)
    print(head); print("-" * len(head))
    for N in N_GRID:
        print(f"{N:<5}({2*N:>5}) " + "".join(f"{rate(N, l, a.trials):>9.3f}" for l in L_GRID))
    print("  the byte codec is 1.000 up to pos_dim and 0.000 past it, at D = 256*pos_dim")

    print(f"\nwhole-token round-trip on real tokens ({a.vocab}), N=4096 (D=8192 = released D)")
    items = vocab.load(a.vocab)
    phi, theta = tables(4096, device=DEV)
    rng = np.random.default_rng(0)
    for lo, hi, label in ((1, 32, "L <= 32 (fits the byte codec)"),
                          (33, 10 ** 6, "L >  32 (byte codec crops)")):
        pool = [b for _, b in items if lo <= len(b) <= hi]
        if not pool:
            continue
        pick = [pool[i] for i in rng.choice(len(pool), min(300, len(pool)), replace=False)]
        f_ok = sum(decode(encode(b, phi, theta), len(b), phi, theta) == b for b in pick)
        b_ok = sum(byte_decode(byte_encode(b, 32, "trunc"), 32) == b for b in pick)
        print(f"  {label:<32} n={len(pick):>4}   FHRR {100*f_ok/len(pick):6.1f}%"
              f"   byte codec {100*b_ok/len(pick):6.1f}%")

    print("\nprecision: per-byte recovery with the CODE cast, tables kept fp32 (N=256,")
    print("chosen because N=4096 is saturated -- a comfortable code hides precision loss)")
    head = f"{'dtype':<10}" + "".join(f"{'L=' + str(l):>9}" for l in L_GRID)
    print(head); print("-" * len(head))
    for name, dt in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        print(f"{name:<10}" + "".join(f"{rate(256, l, a.trials, dtype=dt):>9.3f}"
                                      for l in L_GRID))


if __name__ == "__main__":
    main()
