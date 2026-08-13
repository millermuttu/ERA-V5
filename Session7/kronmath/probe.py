"""
Is the operation a *bilinear* function of the two operand embeddings?

The transformer experiment measures whether a model happens to find the algorithm. This
probe asks the sharper question underneath it, and answers it without any of the optimisation
luck: fit the smallest readout that could possibly express the operation, and see whether the
representation supports it.

    score[c] = sum_{i,j} W[c,i,j] * emb(a)[i] * emb(b)[j]

i.e. a linear map on the outer product emb(a) (x) emb(b). Nothing else -- no depth, no
attention, no nonlinearity. Fit on operand pairs drawn from the SEEN values only, then
evaluated on pairs containing a held-out operand.

Why this is the right question. Adding phases is complex multiplication, which is bilinear in
the two (cos, sin) pairs:

    cos(x+y) = cos x cos y - sin x sin y
    sin(x+y) = sin x cos y + cos x sin y

So whenever an operation is a phase addition in some carried channel, a bilinear readout can
express it exactly, and it extrapolates to operand values never seen because it is a
continuous function of the phase rather than a table lookup.

Expected, and the whole point:
    A-channels  carry addition        -> bilinear readout gets `+`, fails `*`
    M-channels  carry multiplication  -> bilinear readout gets `*` as well
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from codec import MathChannels


def fit_bilinear(emb: torch.Tensor, p: int, op: str, seen, held, steps=3000, lr=0.05, seed=0):
    """Fit score[c] = <W_c, emb(a) (x) emb(b)> on seen pairs; report accuracy on held pairs."""
    torch.manual_seed(seed)
    d = emb.size(1)
    W = torch.zeros(d * d, p, requires_grad=True)
    torch.nn.init.normal_(W, std=0.1)
    opt = torch.optim.Adam([W], lr=lr)

    def batch(a_vals, b_vals, n, g):
        a = a_vals[torch.randint(0, len(a_vals), (n,), generator=g)]
        b = b_vals[torch.randint(0, len(b_vals), (n,), generator=g)]
        y = (a * b) % p if op == "mul" else (a + b) % p
        feats = (emb[a].unsqueeze(2) * emb[b].unsqueeze(1)).reshape(n, d * d)
        return feats, y

    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        x, y = batch(seen, seen, 512, g)
        loss = F.cross_entropy(x @ W, y)
        opt.zero_grad(); loss.backward(); opt.step()

    ge = torch.Generator().manual_seed(999)
    with torch.no_grad():
        xtr, ytr = batch(seen, seen, 4000, ge)
        train_acc = ((xtr @ W).argmax(1) == ytr).float().mean().item()
        # test: at least one operand held out
        parts = [batch(held, seen, 2000, ge), batch(seen, held, 2000, ge),
                 batch(held, held, 2000, ge)]
        xte = torch.cat([q[0] for q in parts]); yte = torch.cat([q[1] for q in parts])
        test_acc = ((xte @ W).argmax(1) == yte).float().mean().item()
    return train_acc, test_acc


def analytic_bilinear(p: int, order: int) -> torch.Tensor:
    """
    The exact bilinear readout, written down rather than fitted.

    With x = (cos t, sin t), the outer product emb(a) (x) emb(b) has components
        f00 = cos ta cos tb   f01 = cos ta sin tb   f10 = sin ta cos tb   f11 = sin ta sin tb
    and the angle-addition identities are linear in exactly those:
        cos(ta+tb) = f00 - f11        sin(ta+tb) = f01 + f10
    so setting  W[:, c] = cos(tc) * (1,0,0,-1) + sin(tc) * (0,1,1,0)  gives
        logit_c = cos(ta + tb - tc),
    which is maximised precisely when tc is the phase of the result. `order` is the order of
    the cyclic group the phase lives in: p for the additive channel, p-1 for the discrete-log
    channel. No fitting, no data.
    """
    cos_part = torch.tensor([1.0, 0.0, 0.0, -1.0])
    sin_part = torch.tensor([0.0, 1.0, 1.0, 0.0])
    t = 2 * torch.pi * torch.arange(order) / order
    return torch.outer(cos_part, t.cos()) + torch.outer(sin_part, t.sin())   # (4, order)


def eval_analytic(p: int, fams: str, op: str, seen, held):
    """Accuracy of the constructed readout on held-out operands. Nothing is trained."""
    ch = MathChannels(primes=(p,), magnitude=False, families=fams)
    emb = ch.encode(torch.arange(p))
    family = "A" if op == "add" else "M"
    order = p if family == "A" else p - 1
    W = analytic_bilinear(p, order)

    # decode a predicted phase index back to an integer: identity for A, g**e for M
    g_root = ch._roots[0]
    lut = torch.tensor([e for e in range(order)]) if family == "A" \
        else torch.tensor([pow(g_root, e, p) for e in range(order)])

    gen = torch.Generator().manual_seed(999)
    def acc(a_vals, b_vals, n):
        a = a_vals[torch.randint(0, len(a_vals), (n,), generator=gen)]
        b = b_vals[torch.randint(0, len(b_vals), (n,), generator=gen)]
        y = (a * b) % p if op == "mul" else (a + b) % p
        xa, xb = emb[a][:, ch._idx(family, 0)], emb[b][:, ch._idx(family, 0)]
        feats = (xa.unsqueeze(2) * xb.unsqueeze(1)).reshape(n, 4)
        pred = lut[(feats @ W).argmax(1)]
        return (pred == y).float().mean().item()

    parts = [acc(held, seen, 3000), acc(seen, held, 3000), acc(held, held, 3000)]
    return sum(parts) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prime", type=int, default=97)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args()

    p = a.prime
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(p, generator=g)
    n_hold = int(p * a.holdout)
    held, seen = perm[:n_hold], perm[n_hold:]
    print(f"p={p}  seen operands={len(seen)}  held out={len(held)}  chance={1/p:.4f}\n")

    print("[1] fitted bilinear readout on the outer product\n")
    print(f"{'channels':10s} {'dims':>5s} {'op':>4s} {'train':>8s} {'test':>8s} {'test/chance':>12s}")
    for fams in ("A", "M", "AM"):
        ch = MathChannels(primes=(p,), magnitude=False, families=fams)
        emb = ch.encode(torch.arange(p))
        for op in ("add", "mul"):
            tr, te = fit_bilinear(emb, p, op, seen, held, steps=a.steps)
            print(f"{fams:10s} {ch.dim:5d} {op:>4s} {tr:8.3f} {te:8.3f} {te*p:11.1f}x")
        print()

    print("[2] the exact bilinear readout, constructed rather than fitted\n")
    print(f"{'channel':10s} {'op':>4s} {'test acc on held-out operands':>32s}")
    for fams, op in (("A", "add"), ("M", "mul")):
        print(f"{fams:10s} {op:>4s} {eval_analytic(p, fams, op, seen, held):32.4f}")
    print("\n(the residual gap from 1.0 is the pairs where an operand is 0 mod p: its "
          "multiplicative\n channel is the zero vector, so every logit ties.)")


if __name__ == "__main__":
    main()
