"""
Controlled comparison: does appending the math channels let a transformer do arithmetic on
operands it has never seen?

Task        a OP b (mod M),  OP in {+, *},  M = product of the carried primes
Split       a set H of operand VALUES never appears as an operand during training.
            Train samples a,b not in H. Test uses pairs with at least one operand in H.

Why modular. An earlier integer-valued version was discarded after measuring it: with an
untied output head only 45.6% of multiplication test targets ever appeared as a training
target, so `mul` accuracy was capped by output-head coverage rather than by the input
representation. Mod M every target is dense and fully covered, so the input representation
is the only thing under test.

Why CRT rather than one large prime. Covering 1001 values with primes 7x11x13 gives a
minimum separation of 0.705 between distinct numbers; covering only 97 values with a single
p=97 gives 0.092. The CRT codec represents 10x more integers, in 12 dimensions instead of 4,
while being 7.7x better conditioned -- each channel only has to resolve at most 13 phases.

Why the split is the right one. Session 7 section 2: an embedding row only moves when its
token is gathered, so a dense row for a held-out operand sits at its initialisation forever.
A byte codec at least gives that token a spelling. Only the math channels give it a value.

Arms (identical transformer body, identical untied output head):
    dense       nn.Embedding(V, d)                     reference point
    kron        byte codec 256 x 8                     spelling, no arithmetic
    kron+A      byte + additive characters             the published baseline (PFE / FoNE)
    kron+AM     byte + additive + multiplicative       this work
    math+A      additive characters alone              diagnostic
    math+AM     additive + multiplicative alone        diagnostic

kron / kron+A / kron+AM differ by 12 input dimensions at most -- ~1.5 K projection
parameters out of 262 K. Any gap between them is not a capacity gap. The expectation is
that kron+A learns addition and cannot learn multiplication, because additive characters
carry no multiplicative structure, while kron+AM learns both.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import MathChannels

DEV = "cuda" if torch.cuda.is_available() else "cpu"
POS_DIM = 8


@dataclass
class Task:
    primes: tuple[int, ...]

    @property
    def mod(self) -> int:
        m = 1
        for p in self.primes:
            m *= p
        return m

    @property
    def tok_add(self) -> int:
        return self.mod

    @property
    def tok_mul(self) -> int:
        return self.mod + 1

    @property
    def vocab(self) -> int:
        return self.mod + 2

    def text(self, i: int) -> str:
        return {self.tok_add: "+", self.tok_mul: "*"}.get(i, str(i))

    def byte_table(self) -> torch.Tensor:
        """The Kronecker byte grid for every token, as the released codec builds it."""
        tbl = torch.zeros(self.vocab, 256 * POS_DIM)
        for i in range(self.vocab):
            b = self.text(i).encode()[:POS_DIM]
            for p, val in enumerate(b):
                tbl[i, val * POS_DIM + p] = 1.0 / math.sqrt(len(b))
            row = tbl[i]
            tbl[i] = (row - row.mean()) / (row.std() + 1e-6)
        return tbl

    def math_table(self, families: str) -> torch.Tensor:
        """Math channels for numeric tokens; zeros for the operator tokens."""
        ch = MathChannels(primes=self.primes, magnitude=False, families=families)
        tbl = torch.zeros(self.vocab, ch.dim)
        tbl[:self.mod] = ch.encode(torch.arange(self.mod))
        return tbl


# arm -> (math families or None, include the byte grid?)
ARMS = {
    "dense":   (None, False),
    "kron":    (None, True),
    "kron+A":  ("A", True),
    "kron+AM": ("AM", True),
    "math+A":  ("A", False),
    "math+AM": ("AM", False),
}


class DenseEmb(nn.Module):
    def __init__(self, vocab: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)

    def forward(self, ids):
        return self.emb(ids)


class CodecEmb(nn.Module):
    """Fixed codec table (buffer, never trained) + one learned projection."""

    def __init__(self, table: torch.Tensor, d_model: int):
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.projection = nn.Linear(table.size(1), d_model, bias=False)

    def forward(self, ids):
        return self.projection(self.table[ids])


class Tiny(nn.Module):
    def __init__(self, emb: nn.Module, vocab: int, d_model=128, layers=4, heads=4):
        super().__init__()
        self.emb = emb
        self.pos = nn.Parameter(torch.zeros(1, 3, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)   # untied

    def forward(self, ids):
        x = self.emb(ids) + self.pos
        return self.head(self.norm(self.body(x))[:, -1])


def _unit(block: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row of a block so blocks contribute equally regardless of width."""
    return block / block.norm(dim=1, keepdim=True).clamp_min(1e-6)


def build_emb(task: Task, arm: str, d_model: int) -> nn.Module:
    if arm == "dense":
        return DenseEmb(task.vocab, d_model)
    fams, with_bytes = ARMS[arm]
    # Without per-block normalization the 2048-dim byte grid carries ~30x the norm of a
    # 12-dim math block and the projection never learns to look at the math channels.
    # A fixed codec choice, applied identically to every arm.
    blocks = ([_unit(task.byte_table())] if with_bytes else []) \
        + ([_unit(task.math_table(fams))] if fams else [])
    return CodecEmb(torch.cat(blocks, dim=1), d_model)


def split_operands(task: Task, holdout_frac: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(task.mod, generator=g)
    n_hold = int(task.mod * holdout_frac)
    held = perm[:n_hold]
    seen = perm[n_hold:]
    return seen, held


def make_examples(task: Task, a_vals: torch.Tensor, b_vals: torch.Tensor, n: int, gen):
    """Sample n (a, op, b) -> result examples with a from a_vals and b from b_vals."""
    a = a_vals[torch.randint(0, len(a_vals), (n,), generator=gen)]
    b = b_vals[torch.randint(0, len(b_vals), (n,), generator=gen)]
    is_mul = torch.randint(0, 2, (n,), generator=gen).bool()
    op = torch.where(is_mul, task.tok_mul, task.tok_add)
    y = torch.where(is_mul, (a * b) % task.mod, (a + b) % task.mod)
    return torch.stack([a, op, b], dim=1), y


def build_eval_sets(task: Task, seen, held, n_eval: int, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    train_ev = make_examples(task, seen, seen, n_eval, g)
    # test: at least one operand held out -- half held x seen, half seen x held, plus held x held
    parts = [make_examples(task, held, seen, n_eval // 3, g),
             make_examples(task, seen, held, n_eval // 3, g),
             make_examples(task, held, held, n_eval - 2 * (n_eval // 3), g)]
    xte = torch.cat([p[0] for p in parts]); yte = torch.cat([p[1] for p in parts])
    return train_ev, (xte, yte)


def run(task: Task, arm: str, seed: int, steps, bs=512, d_model=128, lr=3e-4, wd=1.0,
        holdout=0.2, n_eval=12000, trace=0, quiet=False):
    torch.manual_seed(seed)
    seen, held = split_operands(task, holdout, seed=0)   # identical split for every arm/seed
    (xtr_ev, ytr_ev), (xte, yte) = build_eval_sets(task, seen, held, n_eval)
    xtr_ev, ytr_ev, xte, yte = (t.to(DEV) for t in (xtr_ev, ytr_ev, xte, yte))

    model = Tiny(build_emb(task, arm, d_model), task.vocab, d_model).to(DEV)
    n_in = sum(p.numel() for p in model.emb.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=steps)
    gen = torch.Generator().manual_seed(seed + 1)
    is_mul = xte[:, 1] == task.tok_mul

    def acc(x, y):
        if x.size(0) == 0:
            return float("nan")
        model.eval()
        with torch.no_grad():
            hits = sum((model(x[i:i + 8192]).argmax(-1) == y[i:i + 8192]).sum().item()
                       for i in range(0, x.size(0), 8192))
        model.train()
        return hits / x.size(0)

    for step in range(steps):
        xb, yb = make_examples(task, seen, seen, bs, gen)
        loss = F.cross_entropy(model(xb.to(DEV)), yb.to(DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if trace and (step + 1) % trace == 0:
            print(f"      step {step+1:6d}  train={acc(xtr_ev, ytr_ev):.3f}  "
                  f"+:{acc(xte[~is_mul], yte[~is_mul]):.3f}  "
                  f"*:{acc(xte[is_mul], yte[is_mul]):.3f}", flush=True)

    out = dict(train=acc(xtr_ev, ytr_ev), test=acc(xte, yte),
               add=acc(xte[~is_mul], yte[~is_mul]), mul=acc(xte[is_mul], yte[is_mul]),
               params=n_in)
    if not quiet:
        print(f"  {arm:9s} seed{seed}  in={n_in/1e3:6.1f}K  train={out['train']:.3f}  "
              f"test={out['test']:.3f}   +:{out['add']:.3f}  *:{out['mul']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()

    task = Task(tuple(a.primes))
    seen, held = split_operands(task, a.holdout)
    print(f"device={DEV}  primes={task.primes}  mod {task.mod}  vocab={task.vocab}")
    print(f"operands seen={len(seen)}  held out={len(held)}  "
          f"(chance = {1/task.mod:.4f})\n")

    rows = {}
    for arm in a.arms:
        runs = [run(task, arm, s, steps=a.steps, wd=a.wd, lr=a.lr,
                    holdout=a.holdout, trace=a.trace) for s in range(a.seeds)]
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("train", "test", "add", "mul")}
        sd = (sum((r["test"] - mean["test"]) ** 2 for r in runs) / len(runs)) ** 0.5
        rows[arm] = (mean, sd, runs[0]["params"])
        print(f"  -> {arm}: test {mean['test']:.3f} +- {sd:.3f}\n", flush=True)

    print(f"{'arm':9s} {'in params':>10s} {'train':>7s} {'test':>7s} {'+-':>6s} "
          f"{'test +':>7s} {'test *':>7s}")
    for arm, (m, sd, n) in rows.items():
        print(f"{arm:9s} {n/1e3:9.1f}K {m['train']:7.3f} {m['test']:7.3f} {sd:6.3f} "
              f"{m['add']:7.3f} {m['mul']:7.3f}")


if __name__ == "__main__":
    main()
