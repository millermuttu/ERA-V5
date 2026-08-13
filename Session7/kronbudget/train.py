"""
Does the collision count actually cost anything downstream?

A tiny LM over this repo's own Session 2 BPE tokenizer and its Kannada / Telugu / Hindi /
English corpora, with the input path swapped between role schemes. Everything except the
role map is held fixed: same tokenizer, same data, same model, same steps, same seeds, and
`D = 256 * R` so the projection is the same size within a budget.

Where the harm should show. Two tokens with identical codes are one token as far as the
model's input is concerned, so the damage lands on *context*: whatever follows a collided
token has to be predicted from an ambiguous history. So val loss is reported twice -- overall,
and restricted to positions whose preceding token is in a collision group. The second column
is the one carrying the signal; the first is diluted by every position the collision never
touched, and reporting only the first would hide the effect.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab
from collide import key
from roles import encode

ERA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ERA / "Session2/src"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CORPORA = ["kn_india", "te_india", "hi_india", "en_india"]
SEQ = 64
HOLDOUT = 0.2


def load_corpus():
    """-> (train ids, {corpus: val ids}). Split per corpus, not on the concatenation: the
    files are stored kn, te, hi, en, so a tail split would make the whole validation set
    English -- the one script with no collisions at any budget."""
    from bpe_tokenizer import BalancedBPETokenizer
    tk = BalancedBPETokenizer.load(str(ERA / "Session2/src/tokenizer_10k.json"))
    train, val = [], {}
    for name in CORPORA:
        ids = tk.encode((ERA / f"Session2/src/data/{name}.txt").read_text())
        cut = int((1 - HOLDOUT) * len(ids))
        train.extend(ids[:cut])
        val[name] = np.array(ids[cut:], dtype=np.int64)
    return np.array(train, dtype=np.int64), val


def collided_types(items, R, scheme):
    """Token ids that share a code with at least one other id."""
    groups = defaultdict(list)
    for i, (_, b) in enumerate(items):
        groups[key(b, R, "trunc" if scheme != "trunc" and len(b) <= R else scheme)].append(i)
    return {i for g in groups.values() if len(g) > 1 for i in g}


def code_table(items, R, scheme):
    return torch.stack([encode(b, R, scheme) for _, b in items])


class LM(nn.Module):
    def __init__(self, codes, V, d_model=192, layers=2, heads=4):
        super().__init__()
        self.codes = nn.Embedding.from_pretrained(codes, freeze=True)   # the fixed codec
        self.projection = nn.Linear(codes.shape[1], d_model, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V)

    def forward(self, x):
        h = self.projection(self.codes(x)) + self.pos[:, :x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.head(self.norm(self.body(h, mask=mask, is_causal=True)))


def batches(data, bs, n, rng):
    for _ in range(n):
        i = rng.integers(0, len(data) - SEQ - 1, bs)
        x = np.stack([data[j:j + SEQ] for j in i])
        y = np.stack([data[j + 1:j + SEQ + 1] for j in i])
        yield torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


def run(arm, items, train, val, steps, seed, ref_hit, bs=16, lr=3e-4):
    """`ref_hit` is the BASELINE's collision set, shared by every arm at this budget. Scoring
    each arm on its own collision set would compare losses on different positions, which is
    not a comparison at all -- the question is what the schemes do on the positions the
    released codec cannot disambiguate."""
    R, scheme = arm
    torch.manual_seed(seed)
    hit = ref_hit
    codes = code_table(items, R, scheme)
    model = LM(codes.to(DEV), len(items)).to(DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1)
    model.train()
    for x, y in batches(train, bs, steps, rng):
        loss = F.cross_entropy(model(x).reshape(-1, len(items)), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    hit_t = torch.tensor(sorted(hit) or [-1], device=DEV)
    per, amb, amb_n = {}, 0.0, 0
    with torch.no_grad():
        for name, ids in val.items():
            s, n = 0.0, 0
            for i in range(0, len(ids) - SEQ - 1, SEQ):      # contiguous, not sampled
                x = torch.from_numpy(ids[None, i:i + SEQ]).to(DEV)
                y = torch.from_numpy(ids[None, i + 1:i + SEQ + 1]).to(DEV)
                l = F.cross_entropy(model(x).reshape(-1, len(items)), y.reshape(-1),
                                    reduction="none")
                s += float(l.sum()); n += l.numel()
                # positions whose preceding token is ambiguous under this scheme
                m = torch.isin(x.reshape(-1), hit_t)
                amb += float(l[m].sum()); amb_n += int(m.sum())
            per[name] = s / max(n, 1)
    return per, (amb / amb_n if amb_n else float("nan")), len(hit), amb_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="+",
                    default=["16:trunc", "16:fit-ends", "16:fit-wrap", "16:fit-rel",
                             "32:trunc", "32:fit-ends"])
    a = ap.parse_args()

    train, val = load_corpus()
    items = vocab.load_session2()
    print(f"device={DEV}  {len(train):,} train tokens, vocab {len(items):,}, seq={SEQ}, "
          f"{a.steps} steps x {a.seeds} seeds")
    print("val tokens: " + "  ".join(f"{k.split('_')[0]}={len(v):,}" for k, v in val.items())
          + "\n")

    head = (f"{'arm':<15}{'proj':>8}{'collided':>9}{'ambig':>7}" +
            "".join(f"{k.split('_')[0]:>14}" for k in CORPORA) + f"{'ambig ctx':>14}")
    print(head); print("-" * len(head))
    for spec in a.arms:
        R, scheme = int(spec.split(":")[0]), spec.split(":")[1]
        ref = collided_types(items, R, "trunc")
        own = len(collided_types(items, R, scheme))
        res = [run((R, scheme), items, train, val, a.steps, s, ref) for s in range(a.seeds)]
        row = f"{spec:<15}{256*R*192/1e6:>7.2f}M{own:>9,}{res[0][3]:>7,}"
        for k in CORPORA:
            v = np.array([r[0][k] for r in res])
            row += f"{v.mean():>9.3f}±{v.std():.2f}"
        amb = np.array([r[1] for r in res])
        row += f"{amb.mean():>9.3f}±{amb.std():.2f}"
        print(row)
    print("\nval cross-entropy, lower is better. 'ambig' = val positions whose preceding "
          "token\nis one the scheme cannot tell apart from another; 0 means the scheme has "
          "no such token.")


if __name__ == "__main__":
    main()
