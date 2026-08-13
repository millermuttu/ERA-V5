"""
Does a model actually work without a `d_model x V` output head?

Three output sides, everything else held fixed -- same tokenizer, same data, same body, same
steps, same seeds:

    tied-head   Linear(d_model, V). The baseline, and the thing P5 wants to delete. Its cost
                grows with the vocabulary and nothing else about it does.
    reuse-W     logits = h @ W_proj, reshaped to (256, pos_dim), softmax down the byte axis.
                ZERO output parameters -- the input projection is re-read backwards. This is
                the purest form of the assignment's request.
    byte-head   Linear(d_model, 257 * pos_dim), softmax down the byte axis, with a 257th
                "no byte here" class so the token's length is predicted rather than inferred.
                Costs d_model * 8224, which does not depend on V either.

What the last two predict is a distribution over BYTE STRINGS that factorises across
positions. That is the honest catch and it is measured below: a factorised distribution can
put mass on byte strings that are not tokens, and normalising it over the vocabulary would
cost O(V) again -- the parameters go away, the normalisation does not, unless you accept
per-position sampling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from invert import POS_DIM, D, projection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronbudget"))
import vocab                                                        # noqa: E402
from roles import CHAR_DIM, encode                                  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 64          # small, because a V=151k logit tensor is what fills a 4 GB card
PAD = CHAR_DIM                                                      # the 257th class
CACHE = Path("/tmp/claude-1000/-home-muttu-Desktop-ERA/wikitext_qwen.npy")


def load_data(rows: int = 20000):
    """wikitext-103 under the Qwen2.5 tokenizer -- a real corpus and a 151k vocabulary, which
    is the regime where the output head is worth deleting."""
    if CACHE.exists():
        return np.load(CACHE)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"train[:{rows}]")
    text = "".join(t for t in ds["text"] if t.strip())
    ids = np.array(tk(text)["input_ids"], dtype=np.int32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, ids)
    return ids


def byte_targets(items):
    """[V, pos_dim] of byte values, PAD past the token's length."""
    T = np.full((len(items), POS_DIM), PAD, dtype=np.int64)
    for i, (_, b) in enumerate(items):
        bb = b[:POS_DIM]
        T[i, :len(bb)] = list(bb)
    return torch.from_numpy(T)


def codes_of(byte_tbl: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Build the codec grid for a batch of token ids, by scatter.

    A materialised [V, 8192] table is 5 GB at V=151k, which does not fit -- and the released
    module never builds one either: it keeps a byte buffer per token and scatters at forward
    time. This is that, vectorised. `_demo` checks it against roles.encode."""
    b = byte_tbl[ids].reshape(-1, POS_DIM)                          # [N, pos_dim], PAD=256
    valid = b < CHAR_DIM
    n = valid.sum(1, keepdim=True).clamp(min=1).float()
    src = (valid.float() / n.sqrt()).unsqueeze(1)                   # zero at PAD positions
    g = torch.zeros(b.shape[0], CHAR_DIM, POS_DIM, device=ids.device)
    g.scatter_(1, b.clamp(max=CHAR_DIM - 1).unsqueeze(1), src)
    k = g.reshape(b.shape[0], -1)
    return ((k - k.mean(1, keepdim=True)) / (k.std(1, keepdim=True) + 1e-8)) \
        .reshape(*ids.shape, -1)


class Body(nn.Module):
    def __init__(self, byte_tbl, d_model, layers=4, heads=4):
        super().__init__()
        self.register_buffer("byte_tbl", byte_tbl)                        # [V, pos_dim]
        self.W = nn.Parameter(projection(d_model, "orth").clone())        # W_proj, trainable
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        with torch.no_grad():                       # the codec is fixed; no graph needed here
            c = codes_of(self.byte_tbl, x)
        h = c @ self.W.T + self.pos[:, :x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.norm(self.body(h, mask=mask, is_causal=True))


class Model(nn.Module):
    def __init__(self, arm, byte_tbl, V, d_model):
        super().__init__()
        self.arm, self.trunk = arm, Body(byte_tbl, d_model)
        if arm == "tied-head":
            self.head = nn.Linear(d_model, V, bias=False)
        elif arm == "byte-head":
            self.head = nn.Linear(d_model, (CHAR_DIM + 1) * POS_DIM, bias=False)
        else:
            self.head = None                                        # reuse-W: nothing at all

    def logits(self, x):
        h = self.trunk(x)
        if self.arm == "tied-head":
            return self.head(h)
        if self.arm == "byte-head":
            return self.head(h).reshape(*h.shape[:-1], CHAR_DIM + 1, POS_DIM)
        return (h @ self.trunk.W).reshape(*h.shape[:-1], CHAR_DIM, POS_DIM)

    def out_params(self):
        return 0 if self.head is None else sum(p.numel() for p in self.head.parameters())


def loss_of(model, x, y, tgt_bytes):
    lg = model.logits(x)
    if model.arm == "tied-head":
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
    t = tgt_bytes[y.reshape(-1)]                                    # [B*T, pos_dim]
    lg = lg.reshape(-1, lg.shape[-2], POS_DIM)
    if model.arm == "reuse-W":                                      # no PAD class available
        keep = t < CHAR_DIM
        return F.cross_entropy(lg.permute(0, 2, 1)[keep], t[keep])
    return F.cross_entropy(lg, t)


@torch.no_grad()
def evaluate(model, val, tgt_bytes, byte_key, bs=8, batches=40):
    """Top-1 next-token accuracy for every arm, plus the two failure modes that only the
    factorised arms can have: decoding to a byte string that is not any token, and getting
    the length wrong."""
    rng = np.random.default_rng(7)
    hit = tot = notok = wrong_len = 0
    for _ in range(batches):
        i = rng.integers(0, len(val) - SEQ - 1, bs)
        x = torch.from_numpy(np.stack([val[j:j + SEQ] for j in i]).astype(np.int64)).to(DEV)
        y = torch.from_numpy(np.stack([val[j + 1:j + SEQ + 1] for j in i])
                             .astype(np.int64)).to(DEV)
        lg = model.logits(x)
        if model.arm == "tied-head":
            hit += int((lg.argmax(-1) == y).sum()); tot += y.numel(); continue
        pred = lg.reshape(-1, lg.shape[-2], POS_DIM).argmax(1)      # [B*T, pos_dim] bytes
        true = tgt_bytes[y.reshape(-1)]
        if model.arm == "reuse-W":                                  # length by the gap rule
            peak = lg.reshape(-1, lg.shape[-2], POS_DIM).max(1).values
            L = (peak[:, :-1] - peak[:, 1:]).argmax(1) + 1
            mask = torch.arange(POS_DIM, device=DEV)[None, :] >= L[:, None]
            pred = pred.masked_fill(mask, PAD)
        for row, tr in zip(pred.cpu().numpy(), true.cpu().numpy()):
            tot += 1
            key = tuple(row.tolist())
            if key == tuple(tr.tolist()):
                hit += 1
            else:
                if (row < CHAR_DIM).sum() != (tr < CHAR_DIM).sum():
                    wrong_len += 1
                if key not in byte_key:
                    notok += 1
    return hit / tot, notok / tot, wrong_len / tot


def run(arm, ids, items, tgt_bytes, byte_key, steps, seed, d_model=256, bs=8, lr=3e-4):
    torch.manual_seed(seed)
    cut = int(0.95 * len(ids))
    train, val = ids[:cut], ids[cut:]
    model = Model(arm, tgt_bytes, len(items), d_model).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1)
    tb = tgt_bytes.to(DEV)
    for _ in range(steps):
        i = rng.integers(0, len(train) - SEQ - 1, bs)
        x = torch.from_numpy(np.stack([train[j:j + SEQ] for j in i]).astype(np.int64)).to(DEV)
        y = torch.from_numpy(np.stack([train[j + 1:j + SEQ + 1] for j in i])
                             .astype(np.int64)).to(DEV)
        loss = loss_of(model, x, y, tb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return (*evaluate(model, val, tb, byte_key), model.out_params())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--arms", nargs="+", default=["tied-head", "byte-head", "reuse-W"])
    a = ap.parse_args()

    ids = load_data()
    items = vocab.load("Qwen/Qwen2.5-0.5B-Instruct")
    V = len(items)
    ids = ids[ids < V]
    tgt_bytes = byte_targets(items)
    byte_key = {tuple(r) for r in tgt_bytes.tolist()}

    # the scatter must reproduce the released codec exactly, or nothing below means anything
    probe = torch.tensor([7, 1234, 99_999, V - 1])
    got = codes_of(tgt_bytes, probe)
    for j, i in enumerate(probe.tolist()):
        assert torch.allclose(got[j], encode(items[i][1], POS_DIM, "trunc"), atol=1e-5), i
    print(f"device={DEV}  {len(ids):,} tokens  V={V:,}  d_model={a.d_model}  "
          f"D={D}  {a.steps} steps x {a.seeds} seeds\n")

    head = (f"{'arm':<12}{'out params':>12}{'top-1':>9}{'not a token':>13}"
            f"{'wrong length':>14}")
    print(head); print("-" * len(head))
    for arm in a.arms:
        res = [run(arm, ids, items, tgt_bytes, byte_key, a.steps, s)
               for s in range(a.seeds)]
        acc = np.array([r[0] for r in res])
        print(f"{arm:<12}{res[0][3]/1e6:>10.2f}M{acc.mean():>7.3f}±{acc.std():.3f}"
              f"{100*np.mean([r[1] for r in res]):>12.1f}%"
              f"{100*np.mean([r[2] for r in res]):>13.1f}%")


if __name__ == "__main__":
    main()
