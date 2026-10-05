"""Session 14 assignment: train a dense model, convert it into a mixture-of-experts, keep training.

Section 21 of the lesson: "Train a Linear model and convert that into an MoE! ... must show they
continue to train and reduce loss!" The lesson calls the dense model the "standard model"; the
conversion is Section 15's upcycling.

  phase 1  a dense nanoGPT (6 layers, d 384) trains on the first 20M WikiText-103 tokens at a
           constant learning rate, and is saved.
  phase 2  three branches start from that checkpoint and see the same next 15M tokens:
             dense  the dense model, continued                       (the baseline to beat)
             copy   sparse upcycling: every expert = the dense MLP   (Google, 2022)
             drop   drop-upcycling: each copy with half its neurons redrawn (r = 0.5, 2025)
           8 experts, top-2, a new router, and the Section 13 bias balancing (choosing only).

Copy-upcycling is exact at the moment of conversion: all experts are the same function and the
kept weights sum to 1, so the MoE computes the dense model. `selfcheck` asserts that, and the run
records the validation loss on both sides of the conversion.

The data is Session 13's: 8,192-token BPE on WikiText-103, already tokenized to train.bin/val.bin.

  python assignment.py selfcheck   # conversion is exact, dropless routing is exact, CPU float64
  python assignment.py all         # dense, then the three branches -> results.json
  python assignment.py dense       # phase 1 only
  python assignment.py branch copy # one phase 2 branch (dense | copy | drop)
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "results.json"
CKPT = HERE / "data" / "dense.pt"
S13 = HERE.parent.parent / "Session13" / "assignment" / "data"
NANOGPT = HERE.parent.parent / "Session10" / "assignment" / "nanoGPT"
sys.path.insert(0, str(NANOGPT))
from model import GPT, GPTConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337

VOCAB, BLOCK = 8192, 512
N_LAYER, N_HEAD, N_EMBD = 6, 6, 384
BATCH = 16                                   # sequences of 512 tokens per step
MICRO = 8                                    # per forward pass; the MoE at 16 does not fit 4 GiB
DENSE_TOKENS, BRANCH_TOKENS = 20_000_000, 15_000_000
LR, MIN_LR_FRAC, WD, CLIP = 6e-4, 0.1, 0.1, 1.0
WARMUP_FRAC, REWARM_FRAC = 0.02, 0.01
E, K = 8, 2                                  # experts, top-k
GAMMA = 1e-3                                 # bias update speed (Section 13 / V5 decision)
DROP_R = 0.5
EVALS = 6                                    # validation points per phase
EVAL_BATCHES = 50


def save_result(key, value):
    r = json.loads(OUT.read_text()) if OUT.exists() else {}
    r[key] = value
    OUT.write_text(json.dumps(r, indent=1))


class Batches:
    """Sequential, non-overlapping blocks, starting at a given token offset."""

    def __init__(self, split, start=0):
        self.d = np.memmap(S13 / f"{split}.bin", dtype=np.uint16, mode="r")
        self.pos = start

    def next(self, b):
        n = b * BLOCK
        a = torch.from_numpy(self.d[self.pos: self.pos + n + 1].astype(np.int64))
        self.pos += n
        return a[:-1].view(b, BLOCK).to(DEVICE), a[1:].view(b, BLOCK).to(DEVICE)


# --------------------------------------------------------------------------------------
# the mixture-of-experts layer and the two conversions
# --------------------------------------------------------------------------------------

class MoE(nn.Module):
    """Softmax router, top-k, kept weights renormalized, dropless (every token reaches its experts).

    `bias` is the auxiliary-loss-free balancing term: added to the scores only when choosing, never
    to the weights, nudged after each step toward an even load.
    """

    def __init__(self, experts, n_embd, k=K):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.k = k
        self.router = nn.Linear(n_embd, len(experts), bias=False)
        nn.init.normal_(self.router.weight, 0.0, 0.02)
        self.register_buffer("bias", torch.zeros(len(experts)))
        self.register_buffer("load", torch.zeros(len(experts)))

    def forward(self, x):
        h = x.reshape(-1, x.shape[-1])
        p = self.router(h).softmax(-1)          # the whole model runs in fp32
        idx = (p + self.bias).topk(self.k, dim=-1).indices
        w = p.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(h)
        for e, ex in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, ex(h[tok]) * w[tok, slot, None])
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=len(self.experts))
        return out.view_as(x)

    @torch.no_grad()
    def balance(self):
        """One bias step from this optimizer step's load; returns the load it used."""
        load = self.load.clone()
        self.bias += GAMMA * torch.sign(load.mean() - load)
        self.load.zero_()
        return load


def redraw(mlp, r, gen):
    """Drop-upcycling: re-initialise a fraction r of the hidden neurons (rows of c_fc, columns of
    c_proj) from a normal with the original matrix's mean and std."""
    hid = mlp.c_fc.weight.shape[0]
    pick = torch.randperm(hid, generator=gen)[: int(r * hid)]
    with torch.no_grad():
        for wt, sl in ((mlp.c_fc.weight, (pick, slice(None))), (mlp.c_proj.weight, (slice(None), pick))):
            m, s = wt.mean().item(), wt.std().item()
            wt[sl] = (torch.randn(wt[sl].shape, generator=gen, dtype=wt.dtype) * s + m).to(wt.device)
    return pick


def upcycle(model, method, e=E, k=K, r=DROP_R, seed=SEED):
    """Swap every block's MLP for an MoE built from copies of it. In place; returns the model."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                         # router init
    for blk in model.transformer.h:
        experts = [copy.deepcopy(blk.mlp) for _ in range(e)]
        if method == "drop":
            for ex in experts:
                redraw(ex, r, gen)
        blk.mlp = MoE(experts, blk.ln_2.weight.shape[0], k).to(blk.ln_2.weight.device,
                                                               blk.ln_2.weight.dtype)
    return model


def moes(model):
    return [b.mlp for b in model.transformer.h if isinstance(b.mlp, MoE)]


def active_params(model):
    """Parameters a token touches: everything except the experts it was not routed to."""
    n = model.get_num_params(non_embedding=False)
    for m in moes(model):
        per = sum(p.numel() for p in m.experts[0].parameters())
        n -= per * (len(m.experts) - m.k)
    return n


def new_model(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, vocab=VOCAB, block=BLOCK):
    cfg = GPTConfig(block_size=block, vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, dropout=0.0, bias=False)
    return GPT(cfg)


# --------------------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------------------

def selfcheck():
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 16))
    dense = new_model(n_layer=2, n_head=2, n_embd=16, vocab=64, block=16).double().eval()
    want = dense(idx, idx)[1].item()

    # copy-upcycling computes the dense model exactly, whatever the router does
    for e, k in ((8, 2), (4, 1), (1, 1)):
        moe = upcycle(copy.deepcopy(dense), "copy", e=e, k=k).double().eval()
        for m in moes(moe):
            m.bias.normal_()                        # any choice of experts
        got = moe(idx, idx)[1].item()
        print(f"copy  E={e} k={k}  |loss(moe) - loss(dense)| = {abs(got - want):.2e}")
        assert abs(got - want) < 1e-12

    # dropless routing is exact: equals computing every expert densely and mixing the top-k
    moe = upcycle(copy.deepcopy(dense), "drop", e=4, k=2).double()
    m = moes(moe)[0]
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    p = m.router(x).softmax(-1)
    top = p.topk(2, -1)
    w = top.values / top.values.sum(-1, keepdim=True)
    allout = torch.stack([ex(x) for ex in m.experts], -2)          # (3, 5, E, 16)
    ref = (allout.gather(-2, top.indices[..., None].expand(-1, -1, -1, 16)) * w[..., None]).sum(-2)
    err = (m(x) - ref).abs().max().item()
    print(f"drop  dropless dispatch vs dense mix, max err = {err:.2e}")
    assert err < 1e-12

    # drop-upcycling redraws exactly r of the neurons and leaves the rest untouched
    base = dense.transformer.h[0].mlp
    ex = copy.deepcopy(base)
    pick = redraw(ex, 0.5, torch.Generator().manual_seed(0))
    same = (ex.c_fc.weight == base.c_fc.weight).all(1)
    assert same.sum().item() == 32 and not same[pick].any()
    assert torch.equal(ex.c_proj.weight[:, same], base.c_proj.weight[:, same])
    print("drop  r=0.5 redraws 32 of 64 neurons, keeps the other 32 bit-exact")
    print("selfcheck ok")


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------

def make_optimizer(model, lr):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                              {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), fused=DEVICE == "cuda")


def lr_dense(step, total):
    """Phase 1: warmup, then constant, so any branch can start from the end of it."""
    w = max(1, int(WARMUP_FRAC * total))
    return LR * min(1.0, (step + 1) / w)


def lr_branch(step, total):
    """Phase 2: a short re-warm (the optimizer state is new), then cosine down to 10%."""
    w = max(1, int(REWARM_FRAC * total))
    if step < w:
        return LR * (step + 1) / w
    r = (step - w) / max(1, total - w)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * r)))


@torch.no_grad()
def evaluate(model):
    model.eval()
    val = Batches("val")
    tot = sum(model(*val.next(8))[1].item() for _ in range(EVAL_BATCHES))
    model.train()
    return tot / EVAL_BATCHES


def run(model, tag, start, tokens, lr_fn, log_every=50):
    """Train `model` on train tokens [start, start + tokens). Returns logs and routing stats."""
    torch.manual_seed(SEED)
    opt = make_optimizer(model, LR)
    data = Batches("train", start)
    steps = tokens // (BATCH * BLOCK)
    eval_at = {int(steps * j / EVALS) for j in range(1, EVALS + 1)}
    torch.cuda.reset_peak_memory_stats()
    log, curve, route = [], [[start, evaluate(model)]], []
    print(f"[{tag}] val {curve[0][1]:.4f} at {start:,} tokens (before training)", flush=True)
    t_train, seen = 0.0, start
    win = None                                    # per-layer load summed over the last 50 steps
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_fn(step, steps)
        torch.cuda.synchronize()
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(BATCH // MICRO):
            _, loss = model(*data.next(MICRO))
            (loss * MICRO / BATCH).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        opt.step()
        loads = torch.stack([m.balance() for m in moes(model)]) if moes(model) else None
        torch.cuda.synchronize()
        t_train += time.time() - ts
        seen += BATCH * BLOCK
        if loads is not None:
            win = loads if win is None else win + loads
        if step % log_every == 0 or step == steps - 1:
            li = loss.item()
            row = [step, seen, li, gn]
            msg = ""
            if win is not None:
                # MaxVio: busiest expert's excess over an even share; dead: no tokens in the window
                maxvio = ((win.max(1).values - win.mean(1)) / win.mean(1)).mean().item()
                dead = int((win == 0).sum().item())
                route.append([seen, maxvio, dead, (win / win.sum(1, keepdim=True)).tolist()])
                msg = f" maxvio {maxvio:.2f} dead {dead}"
                win = None
            log.append(row)
            print(f"[{tag}] step {step}/{steps} loss {li:.4f} gnorm {gn:.2f}{msg} "
                  f"{(seen - start) / t_train:,.0f} tok/s", flush=True)
        if step + 1 in eval_at:
            curve.append([seen, evaluate(model)])
            print(f"[{tag}] val {curve[-1][1]:.4f} at {seen:,} tokens", flush=True)
    return {
        "steps": steps, "tokens": seen - start, "start": start,
        "params_total": model.get_num_params(non_embedding=False),
        "params_active": active_params(model),
        "val_start": curve[0][1], "val_end": curve[-1][1],
        "train_end": float(np.mean([r[2] for r in log[-5:]])),
        "tokens_per_s": (seen - start) / t_train, "train_seconds": t_train,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "log": log, "val_curve": curve, "routing": route,
        "router_bias": [m.bias.tolist() for m in moes(model)],
    }


def part_dense():
    torch.manual_seed(SEED)
    model = new_model().to(DEVICE)
    res = run(model, "dense", 0, DENSE_TOKENS, lr_dense)
    CKPT.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    save_result("phase1_dense", res)


def part_branch(name):
    model = new_model()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.to(DEVICE)
    if name != "dense":
        before = evaluate(model)
        upcycle(model, name)
        model.to(DEVICE)
        print(f"[{name}] val {before:.4f} dense -> {evaluate(model):.4f} right after conversion")
    res = run(model, name, DENSE_TOKENS, BRANCH_TOKENS, lr_branch)
    save_result(f"branch_{name}", res)
    del model
    torch.cuda.empty_cache()


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part == "selfcheck":
        return selfcheck()
    if part in ("dense", "all"):
        part_dense()
    if part == "branch":
        part_branch(sys.argv[2])
    if part == "all":
        for name in ("dense", "copy", "drop"):
            part_branch(name)


if __name__ == "__main__":
    main()
