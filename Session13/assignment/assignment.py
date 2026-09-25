"""Session 13 assignment: a reversible transformer, trained three times.

Section 18 of the lesson asks for a ~20M LLM trained for 50M tokens:

  A  standard stack, at a fixed batch size it can run
  B  the same batch size, reversible (and report which reversible variant worked)
  C  reversible, pushed to the largest batch that fits

and a report of final loss, tokens/s and peak memory for each.

The blocks are nanoGPT's (karpathy/nanoGPT, reused from Session 10). What changes between the
arms is only how the blocks are chained and how the backward pass gets its activations:

  standard  p <- p + f(p)                           autograd stores every layer's activations
  midpoint  p[l+1] = p[l-1] + 2h f_l(p[l])          (Gal et al. 2025, section 16) rebuilt backward
  coupling  y1 = x1 + A(x2),  y2 = x2 + M(y1)       RevNet / Reformer, rebuilt backward

Both reversible stacks run inside one torch.autograd.Function that saves only the two boundary
states and walks the stack in reverse during backward, so the memory saving is real, not an
estimate.

  python assignment.py selfcheck   # reversible gradients == ordinary autograd, float64, CPU
  python assignment.py data        # tokenizer + 50M train tokens from WikiText-103
  python assignment.py probe       # short runs of each variant, pick the one that trains
  python assignment.py maxbatch    # largest batch per stack on this GPU
  python assignment.py train A|B|C # one of the three runs
  python assignment.py all         # data, probe, maxbatch, A, B, C
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results.json"
NANOGPT = HERE.parent.parent / "Session10" / "assignment" / "nanoGPT"
NANOGPT_URL = "https://github.com/karpathy/nanoGPT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337

VOCAB = 8192
BLOCK = 512
N_LAYER, N_HEAD, N_EMBD = 10, 6, 384
TOKENS = 50_000_000
LR, MIN_LR_FRAC, WARMUP_FRAC, WD, CLIP = 6e-4, 0.1, 0.02, 0.1, 1.0
EVAL_BATCHES = 50
H = {"midpoint": 0.25}          # step size, overwritten by the probe's winner


def ensure_nanogpt():
    if not NANOGPT.exists():
        subprocess.run(["git", "clone", "--depth", "1", NANOGPT_URL, str(NANOGPT)], check=True)
    if str(NANOGPT) not in sys.path:
        sys.path.insert(0, str(NANOGPT))


def load_results():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_result(key, value):
    r = load_results()
    r[key] = value
    OUT.write_text(json.dumps(r, indent=1))


# --------------------------------------------------------------------------------------
# data: an 8,192-token byte-level BPE trained on WikiText-103, then 50M train tokens
# --------------------------------------------------------------------------------------

def prepare_data():
    if (DATA / "train.bin").exists():
        return
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    DATA.mkdir(exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def docs(split):
        # wikitext is one line per paragraph; blank lines separate them
        return (t for t in ds[split]["text"] if t.strip())

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|eot|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs("train"), trainer)
    tok.save(str(DATA / "tokenizer.json"))

    def encode(split, limit):
        out, n, batch = [], 0, []
        for t in docs(split):
            batch.append(t)
            if len(batch) == 4096:
                for e in tok.encode_batch(batch):
                    out.append(np.asarray(e.ids, dtype=np.uint16))
                    n += len(e.ids)
                batch = []
                if limit and n >= limit:
                    break
        for e in tok.encode_batch(batch):
            out.append(np.asarray(e.ids, dtype=np.uint16))
        return np.concatenate(out)

    # one epoch: the training slice is 50M tokens plus one block, never repeated
    train = encode("train", TOKENS + BLOCK + 1)[: TOKENS + BLOCK + 1]
    val = encode("validation", 0)
    train.tofile(DATA / "train.bin")
    val.tofile(DATA / "val.bin")
    save_result("data", {"dataset": "Salesforce/wikitext wikitext-103-raw-v1", "vocab": VOCAB,
                         "train_tokens": int(len(train)), "val_tokens": int(len(val))})
    print(f"train {len(train):,} tokens, val {len(val):,} tokens")


class Batches:
    """Sequential, non-overlapping blocks: 50M tokens is exactly one pass."""

    def __init__(self, split):
        self.d = np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")
        self.pos = 0

    def next(self, b):
        n = b * BLOCK
        if self.pos + n + 1 > len(self.d):
            self.pos = 0
        a = torch.from_numpy(self.d[self.pos: self.pos + n + 1].astype(np.int64))
        self.pos += n
        return (a[:-1].view(b, BLOCK).to(DEVICE, non_blocking=True),
                a[1:].view(b, BLOCK).to(DEVICE, non_blocking=True))


# --------------------------------------------------------------------------------------
# the model: nanoGPT blocks, three ways of chaining them
# --------------------------------------------------------------------------------------

def attn(b, x):
    return b.attn(b.ln_1(x))


def mlp(b, x):
    return b.mlp(b.ln_2(x))


def f_block(b, x):
    """The whole block as one residual update: attention, then the MLP on x + attention."""
    a = attn(b, x)
    return a + mlp(b, x + a)


def params_of(blocks):
    return [p for b in blocks for p in b.parameters()]


class MidpointStack(torch.autograd.Function):
    """p[l+1] = p[l-1] + 2h f_l(p[l]) over `blocks`; saves only the last two states."""

    @staticmethod
    def forward(ctx, a, b, h, blocks, *params):
        ctx.h, ctx.blocks = h, blocks
        with torch.no_grad():
            for blk in blocks:
                a, b = b, a + 2 * h * f_block(blk, b)
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    def backward(ctx, ga, gb):
        h, blocks = ctx.h, ctx.blocks
        x, y = ctx.saved_tensors                      # (p[l], p[l+1]) for the top block
        gx = torch.zeros_like(x) if ga is None else ga
        gy = torch.zeros_like(y) if gb is None else gb
        grads = []
        for blk in reversed(blocks):
            ps = list(blk.parameters())
            with torch.enable_grad():
                xd = x.detach().requires_grad_(True)
                fo = f_block(blk, xd)
                d = torch.autograd.grad(fo, [xd] + ps, grad_outputs=2 * h * gy)
            prev = y - 2 * h * fo.detach()            # p[l-1], rebuilt from the two we hold
            x, y = prev, x
            gx, gy = gy, gx + d[0]
            grads.append(d[1:])
        flat = [g for d in reversed(grads) for g in d]
        return (gx, gy, None, None, *flat)


class CouplingStack(torch.autograd.Function):
    """y1 = x1 + A(x2), y2 = x2 + M(y1), per block; saves only the last pair."""

    @staticmethod
    def forward(ctx, x1, x2, blocks, *params):
        ctx.blocks = blocks
        with torch.no_grad():
            for blk in blocks:
                x1 = x1 + attn(blk, x2)
                x2 = x2 + mlp(blk, x1)
        ctx.save_for_backward(x1, x2)
        return x1, x2

    @staticmethod
    def backward(ctx, g1, g2):
        y1, y2 = ctx.saved_tensors
        g1 = torch.zeros_like(y1) if g1 is None else g1
        g2 = torch.zeros_like(y2) if g2 is None else g2
        grads = []
        for blk in reversed(ctx.blocks):
            pa = list(blk.ln_1.parameters()) + list(blk.attn.parameters())
            pm = list(blk.ln_2.parameters()) + list(blk.mlp.parameters())
            with torch.enable_grad():
                z = y1.detach().requires_grad_(True)
                m = mlp(blk, z)
                dm = torch.autograd.grad(m, [z] + pm, grad_outputs=g2)
            x2 = y2 - m.detach()
            g1 = g1 + dm[0]
            with torch.enable_grad():
                w = x2.detach().requires_grad_(True)
                a = attn(blk, w)
                da = torch.autograd.grad(a, [w] + pa, grad_outputs=g1)
            x1 = y1 - a.detach()
            g2 = g2 + da[0]
            y1, y2 = x1, x2
            # block.parameters() order is ln_1, attn, ln_2, mlp
            grads.append(list(da[1:]) + list(dm[1:]))
        flat = [g for d in reversed(grads) for g in d]
        return (g1, g2, None, *flat)


class RevGPT(nn.Module):
    def __init__(self, mode, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                 vocab=VOCAB, block=BLOCK, h=None, store=False):
        super().__init__()
        ensure_nanogpt()
        from model import Block, GPTConfig, LayerNorm
        cfg = GPTConfig(block_size=block, vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                        n_embd=n_embd, dropout=0.0, bias=False)
        self.mode, self.h = mode, (H["midpoint"] if h is None else h)
        self.store = store                        # True: same maths, ordinary autograd (selfcheck)
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(n_layer))
        self.ln_f = LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - (self.wte.weight.numel() + self.wpe.weight.numel() if non_embedding else 0)

    def trunk(self, x):
        bl = list(self.blocks)
        if self.mode == "standard":
            for b in bl:
                x = x + f_block(b, x)
            return x
        if self.mode == "midpoint":
            # the rule needs two states; the first step is an ordinary (stored) Euler step
            a, b = x, x + self.h * f_block(bl[0], x)
            rest = bl[1:]
            if self.store:
                for blk in rest:
                    a, b = b, a + 2 * self.h * f_block(blk, b)
                return b
            return MidpointStack.apply(a, b, self.h, rest, *params_of(rest))[1]
        if self.mode == "coupling":
            if self.store:
                x1, x2 = x, x
                for blk in bl:
                    x1 = x1 + attn(blk, x2)
                    x2 = x2 + mlp(blk, x1)
            else:
                x1, x2 = CouplingStack.apply(x, x, bl, *params_of(bl))
            return 0.5 * (x1 + x2)
        raise ValueError(self.mode)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        logits = self.head(self.ln_f(self.trunk(x)))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# --------------------------------------------------------------------------------------
# selfcheck: the reversible backward must give the gradients ordinary autograd gives
# --------------------------------------------------------------------------------------

def selfcheck():
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 16))
    tgt = torch.randint(0, 64, (2, 16))
    for mode in ("midpoint", "coupling"):
        grads = []
        for store in (True, False):
            torch.manual_seed(1)
            m = RevGPT(mode, n_layer=5, n_head=2, n_embd=16, vocab=64, block=16, h=0.25,
                       store=store).double()
            _, loss = m(idx, tgt)
            loss.backward()
            grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
        worst = max((grads[0][n] - grads[1][n]).abs().max().item() for n in grads[0])
        print(f"{mode:9s} max |grad(reversible) - grad(autograd)| = {worst:.2e}")
        assert worst < 1e-10, mode
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


def lr_at(step, total, peak):
    w = max(1, int(WARMUP_FRAC * total))
    if step < w:
        return peak * (step + 1) / w
    r = (step - w) / max(1, total - w)
    return peak * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * r)))


@torch.no_grad()
def evaluate(model, n=EVAL_BATCHES, b=8):
    model.eval()
    val = Batches("val")
    tot = 0.0
    for _ in range(n):
        x, y = val.next(b)
        tot += model(x, y)[1].item()
    model.train()
    return tot / n


def train(mode, batch, tokens, lr, tag, h=None, evals=10, log_every=50):
    torch.manual_seed(SEED)
    model = RevGPT(mode, h=h).to(DEVICE)
    opt = make_optimizer(model, lr)
    data = Batches("train")
    steps = tokens // (batch * BLOCK)
    eval_at = {int(steps * k / evals) for k in range(1, evals + 1)}
    torch.cuda.reset_peak_memory_stats()
    log, curve = [], []
    t_train, seen, t0 = 0.0, 0, time.time()
    diverged = False
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        torch.cuda.synchronize()
        ts = time.time()
        x, y = data.next(batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
        opt.step()
        torch.cuda.synchronize()
        t_train += time.time() - ts
        seen += batch * BLOCK
        li = loss.item()
        if not math.isfinite(li):
            diverged = True
            print(f"[{tag}] diverged at step {step}")
            break
        if step % log_every == 0 or step == steps - 1:
            log.append([step, seen, li, gn])
            print(f"[{tag}] step {step}/{steps} loss {li:.4f} gnorm {gn:.2f} "
                  f"{seen / t_train:,.0f} tok/s  {time.time() - t0:.0f}s", flush=True)
        if step + 1 in eval_at:
            v = evaluate(model)
            curve.append([seen, v])
            print(f"[{tag}] val {v:.4f} at {seen:,} tokens", flush=True)
    res = {
        "mode": mode, "h": model.h if mode == "midpoint" else None, "batch": batch,
        "tokens_per_step": batch * BLOCK, "steps": steps, "lr": lr, "tokens": seen,
        "params": model.n_params(), "params_non_embedding": model.n_params(True),
        "final_train_loss": float(np.mean([r[2] for r in log[-5:]])) if log else None,
        "final_val_loss": curve[-1][1] if curve and not diverged else None,
        "diverged": diverged,
        "tokens_per_s": seen / t_train, "train_seconds": t_train, "wall_seconds": time.time() - t0,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "log": log, "val_curve": curve,
    }
    if mode != "standard":
        res["reconstruction"] = reconstruction_error(model)
    del model, opt
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def reconstruction_error(model, b=4):
    """Run the stack up in fp32, walk it back down, compare with the input it started from."""
    x, _ = Batches("val").next(b)
    p = model.wte(x) + model.wpe(torch.arange(BLOCK, device=x.device))
    bl = list(model.blocks)
    if model.mode == "midpoint":
        a0, b0 = p, p + model.h * f_block(bl[0], p)
        a, c = a0, b0
        for blk in bl[1:]:
            a, c = c, a + 2 * model.h * f_block(blk, c)
        for blk in reversed(bl[1:]):
            a, c = c - 2 * model.h * f_block(blk, a), a
        ref, got = torch.stack([a0, b0]), torch.stack([a, c])
    else:
        x1, x2 = p, p
        for blk in bl:
            x1 = x1 + attn(blk, x2)
            x2 = x2 + mlp(blk, x1)
        for blk in reversed(bl):
            x2 = x2 - mlp(blk, x1)
            x1 = x1 - attn(blk, x2)
        ref, got = torch.stack([p, p]), torch.stack([x1, x2])
    err = (got - ref).abs()
    return {"max_abs": err.max().item(), "rel": (err.norm() / ref.norm()).item()}


# --------------------------------------------------------------------------------------
# largest batch that runs one full optimizer step
# --------------------------------------------------------------------------------------

def fits(mode, b, h=None):
    """One trial per fresh process: in-process, the first OOM left every later trial failing."""
    r = subprocess.run([sys.executable, __file__, "_fits", mode, str(b), str(h)],
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["OOM"])[-1]
    return None if last == "OOM" else float(last)


def _fits(mode, b, h=None):
    model = opt = None
    try:
        torch.manual_seed(SEED)
        model = RevGPT(mode, h=h).to(DEVICE)
        opt = make_optimizer(model, LR)
        x = torch.randint(0, VOCAB, (b, BLOCK), device=DEVICE)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):                         # the second step has the optimizer state
            _, loss = model(x, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, opt
        torch.cuda.empty_cache()


def max_batch(mode, h=None):
    lo, hi, mem = 0, 1, {}
    while (m := fits(mode, hi, h)) is not None:
        mem[hi] = m
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (m := fits(mode, mid, h)) is not None:
            mem[mid], lo = m, mid
        else:
            hi = mid
    return lo, {str(k): v for k, v in sorted(mem.items())}


# --------------------------------------------------------------------------------------
# parts
# --------------------------------------------------------------------------------------

PROBE_TOKENS = 4_000_000
PROBE_BATCH = 8
PROBE_ARMS = [("standard", None), ("midpoint", 0.25), ("midpoint", 0.5), ("coupling", None)]


def part_probe():
    out = {}
    for mode, h in PROBE_ARMS:
        tag = mode if h is None else f"{mode}_h{h}"
        r = train(mode, PROBE_BATCH, PROBE_TOKENS, LR, f"probe:{tag}", h=h, evals=4, log_every=100)
        out[tag] = r
        save_result("probe", out)
    return out


def chosen_variant():
    """The reversible arm B and C use: the probe's lowest final val loss."""
    probe = load_results().get("probe", {})
    rev = {k: v for k, v in probe.items() if k != "standard" and v["final_val_loss"]}
    if not rev:
        return "midpoint", H["midpoint"]
    k = min(rev, key=lambda k: rev[k]["final_val_loss"])
    return rev[k]["mode"], rev[k]["h"]


def part_maxbatch():
    mode, h = chosen_variant()
    out = {}
    for m, hh in (("standard", None), (mode, h)):
        b, mem = max_batch(m, hh)
        out[m] = {"max_batch": b, "peak_gib_by_batch": mem}
        print(f"{m}: max batch {b}")
        save_result("maxbatch", out)
    return out


def pow2_floor(n):
    return 1 << (n.bit_length() - 1)


def part_train(arm):
    r = load_results()
    mb = r["maxbatch"]
    mode, h = chosen_variant()
    base = pow2_floor(mb["standard"]["max_batch"])
    if arm == "A":
        cfg = ("standard", base, LR, None)
    elif arm == "B":
        cfg = (mode, base, LR, h)
    else:
        big = pow2_floor(mb[mode]["max_batch"])
        cfg = (mode, big, LR * math.sqrt(big / base), h)   # sqrt scaling for the larger batch
    res = train(cfg[0], cfg[1], TOKENS, cfg[2], arm, h=cfg[3])
    save_result(f"run_{arm}", res)
    return res


def main():
    if sys.argv[1:2] == ["_fits"]:
        mode, b, h = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        m = _fits(mode, b, None if h == "None" else float(h))
        print("OOM" if m is None else m)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["selfcheck", "data", "probe", "maxbatch", "train", "all"])
    ap.add_argument("arm", nargs="?", choices=["A", "B", "C"])
    a = ap.parse_args()
    if a.part == "selfcheck":
        return selfcheck()
    torch.backends.cuda.matmul.allow_tf32 = True
    prepare_data()
    if a.part == "probe" or a.part == "all":
        part_probe()
    if a.part == "maxbatch" or a.part == "all":
        part_maxbatch()
    if a.part == "train":
        part_train(a.arm)
    if a.part == "all":
        for arm in "ABC":
            part_train(arm)


if __name__ == "__main__":
    main()
