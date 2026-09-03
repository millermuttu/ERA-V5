"""Session 10 assignment: make a real training loop tell you the truth about itself.

The model is nanoGPT (karpathy/nanoGPT). This file does not fork its trainer; it imports
its GPT and drives it from an instrumented loop of our own, which is what parts 3 to 5
need anyway.

Run `python assignment.py all` to produce results.json and the numbers quoted in the
README. `python assignment.py selfcheck` runs the asserts and nothing else.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NANOGPT = HERE / "nanoGPT"
NANOGPT_URL = "https://github.com/karpathy/nanoGPT"
OUT = HERE / "results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337


# --------------------------------------------------------------------------------------
# bootstrap: nanoGPT itself, and the shakespeare_char dataset it ships a prepare.py for
# --------------------------------------------------------------------------------------

def ensure_nanogpt():
    if not NANOGPT.exists():
        print(f"cloning {NANOGPT_URL} -> {NANOGPT}")
        subprocess.run(["git", "clone", "--depth", "1", NANOGPT_URL, str(NANOGPT)], check=True)
    if str(NANOGPT) not in sys.path:
        sys.path.insert(0, str(NANOGPT))


def ensure_data():
    d = NANOGPT / "data" / "shakespeare_char"
    if not (d / "train.bin").exists():
        print("preparing shakespeare_char")
        subprocess.run([sys.executable, "prepare.py"], cwd=str(d), check=True)
    import pickle
    with open(d / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    return train, meta["vocab_size"]


def make_model(vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.0,
               device=DEVICE, dtype=torch.float32):
    """nanoGPT's GPT, at the shakespeare_char size. dropout stays 0: a gradient check
    needs a repeatable forward pass."""
    from model import GPT, GPTConfig
    cfg = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=dropout, bias=False)
    torch.manual_seed(SEED)
    return GPT(cfg).to(device=device, dtype=dtype)


def get_batch(data, batch_size, block_size, rng, device=DEVICE):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)


def ragged_batch(data, batch_size, block_size, rng, device=DEVICE, valid_len=None):
    """A micro-batch holding `valid_len` real tokens per row, the rest masked out.

    nanoGPT packs fixed-length chunks, so every micro-batch holds exactly B*T valid
    tokens and section 8's bug reads 0.0%. The length is drawn per micro-batch rather
    than per row on purpose: rows averaged inside one micro-batch wash the difference
    out, and it is the spread *between* micro-batches that the naive normaliser gets
    wrong. A short micro-batch also carries less context per token, so its mean loss is
    genuinely higher, which is the lesson's [4, 4, 2] against [2.0, 2.0, 5.0].
    """
    x, y = get_batch(data, batch_size, block_size, rng, device)
    if valid_len is None:
        valid_len = block_size
    y[:, valid_len:] = -1                  # nanoGPT's model.py already passes ignore_index=-1
    return x, y, batch_size * valid_len


def micro_lengths(block_size, micro, rng, min_len=8):
    """Log-uniform lengths, so a run sees both near-equal and badly skewed steps."""
    lo, hi = math.log(min_len), math.log(block_size)
    return [int(round(math.exp(rng.uniform(lo, hi)))) for _ in range(micro)]


# --------------------------------------------------------------------------------------
# part 1 — every tensor shape in the step, and what each dimension means
# --------------------------------------------------------------------------------------

def part1_shapes(data, vocab_size):
    B, T = 4, 256
    model = make_model(vocab_size, block_size=T)
    rng = np.random.default_rng(SEED)
    x, y = get_batch(data, B, T, rng)
    logits, loss = model(x, y)

    n_params = sum(p.numel() for p in model.parameters())
    rows = [
        ("tokens x", tuple(x.shape), "B=batch sequences, T=positions in each"),
        ("targets y", tuple(y.shape), "same grid, each entry the token one position later"),
        ("tok_emb", (B, T, model.config.n_embd), "B, T, D=channels per token"),
        ("logits", tuple(logits.shape), "B, T, V=one score per vocabulary token"),
        ("logits, flattened", (B * T, vocab_size), "every position becomes an independent example"),
        ("targets, flattened", (B * T,), "one correct token id per example"),
        ("loss", tuple(loss.shape), "scalar: the mean over contributing positions"),
        ("lm_head.weight", tuple(model.lm_head.weight.shape), "V, D: one learned row per token"),
    ]
    print(f"\n[part 1] shapes  (B={B} T={T} D={model.config.n_embd} V={vocab_size}, {n_params/1e6:.2f}M params)")
    for name, shape, meaning in rows:
        print(f"  {name:22s} {str(shape):18s} {meaning}")
    return {
        "B": B, "T": T, "D": model.config.n_embd, "V": vocab_size,
        "n_params": n_params,
        "shapes": [{"tensor": n, "shape": list(s), "meaning": m} for n, s, m in rows],
        "loss": loss.item(),
    }


# --------------------------------------------------------------------------------------
# part 2 — verify one gradient by hand
# --------------------------------------------------------------------------------------

def part2_gradcheck(data, vocab_size, h=1e-4):
    """Nudge one weight, measure the loss change, compare against backward().

    float64 on CPU. In bf16 this check cannot pass: 2.4 decimal digits is not
    'several decimals', which is section 10's point made from the other side.
    """
    torch.manual_seed(SEED)
    model = make_model(vocab_size, n_layer=2, n_head=2, n_embd=64, block_size=32,
                       device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(SEED)
    x, y = get_batch(data, 2, 32, rng, device="cpu")

    def loss_at():
        with torch.no_grad():
            return float(model(x, y)[1])

    # analytic
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    w = model.transformer.h[0].mlp.c_fc.weight
    idx = (0, 0)
    analytic = float(w.grad[idx])

    base = loss_at()
    with torch.no_grad():
        w[idx] += h
    up = loss_at()
    with torch.no_grad():
        w[idx] -= 2 * h
    down = loss_at()
    with torch.no_grad():
        w[idx] += h                                    # put it back

    forward = (up - base) / h                          # the lesson's own 0.064/0.001 shape
    central = (up - down) / (2 * h)
    print(f"\n[part 2] gradient of transformer.h[0].mlp.c_fc.weight[0,0], float64 on CPU")
    print(f"  loss at w        {base:.12f}")
    print(f"  loss at w+{h:g}  {up:.12f}")
    print(f"  analytic (backward)      {analytic:.12f}")
    print(f"  forward difference       {forward:.12f}   rel err {abs(forward-analytic)/abs(analytic):.3e}")
    print(f"  central difference       {central:.12f}   rel err {abs(central-analytic)/abs(analytic):.3e}")
    return {"h": h, "loss": base, "loss_up": up, "loss_down": down,
            "analytic": analytic, "forward_diff": forward, "central_diff": central,
            "rel_err_forward": abs(forward - analytic) / abs(analytic),
            "rel_err_central": abs(central - analytic) / abs(analytic)}


# --------------------------------------------------------------------------------------
# part 3 — break gradient accumulation on purpose
# --------------------------------------------------------------------------------------

def _accum_step(model, batches, mode):
    """One optimiser step's worth of accumulation. Returns the reported loss.

    correct : sum of all token losses / total valid tokens        (every token one vote)
    naive   : mean of the per-micro-batch means                   (every micro-batch one vote)
              which is nanoGPT's train.py:301, loss / gradient_accumulation_steps
    """
    total_tokens = sum(n for _, _, n in batches)
    reported = 0.0
    for x, y, n in batches:
        logits, _ = model(x, y)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_y = y.view(-1)
        if mode == "correct":
            s = F.cross_entropy(flat_logits, flat_y, ignore_index=-1, reduction="sum")
            loss = s / total_tokens
        else:
            loss = F.cross_entropy(flat_logits, flat_y, ignore_index=-1) / len(batches)
        loss.backward()
        reported += float(loss)
    return reported


def part3_accumulation(data, vocab_size, steps=60, micro=4, B=8, T=128, lr=1e-3):
    def batches_for(rng):
        lens = micro_lengths(T, micro, rng)
        return [ragged_batch(data, B, T, rng, valid_len=L) for L in lens]

    # the damage, measured on the gradient rather than on the reported number: same
    # initialisation, same data, one accumulation step each way
    grads = {}
    for mode in ("correct", "naive"):
        model = make_model(vocab_size, n_layer=4, n_head=4, n_embd=192, block_size=T)
        rng = np.random.default_rng(SEED)
        bs = batches_for(rng)
        model.zero_grad(set_to_none=True)
        _accum_step(model, bs, mode)
        grads[mode] = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    gc, gn = grads["correct"], grads["naive"]
    cos = float(F.cosine_similarity(gc, gn, dim=0))
    ratio = float(gn.norm() / gc.norm())
    rel = float((gn - gc).norm() / gc.norm())

    results = {}
    for mode in ("correct", "naive"):
        model = make_model(vocab_size, n_layer=4, n_head=4, n_embd=192, block_size=T)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
        rng = np.random.default_rng(SEED)                 # identical data order in both runs
        curve, counts, honest = [], [], []
        for _ in range(steps):
            bs = batches_for(rng)
            opt.zero_grad(set_to_none=True)
            curve.append(_accum_step(model, bs, mode))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            counts.append([n for _, _, n in bs])
            # what the loss would have read on this step if it had been normalised properly
            with torch.no_grad():
                tot = sum(n for _, _, n in bs)
                s_ = sum(float(F.cross_entropy(model(x, y)[0].view(-1, vocab_size), y.view(-1),
                                               ignore_index=-1, reduction="sum")) for x, y, _ in bs)
                honest.append(s_ / tot)
        results[mode] = {"curve": curve, "token_counts": counts, "honest_curve": honest}

    # the static demonstration, on the lesson's own numbers
    L, n = [2.0, 2.0, 5.0], [4, 4, 2]
    correct = sum(li * ni for li, ni in zip(L, n)) / sum(n)
    naive = sum(L) / len(L)
    gap_pct = (naive - correct) / correct * 100
    n_eq = [4, 4, 4]
    correct_eq = sum(li * ni for li, ni in zip(L, n_eq)) / sum(n_eq)
    gap_eq = (naive - correct_eq) / correct_eq * 100

    c, na = results["correct"]["curve"], results["naive"]["curve"]
    step_gaps = [(b - a) / a * 100 for a, b in zip(results["naive"]["honest_curve"], na)]
    worst = max(range(len(step_gaps)), key=lambda i: abs(step_gaps[i]))
    print(f"\n[part 3] gradient accumulation, {steps} steps, {micro} ragged micro-batches each")
    print(f"  lesson's static case  correct {correct:.4f}  naive {naive:.4f}  gap {gap_pct:.1f}%")
    print(f"  same losses, equal counts        gap {gap_eq:.1f}%   <- how it hid")
    print(f"  one step, gradient damage: cos {cos:.6f}, |naive|/|correct| {ratio:.4f}, rel diff {rel:.4f}")
    print(f"  naive reported loss vs the honest number, same weights:")
    print(f"    worst step {worst}: reported {na[worst]:.4f} against {results['naive']['honest_curve'][worst]:.4f}"
          f"  ({step_gaps[worst]:+.1f}%),  token counts {results['naive']['token_counts'][worst]}")
    print(f"    mean |gap| over the run {np.mean(np.abs(step_gaps)):.2f}%, max {max(map(abs, step_gaps)):.2f}%")
    print(f"  final loss   correct {c[-1]:.4f}   naive {na[-1]:.4f}")
    results["static"] = {"losses": L, "tokens": n, "correct": correct, "naive": naive,
                         "gap_pct": gap_pct, "gap_pct_equal_counts": gap_eq}
    results["gradient_damage"] = {"cosine": cos, "norm_ratio": ratio, "rel_diff": rel}
    results["step_gaps_pct"] = step_gaps
    results["worst_step"] = {"step": worst, "gap_pct": step_gaps[worst],
                             "reported": na[worst], "honest": results["naive"]["honest_curve"][worst],
                             "token_counts": results["naive"]["token_counts"][worst]}
    results["mean_abs_gap_pct"] = float(np.mean(np.abs(step_gaps)))
    return results


# --------------------------------------------------------------------------------------
# part 4 — log the grad norm every step, find one where it moved before the loss did
# --------------------------------------------------------------------------------------

def _run_with_norms(data, vocab_size, steps, B, T, lr, clip):
    """One run, logging the pre-clip grad norm every step. clip=None disables clipping
    but still measures the norm, so both runs see the same trace definition."""
    model = make_model(vocab_size, block_size=T)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(SEED)                     # same init, same data both ways
    losses, norms, clipped = [], [], 0
    for _ in range(steps):
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        # clip_grad_norm_ returns the norm *before* clipping, and train.py:309 throws it
        # away. Measured before the update is why it can warn first.
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  clip if clip is not None else float("inf")))
        opt.step()
        losses.append(loss.item())
        norms.append(gn)
        if clip is not None and gn > clip:
            clipped += 1
    return losses, norms, clipped


def _find_leads(losses, norms, steps, W=20, spike_mult=3.0, horizon=20):
    """A spike is a norm above spike_mult x its own trailing median. A reaction is the
    first later step whose loss clears the run's own step-to-step noise by 2 sigma --
    without that bar, any upward wiggle counts and the 'lead' means nothing."""
    leads = []
    for i in range(W, steps - horizon):
        base_n = float(np.median(norms[i - W:i]))
        if norms[i] < spike_mult * base_n:
            continue
        ref = max(losses[i - 3:i + 1])
        sigma = float(np.std(np.diff(losses[i - W:i])))
        for j in range(i + 1, min(i + horizon, steps)):
            if losses[j] > ref + 2 * sigma:
                leads.append({"norm_step": i, "norm": norms[i], "baseline_norm": base_n,
                              "loss_step": j, "lead": j - i, "sigma": sigma,
                              "loss_before": ref, "loss_at": losses[j],
                              "rise_in_sigma": (losses[j] - ref) / sigma})
                break
    return leads


def part4_gradnorm(data, vocab_size, steps=600, B=16, T=256, lr=6e-3, clip=1.0):
    """lr is deliberately above nanoGPT's shakespeare_char default (1e-3), so the run
    produces real spikes rather than a manufactured one.

    Both ways, from an identical initialisation and data order, which is the comparison
    section 12's widget draws. What this box actually produced is worth stating plainly
    rather than predicting: the cap left the run *better* on both final and worst loss,
    and the norm-before-loss lead showed up in the capped run, not the uncapped one. The
    uncapped run is noisier step to step, so the 2-sigma reaction bar sits higher there
    and a spike has more to clear -- a likely reason for the difference, not a measured
    one.
    """
    out = {}
    for name, c in (("clipped", clip), ("unclipped", None)):
        losses, norms, nclip = _run_with_norms(data, vocab_size, steps, B, T, lr, c)
        leads = _find_leads(losses, norms, steps)
        out[name] = {"losses": losses, "norms": norms, "clipped_steps": nclip,
                     "leads": leads, "median_norm": float(np.median(norms)),
                     "max_norm": max(norms), "final_loss": losses[-1],
                     "max_loss_after_100": max(losses[100:])}

    print(f"\n[part 4] grad norm, {steps} steps at lr {lr}, same init and data both ways")
    for name in ("clipped", "unclipped"):
        d = out[name]
        tag = f"clip {clip}" if name == "clipped" else "no cap"
        print(f"  {tag:9s} median norm {d['median_norm']:.4f}  max {d['max_norm']:>8.2f}  "
              f"final loss {d['final_loss']:.4f}  worst loss after step 100 {d['max_loss_after_100']:.4f}  "
              f"spikes leading the loss: {len(d['leads'])}")
    print(f"  steps clipped: {out['clipped']['clipped_steps']}/{steps}")
    lead_src = out["unclipped"]["leads"] or out["clipped"]["leads"]
    if lead_src:
        first = lead_src[0]
        best = max(lead_src, key=lambda d: d["lead"])
        print(f"  norm rose to {first['norm']:.3f} at step {first['norm_step']} "
              f"({first['norm']/first['baseline_norm']:.1f}x its trailing median); the loss "
              f"cleared 2 sigma at step {first['loss_step']}, {first['lead']} steps later")
        print(f"  longest lead {best['lead']} steps (norm {best['norm_step']} -> loss {best['loss_step']}, "
              f"a {best['rise_in_sigma']:.1f} sigma rise)")
        print(f"  mean lead over {len(lead_src)} spikes: {np.mean([d['lead'] for d in lead_src]):.2f} steps")
    else:
        print("  no spike cleared the noise bar in either run")
    out["lr"] = lr
    out["steps"] = steps
    out["clip"] = clip
    return out


# --------------------------------------------------------------------------------------
# part 5 — compute your own MFU
# --------------------------------------------------------------------------------------

def measure_peak_flops(dtype=torch.float32, n=4096, iters=30):
    """The best large-matmul rate this box actually sustains. Not a datasheet peak:
    it is what the same code path reaches, so it is the friendlier denominator."""
    if DEVICE != "cuda":
        return None
    a = torch.randn(n, n, device=DEVICE, dtype=dtype)
    b = torch.randn(n, n, device=DEVICE, dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * iters / (time.time() - t0)


def theoretical_fp32_peak():
    """Derived, not quoted: SMs x 64 FP32 lanes per SM x 2 flops per FMA x the max SM
    clock nvidia-smi reports. Returns None if that clock is unavailable, rather than
    guessing a datasheet figure this run cannot verify."""
    if DEVICE != "cuda":
        return None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        mhz = float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None, None
    p = torch.cuda.get_device_properties(0)
    return p.multi_processor_count * 64 * 2 * mhz * 1e6, mhz


def part5_mfu(data, vocab_size, steps=40, B=16, T=256):
    model = make_model(vocab_size, block_size=T)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(SEED)
    for _ in range(5):                                    # warm up, do not time these
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        model(x, y)[1].backward()
        opt.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        model(x, y)[1].backward()
        opt.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps

    N = sum(p.numel() for p in model.parameters())
    tps = B * T / dt
    achieved = 6 * N * tps
    peak_fp32 = measure_peak_flops(torch.float32)
    peak_fp16 = measure_peak_flops(torch.float16)
    theo, mhz = theoretical_fp32_peak()
    mfu = achieved / theo if theo else None               # MFU proper: against the hardware
    util = achieved / peak_fp32 if peak_fp32 else None    # against what matmul here reaches

    nano = float(model.estimate_mfu(B, dt))               # PaLM formula, A100 denominator
    cfg = model.config
    palm_per_token = 6 * N + 12 * cfg.n_layer * cfg.n_head * (cfg.n_embd // cfg.n_head) * cfg.block_size

    print(f"\n[part 5] MFU  ({N/1e6:.2f}M params, B={B} T={T}, fp32)")
    print(f"  {dt*1000:.1f} ms/step, {tps:,.0f} tokens/s")
    print(f"  6N per token  {6*N/1e6:.1f} MFLOP  ->  achieved {achieved/1e12:.4f} TFLOP/s")
    print(f"  nanoGPT's 6N + 12*L*H*Q*T  {palm_per_token/1e6:.1f} MFLOP per token ({palm_per_token/(6*N):.2f}x)")
    if theo:
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"  theoretical fp32 peak: {sm} SM x 64 x 2 x {mhz:.0f} MHz = {theo/1e12:.2f} TFLOP/s")
        print(f"  MFU = achieved / theoretical peak                  {mfu*100:.2f}%")
    print(f"  measured matmul ceiling   fp32 {peak_fp32/1e12:.2f}   fp16 {peak_fp16/1e12:.2f} TFLOP/s")
    print(f"  fraction of the fp32 matmul ceiling                {util*100:.2f}%")
    print(f"  nanoGPT estimate_mfu, A100 312 TFLOP/s denominator  {nano*100:.2f}%  <- wrong card")
    return {"n_params": N, "dt_s": dt, "tokens_per_s": tps,
            "achieved_6n_flops": achieved, "peak_fp32_measured": peak_fp32,
            "peak_fp16_measured": peak_fp16, "theoretical_fp32_peak": theo, "sm_clock_mhz": mhz,
            "mfu_vs_theoretical": mfu, "frac_of_measured_matmul": util,
            "nanogpt_estimate_mfu": nano, "palm_flops_per_token": palm_per_token,
            "six_n_per_token": 6 * N,
            "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}


# --------------------------------------------------------------------------------------
# part 6 — 0.1 by hand in fp32, bf16 and fp8 E4M3
# --------------------------------------------------------------------------------------

def encode_float(x, ebits, mbits):
    """Encode by hand: sign, biased exponent, mantissa. Round-half-to-even, subnormals
    handled, no library call anywhere in here."""
    bias = (1 << (ebits - 1)) - 1
    if x == 0:
        return 0, 0, 0
    s = 1 if x < 0 else 0
    a = abs(x)
    e = math.floor(math.log2(a))
    while a < 2.0 ** e:
        e -= 1
    while a >= 2.0 ** (e + 1):
        e += 1
    emin = 1 - bias

    def rne(v):
        f = math.floor(v)
        d = v - f
        if d > 0.5:
            return f + 1
        if d < 0.5:
            return f
        return f + 1 if f % 2 else f

    if e >= emin:
        m = rne((a / 2.0 ** e - 1) * (1 << mbits))
        if m == (1 << mbits):
            m = 0
            e += 1
        return s, e + bias, m
    return s, 0, rne(a / 2.0 ** emin * (1 << mbits))


def decode_float(s, E, M, ebits, mbits):
    bias = (1 << (ebits - 1)) - 1
    sign = -1.0 if s else 1.0
    if E == 0:
        return sign * (M / (1 << mbits)) * 2.0 ** (1 - bias)
    return sign * (1 + M / (1 << mbits)) * 2.0 ** (E - bias)


def bits(v, w):
    return format(v, f"0{w}b")


def part6_floats(value=0.1):
    formats = [("fp32", 8, 23), ("bf16", 8, 7), ("fp8 E4M3", 4, 3)]
    rows = []
    print(f"\n[part 6] {value} written out by hand")
    for name, eb, mb in formats:
        s, E, M = encode_float(value, eb, mb)
        back = decode_float(s, E, M, eb, mb)
        err = abs(back - value) / abs(value) * 100
        pattern = f"{s} {bits(E, eb)} {bits(M, mb)}"
        bias = (1 << (eb - 1)) - 1
        rows.append({"format": name, "ebits": eb, "mbits": mb, "sign": s, "E": E, "M": M,
                     "bits": pattern, "value": back, "rel_err_pct": err,
                     "exponent": E - bias, "scale": 2.0 ** (E - bias),
                     "mantissa_fraction": M / (1 << mb)})
        print(f"  {name:9s} {pattern:38s} = (1 + {M}/{1 << mb}) x 2^{E - bias:<4d} = {back!r}")
        print(f"  {'':9s} {'':38s}   off by {err:.4f}% against {value}")
    return rows


def cross_check_part6(rows, value=0.1):
    """Hand encoder against the machine, so the by-hand answer is checked rather than
    asserted. fp8 E4M3 is torch's own dtype; fp32 is struct's."""
    fp32_ref = struct.unpack("<I", struct.pack("<f", value))[0]
    fp32 = next(r for r in rows if r["format"] == "fp32")
    mine = (fp32["sign"] << 31) | (fp32["E"] << 23) | fp32["M"]
    assert mine == fp32_ref, f"fp32 bits {mine:#010x} != {fp32_ref:#010x}"

    bf16 = next(r for r in rows if r["format"] == "bf16")
    t_bf16 = float(torch.tensor([value], dtype=torch.float32).to(torch.bfloat16))
    assert bf16["value"] == t_bf16, f"bf16 {bf16['value']} != torch {t_bf16}"

    f8 = next(r for r in rows if r["format"] == "fp8 E4M3")
    t_f8 = float(torch.tensor([value], dtype=torch.float32).to(torch.float8_e4m3fn))
    assert f8["value"] == t_f8, f"fp8 e4m3 {f8['value']} != torch {t_f8}"
    return {"fp32_bits_hex": f"{fp32_ref:#010x}", "torch_bf16": t_bf16, "torch_fp8_e4m3": t_f8}


# --------------------------------------------------------------------------------------

def selfcheck():
    """One runnable check per piece of non-trivial logic."""
    rows = part6_floats()
    xr = cross_check_part6(rows)
    print(f"  cross-check ok: {xr}")

    # the encoder on values whose bits are known independently
    assert encode_float(1.0, 8, 23) == (0, 127, 0)
    assert encode_float(-2.0, 8, 23) == (1, 128, 0)
    assert encode_float(0.0, 8, 23) == (0, 0, 0)
    # a value below the fp8 E4M3 normal floor lands in the subnormal branch
    s, E, M = encode_float(2.0 ** -9, 4, 3)
    assert E == 0 and M > 0, (s, E, M)

    # the two normalisers: unequal counts disagree by the lesson's 15.4%, equal counts agree
    L, n = [2.0, 2.0, 5.0], [4, 4, 2]
    correct = sum(a * b for a, b in zip(L, n)) / sum(n)
    naive = sum(L) / 3
    assert abs(correct - 2.6) < 1e-12 and abs(naive - 3.0) < 1e-12
    assert abs((naive - correct) / correct * 100 - 15.3846) < 1e-3
    eq = sum(a * 4 for a in L) / 12
    assert abs(eq - naive) < 1e-12, "equal token counts must make the two routes agree"

    # the lesson's own two-weight chain, which part 2 generalises
    x, w1, w2, t = 2.0, 3.0, 4.0, 20.0
    y = w2 * (w1 * x)
    gw1 = 2 * (y - t) * w2 * x
    assert gw1 == 64.0
    h = 1e-6
    num = (((w2 * ((w1 + h) * x)) - t) ** 2 - (y - t) ** 2) / h
    assert abs(num - 64.0) < 1e-3, num
    print("\nselfcheck: all asserts passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all",
                    choices=["all", "selfcheck", "1", "2", "3", "4", "5", "6"])
    ap.add_argument("--steps4", type=int, default=600)
    ap.add_argument("--steps3", type=int, default=60)
    args = ap.parse_args()

    if args.part == "selfcheck":
        selfcheck()
        return

    ensure_nanogpt()
    torch.manual_seed(SEED)
    res = {}
    need_data = args.part in ("all", "1", "2", "3", "4", "5")
    data = vocab = None
    if need_data:
        data, vocab = ensure_data()

    if args.part in ("all", "1"):
        res["part1_shapes"] = part1_shapes(data, vocab)
    if args.part in ("all", "2"):
        res["part2_gradcheck"] = part2_gradcheck(data, vocab)
    if args.part in ("all", "3"):
        res["part3_accumulation"] = part3_accumulation(data, vocab, steps=args.steps3)
    if args.part in ("all", "4"):
        res["part4_gradnorm"] = part4_gradnorm(data, vocab, steps=args.steps4)
    if args.part in ("all", "5"):
        res["part5_mfu"] = part5_mfu(data, vocab)
    if args.part in ("all", "6"):
        rows = part6_floats()
        res["part6_floats"] = {"rows": rows, "cross_check": cross_check_part6(rows)}

    if args.part == "all":
        OUT.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
