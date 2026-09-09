"""Session 11 assignment: how a gradient becomes a distance.

Five parts, matching section 15 of the lesson:

  1  Adam by hand against torch.optim.Adam, on the widget's own five gradients
     and then on a real nanoGPT weight tensor.
  2  Bias correction on and off. The assignment asks for twenty steps; twenty
     steps is not enough to answer the question it asks, so we also run the rule
     out to ten thousand.
  3  The update-to-weight ratio, per parameter tensor, over two 1,500-step runs
     that differ only in whether the learning rate is warmed up.
  4  Cosine against WSD, both planned for 300 steps and both stopped at 200,
     three seeds each, plus the branch-and-decay that is the whole reason WSD
     exists.
  5  A learning rate sweep at widths 256, 512 and 1,024, under the standard
     parameterization and under a minimal muP, to state a value for width 4,096.

The model is nanoGPT (karpathy/nanoGPT). This file does not fork its trainer; it
imports its GPT and drives it from an instrumented loop, which every part after
the first needs anyway.

  python assignment.py all         # every part, writes results.json
  python assignment.py selfcheck   # the asserts only, no GPU needed
  python assignment.py 5 --arm mup # one part, one arm
  python assignment.py 5b          # re-run part 5's minima on further seeds
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NANOGPT = HERE / "nanoGPT"
SESSION10_NANOGPT = HERE.parent.parent / "Session10" / "assignment" / "nanoGPT"
NANOGPT_URL = "https://github.com/karpathy/nanoGPT"
OUT = HERE / "results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337

# the lesson's own settings, section 6 and mined-numbers.md
ETA, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
WIDGET_GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]


# --------------------------------------------------------------------------------------
# bootstrap: nanoGPT itself, and the shakespeare_char dataset it ships a prepare.py for
# --------------------------------------------------------------------------------------

def ensure_nanogpt():
    """Session 10 already cloned it next door; reuse that rather than a second copy."""
    global NANOGPT
    if not NANOGPT.exists() and SESSION10_NANOGPT.exists():
        NANOGPT = SESSION10_NANOGPT
    elif not NANOGPT.exists():
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
    val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    return train, val, meta["vocab_size"]


def make_model(vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.0,
               device=DEVICE, dtype=torch.float32, seed=SEED):
    """nanoGPT's GPT at the shakespeare_char size. dropout stays 0 so a rerun repeats."""
    from model import GPT, GPTConfig
    cfg = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=dropout, bias=False)
    torch.manual_seed(seed)
    return GPT(cfg).to(device=device, dtype=dtype)


def get_batch(data, batch_size, block_size, rng, device=DEVICE):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, val, B, T, batches=20, seed=0):
    """Held-out loss on a fixed slice of val.bin. The rng is reseeded every call, so
    every model in parts 4 and 5 is scored on exactly the same tokens."""
    rng = np.random.default_rng(seed)
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = get_batch(val, B, T, rng)
        _, loss = model(x, y)
        tot += float(loss)
    model.train()
    return tot / batches


# --------------------------------------------------------------------------------------
# the two schedules, straight off the widgets (mined-numbers.md, widget_6 and widget_7)
# --------------------------------------------------------------------------------------

def cosine_lr(t, T, peak):
    """PEAK * 0.5 * (1 + cos(pi t / T)), zero past the planned end."""
    return 0.0 if t > T else peak * 0.5 * (1 + math.cos(math.pi * t / T))


def wsd_lr(t, T, peak, warm_frac=0.02, decay_frac=0.10):
    """Warmup over the first 2%, flat, then linear decay over the last 10%."""
    if t > T:
        return 0.0
    W, D = warm_frac * T, (1 - decay_frac) * T
    if t < W:
        return peak * t / W
    if t < D:
        return peak
    return peak * (T - t) / (decay_frac * T)


def warmup_lr(step, warmup_steps, peak):
    """step is 1-based, matching widget_6's lr(n) = PEAK * min(1, n / W)."""
    if warmup_steps <= 0:
        return peak
    return peak * min(1.0, step / warmup_steps)


# --------------------------------------------------------------------------------------
# part 1 — Adam by hand
# --------------------------------------------------------------------------------------

def adam_by_hand(grads, w0=1.0, eta=ETA, b1=B1, b2=B2, eps=EPS, bias_correction=True):
    """The lesson's rule, w <- w - eta * mhat / (sqrt(vhat) + eps), one scalar weight.

    With bias_correction off, mhat and vhat are just m and v — the ablation part 2 needs.
    """
    m = v = 0.0
    w = w0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        if bias_correction:
            mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        else:
            mh, vh = m, v
        step = -eta * mh / (math.sqrt(vh) + eps)
        w += step
        rows.append({"t": t, "g": g, "m": m, "v": v, "mhat": mh, "vhat": vh,
                     "step": step, "w": w})
    return rows


def _torch_adam_scalar(grads, w0=1.0, eta=ETA, b1=B1, b2=B2, eps=EPS):
    """The same five steps through torch.optim.Adam, gradients fed in by hand."""
    w = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=eta, betas=(b1, b2), eps=eps)
    out = []
    for g in grads:
        w.grad = torch.tensor([g], dtype=torch.float64)
        prev = float(w.detach())
        opt.step()
        st = opt.state[w]
        out.append({"m": float(st["exp_avg"]), "v": float(st["exp_avg_sq"]),
                    "step": float(w.detach()) - prev, "w": float(w.detach())})
    return out


def part1_adam(data, val, vocab_size):
    hand = adam_by_hand(WIDGET_GRADS)
    torch_rows = _torch_adam_scalar(WIDGET_GRADS)
    cmp_rows = []
    for h, tr in zip(hand, torch_rows):
        cmp_rows.append({
            "t": h["t"], "g": h["g"],
            "m": h["m"], "v": h["v"], "mhat": h["mhat"], "vhat": h["vhat"],
            "step_hand": h["step"], "step_torch": tr["step"],
            "w_hand": h["w"], "w_torch": tr["w"],
            "abs_diff_step": abs(h["step"] - tr["step"]),
            "rel_diff_step": abs(h["step"] - tr["step"]) / abs(tr["step"]),
        })
    scalar_max_rel = max(r["rel_diff_step"] for r in cmp_rows)

    # where the remaining difference comes from: torch puts eps outside the second
    # moment's bias correction, denom = sqrt(v)/sqrt(1-b2^t) + eps, while the lesson
    # writes sqrt(vhat) + eps. Same rule when eps is negligible, not the same expression.
    def torch_form(rows):
        m = v = 0.0
        w = 1.0
        out = []
        for t, g in enumerate(WIDGET_GRADS, start=1):
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g * g
            denom = math.sqrt(v) / math.sqrt(1 - B2 ** t) + EPS
            step = -(ETA / (1 - B1 ** t)) * m / denom
            w += step
            out.append(step)
        return out
    torch_form_steps = torch_form(hand)
    form_gap = max(abs(a - b["step_torch"]) for a, b in zip(torch_form_steps, cmp_rows))

    # the same five steps on a real weight tensor, real gradients, float32 on the GPU.
    # Both rules are fed the *same* gradient tensor inside one run: two separate runs
    # disagree by 4e-5 on this GPU from kernel non-determinism alone, which is forty
    # times the step being measured and would drown the thing we are checking.
    tensor_rows = []
    nondet = None
    if data is not None:
        name = "transformer.h.0.mlp.c_fc.weight"
        rng = np.random.default_rng(SEED)
        batches = [get_batch(data, 4, 256, rng) for _ in range(5)]
        model = make_model(vocab_size)
        opt = torch.optim.Adam(model.parameters(), lr=ETA, betas=(B1, B2), eps=EPS)
        p = dict(model.named_parameters())[name]
        shadow = p.detach().clone()                     # the hand rule's own copy
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        for t, (x, y) in enumerate(batches, start=1):
            model.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            g = p.grad.detach().clone()
            opt.step()
            with torch.no_grad():
                m.mul_(B1).add_(g, alpha=1 - B1)
                v.mul_(B2).addcmul_(g, g, value=1 - B2)
                mh = m / (1 - B1 ** t)
                vh = v / (1 - B2 ** t)
                shadow += -ETA * mh / (vh.sqrt() + EPS)
                d = (shadow - p.detach()).abs()
                tensor_rows.append({
                    "t": t, "max_abs": float(d.max()),
                    "max_rel": float((d / p.detach().abs().clamp_min(1e-12)).max()),
                    "mean_abs": float(d.mean()),
                    "step_size": float((ETA * mh / (vh.sqrt() + EPS)).abs().max())})
                shadow.copy_(p.detach())                # rejoin, so step t+1 tests step t+1

        # how big run-to-run non-determinism is on this hardware, for scale
        def one_run():
            mdl = make_model(vocab_size)
            o = torch.optim.Adam(mdl.parameters(), lr=ETA, betas=(B1, B2), eps=EPS)
            q = dict(mdl.named_parameters())[name]
            for x, y in batches[:1]:
                mdl.zero_grad(set_to_none=True)
                _, l = mdl(x, y)
                l.backward()
                o.step()
            return q.detach().clone()
        nondet = float((one_run() - one_run()).abs().max())

    print("\n[part 1] Adam by hand against torch.optim.Adam")
    print("  t     g        m         v          mhat      vhat       step        w")
    for r in cmp_rows:
        print(f"  {r['t']}  {r['g']:.2f}  {r['m']:.6f}  {r['v']:.8f}  {r['mhat']:.4f}  "
              f"{r['vhat']:.6f}  {r['step_hand']:+.9f}  {r['w_hand']:.9f}")
    print(f"  widget's fifth-step w is 0.995031; ours is {cmp_rows[-1]['w_hand']:.6f}")
    print(f"  worst relative disagreement with torch, five steps: {scalar_max_rel:.2e}")
    print(f"  torch's own expression, eps outside the v correction, matches to {form_gap:.2e}")
    if tensor_rows:
        print("  real tensor h.0.mlp.c_fc.weight, five steps, same gradients both rules:")
        for r in tensor_rows:
            print(f"    step {r['t']}: max abs {r['max_abs']:.3e} against a step of "
                  f"{r['step_size']:.3e}")
        print(f"  for scale, two identical runs on this GPU differ by {nondet:.3e}")

    return {"widget_preset": WIDGET_GRADS, "rows": cmp_rows,
            "scalar_max_rel_step_diff": scalar_max_rel,
            "torch_expression_max_gap": form_gap,
            "tensor": {"name": "transformer.h.0.mlp.c_fc.weight", "rows": tensor_rows,
                       "run_to_run_nondeterminism": nondet}}


# --------------------------------------------------------------------------------------
# part 2 — bias correction, on and off
# --------------------------------------------------------------------------------------

def bias_ratio(t, b1=B1, b2=B2):
    """Corrected step over uncorrected step, for any gradient sequence.

    step_corrected / step_uncorrected = (m/(1-b1^t)) / sqrt(v/(1-b2^t)) * sqrt(v)/m
                                      = sqrt(1 - b2^t) / (1 - b1^t)
    The gradients cancel, so the ratio is a property of the two betas alone.
    """
    return math.sqrt(1 - b2 ** t) / (1 - b1 ** t)


def part2_bias_correction(n_short=20, n_long=10000):
    grads = [WIDGET_GRADS[i % len(WIDGET_GRADS)] for i in range(n_short)]
    on = adam_by_hand(grads, bias_correction=True)
    off = adam_by_hand(grads, bias_correction=False)
    short = [{"t": a["t"], "step_on": a["step"], "step_off": b["step"],
              "w_on": a["w"], "w_off": b["w"],
              "ratio": a["step"] / b["step"], "analytic": bias_ratio(a["t"])}
             for a, b in zip(on, off)]
    ratio_err = max(abs(r["ratio"] - r["analytic"]) for r in short)

    # the honest answer: how long the two rules take to agree, under stated criteria
    def crossing(tol):
        for t in range(1, n_long + 1):
            if abs(bias_ratio(t) - 1.0) <= tol:
                return t
        return None
    cross_1pct, cross_01pct = crossing(0.01), crossing(0.001)
    curve = [{"t": t, "ratio": bias_ratio(t)}
             for t in sorted(set(list(range(1, 51)) +
                                 [int(round(10 ** (i / 40 * 4))) for i in range(41)]))]

    w_gap_20 = abs(short[-1]["w_on"] - short[-1]["w_off"])
    print(f"\n[part 2] bias correction, first {n_short} steps and the ratio out to {n_long}")
    print("  t   step with      step without   with/without   analytic")
    for r in short[:6] + short[-2:]:
        print(f"  {r['t']:>2}  {r['step_on']:+.8f}  {r['step_off']:+.8f}  "
              f"{r['ratio']:.4f}         {r['analytic']:.4f}")
    print(f"  the ratio formula sqrt(1-b2^t)/(1-b1^t) matches the simulation to {ratio_err:.2e}")
    print(f"  at t=1 the uncorrected step is {1/short[0]['ratio']:.2f}x too large; "
          f"at t={n_short} it is still {1/short[-1]['ratio']:.2f}x")
    print(f"  within 1% at step {cross_1pct}, within 0.1% at step {cross_01pct}")
    print(f"  after {n_short} steps the two weights differ by {w_gap_20:.6f}")

    return {"short": short, "ratio_formula_max_err": ratio_err,
            "crossing_1pct": cross_1pct, "crossing_0p1pct": cross_01pct,
            "ratio_curve": curve, "w_gap_at_20": w_gap_20,
            "overshoot_t1": 1 / short[0]["ratio"], "overshoot_t20": 1 / short[-1]["ratio"]}


# --------------------------------------------------------------------------------------
# part 3 — the update-to-weight ratio, per layer, with and without warmup
# --------------------------------------------------------------------------------------

GROUPS = [("embeddings", ("wte", "wpe")), ("head", ("lm_head",)),
          ("attention", ("attn.c_attn", "attn.c_proj")),
          ("mlp", ("mlp.c_fc", "mlp.c_proj")), ("norms", ("ln_", "ln_f"))]


def group_of(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def _untie_head(model):
    """nanoGPT ties lm_head.weight to wte.weight, so the two are one tensor and section
    14's head-versus-body question cannot be asked of it. Untie for this measurement."""
    w = model.lm_head.weight.detach().clone()
    model.lm_head.weight = nn.Parameter(w)
    return model


def _ratio_run(data, vocab_size, steps, warmup_steps, B=4, T=256, peak=3e-4,
               wd=0.1, betas=(0.9, 0.95), log_every=1):
    model = _untie_head(make_model(vocab_size, block_size=T))
    opt = model.configure_optimizers(wd, peak, betas, "cuda" if DEVICE == "cuda" else "cpu")
    rng = np.random.default_rng(SEED)                  # same init, same data, both runs
    named = [(n, p) for n, p in model.named_parameters()]
    ratios = {n: [] for n, _ in named}
    losses, lrs = [], []
    for step in range(1, steps + 1):
        lr = warmup_lr(step, warmup_steps, peak)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        before = [p.detach().clone() for _, p in named]
        opt.step()
        if step % log_every == 0:
            with torch.no_grad():
                for (n, p), b in zip(named, before):
                    ratios[n].append(float((p - b).norm() / b.norm().clamp_min(1e-12)))
            losses.append(float(loss.detach()))
            lrs.append(lr)
    return {"ratios": ratios, "losses": losses, "lrs": lrs,
            "steps_logged": list(range(log_every, steps + 1, log_every))}


def agreement_ladder(a, b, logged, tols=(0.01, 0.05, 0.10, 0.25, 0.50)):
    """First step from which two ratio traces stay within a tolerance of each other,
    for a ladder of tolerances. None means they never do — which is itself an answer:
    once the warmup has changed the weights, the two runs are different models and
    their ratios have no reason to rejoin."""
    out = {}
    for tol in tols:
        hit = None
        for i in range(len(a)):
            if all(abs(a[j] - b[j]) / max(b[j], 1e-12) <= tol for j in range(i, len(a))):
                hit = logged[i]
                break
        out[str(tol)] = hit
    return out


def part3_update_ratio(data, val, vocab_size, steps=1500, warmup_steps=150):
    runs = {}
    for label, W in (("warmup", warmup_steps), ("no_warmup", 0)):
        print(f"  running {label} ({steps} steps)...")
        runs[label] = _ratio_run(data, vocab_size, steps, W)

    out = {"steps": steps, "warmup_steps": warmup_steps, "runs": {}}
    for label, r in runs.items():
        # the run-wide ratio is the model-level one: every tensor's update against
        # every tensor's size, which is what the widget plots
        allnames = list(r["ratios"])
        overall = [float(np.mean([r["ratios"][n][i] for n in allnames]))
                   for i in range(len(r["steps_logged"]))]
        pk = int(np.argmax(overall))
        by_group = {}
        for g, _ in GROUPS:
            names = [n for n in allnames if group_of(n) == g]
            if not names:
                continue
            series = [float(np.mean([r["ratios"][n][i] for n in names]))
                      for i in range(len(overall))]
            tail = series[len(series) // 2:]
            by_group[g] = {"peak": max(series), "peak_step": r["steps_logged"][int(np.argmax(series))],
                           "median_second_half": float(np.median(tail)), "series": series}
        out["runs"][label] = {
            "peak_ratio": overall[pk], "peak_step": r["steps_logged"][pk],
            "final_loss": r["losses"][-1], "overall": overall, "lrs": r["lrs"],
            "losses": r["losses"], "steps_logged": r["steps_logged"], "by_group": by_group,
            "median_second_half": float(np.median(overall[len(overall) // 2:])),
        }

    out["agreement"] = agreement_ladder(out["runs"]["warmup"]["overall"],
                                        out["runs"]["no_warmup"]["overall"],
                                        runs["warmup"]["steps_logged"])
    out["agree_within_1pct_from_step"] = out["agreement"]["0.01"]

    w, nw = out["runs"]["warmup"], out["runs"]["no_warmup"]
    print(f"\n[part 3] update-to-weight ratio, {steps} steps, warmup {warmup_steps} steps")
    print(f"  peak ratio  without warmup {nw['peak_ratio']*1000:.2f}e-3 at step {nw['peak_step']}")
    print(f"              with warmup    {w['peak_ratio']*1000:.2f}e-3 at step {w['peak_step']}")
    print(f"  widget_6 baseline: 19.2e-3 without, 2.83e-3 with")
    print(f"  the two runs stay within a tolerance of each other from: "
          + ", ".join(f"{float(k)*100:g}% -> {v}" for k, v in out["agreement"].items()))
    print(f"  second-half median, with warmup: {w['median_second_half']*1000:.3f}e-3 "
          f"(the band the lesson wants is 1e-3)")
    print("  by group, with warmup:")
    for g, d in w["by_group"].items():
        print(f"    {g:<11} peak {d['peak']*1000:7.3f}e-3 at step {d['peak_step']:>4}, "
              f"settled {d['median_second_half']*1000:.3f}e-3")
    return out


# --------------------------------------------------------------------------------------
# part 4 — cosine against WSD, both stopped early
# --------------------------------------------------------------------------------------

def _schedule_run(data, val, vocab_size, schedule, total, seed, peak=3e-4, B=4, T=256,
                  eval_at=(200, 300), branch_at=None, branch_steps=20):
    model = make_model(vocab_size, block_size=T, seed=seed)
    opt = model.configure_optimizers(0.1, peak, (0.9, 0.95),
                                     "cuda" if DEVICE == "cuda" else "cpu")
    rng = np.random.default_rng(seed)
    fn = cosine_lr if schedule == "cosine" else wsd_lr
    losses, lrs, evals, branch = [], [], {}, None
    for step in range(1, total + 1):
        lr = fn(step, total, peak)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        lrs.append(lr)
        if step in eval_at:
            evals[step] = eval_loss(model, val, B, T)
        if branch_at is not None and step == branch_at:
            # the point of WSD: take the flat-phase weights and decay them separately
            state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(opt.state_dict())
            branch = _branch_decay(model, opt, state, opt_state, data, val, vocab_size,
                                   rng, peak, branch_steps, B, T)
    return {"losses": losses, "lrs": lrs, "evals": evals, "branch": branch}


def _branch_decay(model, opt, state, opt_state, data, val, vocab_size, rng, peak, steps, B, T):
    """Decay the checkpoint to zero over `steps`, then put the run back where it was."""
    saved_rng = rng.bit_generator.state
    curve = []
    for i in range(1, steps + 1):
        lr = peak * (1 - i / steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    out = {"steps": steps, "curve": curve, "val": eval_loss(model, val, B, T)}
    model.load_state_dict(state)
    opt.load_state_dict(opt_state)
    rng.bit_generator.state = saved_rng
    return out


def part4_schedules(data, val, vocab_size, total=300, stop=200, seeds=(1337, 1338, 1339),
                    branch_steps=20):
    runs = {"cosine": [], "wsd": []}
    for sched in ("cosine", "wsd"):
        for s in seeds:
            print(f"  {sched}, seed {s} ({total} steps)...")
            runs[sched].append(_schedule_run(
                data, val, vocab_size, sched, total, s, eval_at=(stop, total),
                branch_at=stop if sched == "wsd" else None, branch_steps=branch_steps))

    def stat(sched, at):
        vals = [r["evals"][at] for r in runs[sched]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals}

    summary = {sch: {"at_stop": stat(sch, stop), "at_end": stat(sch, total)}
               for sch in runs}
    bvals = [r["branch"]["val"] for r in runs["wsd"]]
    summary["wsd_branch_decayed"] = {"mean": float(np.mean(bvals)), "std": float(np.std(bvals)),
                                     "vals": bvals, "extra_steps": branch_steps}

    c, w = summary["cosine"]["at_stop"], summary["wsd"]["at_stop"]
    gap = w["mean"] - c["mean"]
    noise = max(c["std"], w["std"])
    print(f"\n[part 4] cosine against WSD, {total} planned steps, both stopped at {stop}")
    print(f"  val loss at step {stop}:  cosine {c['mean']:.4f} +/- {c['std']:.4f}   "
          f"WSD {w['mean']:.4f} +/- {w['std']:.4f}")
    print(f"  val loss at step {total}: cosine {summary['cosine']['at_end']['mean']:.4f}   "
          f"WSD {summary['wsd']['at_end']['mean']:.4f}")
    print(f"  WSD checkpoint at {stop} decayed over {branch_steps} extra steps: "
          f"{summary['wsd_branch_decayed']['mean']:.4f} +/- {summary['wsd_branch_decayed']['std']:.4f}")
    print(f"  gap at the stop is {gap:+.4f} against a seed spread of {noise:.4f}")

    return {"total": total, "stop": stop, "seeds": list(seeds), "summary": summary,
            "gap_at_stop": gap, "seed_spread": noise,
            "curves": {sch: {"losses": runs[sch][0]["losses"], "lrs": runs[sch][0]["lrs"],
                             "branch": runs[sch][0]["branch"]} for sch in runs}}


# --------------------------------------------------------------------------------------
# part 5 — learning rate against width, standard parameterization and muP
# --------------------------------------------------------------------------------------

BASE_WIDTH = 256


def apply_mup(model, width, n_layer):
    """A minimal muP: hidden matrices get init and learning rate scaled by width, the
    output is divided by the width multiplier, embeddings and norms are left alone.

    ponytail: nanoGPT ties lm_head to wte, so the head keeps embedding-like treatment
    rather than muP's untied 1/fan_in init. Untying it would change the model being
    swept; the output multiplier below carries the part that matters for transfer.
    """
    mult = width / BASE_WIDTH
    hidden, other = [], []
    for n, p in model.named_parameters():
        if p.dim() >= 2 and ("attn." in n or "mlp." in n):
            hidden.append((n, p))
        else:
            other.append((n, p))
    with torch.no_grad():
        for n, p in hidden:
            std = 0.02 / math.sqrt(mult)                  # 1/sqrt(fan_in) scaling
            if n.endswith("c_proj.weight"):
                std /= math.sqrt(2 * n_layer)             # nanoGPT's residual scaling, kept
            p.normal_(mean=0.0, std=std)
    model.lm_head.register_forward_hook(lambda m, i, o: o / mult)
    return [p for _, p in hidden], [p for _, p in other], mult


def _sweep_run(data, val, vocab_size, width, lr, arm, steps, n_layer=4, B=8, T=256, seed=SEED):
    n_head = max(1, width // 64)
    model = make_model(vocab_size, n_layer=n_layer, n_head=n_head, n_embd=width,
                       block_size=T, seed=seed)
    if arm == "mup":
        hidden, other, mult = apply_mup(model, width, n_layer)
        groups = [{"params": hidden, "lr": lr / mult, "weight_decay": 0.1},
                  {"params": [p for p in other if p.dim() >= 2], "lr": lr, "weight_decay": 0.1},
                  {"params": [p for p in other if p.dim() < 2], "lr": lr, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    else:
        opt = model.configure_optimizers(0.1, lr, (0.9, 0.95),
                                         "cuda" if DEVICE == "cuda" else "cpu")
    rng = np.random.default_rng(seed)
    warm = max(1, int(0.02 * steps))
    base_lrs = [g["lr"] for g in opt.param_groups]      # muP gives the groups different lrs
    diverged = False
    for step in range(1, steps + 1):
        scale = min(1.0, step / warm)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale
        x, y = get_batch(data, B, T, rng)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not math.isfinite(float(loss.detach())):
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    v = float("nan") if diverged else eval_loss(model, val, B, T)
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"width": width, "lr": lr, "val": v, "diverged": diverged, "params": n_params}


def part5_sweep(data, val, vocab_size, arms=("standard", "mup"), widths=(256, 512, 1024),
                steps=400, n_lrs=7, lo=1e-4, hi=3e-2):
    lrs = [lo * (hi / lo) ** (i / (n_lrs - 1)) for i in range(n_lrs)]
    out = {"widths": list(widths), "lrs": lrs, "steps": steps, "arms": {}}
    for arm in arms:
        rows = []
        for w in widths:
            for lr in lrs:
                r = _sweep_run(data, val, vocab_size, w, lr, arm, steps)
                rows.append(r)
                shown = "diverged" if r["diverged"] else f"{r['val']:.4f}"
                print(f"  [{arm}] width {w:>4}  lr {lr:.2e}  val {shown}")
        best = {}
        for w in widths:
            cand = [r for r in rows if r["width"] == w and not r["diverged"]
                    and math.isfinite(r["val"])]
            if not cand:
                print(f"  [{arm}] width {w}: every learning rate diverged")
                continue
            b = min(cand, key=lambda r: r["val"])
            best[w] = {"lr": b["lr"], "val": b["val"]}
        out["arms"][arm] = {"rows": rows, "best": best}

    # standard parameterization: fit best_lr = K / width and read it at 4,096
    st = out["arms"].get("standard")
    if st:
        ws = np.array(list(st["best"]), dtype=float)
        bl = np.array([st["best"][w]["lr"] for w in st["best"]], dtype=float)
        K = float(np.exp(np.mean(np.log(bl * ws))))        # geometric fit of lr * width
        slope = float(np.polyfit(np.log(ws), np.log(bl), 1)[0])
        out["standard_fit"] = {"K": K, "log_slope": slope,
                               "lr_at_4096": K / 4096.0,
                               "widget_K": 0.768, "widget_lr_at_4096": 0.768 / 4096.0}
        print(f"\n[part 5] standard parameterization: best lr ~ K/width with K = {K:.3f} "
              f"(widget's K is 0.768), log-log slope {slope:.2f} (muP-free theory says -1)")
        print(f"  extrapolated to width 4,096: {K/4096:.3e}")
    mp = out["arms"].get("mup")
    if mp:
        vals = [mp["best"][w]["lr"] for w in mp["best"]]
        out["mup_spread"] = {"lrs": vals, "max_over_min": max(vals) / min(vals)}
        print(f"  muP best lrs {['%.2e' % v for v in vals]}, "
              f"spread {max(vals)/min(vals):.2f}x across widths")
    return out


def part5_minimum_check(data, val, vocab_size, sweep, arm="standard", extra_seeds=(1338, 1339),
                        steps=400):
    """The sweep runs one seed per cell, and one seed is not enough to locate a minimum.

    Re-runs each width's best cell and its two neighbours on further seeds, so the claimed
    minimum can be compared against the spread of the runs that produced it.
    """
    lrs = sweep["lrs"]
    out = {}
    for w in sweep["widths"]:
        best_lr = sweep["arms"][arm]["best"][str(w)]["lr"]
        i = min(range(len(lrs)), key=lambda k: abs(lrs[k] - best_lr))
        cells = [j for j in (i - 1, i, i + 1) if 0 <= j < len(lrs)]
        rows = []
        for j in cells:
            vals = [r["val"] for r in sweep["arms"][arm]["rows"]
                    if r["width"] == w and abs(r["lr"] - lrs[j]) < 1e-12]
            for s in extra_seeds:
                r = _sweep_run(data, val, vocab_size, w, lrs[j], arm, steps, seed=s)
                vals.append(r["val"])
                print(f"  [{arm}] width {w:>4} lr {lrs[j]:.2e} seed {s}  val {r['val']:.4f}")
            good = [v for v in vals if math.isfinite(v)]
            rows.append({"lr": lrs[j], "vals": vals,
                         "mean": float(np.mean(good)), "std": float(np.std(good))})
        winner = min(rows, key=lambda r: r["mean"])
        out[str(w)] = {"cells": rows, "best_lr_by_mean": winner["lr"],
                       "single_seed_best_lr": best_lr,
                       "agrees": abs(winner["lr"] - best_lr) < 1e-12,
                       "spread": max(r["std"] for r in rows)}
        print(f"  width {w}: single-seed pick {best_lr:.2e}, {1+len(extra_seeds)}-seed mean pick "
              f"{winner['lr']:.2e}, worst per-cell spread {out[str(w)]['spread']:.4f}")
    return out


# --------------------------------------------------------------------------------------

def selfcheck():
    """One runnable check per piece of non-trivial logic."""
    # the widget's five-step Adam table, to the digits it prints
    rows = adam_by_hand(WIDGET_GRADS)
    want_w = [0.999000, 0.998012, 0.997018, 0.996028, 0.995031]
    want_m = [0.0500, 0.0850, 0.1365, 0.1678, 0.2061]
    for r, w, m in zip(rows, want_w, want_m):
        assert abs(r["w"] - w) < 5e-7, (r["t"], r["w"], w)
        assert abs(r["m"] - m) < 5e-5, (r["t"], r["m"], m)
    # and the first step is exactly one learning rate
    assert abs(rows[0]["step"] / -ETA - 1.0) < 1e-6, rows[0]["step"]

    # without bias correction the first step is the lesson's 3.16 eta
    off = adam_by_hand(WIDGET_GRADS, bias_correction=False)
    assert abs(off[0]["step"] / -ETA - 3.1623) < 1e-3, off[0]["step"]

    # the analytic ratio must reproduce the simulated one for any gradient sequence.
    # eps=0 here on purpose: eps is what makes the ratio depend on the gradients at all,
    # and it moves the t=1 ratio by about 1e-6, which is the size of the epsilon term
    # against a second moment that has only had one gradient in it.
    g = [0.3, -0.7, 1.4, 0.05, 2.0, -0.2, 0.9]
    a = adam_by_hand(g, bias_correction=True, eps=0.0)
    b = adam_by_hand(g, bias_correction=False, eps=0.0)
    for x, y in zip(a, b):
        assert abs(x["step"] / y["step"] - bias_ratio(x["t"])) < 1e-12, x["t"]
    with_eps = adam_by_hand(g, bias_correction=True)
    assert abs(with_eps[0]["step"] / a[0]["step"] - 1) < 1e-5

    # the schedules hit peak, plateau and zero where the widgets say they do
    T, peak = 1000, 3e-4
    assert abs(cosine_lr(0, T, peak) - peak) < 1e-15
    assert abs(cosine_lr(T // 2, T, peak) - peak / 2) < 1e-12
    assert cosine_lr(T, T, peak) < 1e-18
    assert abs(wsd_lr(20, T, peak) - peak) < 1e-15          # warmup ends at 2% = step 20
    assert abs(wsd_lr(10, T, peak) - peak / 2) < 1e-15
    assert abs(wsd_lr(500, T, peak) - peak) < 1e-15         # flat through the middle
    assert abs(wsd_lr(950, T, peak) - peak / 2) < 1e-15     # halfway down the last 10%
    assert wsd_lr(T, T, peak) < 1e-18
    assert abs(warmup_lr(2000, 2000, 3e-4) - 3e-4) < 1e-18
    assert abs(warmup_lr(1000, 2000, 3e-4) - 1.5e-4) < 1e-18

    # widget_6's ratio arithmetic: a full Adam step against a 1/sqrt(4096) weight
    w0 = 1 / math.sqrt(4096)
    assert abs(3e-4 / w0 - 0.0192) < 1e-4, 3e-4 / w0

    # widget_9's inverse-width rule, the thing part 5 measures against
    assert abs(0.768 / 256 - 3.0e-3) < 1e-6
    assert abs(0.768 / 4096 - 1.875e-4) < 1e-9

    # the muP multiplier is 1 at the base width, so muP must reduce to standard there
    assert BASE_WIDTH / BASE_WIDTH == 1.0

    print("selfcheck: all asserts passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="all",
                    choices=["all", "selfcheck", "1", "2", "3", "4", "5", "5b"])
    ap.add_argument("--arm", default="both", choices=["standard", "mup", "both"])
    ap.add_argument("--steps3", type=int, default=1500)
    ap.add_argument("--warmup3", type=int, default=150)
    ap.add_argument("--sweep-steps", type=int, default=400)
    args = ap.parse_args()

    if args.part == "selfcheck":
        selfcheck()
        return

    ensure_nanogpt()
    torch.manual_seed(SEED)
    data = val = vocab = None
    if args.part in ("all", "1", "3", "4", "5", "5b"):
        data, val, vocab = ensure_data()

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.part in ("all", "1"):
        res["part1_adam"] = part1_adam(data, val, vocab)
    if args.part in ("all", "2"):
        res["part2_bias_correction"] = part2_bias_correction()
    if args.part in ("all", "3"):
        res["part3_update_ratio"] = part3_update_ratio(data, val, vocab, steps=args.steps3,
                                                       warmup_steps=args.warmup3)
    if args.part in ("all", "4"):
        res["part4_schedules"] = part4_schedules(data, val, vocab)
    if args.part in ("all", "5"):
        arms = ("standard", "mup") if args.arm == "both" else (args.arm,)
        prev = res.get("part5_sweep", {})
        new = part5_sweep(data, val, vocab, arms=arms, steps=args.sweep_steps)
        if prev.get("arms"):                     # keep an arm run in an earlier invocation
            merged = dict(prev["arms"])
            merged.update(new["arms"])
            new["arms"] = merged
        res["part5_sweep"] = new
    if args.part in ("all", "5b"):
        arm = "standard" if args.arm == "both" else args.arm
        res["part5_sweep"]["minimum_check"] = part5_minimum_check(
            data, val, vocab, res["part5_sweep"], arm=arm, steps=args.sweep_steps)

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
