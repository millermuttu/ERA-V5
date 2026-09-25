"""Three figures from results.json: the variant probe, the largest batch per stack,
and the three 50M-token runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

GREEN, RED, VIOLET, CYAN, AMBER, SLATE = "#159447", "#dc2626", "#7c3aed", "#0891b2", "#d97706", "#475569"
COL = {"standard": SLATE, "midpoint_h0.25": VIOLET, "midpoint_h0.5": AMBER, "coupling": CYAN}

# ---- the probe: which reversible variant trains ---------------------------------------
if "probe" in R:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, r in R["probe"].items():
        lg = r["log"]
        ax.plot([x[1] / 1e6 for x in lg], [x[2] for x in lg], color=COL.get(k, GREEN), lw=1.2,
                alpha=.85, label=f"{k}  (val {r['final_val_loss']:.3f})" if r["final_val_loss"]
                else f"{k}  (diverged)")
    ax.set_xlabel("tokens, millions")
    ax.set_ylabel("train loss")
    ax.set_ylim(top=8)
    ax.set_title("Variant probe, 4M tokens at batch 8", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(HERE / "probe.png", dpi=140)

# ---- memory against batch size ---------------------------------------------------------
if "maxbatch" in R:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, r in R["maxbatch"].items():
        b = [int(x) for x in r["peak_gib_by_batch"]]
        m = list(r["peak_gib_by_batch"].values())
        col = SLATE if k == "standard" else VIOLET
        ax.plot(b, m, marker="o", ms=4, color=col, label=f"{k}, max batch {r['max_batch']}")
    ax.axhline(4.0, color=RED, lw=.8, ls="--")
    ax.annotate("GTX 1650 Ti, 4 GiB", xy=(1, 4.0), xytext=(1, 3.75), fontsize=8, color=RED)
    ax.set_xlabel("batch, sequences of 512 tokens")
    ax.set_ylabel("peak allocated, GiB")
    ax.set_title("Peak memory of one optimizer step", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(HERE / "maxbatch.png", dpi=140)

# ---- the three runs --------------------------------------------------------------------
runs = [(a, R[f"run_{a}"]) for a in "ABC" if f"run_{a}" in R]
if runs:
    rc = {"A": SLATE, "B": VIOLET, "C": GREEN}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    ax = axs[0]
    for a, r in runs:
        lab = f"{a}: {r['mode']}, batch {r['batch']}"
        ax.plot([x[0] / 1e6 for x in r["val_curve"]], [x[1] for x in r["val_curve"]],
                marker="o", ms=3.5, color=rc[a], label=lab)
    ax.set_xlabel("tokens, millions")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss over 50M tokens", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)
    for ax, key, title, fmt in ((axs[1], "final_val_loss", "final val loss", "{:.3f}"),
                                (axs[2], "tokens_per_s", "tokens / s", "{:,.0f}"),
                                (axs[3], "peak_mem_gib", "peak memory, GiB", "{:.2f}")):
        v = [r[key] or 0 for _, r in runs]
        bars = ax.bar([a for a, _ in runs], v, color=[rc[a] for a, _ in runs])
        for b, x in zip(bars, v):
            ax.annotate(fmt.format(x), (b.get_x() + b.get_width() / 2, x), ha="center",
                        va="bottom", fontsize=8.5)
        ax.set_title(title, fontsize=11)
        if key == "final_val_loss":
            ax.set_ylim(min(v) * 0.95, max(v) * 1.02)
        ax.grid(alpha=.15, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "runs.png", dpi=140)
