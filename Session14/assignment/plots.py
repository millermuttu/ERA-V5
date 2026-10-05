"""Two figures from results.json: the loss across the conversion, and how the routers balanced."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

SLATE, VIOLET, GREEN, RED = "#475569", "#7c3aed", "#159447", "#dc2626"
COL = {"dense": SLATE, "copy": VIOLET, "drop": GREEN}
LAB = {"dense": "dense, continued", "copy": "MoE, copy-upcycled", "drop": "MoE, drop-upcycled r=0.5"}
branches = [(k, R[f"branch_{k}"]) for k in COL if f"branch_{k}" in R]


def smooth(y, n=5):
    return np.convolve(y, np.ones(n) / n, mode="valid") if len(y) > n else np.asarray(y)


# ---- loss across the conversion ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
d = R["phase1_dense"]
for ax, key, title in ((axs[0], "train", "Train loss"), (axs[1], "val", "Validation loss")):
    if key == "train":
        lg = d["log"]
        ax.plot([x[1] / 1e6 for x in lg][4:], smooth([x[2] for x in lg]), color=SLATE, lw=1)
    else:
        c = d["val_curve"]
        ax.plot([x[0] / 1e6 for x in c], [x[1] for x in c], color=SLATE, marker="o", ms=3)
    for k, r in branches:
        if key == "train":
            lg = r["log"]
            ax.plot([x[1] / 1e6 for x in lg][4:], smooth([x[2] for x in lg]), color=COL[k], lw=1,
                    label=f"{LAB[k]}  (end {r['train_end']:.3f})")
        else:
            c = r["val_curve"]
            ax.plot([x[0] / 1e6 for x in c], [x[1] for x in c], color=COL[k], marker="o", ms=3,
                    label=f"{LAB[k]}  (end {r['val_end']:.3f})")
    ax.axvline(d["tokens"] / 1e6, color=RED, lw=.8, ls="--")
    ax.annotate("convert", xy=(d["tokens"] / 1e6, 0.97), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=RED)
    ax.set_ylim(top=6.0 if key == "train" else 5.6)
    ax.set_xlabel("tokens, millions")
    ax.set_ylabel(f"{key} loss")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig(HERE / "loss.png", dpi=140)

# ---- routing: imbalance over time, and the final share of each expert ----------------------
moe = [(k, r) for k, r in branches if r["routing"]]
if moe:
    fig, axs = plt.subplots(1, 1 + len(moe), figsize=(14, 4.2),
                            gridspec_kw={"width_ratios": [1.6] + [1] * len(moe)})
    for k, r in moe:
        rt = r["routing"]
        axs[0].plot([x[0] / 1e6 for x in rt], [x[1] for x in rt], color=COL[k], lw=1, label=LAB[k])
    axs[0].set_xlabel("tokens, millions")
    axs[0].set_ylabel("MaxVio, mean over layers")
    axs[0].set_title("Load imbalance (busiest expert over an even share)", fontsize=11)
    axs[0].legend(fontsize=8.5, frameon=False)
    axs[0].grid(alpha=.15)
    for ax, (k, r) in zip(axs[1:], moe):
        share = np.array(r["routing"][-1][3])          # layers x experts, last 50 steps
        im = ax.imshow(share, cmap="viridis", vmin=0, vmax=2 / share.shape[1], aspect="auto")
        ax.set_xlabel("expert")
        ax.set_ylabel("layer")
        ax.set_title(f"{k}: share of routed tokens, end", fontsize=10)
    fig.colorbar(im, ax=axs[1:], fraction=.04, label="share (even = 1/8)")
    fig.savefig(HERE / "routing.png", dpi=140, bbox_inches="tight")
