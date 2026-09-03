"""Two figures from results.json: the accumulation gap, and the norm leading the loss."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

# ---- part 3: both curves together, and the per-step gap that produced them -------------
p3 = R["part3_accumulation"]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
ax.plot(p3["correct"]["curve"], color="#159447", lw=2, label="correct: sum(loss) / sum(tokens)")
ax.plot(p3["naive"]["curve"], color="#dc2626", lw=2, label="naive: mean of the micro-batch means")
ax.plot(p3["naive"]["honest_curve"], color="#dc2626", lw=1.2, ls="--", alpha=.75,
        label="the naive run's true loss, normalised properly")
ax.set_ylabel("loss")
ax.legend(fontsize=8.5, frameon=False)
ax.set_title("Gradient accumulation with unequal micro-batch token counts", fontsize=11)
ax.grid(alpha=.15)

gaps = p3["step_gaps_pct"]
w = p3["worst_step"]
ax2.bar(range(len(gaps)), gaps, color=["#d97706" if abs(g) > 5 else "#c4b5fd" for g in gaps])
ax2.axhline(0, color="#94a3b8", lw=.8)
ax2.annotate(f"step {w['step']}: {w['gap_pct']:+.1f}%\ntokens {w['token_counts']}",
             xy=(w["step"], w["gap_pct"]), xytext=(w["step"] + 3, w["gap_pct"]),
             fontsize=8, va="center", color="#b45309")
ax2.set_ylabel("reported vs true, %")
ax2.set_xlabel("optimiser step")
ax2.grid(alpha=.15)
fig.tight_layout()
fig.savefig(HERE / "part3_accumulation.png", dpi=140)

# ---- part 4: the norm moving before the loss, capped against uncapped --------------
p4 = R["part4_gradnorm"]
fig, axes = plt.subplots(2, 2, figsize=(12, 5.6), sharex=True)
for col, (name, tag) in enumerate((("clipped", f"clip {p4['clip']}"), ("unclipped", "no cap"))):
    d = p4[name]
    ax, ax2 = axes[0][col], axes[1][col]
    ax.semilogy(d["norms"], color="#7c3aed", lw=1.0)
    if name == "clipped":
        ax.axhline(p4["clip"], color="#159447", ls="--", lw=1.4, label=f"cap {p4['clip']}")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.axhline(d["median_norm"], color="#94a3b8", ls=":", lw=1)
    ax.set_title(f"{tag}: final loss {d['final_loss']:.4f}, worst after step 100 "
                 f"{d['max_loss_after_100']:.4f}", fontsize=9.5)
    ax.grid(alpha=.15)
    ax2.plot(d["losses"], color="#dc2626", lw=1.0)
    ax2.grid(alpha=.15)
    ax2.set_xlabel("training step")
    if col == 0:
        ax.set_ylabel("grad norm (pre-clip)")
        ax2.set_ylabel("loss")
    for L in d["leads"]:
        ax.axvline(L["norm_step"], color="#d97706", lw=1.0)
        ax2.axvline(L["loss_step"], color="#dc2626", lw=1.0, alpha=.5)
    if d["leads"]:
        b = max(d["leads"], key=lambda z: z["lead"])
        ax2.annotate(f"norm spiked {b['norm_step']}\nloss reacted {b['loss_step']}"
                     f" (+{b['lead']} steps)",
                     xy=(b["loss_step"], b["loss_at"]), xytext=(0.05, 0.78),
                     textcoords="axes fraction", fontsize=8, color="#b91c1c",
                     arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=.9))
    else:
        ax2.text(0.05, 0.85, "no spike cleared the 2-sigma bar", transform=ax2.transAxes,
                 fontsize=8, color="#64748b")
fig.suptitle(f"Grad norm and loss, {p4['steps']} steps at lr {p4['lr']}, same init and data",
             fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "part4_gradnorm.png", dpi=140)
print("wrote part3_accumulation.png and part4_gradnorm.png")
