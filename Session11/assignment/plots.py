"""Four figures from results.json: bias correction, the update-to-weight ratio,
the two schedules stopped early, and the learning rate against width."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

GREEN, RED, VIOLET, CYAN, AMBER = "#159447", "#dc2626", "#7c3aed", "#0891b2", "#d97706"

# ---- part 2: twenty steps as asked, and the ratio the twenty steps cannot show --------
if "part2_bias_correction" in R:
    p2 = R["part2_bias_correction"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t = [r["t"] for r in p2["short"]]
    ax.plot(t, [-r["step_on"] * 1e3 for r in p2["short"]], color=GREEN, marker="o", ms=3.5,
            label="with bias correction")
    ax.plot(t, [-r["step_off"] * 1e3 for r in p2["short"]], color=RED, marker="o", ms=3.5,
            label="without")
    ax.axhline(1.0, color="#94a3b8", lw=.8, ls="--")
    ax.annotate("one learning rate", xy=(13, 1.0), xytext=(11, 1.9), fontsize=8, color="#64748b")
    ax.set_xlabel("step")
    ax.set_ylabel("step size, thousandths (eta = 1e-3)")
    ax.set_title("The first twenty steps", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)

    c = p2["ratio_curve"]
    ax2.semilogx([r["t"] for r in c], [r["ratio"] for r in c], color=VIOLET, lw=2)
    ax2.axhline(1.0, color="#94a3b8", lw=.8, ls="--")
    for step, tol, col in ((p2["crossing_1pct"], "1%", AMBER),
                           (p2["crossing_0p1pct"], "0.1%", CYAN)):
        ax2.axvline(step, color=col, lw=1.2, ls=":")
        ax2.annotate(f"within {tol}\nstep {step:,}", xy=(step, 0.55), fontsize=8, color=col,
                     ha="right" if tol == "1%" else "left")
    ax2.axvspan(1, 20, color="#e2e8f0", alpha=.6)
    ax2.annotate("the window the\nassignment asks for", xy=(20, 0.35), xytext=(28, 0.22),
                 fontsize=8, color="#64748b")
    ax2.set_xlabel("step")
    ax2.set_ylabel("corrected step / uncorrected step")
    ax2.set_title("sqrt(1 - b2^t) / (1 - b1^t), out to 10,000", fontsize=11)
    ax2.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(HERE / "part2_bias_correction.png", dpi=140)
    plt.close(fig)

# ---- part 3: the ratio with and without warmup, and where each group sits -------------
if "part3_update_ratio" in R:
    p3 = R["part3_update_ratio"]
    w, nw = p3["runs"]["warmup"], p3["runs"]["no_warmup"]
    steps = w["steps_logged"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    ax.semilogy(steps, nw["overall"], color=RED, lw=1.4, label="no warmup")
    ax.semilogy(steps, w["overall"], color=GREEN, lw=1.4, label=f"warmup {p3['warmup_steps']} steps")
    ax.axhline(1e-3, color="#94a3b8", lw=1, ls="--")
    ax.annotate("1e-3, the band the lesson wants", xy=(steps[-1], 1e-3), ha="right", va="bottom",
                fontsize=8, color="#64748b")
    ax.axvline(p3["warmup_steps"], color=AMBER, lw=1, ls=":")
    ax.annotate(f"warmup ends", xy=(p3["warmup_steps"], ax.get_ylim()[1]), rotation=90,
                fontsize=8, color=AMBER, va="top", ha="right")
    for run, col, off in ((nw, RED, (14, 2)), (w, GREEN, (12, -14))):
        ax.plot([run["peak_step"]], [run["peak_ratio"]], "o", color=col, ms=5)
        ax.annotate(f"peak {run['peak_ratio']*1e3:.2f}e-3 at step {run['peak_step']}",
                    xy=(run["peak_step"], run["peak_ratio"]), xytext=off,
                    textcoords="offset points", fontsize=8, color=col)
    ax.set_ylabel("mean |update| / |weight|")
    ax.set_title("Update-to-weight ratio, per optimiser step", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=.15)

    for g, d in w["by_group"].items():
        ax2.semilogy(steps, d["series"], lw=1.2, label=g)
    ax2.axhline(1e-3, color="#94a3b8", lw=1, ls="--")
    ax2.set_xlabel("optimiser step")
    ax2.set_ylabel("ratio, by group")
    ax2.set_title("With warmup, split by parameter group", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, ncol=3)
    ax2.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(HERE / "part3_update_ratio.png", dpi=140)
    plt.close(fig)

# ---- part 4: the two schedules, the stop line, and the branch that follows ------------
if "part4_schedules" in R:
    p4 = R["part4_schedules"]
    stop, total = p4["stop"], p4["total"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 2]})
    for sch, col in (("cosine", VIOLET), ("wsd", CYAN)):
        ax.plot(range(1, total + 1), p4["curves"][sch]["lrs"], color=col, lw=2, label=sch)
    br = p4["curves"]["wsd"]["branch"]
    ax.plot(range(stop, stop + br["steps"]),
            [3e-4 * (1 - i / br["steps"]) for i in range(br["steps"])],
            color=AMBER, lw=2, ls="--", label="WSD branch, decayed from the step-200 checkpoint")
    ax.axvline(stop, color="#94a3b8", lw=1, ls=":")
    ax.set_ylabel("learning rate")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.15)
    ax.set_title(f"Cosine and WSD, both planned for {total} steps, both stopped at {stop}",
                 fontsize=11)

    def smooth(x, k=15):
        return np.convolve(x, np.ones(k) / k, mode="valid")
    for sch, col in (("cosine", VIOLET), ("wsd", CYAN)):
        s = smooth(p4["curves"][sch]["losses"])
        ax2.plot(range(len(s)), s, color=col, lw=1.6, label=f"{sch}, train loss (15-step mean)")
    s = smooth(br["curve"], 5)
    ax2.plot(range(stop, stop + len(s)), s, color=AMBER, lw=1.6, ls="--", label="branch decay")
    ax2.axvline(stop, color="#94a3b8", lw=1, ls=":")
    b = p4["summary"]["wsd_branch_decayed"]
    box = "\n".join([
        f"held-out loss, {len(p4['seeds'])} seeds",
        f"  cosine, stopped at {stop}:  {p4['summary']['cosine']['at_stop']['mean']:.4f}"
        f" +/- {p4['summary']['cosine']['at_stop']['std']:.4f}",
        f"  WSD, stopped at {stop}:     {p4['summary']['wsd']['at_stop']['mean']:.4f}"
        f" +/- {p4['summary']['wsd']['at_stop']['std']:.4f}",
        f"  WSD branch, +{b['extra_steps']} decay steps: {b['mean']:.4f} +/- {b['std']:.4f}",
        f"  cosine, full {total}:        {p4['summary']['cosine']['at_end']['mean']:.4f}",
        f"  WSD, full {total}:           {p4['summary']['wsd']['at_end']['mean']:.4f}",
    ])
    ax2.text(0.985, 0.96, box, transform=ax2.transAxes, ha="right", va="top", ma="left",
             fontsize=8, family="monospace", color="#334155",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1", lw=.8))
    ax2.set_xlabel("step")
    ax2.set_ylabel("training loss")
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    ax2.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(HERE / "part4_schedules.png", dpi=140)
    plt.close(fig)

# ---- part 5: loss against learning rate, standard beside muP -------------------------
if "part5_sweep" in R:
    p5 = R["part5_sweep"]
    arms = [a for a in ("standard", "mup") if a in p5["arms"]]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 4.4), squeeze=False)
    cols = {256: CYAN, 512: VIOLET, 1024: GREEN, 2048: AMBER}
    for ax, arm in zip(axes[0], arms):
        d = p5["arms"][arm]
        for wdt in p5["widths"]:
            rows = [r for r in d["rows"] if r["width"] == wdt and not r["diverged"]
                    and math.isfinite(r["val"])]
            if not rows:
                continue
            ax.semilogx([r["lr"] for r in rows], [r["val"] for r in rows],
                        color=cols.get(wdt, "#64748b"), marker="o", ms=3.5, lw=1.5,
                        label=f"width {wdt}")
            b = d["best"].get(str(wdt), d["best"].get(wdt))
            if b:
                ax.plot([b["lr"]], [b["val"]], "v", color=cols.get(wdt, "#64748b"), ms=9)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("held-out loss")
        ax.set_title({"standard": "standard parameterization",
                      "mup": "minimal muP"}[arm], fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(alpha=.15)
    if "standard_fit" in p5:
        f = p5["standard_fit"]
        axes[0][0].annotate(f"fit: best lr = {f['K']:.2f}/width\nwidth 4,096 -> {f['lr_at_4096']:.2e}"
                            f"\n(widget: 0.768/width -> 1.88e-4)",
                            xy=(0.975, 0.04), xycoords="axes fraction", fontsize=8.5,
                            color="#334155", va="bottom", ha="right",
                            bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec="#cbd5e1", lw=.8))
    fig.tight_layout()
    fig.savefig(HERE / "part5_sweep.png", dpi=140)
    plt.close(fig)

print("wrote the figures next to results.json")
