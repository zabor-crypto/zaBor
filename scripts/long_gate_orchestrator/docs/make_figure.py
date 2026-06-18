#!/usr/bin/env python3
"""Render docs/results.png — the before/after regime-gate validation figure.

Numbers are the out-of-sample validation summary (R-multiples; no account values).
Regenerate:  python3 make_figure.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
GREY, GREEN, INK, MUTED = "#AEAAA2", "#1F9E6E", "#1A1A1A", "#6B6B6B"
STRATS = ["Reversal", "Mean-reversion", "Breakout"]

pf_before,  pf_after  = [1.27, 0.81, 0.98], [1.65, 1.56, 1.31]
ret_before, ret_after = [292, -660, -25],   [368, 558, 190]
retain = ["63%", "40%", "20%"]
sig_before, sig_after = [4114, 6660, 8000], [2612, 2685, 1575]

fig = plt.figure(figsize=(13, 9.2), dpi=150)
fig.patch.set_facecolor("white")
gs = GridSpec(3, 3, height_ratios=[0.9, 1.5, 1.5], hspace=0.55, wspace=0.25,
              left=0.07, right=0.97, top=0.95, bottom=0.06)

# ---- KPI cards ----
kpis = [("Portfolio result (R)", "-404", "+1,128"),
        ("Profit factor", "0.95", "1.45"),
        ("Win rate", "54%", "65%")]
for i, (label, before, after) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F2F1EC",
                               edgecolor="none", zorder=0))
    ax.text(0.06, 0.74, label, transform=ax.transAxes, fontsize=12, color=MUTED, va="center")
    ax.text(0.06, 0.32, before, transform=ax.transAxes, fontsize=27, color=INK,
            va="center", fontweight="bold")
    w = 0.40 if i == 0 else 0.26
    ax.text(0.06 + w, 0.32, "→", transform=ax.transAxes, fontsize=18, color=MUTED, va="center")
    ax.text(0.10 + w, 0.32, after, transform=ax.transAxes, fontsize=27, color=GREEN,
            va="center", fontweight="bold")

def grouped(ax, before, after, title, baseline=None, fmt="{:.2f}"):
    x = range(len(STRATS)); w = 0.38
    ax.bar([i - w/2 for i in x], before, w, color=GREY, label="Before (no gate)")
    ax.bar([i + w/2 for i in x], after, w, color=GREEN, label="After (regime gate)")
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, lw=1, ls="--")
    ax.set_xticks(list(x)); ax.set_xticklabels(STRATS, fontsize=11)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return ax

axpf = fig.add_subplot(gs[1, :])
grouped(axpf, pf_before, pf_after, "Profit factor by strategy — 1.0 = break-even", baseline=1.0)
axpf.legend(loc="upper right", frameon=False, fontsize=10)
axpf.set_ylim(0, 1.8)

axret = fig.add_subplot(gs[2, :])
grouped(axret, ret_before, ret_after, "Total return by strategy — R-multiples (out-of-sample)")
axret.axhline(0, color=MUTED, lw=1)
for i in range(len(STRATS)):
    axret.text(i, -730, f"signals {sig_before[i]:,} → {sig_after[i]:,}  ({retain[i]} kept)",
               ha="center", fontsize=9.5, color=MUTED)
axret.set_ylim(-800, 650)

fig.savefig(HERE / "results.png", facecolor="white", bbox_inches="tight")
print(f"wrote {HERE / 'results.png'}")
