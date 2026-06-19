#!/usr/bin/env python3
"""Supplementary Figure S1: mechanism-level decomposition of provenance
redundancy by channel. Numbers verified against redundancy_full_v2.json."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

channels = [
    "QAR (question answering w/ reasoning)",
    "RE (relation extraction)",
    "Recommendations",
    "Risks",
    "Temporal events",
    "Risk states",
    "Risk-QA",
    "Medications",
    "NER (named-entity recognition)",
    "Summary",
]
copied = np.array([81.0, 80.0, 0.0, 0.0, 47.1, 20.1, 0.0, 0.0, 0.0, 0.0])
dup    = np.array([ 9.7,  7.0, 70.6, 68.1, 20.9, 46.0, 51.3, 25.9, 19.8, 0.0])
total  = copied + dup

BLUE   = "#2f6fb2"
ORANGE = "#f6a01a"

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
y = np.arange(len(channels)); bar_h = 0.62

ax.barh(y, copied, color=BLUE,   height=bar_h, label="Copied context (redundant)")
ax.barh(y, dup, left=copied, color=ORANGE, height=bar_h, label="Duplicated generation (redundant)")

ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(channels)-0.4)
ax.invert_yaxis()
ax.set_yticks(y); ax.set_yticklabels(channels, fontsize=10)
ax.set_xticks([0,20,40,60,80,100])
ax.tick_params(axis="x", labelsize=10, top=True, labeltop=True, bottom=False, labelbottom=False)
ax.tick_params(axis="y", length=0)
ax.set_axisbelow(True)
ax.xaxis.grid(True, linestyle="--", color="#cfcfcf", alpha=0.7)
for sp in ["left","right","bottom"]: ax.spines[sp].set_visible(False)
ax.spines["top"].set_color("#666"); ax.spines["top"].set_linewidth(1.0)
ax.set_title("Share of channel tokens (%)", fontsize=12, pad=22, weight="bold", loc="center")

# in-bar labels: only label a segment when it's wide enough to hold text
for i,(c,d) in enumerate(zip(copied,dup)):
    if c >= 6:
        ax.text(c/2, i, f"{c:.1f}", va="center", ha="center", fontsize=10, color="white", weight="bold")
    if d >= 6:
        ax.text(c+d/2, i, f"{d:.1f}", va="center", ha="center", fontsize=10,
                color="white" if d>=12 else "#333", weight="bold")
# total at right
for i,t in enumerate(total):
    ax.text(101.5, i, f"{t:.1f}", va="center", ha="left", fontsize=10, color="#222", clip_on=False)
ax.text(101.5, -1.15, "Total\nredund. (%)", va="center", ha="left", fontsize=9.5, weight="bold", color="#222")

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False,
          fontsize=10.5, handlelength=2.2, columnspacing=2.0)

plt.tight_layout()
out = Path("/mnt/user-data/outputs")
fig.savefig(out/"si_fig_s1_mechanism_decomposition.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(out/"si_fig_s1_mechanism_decomposition.png", dpi=200, bbox_inches="tight", facecolor="white")
print("[done] wrote si_fig_s1_mechanism_decomposition.{pdf,png}")
