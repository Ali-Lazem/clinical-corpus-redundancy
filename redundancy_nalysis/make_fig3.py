#!/usr/bin/env python3
"""
Redesigned Figure 3: the context-copy redundancy mechanism.
One patient narrative copied into the context field of every extracted
item. Matches the palette and clean aesthetic of the redesigned Figure 2.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

# palette consistent with Fig 2
CTX_COLOR  = "#c0392b"   # copied context (redundant)  - deep red
GEN_COLOR  = "#2e9e5b"   # unique generated            - green
SRC_COLOR  = "#2c6fa6"   # source narrative            - blue
ARROW_COL  = "#c0392b"

fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---- title ----
ax.text(5.0, 9.6, "One patient narrative is copied into every extracted item",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color="#222222")

# ---- source narrative (left) ----
src = FancyBboxPatch((0.4, 4.0), 2.5, 2.1, boxstyle="round,pad=0.10",
                     fc=SRC_COLOR, ec="none")
ax.add_patch(src)
ax.text(1.65, 5.45, "Patient\nnarrative", ha="center", va="center",
        color="white", fontsize=11.5, fontweight="bold")
ax.text(1.65, 4.5, "~600 tokens\nstored once", ha="center", va="center",
        color="#e8f0f7", fontsize=8.5)

# ---- extracted items (right) ----
ys = [8.3, 6.75, 5.2, 3.65, 2.1]
labels = ["QAR item 1", "QAR item 2", "RE item 1", "RE item 2", "item N"]
item_x = 5.7
item_w = 3.9
gen_w = 0.55     # width of the green (unique generated) sliver
for i, (yc, lab) in enumerate(zip(ys, labels)):
    # outer item container
    ax.add_patch(FancyBboxPatch((item_x, yc-0.52), item_w, 1.04,
                 boxstyle="round,pad=0.02", fc="white", ec="#cccccc",
                 lw=0.8, zorder=2))
    # green unique-generated sliver
    ax.add_patch(Rectangle((item_x+0.12, yc-0.38), gen_w, 0.76,
                 fc=GEN_COLOR, ec="none", zorder=3))
    # red copied-context block (the bulk)
    ax.add_patch(Rectangle((item_x+0.12+gen_w+0.06, yc-0.38),
                 item_w-gen_w-0.34, 0.76, fc=CTX_COLOR, ec="none",
                 alpha=0.88, zorder=3))
    # item label
    ax.text(item_x+0.05, yc+0.66, lab, ha="left", va="bottom",
            fontsize=8.5, color="#555555")
    # curved arrow from source to this item's context block
    arr = FancyArrowPatch((2.95, 5.05), (item_x+0.05, yc),
                          arrowstyle="-|>", mutation_scale=11,
                          color=ARROW_COL, lw=1.1, alpha=0.5,
                          connectionstyle="arc3,rad=0.10", zorder=1)
    ax.add_patch(arr)

# "N copies" brace annotation between source and items
ax.text(4.25, 1.5, "the same\nnarrative,\ncopied\n$N$ times",
        ha="center", va="center", fontsize=8.8, color=CTX_COLOR,
        style="italic")

# ---- legend (bottom) ----
ly = 0.55
ax.add_patch(Rectangle((item_x+0.12, ly), 0.32, 0.32, fc=GEN_COLOR, ec="none"))
ax.text(item_x+0.55, ly+0.16, "unique generated content (kept once)",
        va="center", fontsize=8.8, color="#333333")
ax.add_patch(Rectangle((item_x+0.12, ly-0.55), 0.32, 0.32, fc=CTX_COLOR,
             ec="none", alpha=0.88))
ax.text(item_x+0.55, ly-0.39, "copied source context (redundant)",
        va="center", fontsize=8.8, color="#333333")

plt.tight_layout()
plt.savefig("figures/redundancy_fig3_schematic.png", dpi=220,
            bbox_inches="tight")
plt.savefig("figures/redundancy_fig3_schematic.pdf",
            bbox_inches="tight")
print("wrote figures/redundancy_fig3_schematic (png+pdf) redesigned")
