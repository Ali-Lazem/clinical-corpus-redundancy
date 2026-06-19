#!/usr/bin/env python3
"""
Figure 3: One narrative, many copies. A single patient's source narrative is
reproduced verbatim across the context-bearing task channels. Only channels
that copy are shown (NER/medications/risk-QA/summary attach no context and
are omitted). Built from real measured copy-counts (corpus n=2000 sample;
median patient 6028405-1 = 45 copies, matches corpus mean 42.8).

Per-task mean verbatim narrative copies per patient (real):
  RE 25.5, temporal_events 10.9, QAR 6.3.  Corpus mean total 42.8 (max 83).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Only the context-bearing channels (real per-patient mean copies)
CHANNELS = [
    ("Relation\nextraction", "RE",  25.5, "#922B21"),
    ("Temporal\nevents",      "TE", 10.9, "#B03A2E"),
    ("QA with\nreasoning",    "QAR", 6.3, "#C0392B"),
]
CORPUS_MEAN = 42.8
CORPUS_MAX  = 83
CORPUS_MED  = 43

# palette
BLUE = "#1A5276"; BLUE_L = "#D4E6F1"; BLUE_M = "#A9CCE3"
INK  = "#1B2631"; RED = "#C0392B"; CREAM = "#FBEEE6"
PAPER = "#FCF8F5"

fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
fig.patch.set_facecolor("white")

# ============ LEFT: the single source narrative ============
# drawn as a document with text lines
doc_x, doc_y, doc_w, doc_h = 0.6, 3.0, 2.9, 3.4
shadow = FancyBboxPatch((doc_x+0.08, doc_y-0.08), doc_w, doc_h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor="#00000018", edgecolor="none", zorder=1)
ax.add_patch(shadow)
doc = FancyBboxPatch((doc_x, doc_y), doc_w, doc_h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor="white", edgecolor=BLUE, linewidth=2.0, zorder=2)
ax.add_patch(doc)
ax.text(doc_x+doc_w/2, doc_y+doc_h-0.35, "SOURCE NARRATIVE",
        ha="center", fontsize=10, fontweight="bold", color=BLUE, zorder=3)
# faux text lines
rng = np.random.default_rng(7)
ly = doc_y + doc_h - 0.85
while ly > doc_y + 0.5:
    w = rng.uniform(1.7, 2.5)
    ax.add_patch(Rectangle((doc_x+0.3, ly), w, 0.09, facecolor=BLUE_M, edgecolor="none", zorder=3))
    ly -= 0.32
ax.text(doc_x+doc_w/2, doc_y+0.18, "one note  \u00b7  stored once",
        ha="center", fontsize=8, color=BLUE, style="italic", zorder=3)

# ============ CENTER->RIGHT: copies fanning into each channel ============
# Each channel is a "stack of copies" visual whose depth = copy count
ch_x = 6.2
ch_w = 4.8
y_centers = [6.6, 4.4, 2.2]
maxc = max(c for *_ , c, _ in [(a,b,c,d) for a,b,c,d in CHANNELS])

for (name, abbr, copies, color), yc in zip(CHANNELS, y_centers):
    # stacked offset cards to convey "many copies"
    n_show = min(int(round(copies)), 14)   # cap visual stack depth
    depth = 0.9 + 0.85 * (copies / maxc)   # taller stack = more copies
    card_w = 2.0
    # draw stacked rectangles back-to-front
    for k in range(n_show, 0, -1):
        ox = ch_x + k*0.085
        oy = yc - depth/2 + k*0.045
        alpha = 0.25 + 0.5*(1 - k/n_show)
        ax.add_patch(FancyBboxPatch((ox, oy), card_w, depth*0.8,
                     boxstyle="round,pad=0.01,rounding_size=0.04",
                     facecolor=color, alpha=alpha, edgecolor="none", zorder=3))
    # front card (solid)
    front = FancyBboxPatch((ch_x, yc-depth/2), card_w, depth*0.8,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor="white", edgecolor=color, linewidth=1.6, zorder=5)
    ax.add_patch(front)
    # mini text lines inside front card (suggesting it's a copy of the note)
    fy = yc + depth*0.8/2 - 0.28
    for _ in range(3):
        ax.add_patch(Rectangle((ch_x+0.18, fy), rng.uniform(1.2,1.6), 0.06,
                     facecolor=color, alpha=0.35, edgecolor="none", zorder=6))
        fy -= 0.22

    # channel label + big copy number to the right
    ax.text(ch_x + card_w + 1.05, yc+0.18, abbr, fontsize=12, fontweight="bold",
            color=color, va="center", ha="left", zorder=7)
    ax.text(ch_x + card_w + 1.05, yc-0.28, name.replace("\n"," "), fontsize=8,
            color=INK, va="center", ha="left", zorder=7)
    ax.text(ch_x + card_w + 3.1, yc, f"\u00d7{copies:.1f}", fontsize=15,
            fontweight="bold", color=color, va="center", ha="left", zorder=7)

    # curved arrow from the source doc to this channel stack
    arr = FancyArrowPatch((doc_x+doc_w, doc_y+doc_h/2),
                          (ch_x-0.1, yc),
                          arrowstyle="-|>", mutation_scale=15,
                          color=color, lw=1.6, alpha=0.7,
                          connectionstyle=f"arc3,rad={0.25 if yc>4.4 else (-0.25 if yc<4.4 else 0)}",
                          zorder=4)
    ax.add_patch(arr)

# ============ headline banner ============
ax.add_patch(FancyBboxPatch((0.6, 0.25), 11.8, 0.95,
             boxstyle="round,pad=0.03,rounding_size=0.12",
             facecolor=CREAM, edgecolor=RED, linewidth=1.6, zorder=2))
ax.text(6.5, 0.86, f"Each narrative is copied {CORPUS_MEAN:.0f} times on average per patient",
        ha="center", fontsize=12, fontweight="bold", color=RED, zorder=3)
ax.text(6.5, 0.48, f"(median {CORPUS_MED}, up to {CORPUS_MAX} for the most complex presentations) "
        f"\u2014 no copy adds information beyond the single source",
        ha="center", fontsize=8.5, color=INK, zorder=3)

ax.set_title("One narrative, copied verbatim across the context-bearing channels",
             fontsize=13, fontweight="bold", color=INK, pad=14)

plt.tight_layout()
plt.savefig("figures/redundancy_fig3_schematic.pdf", bbox_inches="tight")
plt.savefig("figures/redundancy_fig3_schematic.png", dpi=160, bbox_inches="tight")
print("wrote redundancy_fig3_schematic (redesigned, copy-channels only)")
