#!/usr/bin/env python3
"""
Figure 5: Worked example of context-copy redundancy in one real QAR record
(uid 7665777-11). A document-style visual: the record is drawn as a nested
card showing the tiny extracted fact embedded in a large block of copied
source text. Proportions are real measured field lengths.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Real measured char lengths (record 7665777-11)
L_CTX1, L_CTX2 = 1330, 1330    # context + verification_ctx (two source copies)
L_REASON      = 820            # generated reasoning
L_QUESTION    = 56             # templated question
L_ANSWER      = 11            # "COVID-19"
L_SCAF        = 150           # ids/enums/status
TOTAL = L_CTX1 + L_CTX2 + L_REASON + L_QUESTION + L_ANSWER + L_SCAF

# palette
RED   = "#C0392B"; RED_L = "#F2D7D5"
GREEN = "#1E8449"; GREEN_L = "#D4EFDF"
ORANGE= "#CA6F1E"; ORANGE_L="#FAE5D3"
GREY  = "#7F8C8D"; GREY_L = "#EAEDED"
INK   = "#1B2631"

fig = plt.figure(figsize=(12, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.18)

# ============ LEFT: the record as a document card ============
axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
axL.set_xlim(0, 10); axL.set_ylim(0, 10)

# outer card
card = FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.1,rounding_size=0.25",
                      facecolor="white", edgecolor=INK, linewidth=1.4)
axL.add_patch(card)
axL.text(5, 9.35, "One QAR record  (patient 7665777-11)", ha="center", va="center",
         fontsize=11.5, fontweight="bold", color=INK)

# header row: question (templated)
y = 8.55
axL.add_patch(Rectangle((0.7, y-0.32), 8.6, 0.55, facecolor=ORANGE_L, edgecolor=ORANGE, lw=0.8))
axL.text(0.85, y-0.05, "QUESTION (templated, recurs across all patients)", fontsize=7.5,
         color=ORANGE, va="center", fontweight="bold")
axL.text(0.85, y-0.05, "", fontsize=7)
# the big copied-context block
y2_top = 7.95
y2_bot = 2.05
axL.add_patch(Rectangle((0.7, y2_bot), 8.6, y2_top-y2_bot, facecolor=RED_L, edgecolor=RED, lw=1.0))
axL.text(5.0, y2_top-0.35, "COPIED SOURCE NARRATIVE  \u00d7 2  (context + verification_ctx)",
         ha="center", fontsize=8.5, color=RED, fontweight="bold")
# faux text lines to suggest the copied narrative
import numpy as np
rng = np.random.default_rng(3)
ly = y2_top - 0.85
while ly > y2_bot + 0.85:
    w = rng.uniform(5.5, 8.0)
    axL.add_patch(Rectangle((1.0, ly), w, 0.12, facecolor=RED, alpha=0.30, edgecolor="none"))
    ly -= 0.38
axL.text(5.0, y2_bot+0.45, "2,660 characters  \u2014  72% of the record",
         ha="center", fontsize=8, color=RED, style="italic")

# the tiny extracted fact, highlighted, embedded
axL.add_patch(Rectangle((3.5, 4.7), 3.0, 0.6, facecolor=GREEN, edgecolor=INK, lw=1.2, zorder=5))
axL.text(5.0, 5.0, 'ANSWER: "COVID-19"', ha="center", va="center", fontsize=8.5,
         color="white", fontweight="bold", zorder=6)
axL.annotate("the extracted fact:\n11 characters (0.3%)",
             xy=(6.3, 5.0), xytext=(7.2, 6.2), fontsize=7.5, color=GREEN, fontweight="bold",
             ha="left", zorder=7,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

# reasoning block (generated)
yr = 1.85
axL.add_patch(Rectangle((0.7, yr-1.05), 8.6, 1.1, facecolor=GREEN_L, edgecolor=GREEN, lw=0.9))
axL.text(0.85, yr-0.2, "GENERATED REASONING (820 ch)", fontsize=7.5, color=GREEN,
         va="center", fontweight="bold")
ly = yr - 0.5
for _ in range(2):
    axL.add_patch(Rectangle((1.0, ly), rng.uniform(6,8), 0.10, facecolor=GREEN, alpha=0.30))
    ly -= 0.28

# ============ RIGHT: information-content summary ============
axR = fig.add_subplot(gs[0, 1]); axR.axis("off")
axR.set_xlim(0, 10); axR.set_ylim(0, 10)
axR.text(5, 9.4, "What is actually new?", ha="center", fontsize=11.5,
         fontweight="bold", color=INK)

trainable = L_ANSWER + L_REASON
redundant = L_CTX1 + L_CTX2
overhead  = L_QUESTION + L_SCAF

rows = [
    ("Redundant", "copied source", redundant, RED),
    ("Trainable", "newly generated", trainable, GREEN),
    ("Overhead", "template + scaffold", overhead, GREY),
]
maxv = redundant
y = 7.6
barx0 = 0.7
barmax = 7.5
for name, sub, val, color in rows:
    w = barmax * val / maxv
    axR.add_patch(FancyBboxPatch((barx0, y-0.45), w, 0.7,
                  boxstyle="round,pad=0.02,rounding_size=0.06",
                  facecolor=color, edgecolor="none"))
    if w > 1.6:
        axR.text(barx0+0.15, y-0.1, name, fontsize=9.5, color="white", fontweight="bold", va="center")
    else:
        axR.text(barx0+w+0.15, y-0.1, name, fontsize=9.5, color=color, fontweight="bold", va="center")
    cx = barx0 + w + (0.2 if w > 1.6 else 2.6)
    axR.text(cx, y-0.1, f"{val} ch  ({100*val/TOTAL:.0f}%)",
             fontsize=9, color=INK, va="center")
    axR.text(barx0, y-0.72, sub, fontsize=7.5, color=color, va="center", style="italic")
    y -= 2.05

# headline stat
axR.add_patch(FancyBboxPatch((0.7, 0.5), 8.6, 1.7,
              boxstyle="round,pad=0.05,rounding_size=0.12",
              facecolor="#FBEEE6", edgecolor=ORANGE, lw=1.0))
axR.text(5.0, 1.62, "The extracted fact is", ha="center", fontsize=9, color=INK)
axR.text(5.0, 1.05, "240\u00d7 smaller", ha="center", fontsize=16, color=RED, fontweight="bold")
axR.text(5.0, 0.65, "than the source text copied to support it", ha="center", fontsize=8, color=INK)

plt.savefig("figures/redundancy_fig5_worked_example.pdf", bbox_inches="tight")
plt.savefig("figures/redundancy_fig5_worked_example.png", dpi=160, bbox_inches="tight")
print("rewrote fig5 (document-style)")
print(f"total={TOTAL} redundant={redundant}({100*redundant/TOTAL:.0f}%) "
      f"trainable={trainable}({100*trainable/TOTAL:.0f}%)")
