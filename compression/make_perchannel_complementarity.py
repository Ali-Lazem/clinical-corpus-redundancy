#!/usr/bin/env python3
"""
Per-channel complementarity figure (Option B) - dumbbell/connected-dot design.
The CONNECTING LINE length = the gap between the two methods, which is the
actual subject: provenance varies (where redundancy comes from), compression
saturates high (stylistic predictability everywhere).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---- design tokens (deliberate, derived from the clinical-data subject) ----
INK      = "#1f1b18"   # near-black text
PROV     = "#b5241a"   # deep clinical red  -> provenance (where it came from)
COMP     = "#e8913a"   # warm amber         -> compression (how predictable)
GAPLINE  = "#cfc4ba"   # quiet stone        -> the connecting gap
GRID     = "#ece7e1"
AGREE    = "#7a9a7e"   # muted sage, used once to mark the agreement zone

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# locked numbers: (channel, provenance redundancy %, compression reduction %)
DATA = [
    ("QAR",              90.7, 96.9),
    ("RE",               87.0, 96.3),
    ("recommendations",  70.6, 97.9),
    ("risks",            68.1, 87.1),
    ("temporal events",  68.0, 80.6),
    ("risk states",      66.1, 92.0),
    ("risk QAR",         51.3, 95.5),
    ("medications",      25.9, 79.4),
    ("NER",              19.8, 75.7),
    ("summary",           0.0, 81.0),
]
DATA.sort(key=lambda r: r[1])               # ascending prov, so divergence grows downward
chs  = [d[0] for d in DATA]
prov = np.array([d[1] for d in DATA])
comp = np.array([d[2] for d in DATA])
gap  = comp - prov
y    = np.arange(len(chs))

fig, ax = plt.subplots(figsize=(9.2, 5.4))

# connecting lines (the gap) - thickness encodes gap size subtly
for i in range(len(chs)):
    ax.plot([prov[i], comp[i]], [y[i], y[i]], color=GAPLINE,
            lw=1.0 + 2.6*gap[i]/gap.max(), solid_capstyle="round", zorder=1)

# dots
ax.scatter(prov, y, s=88, color=PROV, edgecolor="white", linewidth=1.0,
           zorder=3, label="Provenance redundancy (copied + duplicated)")
ax.scatter(comp, y, s=88, color=COMP, edgecolor="white", linewidth=1.0,
           zorder=3, label="Compression reduction (LZMA)")

# value labels at each dot (small, only where they don't collide)
for i in range(len(chs)):
    # provenance label to the left of its dot
    ax.text(prov[i]-2.2, y[i], f"{prov[i]:.0f}", va="center", ha="right",
            fontsize=8, color=PROV, fontweight="bold")
    # compression label to the right of its dot
    ax.text(comp[i]+2.2, y[i], f"{comp[i]:.0f}", va="center", ha="left",
            fontsize=8, color=COMP, fontweight="bold")

ax.set_yticks(y); ax.set_yticklabels(chs, fontsize=10)
ax.set_xlim(-8, 112); ax.set_xlabel("% of channel", fontsize=10)
ax.set_xticks([0,20,40,60,80,100])

# right-edge annotation: name the two regimes (the signature element)
ax.annotate("methods agree here\n(redundancy is copied text)",
            xy=(93.5, y[-1]), xytext=(70, y[-1]-1.15),
            fontsize=8.2, color="#5f6b60", ha="center", style="italic",
            arrowprops=dict(arrowstyle="-", color="#b9c4ba", lw=0.8))
ax.annotate("methods diverge\n(compressible by style,\nnot by copying)",
            xy=(81.0, y[0]), xytext=(42, y[0]+0.0),
            fontsize=8.2, color="#8a5a2a", ha="center", style="italic",
            arrowprops=dict(arrowstyle="-", color="#cbb9a6", lw=0.8))

ax.set_title("Provenance and compression measure complementary things",
             fontsize=13, fontweight="bold", loc="left", pad=26)
fig.text(0.125, 0.905,
         "Compression stays high everywhere; provenance reveals which channels are redundant by copying.",
         fontsize=9, color="#6a635c", ha="left", va="bottom")

ax.grid(axis="x", color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
ax.legend(loc="lower right", fontsize=8.6, frameon=False, bbox_to_anchor=(1.0, -0.16), ncol=1)
ax.set_ylim(-0.8, len(chs)-0.2)

plt.tight_layout()
fig.savefig("/mnt/user-data/outputs/redundancy_fig_perchannel_complementarity.pdf", bbox_inches="tight")
fig.savefig("/mnt/user-data/outputs/redundancy_fig_perchannel_complementarity.png", dpi=200, bbox_inches="tight")
print("[done] dumbbell figure written")
