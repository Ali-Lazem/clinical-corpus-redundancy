#!/usr/bin/env python3
"""
make_fig2_pertask.py
====================
Figure 2 (file) / compiled Figure 3 : two redundancy mechanisms per channel.
Extracted verbatim from make_all_figures.py (fig2). Self-contained: the
verified ten-channel numbers (redundancy_full_v2.json) are embedded.

Run:
    python3 make_fig2_pertask.py
Outputs ./figures/redundancy_fig2_pertask.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"; os.makedirs(OUT, exist_ok=True)

# per-channel: (name, total_M, ctx_M, uniqgen_M, dupgen_M, scaffold_M)
CHANNELS = [
    ("QAR",            1267.9, 1026.8, 86.2, 122.6, 32.3),
    ("RE",              732.6,  585.7,  0.0,  51.3, 95.5),
    ("Temporal events", 156.5,   73.8,  0.0,  32.8, 50.0),
    ("Risk-QA",          56.8,    0.0,  2.0,  29.1, 25.6),
    ("Summary",          48.3,    0.0, 48.3,   0.0,  0.0),
    ("NER",              47.5,    0.0, 21.7,   9.4, 16.4),
    ("Risk-states",      44.4,    8.9,  7.6,  20.4,  7.5),
    ("Recommendations",  33.3,    0.0,  1.5,  23.5,  8.2),
    ("Risks",             9.9,    0.0,  0.0,   6.8,  3.2),
    ("Medications",       9.3,    0.0,  1.1,   2.4,  5.8),
]
CTX="#D55E00"; DUP="#E69F00"; INK="#1a1a1a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "text.color":INK,"svg.fonttype":"none"})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  wrote {name}")

def fig2():
    data=[(n, round(100*cx/tot,1), round(100*dp/tot,1)) for (n,tot,cx,_,dp,_) in CHANNELS]
    labels=[d[0] for d in data]; ctx=[d[1] for d in data]; dup=[d[2] for d in data]
    y=np.arange(len(labels)); h=0.38
    fig,ax=plt.subplots(figsize=(10,6.2))
    ax.barh(y+h/2,ctx,height=h,color=CTX,label="Copied context (storage redundancy)",zorder=3)
    ax.barh(y-h/2,dup,height=h,color=DUP,label="Duplicated generation (output-regularity redundancy)",zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=10); ax.invert_yaxis()
    ax.set_xlabel("% of channel tokens",fontsize=10); ax.set_xlim(0,100)
    ax.grid(axis="x",alpha=0.25,zorder=0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for yi,c,d in zip(y,ctx,dup):
        if c>0: ax.text(c+1,yi+h/2,f"{c:.0f}",va="center",fontsize=7.5,color=CTX)
        if d>0: ax.text(d+1,yi-h/2,f"{d:.0f}",va="center",fontsize=7.5,color="#b07d00")
    ax.legend(loc="lower right",fontsize=8.5,framealpha=0.95)
    ax.set_title("Two redundancy mechanisms across ten text-bearing channels",fontsize=12,fontweight="bold",pad=12)
    fig.tight_layout(); save(fig,"redundancy_fig2_pertask")

if __name__=="__main__": fig2()
