#!/usr/bin/env python3
"""
make_fig4_composition.py
========================
Figure 4 (file) / compiled Figure 5 : per-channel token-provenance composition.
Extracted verbatim from make_all_figures.py (fig4). Self-contained.

Run:
    python3 make_fig4_composition.py
Outputs ./figures/redundancy_fig4_composition.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

OUT = "figures"; os.makedirs(OUT, exist_ok=True)

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
SRC="#0072B2"; UNIQ="#009E73"; SCAF="#B0B4B8"; DUP="#E69F00"; CTX="#D55E00"; INK="#1a1a1a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "text.color":INK,"svg.fonttype":"none"})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  wrote {name}")

def fig4():
    rows=[]
    for n,tot,cx,ug,dp,sc in CHANNELS:
        rows.append((n,100*cx/tot,100*dp/tot,100*ug/tot,100*sc/tot))
    labels=[r[0] for r in rows]
    ctx=np.array([r[1] for r in rows]); dup=np.array([r[2] for r in rows])
    uniq=np.array([r[3] for r in rows]); scaf=np.array([r[4] for r in rows])
    y=np.arange(len(labels))[::-1]
    fig,ax=plt.subplots(figsize=(11,6.6)); fig.patch.set_facecolor("white")
    bar_h=0.62; left=np.zeros(len(labels)); geom=[]
    for vals,col,key in [(ctx,CTX,"ctx"),(dup,DUP,"dup"),(uniq,UNIQ,"uniq"),(scaf,SCAF,"scaf")]:
        ax.barh(y,vals,left=left,height=bar_h,color=col,zorder=3,edgecolor="white",lw=1.2)
        for yi,x0,w in zip(y,left,vals): geom.append((yi,x0,w,key,col))
        left=left+vals
    cmap={"ctx":CTX,"dup":DUP,"uniq":UNIQ,"scaf":"#6b6f73"}
    for yi,x0,w,key,col in geom:
        if w<=0: continue
        cx0=x0+w/2
        if w>=8:
            ax.text(cx0,yi,f"{w:.0f}",va="center",ha="center",fontsize=8.5,color="white",fontweight="bold",zorder=5)
        elif w>=2:
            yt=yi+bar_h/2+0.10
            ax.plot([cx0,cx0],[yi+bar_h/2,yt-0.02],color=cmap[key],lw=0.7,zorder=4)
            ax.text(cx0,yt,f"{w:.0f}",va="bottom",ha="center",fontsize=7,color=cmap[key],fontweight="bold",zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=10.5)
    ax.set_xlim(0,100); ax.set_ylim(-0.7,len(labels)-0.2)
    ax.set_xlabel("Share of channel tokens (%)",fontsize=10.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(axis="x",labelsize=9,color="#ccc"); ax.tick_params(axis="y",length=0)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#ccc"); ax.set_axisbelow(True)
    ax.xaxis.grid(True,color="#ededed",lw=0.8,zorder=0)
    leg=[Patch(facecolor=CTX,label="Copied context  (redundant)"),
         Patch(facecolor=DUP,label="Duplicated generation  (redundant)"),
         Patch(facecolor=UNIQ,label="Unique generated  (trainable)"),
         Patch(facecolor=SCAF,label="Scaffold")]
    ax.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,1.04),ncol=4,
              fontsize=8.6,frameon=False,handlelength=1.1,columnspacing=1.6,handletextpad=0.5)
    ax.set_title("Per-channel token-provenance composition",fontsize=13,fontweight="bold",pad=40,loc="left",x=0.0)
    fig.tight_layout(); save(fig,"redundancy_fig4_composition")

if __name__=="__main__": fig4()
