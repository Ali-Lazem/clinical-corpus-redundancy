#!/usr/bin/env python3
"""Three-facet loss-curve figure: one sub-panel per adaptation depth
(10k / 20k / 40k), each overlaying the three conditions' MLM training loss.
Reads the .out job files directly (the {'loss': ...} log lines).

Usage:
  python3 make_loss_panel.py \
    --d10 mlm_A_raw_s1_<10k>.out mlm_B_dedup_s1_<10k>.out mlm_B1_ctxremoved_s1_<10k>.out \
    --d20 mlm_A_raw_s1_<20k>.out mlm_B_dedup_s1_<20k>.out mlm_B1_ctxremoved_s1_<20k>.out \
    --d40 mlm_A_raw_s1_<40k>.out mlm_B_dedup_s1_<40k>.out mlm_B1_ctxremoved_s1_<40k>.out \
    --out .
Pass the three condition files (raw, dedup, ctx) in that order for each depth.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re, argparse
from pathlib import Path

def trace(path):
    losses=[]; pat=re.compile(r"'loss': ([0-9.]+)")
    with open(path) as f:
        for line in f:
            m=pat.search(line)
            if m: losses.append(float(m.group(1)))
    steps=[(i+1)*100 for i in range(len(losses))]
    return steps, losses

ap=argparse.ArgumentParser()
ap.add_argument("--d10",nargs=3,required=True)
ap.add_argument("--d20",nargs=3,required=True)
ap.add_argument("--d40",nargs=3,required=True)
ap.add_argument("--out",default=".")
a=ap.parse_args()

depths=[("10,000 steps",a.d10),("20,000 steps",a.d20),
        ("40,000 steps (primary)",a.d40)]
condlabel=["Raw","De-dup","Ctx-removed"]
colors=["#c0584b","#2f6fb2","#e0a030"]

fig,axes=plt.subplots(1,3,figsize=(14,4.4),dpi=140,sharey=True)
for ax,(title,files) in zip(axes,depths):
    for path,lab,col in zip(files,condlabel,colors):
        s,l=trace(path)
        if l: ax.plot(s,l,lw=1.1,color=col,label=lab,alpha=0.9)
    ax.set_title(title,fontsize=11,weight="bold")
    ax.set_xlabel("MLM step",fontsize=10)
    ax.grid(True,ls="--",color="#ddd",alpha=0.6); ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("MLM training loss",fontsize=11)
axes[2].legend(frameon=False,fontsize=9.5,loc="upper right")
fig.suptitle("MLM adaptation loss by step, across the three adaptation depths "
             "(seed 1; other seeds similar)",
             fontsize=12.5,weight="bold",x=0.04,ha="left",y=1.02)
plt.tight_layout(rect=[0,0,1,0.96])
out=Path(a.out)
fig.savefig(out/"loss_panel.pdf",bbox_inches="tight",facecolor="white")
fig.savefig(out/"loss_panel.png",dpi=170,bbox_inches="tight",facecolor="white")
print("[done] wrote loss_panel.{pdf,png}")
