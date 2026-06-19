#!/usr/bin/env python3
"""3x2 dumbbell grid: rows = depths (10k/20k/40k top->bottom),
columns = benchmarks (NCBI | BC5CDR). Each cell is a dumbbell over the four
slices. Lets the reader follow the effect down the depths and across datasets.

Reads per-depth result JSONs. Two input modes:
  (A) one combined file per dataset keyed by depth ->
      {depth:{cond:{slice:mean}}}   (use --ncbi-all / --bc5cdr-all)
  (B) the real pipeline files: NCBI three flat files + BC5CDR nested file
      (use --ncbi-10k --ncbi-20k --ncbi-40k --bc5cdr  ; see code)

For simplicity on the cluster use mode B (your real files).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, argparse
from pathlib import Path
from matplotlib.lines import Line2D

SLICES=["COMMON","RARE","UNSEEN","ALL"]
SLAB={"COMMON":"Common","RARE":"Rare","UNSEEN":"Unseen","ALL":"All"}
CONDS=["A_raw","B_dedup","B1_ctxremoved"]
RAW="#c0584b"; DED="#2f6fb2"; ABL="#e0a030"
NCBI_N={"COMMON":441,"RARE":152,"UNSEEN":367,"ALL":960}
BC_N={"COMMON":2016,"RARE":353,"UNSEEN":2055,"ALL":4424}

ap=argparse.ArgumentParser()
# mode A (combined test files)
ap.add_argument("--ncbi-all"); ap.add_argument("--bc5cdr-all")
# mode B (real pipeline files)
ap.add_argument("--ncbi-10k"); ap.add_argument("--ncbi-20k"); ap.add_argument("--ncbi-40k")
ap.add_argument("--bc5cdr")
ap.add_argument("--out",default=".")
a=ap.parse_args()

def flat(p):
    pc=json.load(open(p))["per_condition"]
    return {c:{sl:pc[c][sl]["f1_mean"] for sl in SLICES} for c in CONDS}

if a.ncbi_all and a.bc5cdr_all:
    ncbi=json.load(open(a.ncbi_all)); bc=json.load(open(a.bc5cdr_all))
else:
    ncbi={"10k":flat(a.ncbi_10k),"20k":flat(a.ncbi_20k),"40k":flat(a.ncbi_40k)}
    bcraw=json.load(open(a.bc5cdr))["per_depth"]
    bc={d:{c:{sl:bcraw[d]["per_condition"][c][sl]["f1_mean"] for sl in SLICES} for c in CONDS}
        for d in ["10k","20k","40k"]}

depths=["10k","20k","40k"]; dlab={"10k":"10,000 steps","20k":"20,000 steps","40k":"40,000 steps (primary)"}
datasets=[("NCBI-Disease",ncbi,NCBI_N),("BC5CDR-Disease",bc,BC_N)]

fig,axes=plt.subplots(len(depths),2,figsize=(14,11),dpi=140,sharex="col")
y=np.arange(len(SLICES))[::-1]
for ri,depth in enumerate(depths):
    for ci,(dsname,D,NS) in enumerate(datasets):
        ax=axes[ri][ci]; cell=D[depth]
        for yi,sl in zip(y,SLICES):
            a_=cell["A_raw"][sl]; b_=cell["B_dedup"][sl]; c_=cell["B1_ctxremoved"][sl]
            ax.plot([a_,b_],[yi,yi],color=DED,lw=6,alpha=0.18,solid_capstyle="round",zorder=1)
            ax.scatter(a_,yi,s=95,color=RAW,zorder=3,edgecolor="white",linewidth=1.0)
            ax.scatter(c_,yi,s=60,color=ABL,zorder=3,edgecolor="white",linewidth=1.0,marker="D")
            ax.scatter(b_,yi,s=95,color=DED,zorder=4,edgecolor="white",linewidth=1.0)
            ax.text((a_+b_)/2,yi+0.20,f"{b_-a_:+.3f}",ha="center",va="bottom",
                    fontsize=8,color=DED,weight="bold")
            if abs(b_-a_)<0.05:
                ax.text(a_,yi-0.18,f"{a_:.3f}",ha="center",va="top",fontsize=6.8,color=RAW)
                ax.text(b_,yi-0.40,f"{b_:.3f}",ha="center",va="top",fontsize=6.8,color=DED)
            else:
                ax.text(a_,yi-0.18,f"{a_:.3f}",ha="center",va="top",fontsize=6.8,color=RAW)
                ax.text(b_,yi-0.18,f"{b_:.3f}",ha="center",va="top",fontsize=6.8,color=DED)
        ax.set_ylim(-0.8,len(SLICES)-0.25); ax.set_xlim(0.40,0.92)
        ax.set_yticks(y); ax.set_yticklabels([f"{SLAB[s]}\n(n={NS[s]})" for s in SLICES],fontsize=8.5)
        ax.grid(axis="x",ls="--",color="#ddd",alpha=0.7); ax.set_axisbelow(True)
        for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y",length=0)
        if ri==0: ax.set_title(dsname,fontsize=12,weight="bold",pad=10)
        if ci==0: ax.set_ylabel(dlab[depth],fontsize=11,weight="bold")
        if ri==len(depths)-1: ax.set_xlabel("Disease-NER F1",fontsize=10)
legend=[Line2D([0],[0],marker='o',color='w',markerfacecolor=RAW,markersize=10,label='Raw'),
        Line2D([0],[0],marker='D',color='w',markerfacecolor=ABL,markersize=8,label='Ctx-removed'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=DED,markersize=10,label='De-dup')]
axes[0][1].legend(handles=legend,loc="lower right",frameon=False,fontsize=9)
fig.suptitle("De-duplication effect across adaptation depths and two benchmarks",
             fontsize=13,weight="bold",x=0.04,ha="left",y=1.0)
plt.tight_layout(rect=[0,0,1,0.985])
out=Path(a.out)
fig.savefig(out/"downstream_grid.pdf",bbox_inches="tight",facecolor="white")
fig.savefig(out/"downstream_grid.png",dpi=150,bbox_inches="tight",facecolor="white")
print("[done] wrote downstream_grid.{pdf,png}")
