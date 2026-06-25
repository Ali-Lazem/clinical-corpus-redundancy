#!/usr/bin/env python3
"""
make_compression_fig.py  (v3 - professional redesign, all five categories)
===========================================================================
Paper-2 compression figure, two panels in a clean editorial style:

  (a) Corpus composition: horizontal stacked bar over ALL FIVE provenance
      categories (unique source, unique generated, copied context, duplicated
      generation, scaffold), grouped visually into trainable / redundant /
      scaffold with a bracket annotation.
  (b) Compressibility: grouped bars, compression ratio per stream x compressor,
      reduction annotated, with the headline full-vs-trainable factor called out.

Reads compression_full.json (full-corpus run) for panel (b); reads
redundancy_full_v2.json["global"] for the token composition in panel (a).

Usage:
  python3 make_compression_fig.py \
      --json /scratch/.../reports/compression_full.json \
      --prov /scratch/.../reports/redundancy_full_v2.json \
      --out  redundancy_fig_compression.pdf
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ---- editorial palette (cohesive reds/teal/grey, matched to fig2) ----
C_SRC   = "#4a7c8c"   # unique source  (teal -- trainable, informative)
C_GENU  = "#7fb3c4"   # unique generated (light teal -- trainable)
C_CTX   = "#cc2a1e"   # copied context (deep red -- dominant redundancy)
C_DUP   = "#f4a259"   # duplicated generation (orange -- redundancy)
C_SCAF  = "#cfcfcf"   # scaffold (neutral grey)

COMP_COLORS = {"gzip":"#f6c89a","bzip2":"#ef8a52","lzma":"#cc2a1e","ppmd":"#7a1d14"}
COMP_LABELS = {"gzip":"gzip","bzip2":"bzip2","lzma":"LZMA","ppmd":"PPMD"}
COMP_ORDER  = ["gzip","bzip2","lzma","ppmd"]

STREAM_ORDER  = ["full","trainable","copied_ctx","dup_gen"]
STREAM_LABELS = {"full":"Full\ncorpus","trainable":"Trainable-\nunique",
                 "copied_ctx":"Copied\ncontext","dup_gen":"Duplicated\ngeneration"}

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.edgecolor":"#333333","axes.linewidth":0.8,
})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json",required=True)
    ap.add_argument("--prov",required=True,help="redundancy_full_v2.json")
    ap.add_argument("--out",default="redundancy_fig_compression.pdf")
    a=ap.parse_args()

    d=json.load(open(a.json)); sres=d["streams"]; factor=d.get("full_vs_trainable_factor",{})
    g=json.load(open(a.prov))["global"]

    # five categories, in stacked order: source, gen-unique | copied-ctx, dup-gen | scaffold
    cats=[("Unique source",g["unique_source_tokens"],C_SRC),
          ("Unique generated",g["unique_generated_tokens"],C_GENU),
          ("Copied context",g["copied_context_tokens"],C_CTX),
          ("Duplicated generation",g["duplicated_generated_tokens"],C_DUP),
          ("Scaffold",g["scaffold_tokens"],C_SCAF)]
    total=sum(v for _,v,_ in cats)

    fig,(axA,axB)=plt.subplots(1,2,figsize=(12.5,4.4),
                               gridspec_kw={"width_ratios":[1.15,1.0]})

    # ===== Panel (a): stacked composition, all five categories =====
    left=0.0; centers={}
    for name,v,col in cats:
        p=100*v/total
        axA.barh(0,p,left=left,color=col,edgecolor="white",linewidth=1.4,height=0.5)
        centers[name]=left+p/2
        # % label: inside if wide, else below with a small leader
        if p>=5.0:
            txtcol="white" if col in (C_CTX,C_SRC) else "#222"
            axA.text(left+p/2,0,f"{p:.1f}%",ha="center",va="center",
                     fontsize=9,fontweight="bold",color=txtcol)
        else:
            axA.text(left+p/2,-0.34,f"{p:.1f}%",ha="center",va="top",
                     fontsize=7.8,fontweight="bold",color=col)
            axA.plot([left+p/2,left+p/2],[-0.25,-0.05],color=col,lw=0.7,clip_on=False)
        left+=p

    # grouping brackets above: trainable (src+genu), redundant (ctx+dup), scaffold
    tur=100*(cats[0][1]+cats[1][1])/total
    red=100*(cats[2][1]+cats[3][1])/total
    scaf=100*cats[4][1]/total
    def bracket(x0,x1,label,color):
        y=0.40
        axA.plot([x0,x0,x1,x1],[y,y+0.06,y+0.06,y],color=color,lw=1.3,clip_on=False)
        axA.text((x0+x1)/2,y+0.13,label,ha="center",va="bottom",
                 fontsize=9.2,fontweight="bold",color=color,clip_on=False)
    bracket(0,tur,f"Trainable-unique  {tur:.1f}%","#2f6360")
    bracket(tur,tur+red,f"Redundant  {red:.1f}%","#8c1a12")
    bracket(tur+red,100,f"Scaffold  {scaf:.1f}%","#777")

    axA.set_xlim(0,100); axA.set_ylim(-0.62,0.78)
    axA.set_yticks([]); axA.set_xlabel("% of corpus tokens",fontsize=10)
    axA.set_title("(a) Corpus composition by provenance",fontsize=11.5,loc="left",
                  fontweight="bold",pad=26)
    for sp in ["top","right","left"]: axA.spines[sp].set_visible(False)
    # legend (all five) below
    handles=[mpatches.Patch(color=col,label=name) for name,_,col in cats]
    axA.legend(handles=handles,fontsize=8,ncol=3,frameon=False,
               loc="upper center",bbox_to_anchor=(0.5,-0.16),
               handlelength=1.1,columnspacing=1.3)

    # ===== Panel (b): grouped compression bars (polished) =====
    nS,nC=len(STREAM_ORDER),len(COMP_ORDER); bw=0.185; base=np.arange(nS)
    ymax=max(sres[s].get(c,{}).get("ratio",0) for s in STREAM_ORDER for c in COMP_ORDER)
    # subtle vertical band behind every other stream group for readability
    for si in range(nS):
        if si%2==1:
            axB.axvspan(si-0.5,si+0.5,color="#f7f7f7",zorder=0)
    for ci,cp in enumerate(COMP_ORDER):
        xs=base+(ci-(nC-1)/2)*bw
        vals=[sres.get(s,{}).get(cp,{}).get("ratio",np.nan) for s in STREAM_ORDER]
        axB.bar(xs,vals,width=bw,color=COMP_COLORS[cp],edgecolor="#333",
                linewidth=0.5,label=COMP_LABELS[cp],zorder=3)
        # value labels atop bars (small)
        for x,v in zip(xs,vals):
            if v==v:  # not nan
                axB.text(x,v+ymax*0.018,f"{v:.3f}",ha="center",va="bottom",
                         fontsize=6.4,color="#444",rotation=90,zorder=4)
    axB.set_xticks(base); axB.set_xticklabels([STREAM_LABELS[s] for s in STREAM_ORDER],fontsize=9)
    axB.set_ylabel("Compression ratio  (compressed / raw)",fontsize=10)
    axB.set_xlim(-0.5,nS-0.5); axB.set_ylim(0,ymax*1.30)
    axB.set_title("(b) Compressibility by provenance stream",fontsize=11.5,loc="left",
                  fontweight="bold",pad=26)
    axB.legend(fontsize=8.5,ncol=4,frameon=False,loc="upper center",
               bbox_to_anchor=(0.5,1.06),columnspacing=1.4,handlelength=1.1)
    axB.grid(axis="y",color="#e8e8e8",linewidth=0.8,zorder=1); axB.set_axisbelow(True)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False)
    # "lower = more redundant" as a right-aligned x-axis sublabel (no collision)
    axB.annotate("lower ratio = more redundant",xy=(1.0,-0.20),xycoords="axes fraction",
                 ha="right",va="top",fontsize=7.8,style="italic",color="#888")
    # headline factor as a clean text callout in the open upper area between
    # the Full and Trainable groups (no arrow crossing bars)
    fac_lzma=factor.get("lzma")
    if fac_lzma:
        axB.text(0.5,ymax*1.18,
                 f"Full corpus compresses {fac_lzma:.1f}$\\times$ harder\nthan its trainable-unique subset (LZMA)",
                 ha="center",va="top",fontsize=7.8,color="#8c1a12",
                 fontweight="bold",linespacing=1.3,zorder=5)

    fig.suptitle("Compression independently confirms the redundancy decomposition",
                 fontsize=12.5,fontweight="bold",x=0.012,ha="left",y=1.02)
    plt.tight_layout()
    fig.savefig(a.out,bbox_inches="tight",facecolor="white")
    fig.savefig(a.out.replace(".pdf",".png"),dpi=180,bbox_inches="tight",facecolor="white")
    print(f"[done] wrote {a.out}")
    print(f"composition: TUR {tur:.1f}%  redundant {red:.1f}%  scaffold {scaf:.1f}%")
    print(f"factor: {factor}")

if __name__=="__main__": main()
