#!/usr/bin/env python3
"""
make_all_figures.py
===================
Regenerates all four figures for the redundancy paper from the verified
ten-channel provenance counts (redundancy_full_v2.json). Self-contained:
the aggregate numbers are embedded below, so no external data file is
needed to reproduce the figures.

Outputs (PNG + PDF) into ./figures/:
  redundancy_fig1_sankey       - source -> output flow, four-bucket split
  redundancy_fig2_pertask      - two redundancy mechanisms per channel
  redundancy_fig3_schematic    - the context-copy mechanism (illustrative)
  redundancy_fig4_composition  - 100% stacked per-channel provenance

Run:
    pip install matplotlib numpy
    python3 make_all_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# VERIFIED TEN-CHANNEL NUMBERS (from redundancy_full_v2.json, full 167,034)
# ---------------------------------------------------------------------------
# global (millions of tokens)
G = dict(source=104.2, unique_gen=168.5, dup_gen=298.3, ctx=1695.2,
         scaffold=244.6, total=2510.7)
# derived
G_trainable = G["source"] + G["unique_gen"]          # 272.7M ~ 272.6M
G_redundant = G["ctx"] + G["dup_gen"]                # 1993.5M

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

# palette (Okabe-Ito derived; shared across figures)
SRC="#0072B2"; UNIQ="#009E73"; SCAF="#B0B4B8"; DUP="#E69F00"; CTX="#D55E00"
PIPE="#3A3A3A"; FUNNEL="#56B4E9"; INK="#1a1a1a"; RED="#B22222"; GRN="#1B7A57"

plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "text.color":INK,"svg.fonttype":"none"})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")

# ===========================================================================
# FIGURE 1  -  source-to-output Sankey-style flow
# ===========================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1.8, 11.2); ax.set_ylim(-200, 2780); ax.axis("off")
    vs, vu, vsc, vd, vc = 104, 168, 245, 298, 1695
    vt = vs+vu+vsc+vd+vc
    y_src_b, y_src_t = 0, vs
    y_ug_b, y_ug_t = y_src_t, y_src_t+vu
    y_sc_b, y_sc_t = y_ug_t, y_ug_t+vsc
    y_dp_b, y_dp_t = y_sc_t, y_sc_t+vd
    y_cx_b, y_cx_t = y_dp_t, y_dp_t+vc
    lc = vt/2; ylb, ylt = lc-vs/2, lc+vs/2
    ax.add_patch(patches.FancyBboxPatch((-0.5,ylb),0.5,ylt-ylb,
        boxstyle="round,pad=0,rounding_size=8",facecolor=SRC,edgecolor="#0a3d5c",lw=1,zorder=4))
    ax.text(-0.72,lc,"Source\nnarratives\n104\u2009M",ha="right",va="center",fontsize=11,fontweight="bold",linespacing=1.35)
    ax.text(-0.72,ylb-120,"deduplicated",ha="right",va="top",fontsize=8,style="italic",color="#666")
    xc=np.linspace(0,3.5,200); t=xc/3.5; s=3*t**2-2*t**3
    yfb=ylb+(0-ylb)*s; yft=ylt+(vt-ylt)*s
    for a in (0.10,0.12): ax.fill_between(xc,yfb,yft,facecolor=FUNNEL,alpha=a,zorder=1)
    ax.plot(xc,yfb,color=FUNNEL,lw=1.3,alpha=0.55,zorder=2); ax.plot(xc,yft,color=FUNNEL,lw=1.3,alpha=0.55,zorder=2)
    ax.text(1.75,lc+380,"19.1\u00d7 more redundant\nthan source",ha="center",va="center",
            fontsize=10,style="italic",color=PIPE,linespacing=1.25)
    ax.add_patch(patches.Rectangle((3.5,0),0.32,vt,facecolor=PIPE,zorder=3))
    ax.text(3.66,vt+90,"Pipeline output\n2.51\u2009B tokens",ha="center",va="bottom",fontsize=11,fontweight="bold",linespacing=1.2)
    segs=[(y_src_b,y_src_t,SRC),(y_ug_b,y_ug_t,UNIQ),(y_sc_b,y_sc_t,SCAF),(y_dp_b,y_dp_t,DUP),(y_cx_b,y_cx_t,CTX)]
    for b,tp,c in segs:
        ax.add_patch(patches.Rectangle((3.82,b),2.03,tp-b,facecolor=c,edgecolor="white",lw=0.8,alpha=0.95,zorder=4))
        ax.add_patch(patches.Polygon([(3.82,b),(3.82,b),(3.82,tp),(3.82,tp)],closed=True,facecolor=c,alpha=0.16,zorder=2))
    labels=[(y_src_b,y_src_t,"Unique source \u2013 trainable (104\u2009M)"),
            (y_ug_b,y_ug_t,"Unique generated \u2013 trainable (168\u2009M)"),
            (y_sc_b,y_sc_t,"Scaffold (245\u2009M)"),
            (y_dp_b,y_dp_t,"Duplicated generated \u2013 redundant (298\u2009M)"),
            (y_cx_b,y_cx_t,"Copied context \u2013 redundant (1,695\u2009M)")]
    for b,tp,txt in labels: ax.text(6.03,b+(tp-b)/2,txt,ha="left",va="center",fontsize=9.5)
    xbk=9.4
    ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_cx_t,y_cx_t,y_dp_b,y_dp_b],color=RED,lw=1.2)
    ax.text(xbk+0.24,(y_dp_b+y_cx_t)/2,"79.4%\nredundant",ha="left",va="center",fontsize=11,fontweight="bold",color=RED,linespacing=1.2)
    ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_ug_t,y_ug_t,y_src_b,y_src_b],color=GRN,lw=1.2)
    ax.text(xbk+0.24,(y_src_b+y_ug_t)/2,"10.9%\ntrainable",ha="left",va="center",fontsize=11,fontweight="bold",color=GRN,linespacing=1.2)
    ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_sc_t,y_sc_t,y_sc_b,y_sc_b],color="#777",lw=1.0)
    ax.text(xbk+0.24,(y_sc_b+y_sc_t)/2,"9.7%\nscaffold",ha="left",va="center",fontsize=9.5,fontweight="bold",color="#666",linespacing=1.2)
    fig.text(0.5,0.085,"Redundant content is 19.1\u00d7 the unique source (1.99\u2009B vs 104\u2009M)",ha="center",fontsize=10,style="italic",color="#555")
    fig.text(0.5,0.035,"From 104\u2009M source tokens to 2.51\u2009B output tokens: only 10.9% is trainable-unique content",ha="center",fontsize=11.5,fontweight="bold")
    save(fig,"redundancy_fig1_sankey")

# ===========================================================================
# FIGURE 2  -  two redundancy mechanisms (grouped bars)
# ===========================================================================
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

# ===========================================================================
# FIGURE 3  -  context-copy mechanism schematic (illustrative)
# ===========================================================================
def fig3():
    # Fig 3 is a schematic; rendered by the proven standalone make_fig3.py
    import runpy
    runpy.run_path("make_fig3.py")
    print("  wrote redundancy_fig3_schematic")

def fig5():
    # Fig 5 is the single-record worked example; rendered by its standalone script
    import runpy
    runpy.run_path("make_fig5_worked_example.py")
    print("  wrote redundancy_fig5_worked_example")

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

if __name__=="__main__":
    print("Generating figures into ./figures/ ...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("Done.")
