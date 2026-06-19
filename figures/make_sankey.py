#!/usr/bin/env python3
"""Figure 1: source-to-output redundancy flow (original layout, corrected numbers)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PALETTE = {
    "source":"#0072B2","unique_gen":"#009E73","scaffold":"#999999",
    "dup_gen":"#E69F00","copied_ctx":"#D55E00","pipeline_bar":"#3A3A3A",
    "funnel_fill":"#56B4E9","text_dark":"#222222","text_red":"#B22222",
    "text_green":"#1B7A57",
}
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "text.color":PALETTE["text_dark"], "svg.fonttype":"none"})

fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(-1.8,11.2); ax.set_ylim(-200,2680); ax.axis("off")

v_source=104; v_unique_gen=159; v_scaffold=213; v_dup_gen=260; v_copied_ctx=1686
v_total=v_source+v_unique_gen+v_scaffold+v_dup_gen+v_copied_ctx

y_src_b,y_src_t=0,v_source
y_ugen_b,y_ugen_t=y_src_t,y_src_t+v_unique_gen
y_scaf_b,y_scaf_t=y_ugen_t,y_ugen_t+v_scaffold
y_dup_b,y_dup_t=y_scaf_t,y_scaf_t+v_dup_gen
y_ctx_b,y_ctx_t=y_dup_t,y_dup_t+v_copied_ctx

left_center=v_total/2
y_left_b=left_center-v_source/2; y_left_t=left_center+v_source/2

x_l0,x_l1=-0.5,0.0
ax.add_patch(patches.FancyBboxPatch((x_l0,y_left_b),x_l1-x_l0,y_left_t-y_left_b,
    boxstyle="round,pad=0,rounding_size=8",facecolor=PALETTE["source"],
    edgecolor="#0a3d5c",linewidth=1.0,zorder=4))
ax.text(x_l0-0.22,left_center,"Source\nnarratives\n104\u2009M",ha="right",va="center",
    fontsize=11,fontweight="bold",linespacing=1.35)
ax.text(x_l0-0.22,y_left_b-120,"deduplicated",ha="right",va="top",
    fontsize=8,style="italic",color="#666666")

x_f0,x_f1=0.0,3.5
xc=np.linspace(x_f0,x_f1,200); t=(xc-x_f0)/(x_f1-x_f0); s=3*t**2-2*t**3
y_fb=y_left_b+(0-y_left_b)*s; y_ft=y_left_t+(v_total-y_left_t)*s
for a in [0.10,0.12]:
    ax.fill_between(xc,y_fb,y_ft,facecolor=PALETTE["funnel_fill"],alpha=a,zorder=1)
ax.plot(xc,y_fb,color=PALETTE["funnel_fill"],lw=1.3,alpha=0.55,zorder=2)
ax.plot(xc,y_ft,color=PALETTE["funnel_fill"],lw=1.3,alpha=0.55,zorder=2)
ax.text((x_f0+x_f1)/2,left_center+360,"18.7\u00d7 more redundant\nthan source",ha="center",
    va="center",fontsize=10,style="italic",color=PALETTE["pipeline_bar"],linespacing=1.25)

x_m0,x_m1=3.5,3.82
ax.add_patch(patches.Rectangle((x_m0,0),x_m1-x_m0,v_total,
    facecolor=PALETTE["pipeline_bar"],edgecolor=None,zorder=3))
ax.text((x_m0+x_m1)/2,v_total+90,"Pipeline output\n2.42\u2009B tokens",ha="center",
    va="bottom",fontsize=11,fontweight="bold",linespacing=1.2)

x_b0,x_b1=3.82,5.85
segs=[(y_src_b,y_src_t,PALETTE["source"]),
      (y_ugen_b,y_ugen_t,PALETTE["unique_gen"]),
      (y_scaf_b,y_scaf_t,PALETTE["scaffold"]),
      (y_dup_b,y_dup_t,PALETTE["dup_gen"]),
      (y_ctx_b,y_ctx_t,PALETTE["copied_ctx"])]
for b,tp,c in segs:
    ax.add_patch(patches.Rectangle((x_b0,b),x_b1-x_b0,tp-b,facecolor=c,
        edgecolor="#FFFFFF",linewidth=0.8,alpha=0.95,zorder=4))
for b,tp,c in segs:
    ax.add_patch(patches.Polygon([(x_m1,b),(x_b0,b),(x_b0,tp),(x_m1,tp)],
        closed=True,facecolor=c,alpha=0.16,edgecolor="none",zorder=2))

labels=[(y_src_b,y_src_t,"Unique source \u2013 trainable (104\u2009M)"),
        (y_ugen_b,y_ugen_t,"Unique generated \u2013 trainable (159\u2009M)"),
        (y_scaf_b,y_scaf_t,"Scaffold (213\u2009M)"),
        (y_dup_b,y_dup_t,"Duplicated generated \u2013 redundant (260\u2009M)"),
        (y_ctx_b,y_ctx_t,"Copied context \u2013 redundant (1,686\u2009M)")]
for b,tp,txt in labels:
    ax.text(x_b1+0.18,b+(tp-b)/2,txt,ha="left",va="center",fontsize=9.5,linespacing=1.2)

xbk=x_b1+3.55
y_red_c=y_dup_b+(y_ctx_t-y_dup_b)/2
ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_ctx_t,y_ctx_t,y_dup_b,y_dup_b],
    color=PALETTE["text_red"],lw=1.2)
ax.text(xbk+0.24,y_red_c,"80.3%\nredundant",ha="left",va="center",fontsize=11,
    fontweight="bold",color=PALETTE["text_red"],linespacing=1.2)
y_tr_c=y_src_b+(y_ugen_t-y_src_b)/2
ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_ugen_t,y_ugen_t,y_src_b,y_src_b],
    color=PALETTE["text_green"],lw=1.2)
ax.text(xbk+0.24,y_tr_c,"10.9%\ntrainable",ha="left",va="center",fontsize=11,
    fontweight="bold",color=PALETTE["text_green"],linespacing=1.2)
ax.plot([xbk,xbk+0.12,xbk+0.12,xbk],[y_scaf_t,y_scaf_t,y_scaf_b,y_scaf_b],
    color="#777777",lw=1.0)
ax.text(xbk+0.24,y_scaf_b+(y_scaf_t-y_scaf_b)/2,"8.8%\nscaffold",ha="left",
    va="center",fontsize=9.5,fontweight="bold",color="#666666",linespacing=1.2)

fig.text(0.5,0.085,"Redundant content is 18.7\u00d7 the unique source (1.95\u2009B vs 104\u2009M)",
    ha="center",fontsize=10,style="italic",color="#555555")
fig.text(0.5,0.035,"From 104\u2009M source tokens to 2.42\u2009B output tokens: only 10.9% is trainable-unique content",
    ha="center",fontsize=11.5,fontweight="bold")

plt.savefig("/mnt/user-data/outputs/redundancy_fig1_sankey.png",dpi=300,
    bbox_inches="tight",facecolor="white")
plt.savefig("/mnt/user-data/outputs/redundancy_fig1_sankey.pdf",
    bbox_inches="tight",facecolor="white")
print("[OK] rendered redundancy_fig1_sankey")
plt.close()
