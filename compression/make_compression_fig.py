#!/usr/bin/env python3
"""
make_compression_fig.py
=======================
Paper-2 compression figure. Two panels:
  (a) raw size of each provenance stream (FULL / TRAINABLE / COPIED_CTX /
      DUP_GEN) -- shows how much of the corpus is removable.
  (b) compression ratio of each stream under the four compressors -- shows
      FULL and the two redundant streams compress far harder than TRAINABLE.

Reads compression_redundancy_v4.json (full-run output). Falls back to the
aligned 2% test file if the full run is not yet present. Regenerates
automatically from whichever JSON is supplied.

Usage:
  python3 make_compression_fig.py \
      --json /path/to/reports/compression_redundancy_v4.json \
      --out  redundancy_fig_compression.pdf
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STREAM_ORDER  = ["full", "trainable", "copied_ctx", "dup_gen"]
STREAM_LABELS = {"full": "Full corpus", "trainable": "Trainable-unique",
                 "copied_ctx": "Copied context", "dup_gen": "Duplicated\ngeneration"}
COMP_ORDER    = ["gzip", "bzip2", "lzma", "ppmd"]
COMP_LABELS   = {"gzip": "gzip", "bzip2": "bzip2", "lzma": "LZMA", "ppmd": "PPMd"}
# colour-blind-safe palette
STREAM_COLORS = {"full": "#444444", "trainable": "#2c7fb8",
                 "copied_ctx": "#d95f0e", "dup_gen": "#dd3497"}
COMP_COLORS   = {"gzip": "#a6cee3", "bzip2": "#1f78b4",
                 "lzma": "#33a02c", "ppmd": "#6a3d9a"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="redundancy_fig_compression.pdf")
    args = ap.parse_args()

    with open(args.json) as f:
        d = json.load(f)

    raw = d["raw_bytes"]                       # {stream: bytes}
    sres = d["streams"]                        # {stream: {comp: {...}}}
    factor = d.get("full_vs_trainable_factor", {})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel (a): raw stream sizes in GB ----
    gb = [raw[s] / 1e9 for s in STREAM_ORDER]
    xs = np.arange(len(STREAM_ORDER))
    bars = axL.bar(xs, gb, color=[STREAM_COLORS[s] for s in STREAM_ORDER],
                   edgecolor="black", linewidth=0.6, width=0.62)
    axL.set_xticks(xs)
    axL.set_xticklabels([STREAM_LABELS[s] for s in STREAM_ORDER], fontsize=9)
    axL.set_ylabel("Raw size (GB)", fontsize=10)
    axL.set_title("(a) Corpus volume by provenance", fontsize=11, loc="left")
    for b, v in zip(bars, gb):
        axL.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    # annotate trainable as % of full
    tr_pct = 100 * raw["trainable"] / raw["full"]
    axL.text(0.97, 0.95, f"Trainable-unique = {tr_pct:.0f}% of full corpus",
             transform=axL.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7bc", ec="#cc9900", lw=0.6))
    axL.spines[["top", "right"]].set_visible(False)

    # ---- Panel (b): compression ratio per stream, grouped by compressor ----
    nS, nC = len(STREAM_ORDER), len(COMP_ORDER)
    bw = 0.2
    base = np.arange(nS)
    for ci, comp in enumerate(COMP_ORDER):
        vals = []
        for s in STREAM_ORDER:
            r = sres.get(s, {}).get(comp, {})
            vals.append(r.get("ratio", np.nan))
        axR.bar(base + (ci - (nC-1)/2)*bw, vals, width=bw,
                color=COMP_COLORS[comp], edgecolor="black", linewidth=0.4,
                label=COMP_LABELS[comp])
    axR.set_xticks(base)
    axR.set_xticklabels([STREAM_LABELS[s] for s in STREAM_ORDER], fontsize=9)
    axR.set_ylabel("Compression ratio (compressed / raw)", fontsize=10)
    axR.set_title("(b) Compressibility by provenance", fontsize=11, loc="left")
    axR.legend(fontsize=8, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0))
    axR.set_ylim(0, max(0.27, axR.get_ylim()[1]))
    axR.spines[["top", "right"]].set_visible(False)
    # annotate the headline factor (lzma)
    if "lzma" in factor:
        axR.text(0.97, 0.78,
                 f"Full compresses {factor['lzma']}$\\times$ harder\nthan trainable (LZMA)",
                 transform=axR.transAxes, ha="right", va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#deebf7", ec="#3182bd", lw=0.6))

    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[done] wrote {args.out} and .png")
    # echo the numbers used, for the caption
    print("\nNumbers used (for caption):")
    for s in STREAM_ORDER:
        print(f"  {s:11s} raw={raw[s]/1e9:.3f}GB  " +
              "  ".join(f"{c}={sres[s].get(c,{}).get('ratio','-')}" for c in COMP_ORDER))
    print(f"  factor(full/trainable): {factor}")

if __name__ == "__main__":
    main()
