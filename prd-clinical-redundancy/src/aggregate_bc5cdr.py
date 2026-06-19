#!/usr/bin/env python3
"""Aggregate BC5CDR probe metrics: mean +/- s.d. over seeds, per depth/slice,
plus the de-dup gain (B_dedup - A_raw) and difference-in-differences.
Mirrors aggregate_results.py for NCBI.

Usage:
  python3 aggregate_bc5cdr.py --base /scratch/SCWF00175/shared/results \
      --out /scratch/SCWF00175/shared/reports/bc5cdr_results.json
"""
import argparse, json, glob, statistics as st
from pathlib import Path

CONDS=["A_raw","B_dedup","B1_ctxremoved"]; SLICES=["RARE","COMMON","UNSEEN","ALL"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    report={"per_depth":{}}
    for depth in ["10k","20k","40k"]:
        d=Path(a.base)/f"bc5cdr_{depth}"
        pc={}
        for cond in CONDS:
            files=sorted(glob.glob(str(d/f"metrics_{cond}_seed*.json")))
            if not files: continue
            per_slice={sl:[] for sl in SLICES}
            for f in files:
                m=json.load(open(f))["slices"]
                for sl in SLICES: per_slice[sl].append(m[sl]["f1"])
            pc[cond]={sl:{"f1_mean":round(st.mean(v),4),
                          "f1_sd":round(st.pstdev(v),4) if len(v)>1 else 0.0,
                          "n_seeds":len(v)} for sl,v in per_slice.items()}
        if "A_raw" in pc and "B_dedup" in pc:
            gain={sl:round(pc["B_dedup"][sl]["f1_mean"]-pc["A_raw"][sl]["f1_mean"],4) for sl in SLICES}
            did=round(gain["RARE"]-gain["COMMON"],4)
        else:
            gain,did={},None
        report["per_depth"][depth]={"per_condition":pc,"dedup_gain":gain,
                                    "did_rare_vs_common":did}
    json.dump(report,open(a.out,"w"),indent=2)
    # print
    for depth,blk in report["per_depth"].items():
        print(f"\n===== BC5CDR {depth} =====")
        for cond in CONDS:
            if cond in blk["per_condition"]:
                row=blk["per_condition"][cond]
                print(f"  {cond:16s} "+"  ".join(
                    f"{sl}:{row[sl]['f1_mean']:.3f}±{row[sl]['f1_sd']:.3f}" for sl in SLICES))
        print(f"  gain={blk['dedup_gain']}  DiD={blk['did_rare_vs_common']}")
    print(f"\nwrote {a.out}")

if __name__=="__main__":
    main()
