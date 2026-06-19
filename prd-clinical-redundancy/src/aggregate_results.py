#!/usr/bin/env python3
"""
aggregate_results.py
====================
Step 5: aggregate the 9 (condition x seed) NCBI results into the paper's
headline numbers, with bootstrap confidence intervals.

Reads:  results/full/metrics_<cond>_seed<k>.json     (from train_eval_ncbi.py)
        results/full/preds_<cond>_seed<k>.jsonl      (per-mention, for bootstrap)

Produces:
  - per-condition mean +/- s.d. F1 on ALL/RARE/COMMON/UNSEEN slices
  - the HEADLINE difference-in-differences:
        does (B_dedup - A_raw) on RARE exceed (B_dedup - A_raw) on COMMON?
    i.e. does de-duplication help RARE diseases MORE than common ones?
  - bootstrap 95% CIs (resampling test mentions) for each condition-slice F1
    and for the key deltas
  - a results table (markdown) + a JSON dump for the figure

Usage:
  python3 aggregate_results.py \
      --results-dir /scratch/SCWF00175/shared/results/full \
      --conditions A_raw B_dedup B1_ctxremoved \
      --seeds 1 2 3 \
      --out /scratch/SCWF00175/shared/reports/downstream_results.json
"""
import argparse, json, glob, os, random
from pathlib import Path
from collections import defaultdict

def entity_f1(gold_seqs, pred_seqs):
    """Entity-level F1 via seqell if available, else exact-span fallback."""
    try:
        from seqeval.metrics import f1_score
        return float(f1_score(gold_seqs, pred_seqs))
    except Exception:
        # fallback: exact full-span match per mention
        tp=fp=fn=0
        for g,p in zip(gold_seqs,pred_seqs):
            g_has = any(x!="O" for x in g); p_has = any(x!="O" for x in p)
            if g_has and g==p: tp+=1
            elif p_has and g!=p: fp+=1
            if g_has and g!=p: fn+=1
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        return 2*prec*rec/(prec+rec) if prec+rec else 0.0

def load_preds(results_dir, cond, seed):
    fp = os.path.join(results_dir, f"preds_{cond}_seed{seed}.jsonl")
    if not os.path.exists(fp): return None
    rows=[]
    with open(fp) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def bootstrap_f1_by_slice(rows, n_boot=1000, seed=0):
    """Bootstrap entity-F1 per slice by resampling mentions."""
    rng = random.Random(seed)
    by_slice = defaultdict(list)
    for r in rows:
        by_slice[r["bin"]].append(r)
        by_slice["ALL"].append(r)
    out={}
    for sl, items in by_slice.items():
        if not items:
            out[sl]={"f1":None,"lo":None,"hi":None,"n":0}; continue
        boots=[]
        for _ in range(n_boot):
            samp=[items[rng.randrange(len(items))] for _ in range(len(items))]
            g=[s["gold"] for s in samp]; p=[s["pred"] for s in samp]
            boots.append(entity_f1(g,p))
        boots.sort()
        point=entity_f1([s["gold"] for s in items],[s["pred"] for s in items])
        out[sl]={"f1":round(point,4),
                 "lo":round(boots[int(0.025*n_boot)],4),
                 "hi":round(boots[int(0.975*n_boot)],4),
                 "n":len(items)}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir",required=True)
    ap.add_argument("--conditions",nargs="+",default=["A_raw","B_dedup","B1_ctxremoved"])
    ap.add_argument("--seeds",type=int,nargs="+",default=[1,2,3])
    ap.add_argument("--n-boot",type=int,default=1000)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    SLICES=("ALL","RARE","COMMON","UNSEEN")

    # 1) per-condition mean +/- sd across seeds (from metrics_*.json)
    cond_seed_f1 = defaultdict(lambda: defaultdict(list))   # cond -> slice -> [f1 per seed]
    for cond in args.conditions:
        for seed in args.seeds:
            mf=os.path.join(args.results_dir,f"metrics_{cond}_seed{seed}.json")
            if not os.path.exists(mf):
                print(f"[warn] missing {mf}"); continue
            m=json.load(open(mf))
            for sl in SLICES:
                if m.get(sl,{}).get("f1") is not None:
                    cond_seed_f1[cond][sl].append(m[sl]["f1"])

    def mean_sd(xs):
        xs=[x for x in xs if x is not None]
        if not xs: return None,None
        mu=sum(xs)/len(xs)
        sd=(sum((x-mu)**2 for x in xs)/(len(xs)-1))**0.5 if len(xs)>1 else 0.0
        return round(mu,4),round(sd,4)

    summary={"per_condition":{}}
    for cond in args.conditions:
        summary["per_condition"][cond]={}
        for sl in SLICES:
            mu,sd=mean_sd(cond_seed_f1[cond][sl])
            summary["per_condition"][cond][sl]={"f1_mean":mu,"f1_sd":sd,
                                                "n_seeds":len(cond_seed_f1[cond][sl])}

    # 2) bootstrap CIs using pooled predictions across seeds (per condition)
    summary["bootstrap"]={}
    for cond in args.conditions:
        pooled=[]
        for seed in args.seeds:
            rows=load_preds(args.results_dir,cond,seed)
            if rows: pooled.extend(rows)
        if pooled:
            summary["bootstrap"][cond]=bootstrap_f1_by_slice(pooled,args.n_boot)

    # 3) HEADLINE: difference-in-differences (B_dedup vs A_raw, RARE vs COMMON)
    def cond_mean(cond,sl):
        v=summary["per_condition"].get(cond,{}).get(sl,{}).get("f1_mean")
        return v
    head={}
    if "A_raw" in args.conditions and "B_dedup" in args.conditions:
        for sl in SLICES:
            a=cond_mean("A_raw",sl); b=cond_mean("B_dedup",sl)
            head[f"delta_{sl}"]= round(b-a,4) if (a is not None and b is not None) else None
        if head.get("delta_RARE") is not None and head.get("delta_COMMON") is not None:
            head["did_rare_vs_common"]=round(head["delta_RARE"]-head["delta_COMMON"],4)
            head["interpretation"]=(
                "positive did_rare_vs_common => de-duplication helps RARE "
                "diseases MORE than common ones (supports the hypothesis)")
    summary["headline_dedup_effect"]=head

    # 4) markdown table
    md=["# Downstream results: corpus redundancy effect on NCBI disease-NER\n",
        "F1 (mean +/- s.d. across seeds); bootstrap 95% CI in brackets.\n",
        "| Condition | ALL | RARE | COMMON | UNSEEN |",
        "|---|---|---|---|---|"]
    for cond in args.conditions:
        row=[cond]
        for sl in SLICES:
            pc=summary["per_condition"][cond][sl]
            bs=summary.get("bootstrap",{}).get(cond,{}).get(sl,{})
            cell=f"{pc['f1_mean']}±{pc['f1_sd']}"
            if bs.get("lo") is not None: cell+=f" [{bs['lo']},{bs['hi']}]"
            row.append(cell)
        md.append("| "+" | ".join(str(x) for x in row)+" |")
    md.append("")
    if head:
        md.append("## Headline: de-duplication effect (B_dedup - A_raw)\n")
        for sl in SLICES:
            md.append(f"- delta {sl}: {head.get(f'delta_{sl}')}")
        if "did_rare_vs_common" in head:
            md.append(f"\n**Difference-in-differences (RARE - COMMON): "
                      f"{head['did_rare_vs_common']}**")
            md.append(f"\n{head['interpretation']}")
    md_txt="\n".join(md)

    Path(args.out).write_text(json.dumps(summary,indent=2))
    Path(args.out.replace(".json",".md")).write_text(md_txt)
    print(md_txt)
    print(f"\n[done] wrote {args.out} and .md")

if __name__=="__main__":
    main()
