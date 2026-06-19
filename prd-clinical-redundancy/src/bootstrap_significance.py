#!/usr/bin/env python3
"""
bootstrap_significance.py
Paired mention-level bootstrap test of the de-duplication F1 gain.

Reads preds_<cond>_seed<k>.jsonl as written by train_eval_ncbi.py / probe_bc5cdr.py.
Each line is ONE mention:
  {"mention": <str>, "bin": "RARE|COMMON|UNSEEN", "gold": [BIO...], "pred": [BIO...]}

Entity-level F1 is computed by comparing gold vs pred entity spans (seqeval-style:
a predicted entity counts as correct only if its type and full span match the gold).
We resample MENTIONS (with replacement) to get the bootstrap distribution of the
dedup-minus-raw F1 gap, per slice and overall. This uses the full mention count,
so the test is well-powered (unlike a 3-seed test).

Pairing: the test set is fixed, so the k-th mention of a given (slice) is the same
mention across conditions and seeds. We pool seeds within a condition by taking,
for each mention, the majority-vote tag sequence across that condition's seeds
(robust to a single bad seed). Mentions are aligned across conditions by (bin, index).
"""
import json, glob, argparse, numpy as np
from collections import defaultdict, Counter

def spans_from_bio(tags):
    """Extract entity spans (start,end,type) from a BIO tag list."""
    spans=[]; start=None; typ=None
    for i,t in enumerate(tags+["O"]):
        if t.startswith("B-"):
            if start is not None: spans.append((start,i,typ))
            start=i; typ=t[2:]
        elif t.startswith("I-") and start is not None and t[2:]==typ:
            continue
        else:
            if start is not None: spans.append((start,i,typ)); start=None; typ=None
    return set(spans)

def load_condition(results_dir, cond):
    """Return list of mentions in fixed order, each with gold + majority-vote pred.
    Groups the per-seed files; assumes each seed file lists mentions in the SAME order."""
    seed_files=sorted(glob.glob(f"{results_dir}/preds_{cond}_seed*.jsonl"))
    if not seed_files: return []
    per_seed=[]
    for f in seed_files:
        rows=[json.loads(l) for l in open(f) if l.strip()]
        per_seed.append(rows)
    n=min(len(s) for s in per_seed)
    out=[]
    for i in range(n):
        gold=per_seed[0][i]["gold"]; binv=per_seed[0][i]["bin"]
        # majority-vote the predicted tag at each position across seeds
        preds=[s[i]["pred"] for s in per_seed]
        L=len(gold)
        mv=[]
        for j in range(L):
            votes=[p[j] for p in preds if j<len(p)]
            mv.append(Counter(votes).most_common(1)[0][0] if votes else "O")
        out.append({"bin":binv,"gold":gold,"pred":mv})
    return out

def f1_over(idx, rows):
    agg=defaultdict(lambda:[0,0,0])  # slice -> tp,fp,fn
    for i in idx:
        r=rows[i]; g=spans_from_bio(r["gold"]); p=spans_from_bio(r["pred"])
        tp=len(g&p); fp=len(p-g); fn=len(g-p)
        for key in (r["bin"],"ALL"):
            agg[key][0]+=tp; agg[key][1]+=fp; agg[key][2]+=fn
    out={}
    for k,(tp,fp,fn) in agg.items():
        if tp==0: out[k]=0.0
        else:
            pr=tp/(tp+fp); rc=tp/(tp+fn); out[k]=0.0 if pr+rc==0 else 2*pr*rc/(pr+rc)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir",required=True)
    ap.add_argument("--n-boot",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed)

    raw=load_condition(a.results_dir,"A_raw")
    ded=load_condition(a.results_dir,"B_dedup")
    n=min(len(raw),len(ded)); raw=raw[:n]; ded=ded[:n]
    print(f"[info] {n} mentions  ({a.results_dir})")
    if n==0: print("[error] no preds found"); return

    SLICES=["ALL","RARE","COMMON","UNSEEN"]
    fr0=f1_over(range(n),raw); fd0=f1_over(range(n),ded)
    obs={sl:fd0.get(sl,0)-fr0.get(sl,0) for sl in SLICES}
    boot={sl:[] for sl in SLICES}
    for _ in range(a.n_boot):
        idx=rng.integers(0,n,n)
        fr=f1_over(idx,raw); fd=f1_over(idx,ded)
        for sl in SLICES: boot[sl].append(fd.get(sl,0)-fr.get(sl,0))
    print(f"\n{'slice':8s}{'gap':>9s}{'95% CI':>24s}{'p(2-sided)':>13s}")
    for sl in SLICES:
        b=np.array(boot[sl]); lo,hi=np.percentile(b,[2.5,97.5])
        p=2*min((b<=0).mean(),(b>=0).mean()); p=min(p,1.0)
        star=" *" if p<0.05 else ""
        print(f"{sl:8s}{obs[sl]:+9.4f}   [{lo:+.4f},{hi:+.4f}] {p:11.4f}{star}")

if __name__=="__main__": main()
