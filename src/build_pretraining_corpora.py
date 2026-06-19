#!/usr/bin/env python3
"""
build_pretraining_corpora.py
============================
Step 2 of the downstream experiment.

Emits MLM continued-pretraining text corpora from the multi-task clinical
corpus, in conditions that differ ONLY in redundancy, using the SAME PRD
provenance taxonomy and recursive walk as redundancy_full_analysis_v2.py
and compression_redundancy_v3_aligned_v2.py. "De-duplicated" therefore means
exactly what it means everywhere else in the paper.

Conditions produced (one .txt per condition, one document per line-block):
  A_raw        : every text-bearing field, corpus order            (FULL)
  B_dedup      : unique source + first-occurrence generated         (TRAINABLE)
  B1_ctxremoved: copied-context removed, generated duplicates KEPT  (ablation)
  (B2 == B_dedup: both removed)

Each "document" is one patient record's text-bearing content concatenated,
so MLM sees coherent clinical context (not shuffled fields).

EQUAL-TOKEN BUDGETING (confound control):
  --budget-tokens N   truncate EACH corpus to the first N whitespace tokens
                      (so A and B train on the same budget; default: match
                      to the smallest corpus = B_dedup, computed and printed).
  Reports the token count of each corpus so you can set the budget.

Provenance taxonomy is imported-by-copy (kept in sync with the classifier).

Usage:
  python3 build_pretraining_corpora.py \
      --enriched /path/to/multitask_data_enriched.jsonl \
      --risk-dir /path/to/risk_files \
      --out /path/to/pretrain_corpora \
      --budget-tokens 0          # 0 = no truncation; report sizes first

  # then re-run with --budget-tokens set to the printed B_dedup size for
  # an equal-budget A-vs-B comparison.
"""
import argparse, json, os, hashlib, sys
from pathlib import Path

# ---- taxonomy: identical to redundancy_full_analysis_v2.py ----
CONTEXT_FIELDS = {"context", "verification_anchor", "verification_ctx", "sentence"}
SCAFFOLD_FIELDS = {
    "qa_id","rel_id","event_id","uid","start_char","end_char","confidence",
    "llm_confidence","direction","temporal_order","is_anchored",
    "is_negated_source","timepoint_type","event_type","assertion_status",
    "verifier_status","status","head_type","tail_type","label","category",
    "meta_species","verdict_path","med_id","drug","route","frequency","qtype",
    "risk_id","state","severity","volatility_profile","rule","threshold",
    "decision_type","severity_marker","method","actionability","source",
    "provenance","last_updated","rec_id","type","agentic_check",
    "constraints_applied","based_on_risks","level","node_type","size",
}
PLACEHOLDERS = {"context unavailable","","n/a","none","."}


def load_uid_map(path):
    m = {}
    if not path or not os.path.exists(path):
        print(f"[warn] risk file not found: {path}", file=sys.stderr)
        return m
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            u = rec.get("uid")
            if u is not None:
                m[u] = rec
    return m


def walk_collect(obj, key, seen_gen, buckets):
    """Mirror of the classifier walk. Routes each leaf string into the
    per-document buckets: 'raw' (everything), 'src_or_firstgen' (dedup),
    'ctxremoved' (drop copied-context, keep gen dups)."""
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return
        if key in SCAFFOLD_FIELDS:
            return                                   # scaffold: never text
        if key in CONTEXT_FIELDS:
            if s.lower() in PLACEHOLDERS:
                return
            # copied context: in RAW only (not dedup, not ctxremoved)
            buckets["raw"].append(s)
            return
        # generated text
        buckets["raw"].append(s)
        buckets["ctxremoved"].append(s)             # ctx removed but gen dups kept
        hh = hashlib.md5(s.encode("utf-8","replace")).hexdigest()
        if hh not in seen_gen:
            seen_gen.add(hh)
            buckets["dedup"].append(s)              # first occurrence only
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk_collect(v, k, seen_gen, buckets)
    elif isinstance(obj, list):
        for v in obj:
            walk_collect(v, key, seen_gen, buckets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--risk-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-tokens", type=int, default=0,
                    help="truncate each corpus to first N whitespace tokens "
                         "(0 = no truncation; run once to see sizes)")
    ap.add_argument("--limit", type=int, default=0, help="first N records (testing)")
    args = ap.parse_args()

    rd = args.risk_dir or os.path.dirname(args.enriched)
    recs_map   = load_uid_map(os.path.join(rd, "active_v6_recommendations.jsonl"))
    states_map = load_uid_map(os.path.join(rd, "active_v6_risk_states.jsonl"))
    risks_map  = load_uid_map(os.path.join(rd, "active_v6_risks.jsonl"))
    print(f"[load] risk channels: recs={len(recs_map):,} states={len(states_map):,} risks={len(risks_map):,}")

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    seen_gen = set()
    seen_src = set()

    # open three writers
    writers = {
        "A_raw":         open(OUT / "A_raw.txt", "w"),
        "B_dedup":       open(OUT / "B_dedup.txt", "w"),
        "B1_ctxremoved": open(OUT / "B1_ctxremoved.txt", "w"),
    }
    tok_counts = {k: 0 for k in writers}
    n_rec = 0

    def wtok(name, text):
        writers[name].write(text + "\n")
        tok_counts[name] += len(text.split())

    with open(args.enriched) as f:
        for line in f:
            if args.limit and n_rec >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_rec += 1
            uid = rec.get("uid")

            buckets = {"raw": [], "dedup": [], "ctxremoved": []}

            # unique source narrative (dedup across corpus; in all conditions once)
            src = rec.get("text", "")
            if isinstance(src, str) and src.strip():
                buckets["raw"].append(src.strip())
                buckets["ctxremoved"].append(src.strip())
                hh = hashlib.md5(src.encode("utf-8","replace")).hexdigest()
                if hh not in seen_src:
                    seen_src.add(hh)
                    buckets["dedup"].append(src.strip())

            # 7 enriched task channels
            for tname, tval in rec.get("tasks", {}).items():
                walk_collect(tval, None, seen_gen, buckets)
            # 3 risk channels by uid
            if uid is not None:
                if uid in recs_map:
                    walk_collect(recs_map[uid].get("recommendations", []), None, seen_gen, buckets)
                if uid in states_map:
                    walk_collect(states_map[uid].get("states", []), None, seen_gen, buckets)
                if uid in risks_map:
                    walk_collect(risks_map[uid].get("risks", []), None, seen_gen, buckets)

            # one document per record per condition (blank-line separated)
            if buckets["raw"]:
                wtok("A_raw", " ".join(buckets["raw"]))
            if buckets["dedup"]:
                wtok("B_dedup", " ".join(buckets["dedup"]))
            if buckets["ctxremoved"]:
                wtok("B1_ctxremoved", " ".join(buckets["ctxremoved"]))

    for w in writers.values():
        w.close()

    print("\n" + "="*60)
    print("PRETRAINING CORPORA — token counts (whitespace)")
    print("="*60)
    for k in writers:
        print(f"  {k:16s} {tok_counts[k]:>14,} tokens")
    smallest = min(tok_counts.values())
    print(f"\n  smallest corpus: {smallest:,} tokens (B_dedup expected)")
    print(f"  -> for equal-budget A-vs-B, re-run with --budget-tokens {smallest}")

    # equal-token budgeting (truncate each to budget)
    if args.budget_tokens > 0:
        print(f"\n[budget] truncating each corpus to {args.budget_tokens:,} tokens")
        for k in writers:
            src = OUT / f"{k}.txt"
            dst = OUT / f"{k}.budget.txt"
            kept = 0
            with open(src) as r, open(dst, "w") as w:
                for line in r:
                    nt = len(line.split())
                    if kept + nt > args.budget_tokens:
                        break
                    w.write(line); kept += nt
            print(f"  {k:16s} -> {dst.name}  ({kept:,} tokens)")

    manifest = {
        "records": n_rec,
        "token_counts": tok_counts,
        "equal_budget_tokens": args.budget_tokens or None,
        "conditions": {
            "A_raw": "every text-bearing field (FULL) = redundant corpus",
            "B_dedup": "unique source + first-occurrence generated (TRAINABLE)",
            "B1_ctxremoved": "copied-context removed, generated duplicates kept (ablation)",
        },
        "taxonomy": "identical to redundancy_full_analysis_v2.py (PRD)",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] wrote corpora + MANIFEST.json to {OUT}")


if __name__ == "__main__":
    main()
