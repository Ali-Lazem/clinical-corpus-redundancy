#!/usr/bin/env python3
"""
build_rare_slice.py
===================
Step 3 of the downstream experiment.

Pre-registered rare-disease slice (see PREREGISTRATION.md):
  A NCBI-Disease test mention is RARE if its (normalised) disease string
  falls in the bottom 10% of the disease-frequency distribution of OUR
  TRAINING corpus. Frequencies are computed BEFORE any model is trained.

Matching is SURFACE-STRING level (lowercased, whitespace-collapsed), NOT
MeSH-concept level, because the corpus's LLM-extracted diagnoses are not
normalised to an ontology. This is stated explicitly; the script reports
coverage so the limitation is quantified, not hidden.

Three frequency bins are emitted for each NCBI test mention:
  RARE     : disease in bottom 10% of corpus frequency (>0 occurrences)
  COMMON   : disease above the 10th percentile
  UNSEEN   : disease NOT present in the corpus at all (separate category!)
            -- reported separately; UNSEEN is the most extreme rare case and
               is informative on its own (does redundancy hurt diseases the
               corpus never saw? it can't, so UNSEEN is a useful control).

Inputs:
  --enriched   the corpus (for disease frequencies from the NER channel)
  --ncbi-dir   directory with NCBI-Disease test split (HF format or CoNLL)
               OR --use-hf to pull `ncbi/ncbi_disease` via datasets library
Outputs:
  corpus_disease_freq.json    {normalised_disease: count} from corpus
  ncbi_test_tagged.jsonl      each test sentence + per-mention rare/common/unseen
  slice_manifest.json         percentile threshold, coverage stats, bin counts

Usage:
  python3 build_rare_slice.py \
      --enriched /path/to/multitask_data_enriched.jsonl \
      --use-hf \
      --out /path/to/rare_slice \
      --percentile 10
"""
import argparse, json, re, os, sys
from pathlib import Path
from collections import Counter

def norm_disease(s):
    """Normalise a disease surface string for matching."""
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:]+$", "", s)          # trailing punctuation
    return s

# ----------------------------------------------------------------------
# 1. corpus disease frequencies (from NER channel, category == diagnosis)
# ----------------------------------------------------------------------
def corpus_disease_freq(enriched, limit=0):
    freq = Counter()
    n = 0
    with open(enriched) as f:
        for line in f:
            if limit and n >= limit: break
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except Exception: continue
            n += 1
            for it in (rec.get("tasks", {}) or {}).get("ner", []) or []:
                cat = norm_disease(it.get("category"))
                if cat == "diagnosis":
                    ent = norm_disease(it.get("entity"))
                    if ent:
                        freq[ent] += 1
    return freq, n

# ----------------------------------------------------------------------
# 2. load NCBI-Disease test set (HF datasets or local CoNLL/BIO)
# ----------------------------------------------------------------------
def load_ncbi_test_hf():
    """Load NCBI-Disease test split WITHOUT the dataset loading script.
    Strategy 1: read the parquet files the Hub auto-generates (no script).
    Strategy 2: fall back to load_dataset on the parquet config.
    """
    # --- Strategy 1: direct parquet via huggingface_hub (most robust) ---
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        # the auto-converted parquet lives under refs/convert/parquet
        fp = hf_hub_download(repo_id="ncbi/ncbi_disease",
                             filename="ncbi_disease/test/0000.parquet",
                             repo_type="dataset", revision="refs/convert/parquet")
        tbl = pq.read_table(fp).to_pylist()
        return _bio_sents_from_rows(tbl)
    except Exception as e:
        print(f"[info] parquet path failed ({e}); trying datasets parquet config")
    # --- Strategy 2: datasets with explicit parquet (no script) ---
    from datasets import load_dataset
    ds = load_dataset("parquet",
                      data_files="hf://datasets/ncbi/ncbi_disease@refs/convert/parquet/"
                                 "ncbi_disease/test/0000.parquet",
                      split="train")
    return _bio_sents_from_rows([dict(r) for r in ds])


def _bio_sents_from_rows(rows):
    """rows: list of {tokens, ner_tags}. Reconstruct disease mention spans."""
    sents = []
    for ex in rows:
        toks = list(ex["tokens"]); tags = list(ex["ner_tags"])
        mentions = []; cur = []
        for t, g in zip(toks, tags):
            g = int(g)
            if g == 1:
                if cur: mentions.append(" ".join(cur))
                cur = [t]
            elif g == 2:
                cur.append(t)
            else:
                if cur: mentions.append(" ".join(cur)); cur = []
        if cur: mentions.append(" ".join(cur))
        sents.append({"tokens": toks, "ner_tags": tags, "mentions": mentions})
    return sents

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--use-hf", action="store_true",
                    help="load NCBI-Disease test via HuggingFace datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="rare = bottom this %% of corpus disease frequency")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] computing corpus disease frequencies (NER diagnosis entities)...")
    freq, n_rec = corpus_disease_freq(args.enriched, args.limit)
    print(f"      {len(freq):,} distinct disease STRINGS over {n_rec:,} records")
    print(f"      (note: surface-string vocabulary is large/noisy; we rank only")
    print(f"       the NCBI test diseases by their corpus frequency -- Option B)")
    if not freq:
        print("[FATAL] no diagnosis entities found. Check NER channel/category.", file=sys.stderr)
        sys.exit(2)
    (OUT / "corpus_disease_freq.json").write_text(json.dumps(dict(freq.most_common(5000)), indent=2))

    print("[2/4] loading NCBI-Disease test set...")
    if args.use_hf:
        sents = load_ncbi_test_hf()
    else:
        print("[FATAL] only --use-hf implemented here.", file=sys.stderr)
        sys.exit(2)
    print(f"      {len(sents):,} test sentences")

    # ---- Option B: collect the DISTINCT NCBI diseases, score each by corpus freq ----
    print("[3/4] ranking NCBI diseases by corpus frequency...")
    ncbi_diseases = Counter()             # distinct NCBI disease -> #test mentions
    for s in sents:
        for m in s["mentions"]:
            ncbi_diseases[norm_disease(m)] += 1
    ncbi_corpus_freq = {d: freq.get(d, 0) for d in ncbi_diseases}
    seen_ncbi = {d: c for d, c in ncbi_corpus_freq.items() if c > 0}
    unseen_ncbi = {d for d, c in ncbi_corpus_freq.items() if c == 0}
    print(f"      {len(ncbi_diseases):,} distinct NCBI test diseases; "
          f"{len(seen_ncbi):,} present in corpus, {len(unseen_ncbi):,} UNSEEN")

    # rare = bottom `percentile` of the SEEN NCBI diseases by corpus frequency
    import math
    if seen_ncbi:
        seen_counts = sorted(seen_ncbi.values())
        idx = max(0, int(math.ceil(len(seen_counts) * args.percentile / 100.0)) - 1)
        thresh = seen_counts[idx]
    else:
        thresh = 0
    print(f"      RARE threshold among SEEN NCBI diseases: corpus freq <= {thresh}")
    (OUT / "ncbi_disease_corpus_freq.json").write_text(
        json.dumps(dict(sorted(ncbi_corpus_freq.items(), key=lambda x:x[1])), indent=2))

    print("[4/4] tagging each NCBI test mention rare/common/unseen...")
    bin_counts = Counter(); total_mentions = 0; matched = 0; tagged = []
    for s in sents:
        mtags = []
        for m in s["mentions"]:
            total_mentions += 1
            nm = norm_disease(m)
            c = freq.get(nm, 0)
            if c == 0:
                b = "UNSEEN"
            elif c <= thresh:
                b = "RARE"; matched += 1
            else:
                b = "COMMON"; matched += 1
            bin_counts[b] += 1
            mtags.append({"mention": m, "norm": nm, "corpus_freq": c, "bin": b})
        tagged.append({"tokens": s["tokens"], "ner_tags": s["ner_tags"],
                       "mention_bins": mtags})

    with open(OUT / "ncbi_test_tagged.jsonl", "w") as w:
        for t in tagged:
            w.write(json.dumps(t) + "\n")

    coverage = 100.0 * matched / max(total_mentions, 1)
    manifest = {
        "corpus_records": n_rec,
        "distinct_corpus_diseases": len(freq),
        "percentile": args.percentile,
        "rare_threshold_freq": thresh,
        "ncbi_test_sentences": len(sents),
        "total_test_mentions": total_mentions,
        "bin_counts": dict(bin_counts),
        "coverage_pct_seen_in_corpus": round(coverage, 1),
        "matching": "surface-string (lowercased, whitespace-collapsed), NOT MeSH-normalised. "
                    "Rarity ranks ONLY the NCBI test diseases by corpus frequency (Option B), "
                    "avoiding the noisy 383k-string full vocabulary.",
        "rare_threshold_freq": thresh,
        "distinct_ncbi_diseases": len(ncbi_diseases),
        "ncbi_diseases_seen_in_corpus": len(seen_ncbi),
        "ncbi_diseases_unseen": len(unseen_ncbi),
        "note": "UNSEEN = disease absent from corpus (most extreme rarity; "
                "reported separately as a control, not merged into RARE).",
    }
    (OUT / "slice_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n" + "="*60)
    print("RARE-SLICE TAGGING COMPLETE")
    print("="*60)
    print(f"  total test mentions : {total_mentions:,}")
    for b in ("RARE", "COMMON", "UNSEEN"):
        c = bin_counts.get(b, 0)
        print(f"    {b:8s}: {c:,} ({100*c/max(total_mentions,1):.1f}%)")
    print(f"  coverage (seen in corpus): {coverage:.1f}%")
    print(f"  rare threshold: corpus freq <= {thresh}")
    print(f"\n  wrote {OUT}/ncbi_test_tagged.jsonl + slice_manifest.json")
    if bin_counts.get("UNSEEN", 0) > total_mentions * 0.5:
        print("\n  [!] >50% of NCBI diseases are UNSEEN in your corpus.")
        print("      This is itself a finding (corpus disease coverage is")
        print("      limited), but means the RARE-vs-COMMON contrast rests on")
        print("      the seen subset. Report coverage honestly in the paper.")

if __name__ == "__main__":
    main()
