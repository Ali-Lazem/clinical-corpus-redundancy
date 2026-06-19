#!/usr/bin/env python3
"""
prepare_bc5cdr.py
=================
Prepare BC5CDR-Disease (from tner/bc5cdr) as a disease-NER probe set that
mirrors the NCBI-Disease setup, and build the rare/common/unseen frequency
stratification against OUR corpus.

tner/bc5cdr tags: O=0, B-Chemical=1, B-Disease=2, I-Disease=3, I-Chemical=4
We KEEP only Disease (collapse Chemical -> O) so the task matches NCBI exactly.

Outputs (to --out dir):
  bc5cdr_train.jsonl   tokens + disease-only BIO tags (ids: O=0,B=1,I=2)
  bc5cdr_val.jsonl
  bc5cdr_test.jsonl
  bc5cdr_test_tagged.jsonl   test mentions tagged rare/common/unseen
  bc5cdr_stratification.json summary of coverage + thresholds

Usage:
  python3 prepare_bc5cdr.py \
      --corpus-freq /scratch/SCWF00175/shared/datasets/corpus_disease_freq.json \
      --out /scratch/SCWF00175/shared/datasets/bc5cdr_disease
"""
import argparse, json, re
from pathlib import Path
from collections import Counter

# disease-only label scheme (matches NCBI harness: O/B/I)
OUT_LABEL2ID = {"O": 0, "B-Disease": 1, "I-Disease": 2}

# tner/bc5cdr source ids
SRC = {"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}

def remap_tags(src_tags):
    """Collapse Chemical -> O, keep Disease, renumber to O/B/I disease-only."""
    out = []
    for t in src_tags:
        if t == SRC["B-Disease"]:
            out.append(OUT_LABEL2ID["B-Disease"])
        elif t == SRC["I-Disease"]:
            out.append(OUT_LABEL2ID["I-Disease"])
        else:  # O, B-Chemical, I-Chemical all -> O
            out.append(OUT_LABEL2ID["O"])
    return out

def norm(s):
    """Normalised surface string for matching: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_mentions(tokens, tags):
    """Yield normalised disease mention strings from BIO tags (disease-only ids)."""
    cur = []
    for tok, tg in zip(tokens, tags):
        if tg == OUT_LABEL2ID["B-Disease"]:
            if cur: yield norm(" ".join(cur))
            cur = [tok]
        elif tg == OUT_LABEL2ID["I-Disease"] and cur:
            cur.append(tok)
        else:
            if cur: yield norm(" ".join(cur)); cur = []
    if cur: yield norm(" ".join(cur))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-freq", required=True,
                    help="JSON {normalised_disease: count} from OUR corpus "
                         "(same file used for the NCBI stratification)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-name", default="tner/bc5cdr")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"[load] {args.hf-name if False else args.hf_name}")
    ds = load_dataset(args.hf_name, revision="refs/convert/parquet")

    # write disease-only splits in the harness's expected jsonl shape
    split_map = {"train": "bc5cdr_train.jsonl",
                 "validation": "bc5cdr_val.jsonl",
                 "test": "bc5cdr_test.jsonl"}
    for split, fname in split_map.items():
        if split not in ds:  # some versions name validation 'valid'
            alt = "valid" if split == "validation" else split
            split_key = alt if alt in ds else split
        else:
            split_key = split
        rows = ds[split_key]
        with open(out / fname, "w") as f:
            for r in rows:
                toks = r["tokens"]; tags = remap_tags(r["tags"])
                f.write(json.dumps({"tokens": toks, "ner_tags": tags}) + "\n")
        print(f"[write] {fname}: {len(rows):,} sentences")

    # ---- frequency stratification of TEST mentions vs OUR corpus ----
    corpus_freq = json.load(open(args.corpus_freq))   # {norm_disease: count}
    corpus_freq = {norm(k): v for k, v in corpus_freq.items()}

    # gather distinct test diseases + per-mention list
    test_rows = [json.loads(l) for l in open(out / "bc5cdr_test.jsonl")]
    mention_list = []
    for r in test_rows:
        for m in extract_mentions(r["tokens"], r["ner_tags"]):
            mention_list.append(m)
    distinct = sorted(set(mention_list))
    n_mentions = len(mention_list)

    seen = {d for d in distinct if d in corpus_freq}
    unseen = {d for d in distinct if d not in corpus_freq}
    cov = 100.0 * len(seen) / max(len(distinct), 1)

    # quartile threshold on SEEN-disease corpus frequencies (matches NCBI rule)
    seen_freqs = sorted(corpus_freq[d] for d in seen)
    if seen_freqs:
        q1_idx = max(0, int(0.25 * len(seen_freqs)) - 1)
        rare_thr = seen_freqs[q1_idx]
    else:
        rare_thr = 0
    def slice_of(d):
        if d in unseen: return "UNSEEN"
        return "RARE" if corpus_freq[d] <= rare_thr else "COMMON"

    # tag each test mention, count slices
    slice_counts = Counter()
    tagged = []
    for r in test_rows:
        ms = list(extract_mentions(r["tokens"], r["ner_tags"]))
        sl = [slice_of(m) for m in ms]
        for s in sl: slice_counts[s] += 1
        tagged.append({"tokens": r["tokens"], "ner_tags": r["ner_tags"],
                       "mentions": ms, "slices": sl})
    with open(out / "bc5cdr_test_tagged.jsonl", "w") as f:
        for t in tagged: f.write(json.dumps(t) + "\n")

    summary = {
        "dataset": args.hf_name, "task": "disease-NER (Chemical collapsed to O)",
        "distinct_test_diseases": len(distinct),
        "covered_by_corpus": len(seen),
        "coverage_pct": round(cov, 1),
        "rare_threshold_freq_leq": rare_thr,
        "test_mentions_total": n_mentions,
        "slice_counts": dict(slice_counts),
        "label2id": OUT_LABEL2ID,
    }
    json.dump(summary, open(out / "bc5cdr_stratification.json", "w"), indent=2)
    print("\n=== BC5CDR-Disease stratification (vs OUR corpus) ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
