#!/usr/bin/env python3
"""
near_dup_robustness.py
======================

Robustness check for the redundancy paper: exact-hash dedup (used in the
main analysis) misses NEAR-duplicates -- generated strings that differ by
a token or two (e.g. two reasoning chains identical apart from a patient
age). This script measures near-duplicate redundancy on a sample of the
GENERATED content, to show that true redundancy is even higher than the
exact-match figure reported in the paper (i.e. the main estimate is
conservative).

Method: collect generated strings (answers/reasoning) from a sample of
records, then cluster by near-duplicate similarity using either:
  - MinHash + LSH (fast, scalable) if `datasketch` is installed, or
  - a fallback: normalized-text exact match (strips numbers/whitespace),
    which catches the "identical apart from a number" case specifically.

Reports: exact-unique count vs near-dup-unique count, and the extra
redundancy fraction near-dup catches beyond exact-match.

Usage:
    python3 near_dup_robustness.py \\
        --dir /scratch/SCWF00175/shared/code/risk_v7 \\
        --sample 0.02 \\
        --threshold 0.85
"""
import argparse, json, re, hashlib
from pathlib import Path
from collections import defaultdict
import random

# generated-text fields worth checking for near-duplication
GEN_FIELDS = {"answer", "reason", "logic", "level_2_interpretation",
              "summary", "justification"}


def collect_generated(rec, out):
    """Pull free-text generated strings from a record (skip context fields)."""
    CONTEXT = {"context", "verification_anchor", "verification_ctx",
               "sentence", "text"}

    def walk(o, key=None):
        if isinstance(o, str):
            if key in GEN_FIELDS and len(o) > 30:
                out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k in CONTEXT:
                    continue
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
    walk(rec)


def norm_numbers(s):
    """Normalize: lowercase, collapse whitespace, replace digits with #.
    Catches 'identical apart from a number' near-duplicates."""
    s = s.lower()
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def h(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sample", type=float, default=0.02)
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Jaccard threshold for MinHash LSH (if available)")
    args = ap.parse_args()

    enriched = Path(args.dir) / "multitask_data_enriched.jsonl"
    rng = random.Random(42)

    strings = []
    with open(enriched) as f:
        for line in f:
            if rng.random() > args.sample:
                continue
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except: continue
            collect_generated(rec, strings)

    n = len(strings)
    print(f"[collected] {n:,} generated strings from {args.sample*100:.0f}% sample")
    if n == 0:
        print("No generated strings collected; check GEN_FIELDS names against your schema.")
        return

    # 1. exact-unique (the paper's method)
    exact = len({h(s) for s in strings})

    # 2. number-normalized unique (catches 'differ only by a number')
    normnum = len({h(norm_numbers(s)) for s in strings})

    # 3. MinHash near-dup clustering if datasketch available
    minhash_unique = None
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=args.threshold, num_perm=64)
        mhs = []
        for i, s in enumerate(strings):
            m = MinHash(num_perm=64)
            for tok in set(s.lower().split()):
                m.update(tok.encode("utf8"))
            mhs.append(m)
            lsh.insert(str(i), m)
        # count clusters greedily
        seen = set(); clusters = 0
        for i in range(len(strings)):
            if str(i) in seen: continue
            dup = lsh.query(mhs[i])
            for d in dup: seen.add(d)
            clusters += 1
        minhash_unique = clusters
    except ImportError:
        print("[note] datasketch not installed; skipping MinHash. "
              "pip install datasketch  for the full near-dup measure.")

    print("\n" + "="*56)
    print("NEAR-DUPLICATE ROBUSTNESS (generated content sample)")
    print("="*56)
    print(f"  total generated strings        : {n:,}")
    print(f"  exact-unique (paper method)    : {exact:,}  "
          f"({100*exact/n:.1f}% unique)")
    print(f"  number-normalized unique       : {normnum:,}  "
          f"({100*normnum/n:.1f}% unique)")
    extra_num = 100*(exact-normnum)/n
    print(f"    -> {extra_num:.1f}% additional redundancy from "
          f"number-only differences alone")
    if minhash_unique is not None:
        print(f"  MinHash near-dup unique (J>{args.threshold}) : {minhash_unique:,}  "
              f"({100*minhash_unique/n:.1f}% unique)")
        extra_mh = 100*(exact-minhash_unique)/n
        print(f"    -> {extra_mh:.1f}% additional redundancy beyond exact-match")
    print("\n  INTERPRETATION: any value below the exact-unique line means")
    print("  the paper's exact-match redundancy is CONSERVATIVE; true")
    print("  redundancy is higher by the percentages shown.")


if __name__ == "__main__":
    main()
