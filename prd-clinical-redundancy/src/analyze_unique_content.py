#!/usr/bin/env python3
"""
analyze_unique_content.py
=========================

Separates the 167K corpus into:
  (1) unique human source text (the patient narratives)
  (2) unique model-GENERATED content (answers, reasoning, summaries,
      labels) -- the genuinely new signal
  (3) repeated/duplicated context (source narrative copied into per-item
      context/anchor/ctx fields) -- redundant
  (4) structural scaffolding (ids, enums, type labels)

This is the number that determines whether a training-corpus framing is
viable, and quantifies the context-duplication ratio for the
"synthetic-data redundancy" methodological angle.

Approach: for each task item, classify each string field as either
CONTEXT (matches/overlaps the patient narrative) or GENERATED (new),
then deduplicate generated strings by hash to get unique-generated.

Usage
-----
    python3 analyze_unique_content.py \\
        --dir /scratch/SCWF00175/shared/code/risk_v7 \\
        --hf-tokenizer /scratch/SCWF00175/shared/models/llama3-70b \\
        --out /scratch/SCWF00175/shared/reports/unique_content_report.json \\
        --sample 0.05
"""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter
import random

# Fields known to carry COPIED source narrative (not generated content)
CONTEXT_FIELDS = {
    "context", "verification_anchor", "verification_ctx", "sentence",
    "text", "meta",  # meta often holds verification_ctx
}
# Fields that are pure structural scaffolding (ids, enums)
SCAFFOLD_FIELDS = {
    "qa_id", "rel_id", "event_id", "uid", "start_char", "end_char",
    "confidence", "llm_confidence", "direction", "temporal_order",
    "is_anchored", "is_negated_source", "timepoint_type", "event_type",
    "assertion_status", "verifier_status", "status", "head_type",
    "tail_type", "label", "category", "meta_species", "verdict_path",
}


class Tok:
    def __init__(self, hf=None):
        self.mode = "whitespace*1.3"
        self.tok = None
        if hf:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(
                    hf, use_fast=True, trust_remote_code=True)
                self.mode = f"hf:{hf}"
            except Exception as e:
                print(f"[WARN] HF tokenizer failed: {e}")
        if self.tok is None:
            try:
                import tiktoken
                self.tok = tiktoken.get_encoding("cl100k_base")
                self.mode = "tiktoken:cl100k"
            except Exception:
                pass

    def n(self, s):
        if not s: return 0
        if self.mode.startswith("hf:"):
            return len(self.tok.encode(s, add_special_tokens=False))
        if self.mode.startswith("tiktoken"):
            return len(self.tok.encode(s))
        return int(len(s.split()) * 1.3)


def h(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def classify_and_count(obj, tok, key=None,
                       gen_tokens=None, ctx_tokens=None, scaffold_tokens=None,
                       gen_hashes=None, src_text_set=None):
    """Walk an item, bucketing each string into generated/context/scaffold."""
    if isinstance(obj, str):
        if not obj.strip():
            return
        n = tok.n(obj)
        if key in SCAFFOLD_FIELDS:
            scaffold_tokens[0] += n
        elif key in CONTEXT_FIELDS:
            ctx_tokens[0] += n
        else:
            # generated content — count unique only
            hh = h(obj)
            if hh not in gen_hashes:
                gen_hashes.add(hh)
                gen_tokens[0] += n
    elif isinstance(obj, dict):
        for k, v in obj.items():
            classify_and_count(v, tok, k, gen_tokens, ctx_tokens,
                               scaffold_tokens, gen_hashes, src_text_set)
    elif isinstance(obj, list):
        for v in obj:
            classify_and_count(v, tok, key, gen_tokens, ctx_tokens,
                               scaffold_tokens, gen_hashes, src_text_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--hf-tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=float, default=0.05)
    args = ap.parse_args()

    d = Path(args.dir)
    enriched = d / "multitask_data_enriched.jsonl"
    tok = Tok(args.hf_tokenizer)
    print(f"[tokenizer] {tok.mode}")
    print(f"[sample]    {args.sample*100:.1f}%")

    rng = random.Random(42)

    # Unique human source text (dedup narratives by hash)
    src_hashes = set()
    src_tokens = 0

    gen_tokens = [0]      # unique generated content
    ctx_tokens = [0]      # repeated context
    scaffold_tokens = [0] # ids/enums
    gen_hashes = set()

    n_rec = 0
    with open(enriched) as f:
        for line in f:
            if args.sample < 1.0 and rng.random() > args.sample:
                continue
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_rec += 1

            # unique source narrative
            txt = rec.get("text", "")
            if isinstance(txt, str) and txt.strip():
                hh = h(txt)
                if hh not in src_hashes:
                    src_hashes.add(hh)
                    src_tokens += tok.n(txt)

            # classify everything under tasks
            tasks = rec.get("tasks", {}) or {}
            classify_and_count(tasks, tok, None, gen_tokens, ctx_tokens,
                               scaffold_tokens, gen_hashes, src_hashes)

    scale = 1.0 / args.sample if args.sample < 1.0 else 1.0
    def s(x): return int(round(x * scale))

    total = s(src_tokens) + s(gen_tokens[0]) + s(ctx_tokens[0]) + s(scaffold_tokens[0])

    report = {
        "_provenance": {
            "records_processed": n_rec,
            "sample_fraction": args.sample,
            "scale": scale,
            "tokenizer": tok.mode,
            "note": "generated tokens are DEDUPLICATED by md5; context/source also deduped",
        },
        "unique_source_text_tokens": s(src_tokens),
        "unique_generated_tokens": s(gen_tokens[0]),
        "repeated_context_tokens": s(ctx_tokens[0]),
        "scaffold_tokens": s(scaffold_tokens[0]),
        "total_classified_tokens": total,
        "redundancy_ratio": round(
            (s(ctx_tokens[0])) / max(s(src_tokens), 1), 2),
        "trainable_unique_tokens_estimate":
            s(src_tokens) + s(gen_tokens[0]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("UNIQUE CONTENT ANALYSIS" +
          (" (ESTIMATE)" if args.sample < 1.0 else ""))
    print("=" * 60)
    print(f"  unique source narratives : {s(src_tokens):>15,}")
    print(f"  unique GENERATED content : {s(gen_tokens[0]):>15,}")
    print(f"  repeated context (waste) : {s(ctx_tokens[0]):>15,}")
    print(f"  scaffold (ids/enums)     : {s(scaffold_tokens[0]):>15,}")
    print("  " + "-"*40)
    print(f"  total classified         : {total:>15,}")
    print(f"\n  >> TRAINABLE UNIQUE       : "
          f"{report['trainable_unique_tokens_estimate']:>15,}")
    print(f"  >> context redundancy     : {report['redundancy_ratio']}x source")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
