#!/usr/bin/env python3
"""
validate_classifier.py — validates the field->category mapping used by the
token-provenance classifier (Methods "Validation" paragraph).

Schema (confirmed from multitask_data_enriched.jsonl):
  top-level: uid, text (THE SOURCE NARRATIVE), tasks, meta_species
  tasks.qa[].context        -> copied context (should be verbatim source)
  tasks.re[].context        -> copied context (should be verbatim source)
  tasks.temporal_events[].sentence -> copied context (source span)
  tasks.qa[].answer/reason  -> generated content (should NOT be source)
  tasks.summary             -> generated content
  tasks.ner[].entity        -> generated (extracted span; may appear in source)

Checks on a random sample:
  1. context/sentence fields are verbatim substrings of `text`.
  2. answer/reason/summary fields are NOT substrings of `text`.

Run:
  python3 validate_classifier.py \
      --enriched /scratch/SCWF00175/shared/code/risk_v7/multitask_data_enriched.jsonl \
      --sample 200
"""
import json, argparse, random

CONTEXT_FIELDS = {"context", "sentence"}          # copied-source fields
GEN_FIELDS     = {"answer", "reason", "summary"}  # generated fields

def norm(s):
    return " ".join(str(s).split()).strip()

def iter_task_items(rec):
    """Yield (field_name, value) for the fields we classify, per task item."""
    tasks = rec.get("tasks", {})
    for tname, tval in tasks.items():
        items = tval if isinstance(tval, list) else [tval]
        for it in items:
            if not isinstance(it, dict):
                # e.g. summary may be a plain string
                if tname == "summary" and isinstance(it, str):
                    yield "summary", it
                continue
            for k, v in it.items():
                if isinstance(v, str) and (k in CONTEXT_FIELDS or k in GEN_FIELDS):
                    yield k, v
        # summary stored directly as string/list at task level
        if tname == "summary" and isinstance(tval, str):
            yield "summary", tval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    sample = []
    with open(args.enriched) as f:
        for i, line in enumerate(f):
            if len(sample) < args.sample:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < args.sample:
                    sample[j] = line

    ctx_total = ctx_verbatim = 0
    gen_total = gen_novel = 0
    ctx_misses, gen_hits = [], []
    n_used = 0

    for line in sample:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        source = norm(rec.get("text", ""))
        if not source:
            continue
        n_used += 1
        for k, v in iter_task_items(rec):
            vn = norm(v)
            if len(vn) < 20:
                continue
            probe = vn[:200]
            in_source = (probe in source) or (vn in source)
            if k in CONTEXT_FIELDS:
                ctx_total += 1
                if in_source:
                    ctx_verbatim += 1
                else:
                    if len(ctx_misses) < 5:
                        ctx_misses.append((k, vn[:70]))
            elif k in GEN_FIELDS:
                gen_total += 1
                if not in_source:
                    gen_novel += 1
                else:
                    if len(gen_hits) < 5:
                        gen_hits.append((k, vn[:70]))

    print("=" * 62)
    print(f"Records sampled: {len(sample)}  | with source narrative: {n_used}")
    print("-" * 62)
    print("COPIED-CONTEXT fields (context, sentence) -- expect verbatim:")
    if ctx_total:
        print(f"  {ctx_verbatim}/{ctx_total} = {100*ctx_verbatim/ctx_total:.1f}% are exact source substrings")
    else:
        print("  (none found)")
    print("GENERATED fields (answer, reason, summary) -- expect NOT in source:")
    if gen_total:
        print(f"  {gen_novel}/{gen_total} = {100*gen_novel/gen_total:.1f}% are novel (not source substrings)")
    else:
        print("  (none found)")
    print("=" * 62)
    if ctx_misses:
        print(f"\nContext fields NOT matching source ({len(ctx_misses)} shown):")
        for k, s in ctx_misses:
            print(f"   [{k}] {s!r}")
    if gen_hits:
        print(f"\nGenerated fields that WERE in source ({len(gen_hits)} shown):")
        for k, s in gen_hits:
            print(f"   [{k}] {s!r}")

if __name__ == "__main__":
    main()
