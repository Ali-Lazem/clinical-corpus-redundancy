#!/usr/bin/env python3
"""
redundancy_full_analysis_v2.py
==============================

Paper-2 headline analysis, EXTENDED to ten text-bearing channels.

v1 analysed seven channels read from rec["tasks"] in the enriched file.
v2 adds the three risk-derived channels that live in separate files but
ALSO emit generated text and (risk_states) verbatim source-narrative copies:

    recommendations  <- active_v6_recommendations.jsonl
    risk_states      <- active_v6_risk_states.jsonl
    risks            <- active_v6_risks.jsonl

The visualisation channel (visualization_payload.jsonl) remains EXCLUDED:
it emits only graph elements (ids, colours, shapes, levels) and entity
labels that are re-references of NER strings -- no new generated text and
no narrative copies.

Key consistency guarantees:
  * the SAME global gen_hashes set spans all ten channels, so a string that
    first appears in (say) QAR and recurs in risks is correctly DUP. This
    is the corpus-level redundancy definition used throughout the paper.
  * source narrative counted ONCE (deduplicated) -- the asymmetry vs the
    full context-copy count IS the measurement (do not "fix" this).
  * verification_anchor / verification_ctx PLACEHOLDER GUARD: in the three
    risk channels these may hold the literal "Context unavailable" rather
    than a narrative slice; placeholders -> scaffold, real text -> context.
    (Audit shows the seven enriched channels have 0% placeholders, but the
    guard is applied uniformly so the rule is identical everywhere.)

Usage (full run for paper numbers):
    python3 redundancy_full_analysis_v2.py \
        --dir /scratch/SCWF00175/shared/code/risk_v7 \
        --hf-tokenizer /scratch/SCWF00175/shared/models/llama3-70b \
        --out /scratch/SCWF00175/shared/reports/redundancy_full_v2.json \
        --sample 1.0
"""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import random

# ---- field taxonomy (shared by enriched channels AND risk channels) ----
CONTEXT_FIELDS = {"context", "verification_anchor", "verification_ctx",
                  "sentence"}
SCAFFOLD_FIELDS = {
    # enriched-channel scaffold (unchanged from v1)
    "qa_id", "rel_id", "event_id", "uid", "start_char", "end_char",
    "confidence", "llm_confidence", "direction", "temporal_order",
    "is_anchored", "is_negated_source", "timepoint_type", "event_type",
    "assertion_status", "verifier_status", "status", "head_type",
    "tail_type", "label", "category", "meta_species", "verdict_path",
    "med_id", "drug", "route", "frequency", "qtype", "risk_id", "state",
    # risk-channel scaffold (new)
    "severity", "volatility_profile", "rule", "threshold", "decision_type",
    "severity_marker", "method", "actionability", "source", "provenance",
    "last_updated", "rec_id", "type", "agentic_check", "constraints_applied",
    "based_on_risks", "level", "node_type", "size",
}
# placeholder strings that must NOT count as context-copy
PLACEHOLDERS = {"context unavailable", "", "n/a", "none", "."}


class Tok:
    def __init__(self, hf=None):
        self.mode = "whitespace*1.3"; self.tok = None
        if hf:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(hf, use_fast=True,
                                                         trust_remote_code=True)
                self.mode = f"hf:{hf}"
            except Exception as e:
                print(f"[WARN] HF tokenizer: {e}")
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


def walk(obj, tok, key, acc, gen_hashes):
    """Classify every string under obj into acc buckets. Shared by all channels.
    gen_hashes is the GLOBAL set (corpus-level dup definition)."""
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return
        n = tok.n(s)
        if key in SCAFFOLD_FIELDS:
            acc["scaffold"] += n
        elif key in CONTEXT_FIELDS:
            # placeholder guard: only real narrative slices are context-copy
            if s.lower() in PLACEHOLDERS:
                acc["scaffold"] += n
            else:
                acc["context_copied"] += n
                acc["context_field_tokens"][key] += n
        else:
            hh = h(s)
            if hh in gen_hashes:
                acc["generated_dup"] += n
            else:
                gen_hashes.add(hh)
                acc["generated_unique"] += n
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, tok, k, acc, gen_hashes)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, tok, key, acc, gen_hashes)


def load_uid_map(path):
    """Read a *.jsonl keyed by 'uid' into {uid: record}. Missing file -> {}."""
    m = {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] not found: {path}")
        return m
    with open(p) as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--hf-tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=float, default=1.0)
    args = ap.parse_args()

    D = Path(args.dir)
    enriched = D / "multitask_data_enriched.jsonl"
    tok = Tok(args.hf_tokenizer)
    rng = random.Random(42)
    print(f"[tokenizer] {tok.mode}   [sample] {args.sample*100:.0f}%")

    # ---- preload the three risk channels, keyed by uid ----
    print("[load] recommendations / risk_states / risks ...")
    recs_map  = load_uid_map(D / "active_v6_recommendations.jsonl")
    states_map = load_uid_map(D / "active_v6_risk_states.jsonl")
    risks_map  = load_uid_map(D / "active_v6_risks.jsonl")
    print(f"[load] recs={len(recs_map):,}  states={len(states_map):,}  risks={len(risks_map):,}")

    src_hashes = set()
    src_tokens = 0
    task_acc = defaultdict(lambda: {
        "context_copied": 0, "generated_unique": 0, "generated_dup": 0,
        "scaffold": 0, "context_field_tokens": Counter(),
    })
    gen_hashes = set()           # GLOBAL, spans all ten channels
    n_rec = 0

    # seven enriched channels (v1) + three risk channels (v2) = ten
    ENRICHED_TASKS = ["ner", "re", "qa", "temporal_events", "summary",
                      "medications", "risk_qa"]

    with open(enriched) as f:
        for line in f:
            if args.sample < 1.0 and rng.random() > args.sample:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_rec += 1
            uid = rec.get("uid")

            # unique source (deduplicated, counted once)
            txt = rec.get("text", "")
            if isinstance(txt, str) and txt.strip():
                hh = h(txt)
                if hh not in src_hashes:
                    src_hashes.add(hh)
                    src_tokens += tok.n(txt)

            # 7 enriched channels
            tasks = rec.get("tasks", {}) or {}
            for tname in ENRICHED_TASKS:
                tval = tasks.get(tname)
                if tval is None:
                    continue
                walk(tval, tok, None, task_acc[tname], gen_hashes)

            # 3 risk channels, joined by uid, SAME global gen_hashes
            if uid is not None:
                if uid in recs_map:
                    walk(recs_map[uid].get("recommendations", []),
                         tok, None, task_acc["recommendations"], gen_hashes)
                if uid in states_map:
                    walk(states_map[uid].get("states", []),
                         tok, None, task_acc["risk_states"], gen_hashes)
                if uid in risks_map:
                    walk(risks_map[uid].get("risks", []),
                         tok, None, task_acc["risks"], gen_hashes)

    scale = 1.0 / args.sample if args.sample < 1.0 else 1.0
    def s(x): return int(round(x * scale))

    tot_ctx   = sum(a["context_copied"] for a in task_acc.values())
    tot_gen_u = sum(a["generated_unique"] for a in task_acc.values())
    tot_gen_d = sum(a["generated_dup"] for a in task_acc.values())
    tot_scaf  = sum(a["scaffold"] for a in task_acc.values())
    total = s(src_tokens) + s(tot_ctx) + s(tot_gen_u) + s(tot_gen_d) + s(tot_scaf)

    trainable_unique = s(src_tokens) + s(tot_gen_u)
    redundant = s(tot_ctx) + s(tot_gen_d)
    redundancy_ratio = round(redundant / max(s(src_tokens), 1), 2)

    report = {
        "_provenance": {"records": n_rec, "sample": args.sample,
                        "scale": scale, "tokenizer": tok.mode,
                        "channels": ENRICHED_TASKS + ["recommendations",
                                    "risk_states", "risks"],
                        "excluded": ["visualization (structured graph only)"]},
        "global": {
            "unique_source_tokens": s(src_tokens),
            "unique_generated_tokens": s(tot_gen_u),
            "duplicated_generated_tokens": s(tot_gen_d),
            "copied_context_tokens": s(tot_ctx),
            "scaffold_tokens": s(tot_scaf),
            "total_tokens": total,
            "trainable_unique_tokens": trainable_unique,
            "redundant_tokens": redundant,
            "redundancy_ratio_vs_source": redundancy_ratio,
            "pct_trainable": round(100 * trainable_unique / max(total, 1), 1),
            "pct_redundant": round(100 * redundant / max(total, 1), 1),
            "pct_scaffold": round(100 * s(tot_scaf) / max(total, 1), 1),
            "TUR": round(trainable_unique / max(total, 1), 4),
            "CCR": round(s(tot_ctx) / max(total, 1), 4),
        },
        "per_task": {},
    }
    for tname, a in task_acc.items():
        tt = (s(a["context_copied"]) + s(a["generated_unique"])
              + s(a["generated_dup"]) + s(a["scaffold"]))
        if tt == 0:
            continue
        report["per_task"][tname] = {
            "total_tokens": tt,
            "copied_context": s(a["context_copied"]),
            "generated_unique": s(a["generated_unique"]),
            "generated_dup": s(a["generated_dup"]),
            "scaffold": s(a["scaffold"]),
            "pct_context_copied": round(100*s(a["context_copied"])/max(tt,1),1),
            "pct_dup_generated": round(100*s(a["generated_dup"])/max(tt,1),1),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    g = report["global"]
    print("\n" + "="*66)
    print("GLOBAL TOKEN COMPOSITION (TEN CHANNELS)" +
          (" [ESTIMATE]" if args.sample < 1 else ""))
    print("="*66)
    print(f"  unique source text      : {g['unique_source_tokens']:>15,}")
    print(f"  unique generated        : {g['unique_generated_tokens']:>15,}")
    print(f"  duplicated generated    : {g['duplicated_generated_tokens']:>15,}")
    print(f"  copied context          : {g['copied_context_tokens']:>15,}")
    print(f"  scaffold                : {g['scaffold_tokens']:>15,}")
    print(f"  {'-'*44}")
    print(f"  TOTAL                   : {g['total_tokens']:>15,}")
    print(f"\n  >> trainable-unique     : {g['trainable_unique_tokens']:>15,}  ({g['pct_trainable']}%)  TUR={g['TUR']}")
    print(f"  >> redundant            : {g['redundant_tokens']:>15,}  ({g['pct_redundant']}%)")
    print(f"  >> scaffold             : {g['scaffold_tokens']:>15,}  ({g['pct_scaffold']}%)")
    print(f"  >> CCR                  : {g['CCR']}")
    print(f"  >> redundancy vs source : {g['redundancy_ratio_vs_source']}x")
    print(f"  >> check 3 buckets sum  : {g['pct_trainable']+g['pct_redundant']+g['pct_scaffold']:.1f}%")
    print("\n" + "="*66)
    print("PER-CHANNEL (sorted by size)")
    print("="*66)
    print(f"  {'channel':<18}{'total':>13}{'ctx':>13}{'dupgen':>13}{'%ctx':>7}{'%dup':>7}")
    for tname, t in sorted(report["per_task"].items(), key=lambda x: -x[1]["total_tokens"]):
        print(f"  {tname:<18}{t['total_tokens']:>13,}{t['copied_context']:>13,}"
              f"{t['generated_dup']:>13,}{t['pct_context_copied']:>6}%{t['pct_dup_generated']:>6}%")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
