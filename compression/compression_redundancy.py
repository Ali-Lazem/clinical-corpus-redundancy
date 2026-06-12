#!/usr/bin/env python3
"""
compression_redundancy_v3_aligned_v2.py
=======================================
Model-free corroboration of the token-provenance redundancy finding
(Paper 2), via classical lossless compression (suggested by W. Teahan).

This version is FULLY ALIGNED to redundancy_full_analysis_v2.py:
  * identical field taxonomy (CONTEXT_FIELDS / SCAFFOLD_FIELDS / PLACEHOLDERS)
  * identical recursive walk() that descends into dicts/lists, so text nested
    inside structured fields (e.g. the `reason` object) is counted
  * SAME TEN CHANNELS: the 7 enriched channels read from rec["tasks"], PLUS
    the 3 risk-derived channels read from separate files and joined by uid:
        recommendations <- active_v6_recommendations.jsonl
        risk_states     <- active_v6_risk_states.jsonl
        risks           <- active_v6_risks.jsonl
  * source narrative de-duplicated and counted once (same asymmetry)
  * global generated-hash set spans all channels (corpus-level dup definition)

So the compression streams cover EXACTLY the same bytes the provenance
classifier counts. Compression VALIDATES the decomposition; it is not the
contribution.

STREAMS:
  full        every text-bearing field, corpus order (what a consumer stores)
  trainable   unique source + first-occurrence generated (trainable-unique set)
  copied_ctx  copied-context fields only          -> mechanism 1 (context copy)
  dup_gen     duplicate generated strings (2nd+)  -> mechanism 2 (generation dup)

COMPRESSORS: gzip (LZ77), bzip2 (BWT), lzma/xz (LZMA), pyppmd (PPMd, PPM family).

CONTROLS / METRICS:
  shuffled FULL (ordering control), per-channel breakdown,
  compression_gap = trainable_ratio - full_ratio,
  full_vs_trainable factor = trainable_ratio / full_ratio (headline),
  copied_ctx & dup_gen byte fractions (cross-check vs classifier CCR/dup_gen).

PARALLELISM: every (stream x compressor) job runs across the allocated cores.

Usage:
  python3 compression_redundancy_v3_aligned_v2.py \
      --enriched /path/to/pipeline/output/multitask_data_enriched.jsonl \
      --risk-dir /path/to/pipeline/output \
      --sample 1.0 --workers 0 --no-shuffle-control \
      --out /path/to/reports/compression_redundancy_v4.json

  # --workers 0 = all detected cores; --risk-dir defaults to the enriched dir
"""
import argparse, json, gzip, bz2, lzma, hashlib, random, io, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pyppmd
    HAVE_PPMD = True
except ImportError:
    HAVE_PPMD = False

# ============================================================
# Field taxonomy: IDENTICAL to redundancy_full_analysis_v2.py
# ============================================================
CONTEXT_FIELDS = {"context", "verification_anchor", "verification_ctx",
                  "sentence"}
SCAFFOLD_FIELDS = {
    "qa_id", "rel_id", "event_id", "uid", "start_char", "end_char",
    "confidence", "llm_confidence", "direction", "temporal_order",
    "is_anchored", "is_negated_source", "timepoint_type", "event_type",
    "assertion_status", "verifier_status", "status", "head_type",
    "tail_type", "label", "category", "meta_species", "verdict_path",
    "med_id", "drug", "route", "frequency", "qtype", "risk_id", "state",
    "severity", "volatility_profile", "rule", "threshold", "decision_type",
    "severity_marker", "method", "actionability", "source", "provenance",
    "last_updated", "rec_id", "type", "agentic_check", "constraints_applied",
    "based_on_risks", "level", "node_type", "size",
}
PLACEHOLDERS = {"context unavailable", "", "n/a", "none", "."}


# ============================================================
# Helpers
# ============================================================
def progress(done, total, prefix="", width=40, t0=None):
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = (time.time() - t0) if t0 else 0
    eta = (elapsed / frac - elapsed) if frac > 0 else 0
    sys.stdout.write(f"\r{prefix} [{bar}] {done}/{total} "
                     f"({100*frac:5.1f}%)  elapsed {elapsed:5.0f}s  eta {eta:5.0f}s")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n"); sys.stdout.flush()


def load_uid_map(path):
    """Read a *.jsonl keyed by 'uid' into {uid: record}. Missing file -> {}."""
    m = {}
    if not path or not os.path.exists(path):
        print(f"[warn] risk file not found: {path}")
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


# ============================================================
# Stream collection (recursive walk mirrors the classifier exactly)
# ============================================================
def collect_streams(enriched, sample, seed=42, shuffle_records=False,
                    count_lines=None, risk_dir=None):
    rng = random.Random(seed)
    seen_gen = set()       # md5 of generated strings: first occ -> trainable
    seen_src = set()       # md5 of source narratives: dedup
    full       = io.BytesIO()
    trainable  = io.BytesIO()
    copied     = io.BytesIO()
    dup_gen    = io.BytesIO()
    per_channel = {}

    # --- preload the 3 risk-derived channels (separate files), keyed by uid ---
    rd = risk_dir or os.path.dirname(enriched)
    recs_map   = load_uid_map(os.path.join(rd, "active_v6_recommendations.jsonl"))
    states_map = load_uid_map(os.path.join(rd, "active_v6_risk_states.jsonl"))
    risks_map  = load_uid_map(os.path.join(rd, "active_v6_risks.jsonl"))
    print(f"[collect] risk channels: recs={len(recs_map):,} "
          f"states={len(states_map):,} risks={len(risks_map):,}", flush=True)

    def walk(obj, key, ch_buf):
        """Mirror of the classifier's walk(): recurse into dicts/lists; route
        each leaf string into FULL (+channel) and the right provenance stream."""
        if isinstance(obj, str):
            s = obj.strip()
            if not s:
                return
            b = s.encode("utf-8", "replace")
            if key in SCAFFOLD_FIELDS:
                return                                  # scaffold: not corpus text
            if key in CONTEXT_FIELDS:
                if s.lower() in PLACEHOLDERS:
                    return                              # placeholder guard
                full.write(b); ch_buf.write(b); copied.write(b)
                return
            full.write(b); ch_buf.write(b)              # generated text
            hh = hashlib.md5(b).hexdigest()
            if hh not in seen_gen:
                seen_gen.add(hh); trainable.write(b)
            else:
                dup_gen.write(b)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k, ch_buf)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key, ch_buf)

    def line_iter():
        if shuffle_records:
            with open(enriched) as f:
                lines = [ln for ln in f
                         if not (sample < 1.0 and rng.random() > sample)]
            rng.shuffle(lines)
            for ln in lines:
                yield ln
        else:
            with open(enriched) as f:
                for ln in f:
                    if sample < 1.0 and rng.random() > sample:
                        continue
                    yield ln

    n = 0
    t0 = time.time()
    for line in line_iter():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        n += 1
        if count_lines and (n % 5000 == 0):
            progress(min(n, count_lines), count_lines, prefix="[collect]", t0=t0)

        # --- unique source narrative (dedup, counted once) ---
        src_txt = rec.get("text", "")
        if isinstance(src_txt, str) and src_txt.strip():
            b = src_txt.encode("utf-8", "replace")
            full.write(b)
            per_channel.setdefault("source", io.BytesIO()).write(b)
            hh = hashlib.md5(b).hexdigest()
            if hh not in seen_src:
                seen_src.add(hh); trainable.write(b)

        # --- 7 enriched task channels ---
        for tname, tval in rec.get("tasks", {}).items():
            ch = per_channel.setdefault(tname, io.BytesIO())
            walk(tval, None, ch)

        # --- 3 risk-derived channels (separate files), joined by uid ---
        uid = rec.get("uid")
        if uid is not None:
            if uid in recs_map:
                ch = per_channel.setdefault("recommendations", io.BytesIO())
                walk(recs_map[uid].get("recommendations", []), None, ch)
            if uid in states_map:
                ch = per_channel.setdefault("risk_states", io.BytesIO())
                walk(states_map[uid].get("states", []), None, ch)
            if uid in risks_map:
                ch = per_channel.setdefault("risks", io.BytesIO())
                walk(risks_map[uid].get("risks", []), None, ch)

    if count_lines:
        progress(count_lines, count_lines, prefix="[collect]", t0=t0)

    streams = {
        "full":       full.getvalue(),
        "trainable":  trainable.getvalue(),
        "copied_ctx": copied.getvalue(),
        "dup_gen":    dup_gen.getvalue(),
    }
    per_ch = {name: buf.getvalue() for name, buf in per_channel.items()}
    return streams, per_ch, n


# ============================================================
# Compression jobs (run in worker processes)
# ============================================================
def _compress_one(args):
    name, comp, data = args
    raw = len(data)
    try:
        if comp == "gzip":
            c = len(gzip.compress(data, 6))
        elif comp == "bzip2":
            c = len(bz2.compress(data, 9))
        elif comp == "lzma":
            c = len(lzma.compress(data, preset=6))
        elif comp == "ppmd":
            c = len(pyppmd.compress(data))
        else:
            return (name, comp, {"error": "unknown compressor"})
        return (name, comp, {
            "raw_bytes": raw, "compressed_bytes": c,
            "ratio": round(c / raw, 4) if raw else None,
            "reduction_pct": round(100 * (1 - c / raw), 2) if raw else None,
        })
    except Exception as e:
        return (name, comp, {"error": str(e), "raw_bytes": raw})


def run_jobs(blobs, comps, workers, label="compress"):
    jobs = [(name, comp, data) for name, data in blobs.items()
            for comp in comps if len(data) > 0]
    results = {name: {} for name in blobs}
    t0 = time.time(); done = 0; total = len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_compress_one, j) for j in jobs]
        for fut in as_completed(futs):
            name, comp, r = fut.result()
            results[name][comp] = r
            done += 1
            progress(done, total, prefix=f"[{label}]", t0=t0)
    return results


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--risk-dir", default=None,
                    help="dir with active_v6_{recommendations,risk_states,risks}.jsonl; "
                         "defaults to the enriched file's directory")
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for subsampling (vary for Monte-Carlo runs)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-shuffle-control", action="store_true")
    ap.add_argument("--no-per-channel", action="store_true")
    args = ap.parse_args()

    workers = args.workers or os.cpu_count()
    comps = ["gzip", "bzip2", "lzma"] + (["ppmd"] if HAVE_PPMD else [])
    print(f"[setup] workers={workers}  compressors={comps}  "
          f"ppmd={'yes' if HAVE_PPMD else 'MISSING'}", flush=True)

    print("[setup] counting records for progress bar ...", flush=True)
    nlines = sum(1 for _ in open(args.enriched))
    est = int(nlines * args.sample) if args.sample < 1.0 else nlines
    print(f"[setup] ~{est:,} records expected", flush=True)

    print("[collect] building provenance streams (corpus order) ...", flush=True)
    streams, per_ch, n = collect_streams(args.enriched, args.sample, seed=args.seed,
                                         count_lines=est, risk_dir=args.risk_dir)
    for k, v in streams.items():
        print(f"  {k:11s} {len(v):>15,} B", flush=True)

    print(f"[compress] {len(streams)} streams x {len(comps)} compressors "
          f"across {workers} cores ...", flush=True)
    stream_res = run_jobs(streams, comps, workers, label="streams")

    result = {
        "_provenance": {
            "enriched": args.enriched,
            "risk_dir": args.risk_dir or os.path.dirname(args.enriched),
            "sample": args.sample, "seed": args.seed, "records": n, "workers": workers,
            "compressors": comps,
            "channels_note": "7 enriched (rec['tasks']) + 3 risk-derived "
                             "(recommendations/risk_states/risks) joined by uid; "
                             "field taxonomy + recursive walk identical to "
                             "redundancy_full_analysis_v2.py.",
            "framing": "compression VALIDATES the provenance decomposition; "
                       "it is not the contribution.",
        },
        "streams": stream_res,
        "raw_bytes": {k: len(v) for k, v in streams.items()},
    }

    ff = streams["full"]
    result["copied_ctx_byte_fraction"] = round(len(streams["copied_ctx"]) / len(ff), 4) if ff else None
    result["dup_gen_byte_fraction"]    = round(len(streams["dup_gen"]) / len(ff), 4) if ff else None
    result["full_vs_trainable_factor"] = {}
    result["compression_gap"] = {}
    for c in comps:
        rf = stream_res["full"].get(c, {}).get("ratio")
        rt = stream_res["trainable"].get(c, {}).get("ratio")
        if rf and rt:
            result["full_vs_trainable_factor"][c] = round(rt / rf, 2)
            result["compression_gap"][c] = round(rt - rf, 4)

    # ordering control
    if not args.no_shuffle_control:
        print("[shuffle] rebuilding FULL with records shuffled (ordering control) ...",
              flush=True)
        sh_streams, _, _ = collect_streams(args.enriched, args.sample, seed=args.seed,
                                           shuffle_records=True, count_lines=est,
                                           risk_dir=args.risk_dir)
        sh_res = run_jobs({"full_shuffled": sh_streams["full"]}, comps, workers,
                          label="shuffled")
        result["full_shuffled"] = sh_res["full_shuffled"]
        result["ordering_robustness"] = {}
        for c in comps:
            ordered = stream_res["full"].get(c, {}).get("ratio")
            shuf = sh_res["full_shuffled"].get(c, {}).get("ratio")
            if ordered and shuf:
                result["ordering_robustness"][c] = {
                    "ordered": ordered, "shuffled": shuf,
                    "delta": round(shuf - ordered, 4)}

    # per-channel
    if not args.no_per_channel:
        print(f"[per-channel] {len(per_ch)} channels x {len(comps)} compressors ...",
              flush=True)
        pc_res = run_jobs(per_ch, comps, workers, label="channels")
        result["per_channel"] = {
            name: {"raw_bytes": len(per_ch[name]), **pc_res[name]}
            for name in per_ch
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.out}", flush=True)

    # ---- human-readable summary ----
    print("\n" + "=" * 64)
    print("COMPRESSION SUMMARY (compressed/raw; lower = more redundant)")
    print("=" * 64)
    for s in ("full", "trainable", "copied_ctx", "dup_gen"):
        rb = result["raw_bytes"][s]
        print(f"\n{s.upper()}  (raw {rb:,} B)")
        for c in comps:
            r = stream_res[s].get(c, {})
            if r.get("ratio") is not None:
                print(f"  {c:6s} ratio={r['ratio']:.4f}  ({r['reduction_pct']:.1f}% reduction)")
    print("\nHEADLINE  FULL compresses harder than TRAINABLE by factor:")
    for c, v in result["full_vs_trainable_factor"].items():
        print(f"  {c:6s} {v}x   (compression gap {result['compression_gap'][c]})")
    print(f"\nCOPIED_CTX = {100*result['copied_ctx_byte_fraction']:.1f}% of FULL bytes "
          f"(cross-check vs classifier CCR)")
    print(f"DUP_GEN    = {100*result['dup_gen_byte_fraction']:.1f}% of FULL bytes "
          f"(cross-check vs classifier dup_gen)")
    if "ordering_robustness" in result:
        print("\nORDERING CONTROL (FULL ordered vs shuffled; small delta = robust):")
        for c, d in result["ordering_robustness"].items():
            print(f"  {c:6s} ordered={d['ordered']:.4f}  shuffled={d['shuffled']:.4f}  "
                  f"delta={d['delta']:+.4f}")
    if "per_channel" in result:
        print(f"\nPER-CHANNEL: {len(result['per_channel'])} channels covered: "
              f"{sorted(result['per_channel'])}")


if __name__ == "__main__":
    main()
