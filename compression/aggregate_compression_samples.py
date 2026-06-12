#!/usr/bin/env python3
"""
aggregate_compression_samples.py
================================
Aggregate N Monte-Carlo subsample runs of compression_redundancy_v3_aligned_v2.py
into mean +/- std per (stream, compressor), plus the headline factor stats.

This quantifies SAMPLING STABILITY (robustness), and -- if a full-corpus
result is supplied via --full -- also reports how closely the subsample mean
tracks the full-corpus value (ACCURACY). Stability and accuracy are different
claims; we report both.

Usage:
  python3 aggregate_compression_samples.py \
      --glob "/path/to/reports/compression_sample_*.json" \
      --full /path/to/reports/compression_redundancy_v4.json \
      --out  /path/to/reports/compression_montecarlo.json
"""
import argparse, json, glob, math

STREAMS = ["full", "trainable", "copied_ctx", "dup_gen"]
COMPS   = ["gzip", "bzip2", "lzma", "ppmd"]

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)   # sample std
    return m, math.sqrt(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for the sample JSONs")
    ap.add_argument("--full", default=None, help="optional full-corpus JSON (accuracy anchor)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[error] no files match {args.glob}"); return
    runs = [json.load(open(f)) for f in files]
    print(f"[aggregate] {len(runs)} sample runs: {[f.split('/')[-1] for f in files]}")

    full = json.load(open(args.full)) if args.full else None

    out = {"n_samples": len(runs), "files": files, "streams": {}, "factor": {}}

    # per (stream, compressor): mean +/- std of ratio, vs full
    print(f"\n{'stream':<12}{'comp':<7}{'mean':>9}{'std':>9}{'full':>9}{'|dev|':>8}")
    for s in STREAMS:
        out["streams"][s] = {}
        for c in COMPS:
            ratios = [r["streams"].get(s, {}).get(c, {}).get("ratio") for r in runs]
            m, sd = mean_std(ratios)
            fullv = (full["streams"].get(s, {}).get(c, {}).get("ratio")
                     if full else None)
            dev = (abs(m - fullv) if (m is not None and fullv is not None) else None)
            out["streams"][s][c] = {"mean": round(m,4) if m is not None else None,
                                    "std": round(sd,4) if sd is not None else None,
                                    "full": fullv,
                                    "abs_dev_from_full": round(dev,4) if dev is not None else None}
            if m is not None:
                fs = f"{fullv:.4f}" if fullv is not None else "   -  "
                ds = f"{dev:.4f}" if dev is not None else "  -  "
                print(f"{s:<12}{c:<7}{m:>9.4f}{sd:>9.4f}{fs:>9}{ds:>8}")

    # headline factor mean +/- std
    print(f"\n{'factor':<12}{'mean':>9}{'std':>9}{'full':>9}")
    for c in COMPS:
        facs = [r.get("full_vs_trainable_factor", {}).get(c) for r in runs]
        m, sd = mean_std(facs)
        fullv = full.get("full_vs_trainable_factor", {}).get(c) if full else None
        out["factor"][c] = {"mean": round(m,3) if m is not None else None,
                            "std": round(sd,3) if sd is not None else None,
                            "full": fullv}
        if m is not None:
            fs = f"{fullv:.2f}" if fullv is not None else "  -  "
            print(f"{c:<12}{m:>9.3f}{sd:>9.3f}{fs:>9}")

    # cross-check fractions
    cc = [r.get("copied_ctx_byte_fraction") for r in runs]
    dg = [r.get("dup_gen_byte_fraction") for r in runs]
    out["copied_ctx_byte_fraction"] = dict(zip(["mean","std"], mean_std(cc)))
    out["dup_gen_byte_fraction"]    = dict(zip(["mean","std"], mean_std(dg)))
    print(f"\nCOPIED_CTX byte-fraction: {out['copied_ctx_byte_fraction']['mean']:.3f} "
          f"+/- {out['copied_ctx_byte_fraction']['std']:.3f}")
    print(f"DUP_GEN    byte-fraction: {out['dup_gen_byte_fraction']['mean']:.3f} "
          f"+/- {out['dup_gen_byte_fraction']['std']:.3f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}")

    # paper-ready line
    g = out["streams"]["full"]["gzip"]; l = out["streams"]["full"]["lzma"]
    print("\n--- paper-ready ---")
    print(f"gzip FULL ratio: {g['mean']} +/- {g['std']} across {len(runs)} samples")
    print(f"lzma FULL ratio: {l['mean']} +/- {l['std']}")
    fl = out["factor"]["lzma"]
    print(f"full-vs-trainable factor (lzma): {fl['mean']} +/- {fl['std']}")

if __name__ == "__main__":
    main()
