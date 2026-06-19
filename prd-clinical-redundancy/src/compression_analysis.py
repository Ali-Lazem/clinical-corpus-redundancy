#!/usr/bin/env python3
"""
compress_full_corpus.py
=======================
Memory-safe full-corpus compression for the PRD paper (fixes the OOM in the
all-in-memory v3_aligned script). Two stages:

  STAGE 1 (stream to disk): walk the corpus ONCE with the SAME taxonomy and
  recursive walk as redundancy_full_analysis_v2.py, writing the four provenance
  streams to DISK as .txt files (full / trainable / copied_ctx / dup_gen).
  Nothing large is held in RAM -- bytes are written straight to file handles.

  STAGE 2 (compress files one at a time): compress each stream file with
  gzip / bzip2 / lzma / PPMd, SEQUENTIALLY (no ProcessPool), so only ONE
  compressor model lives in memory at a time. PPMd runs at high order via a
  streaming API so it never holds the whole 20 GB plus a huge model at once.

Usage:
  # Stage 1 only (produce the .txt streams):
  python3 compress_full_corpus.py stream \
      --enriched /scratch/.../risk_v7/multitask_data_enriched.jsonl \
      --risk-dir /scratch/.../risk_v7 \
      --outdir   /scratch/.../reports/streams

  # Stage 2 (compress the streams, one job at a time):
  python3 compress_full_corpus.py compress \
      --outdir /scratch/.../reports/streams \
      --report /scratch/.../reports/compression_full.json \
      --ppmd-order 16 --ppmd-mem-mb 2048
"""
import argparse, json, gzip, bz2, lzma, zlib, hashlib, os, sys, time

CONTEXT_FIELDS = {"context", "verification_anchor", "verification_ctx", "sentence"}
SCAFFOLD_FIELDS = {
    "qa_id","rel_id","event_id","uid","start_char","end_char","confidence",
    "llm_confidence","direction","temporal_order","is_anchored","is_negated_source",
    "timepoint_type","event_type","assertion_status","verifier_status","status",
    "head_type","tail_type","label","category","meta_species","verdict_path",
    "med_id","drug","route","frequency","qtype","risk_id","state","severity",
    "volatility_profile","rule","threshold","decision_type","severity_marker",
    "method","actionability","source","provenance","last_updated","rec_id","type",
    "agentic_check","constraints_applied","based_on_risks","level","node_type","size",
}
PLACEHOLDERS = {"context unavailable","","n/a","none","."}

def load_uid_map(path):
    m={}
    if not path or not os.path.exists(path): 
        print(f"[warn] not found: {path}"); return m
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except Exception: continue
            u=rec.get("uid")
            if u is not None: m[u]=rec
    return m

# ---------------- STAGE 1: stream to disk ----------------
def stage_stream(args):
    os.makedirs(args.outdir, exist_ok=True)
    f_full = open(os.path.join(args.outdir,"full.txt"),"wb")
    f_train= open(os.path.join(args.outdir,"trainable.txt"),"wb")
    f_ctx  = open(os.path.join(args.outdir,"copied_ctx.txt"),"wb")
    f_dup  = open(os.path.join(args.outdir,"dup_gen.txt"),"wb")
    seen_gen=set(); seen_src=set()

    rd = args.risk_dir or os.path.dirname(args.enriched)
    recs_map  = load_uid_map(os.path.join(rd,"active_v6_recommendations.jsonl"))
    states_map= load_uid_map(os.path.join(rd,"active_v6_risk_states.jsonl"))
    risks_map = load_uid_map(os.path.join(rd,"active_v6_risks.jsonl"))
    print(f"[stream] risk channels: recs={len(recs_map):,} states={len(states_map):,} risks={len(risks_map):,}",flush=True)

    def walk(obj,key):
        if isinstance(obj,str):
            s=obj.strip()
            if not s: return
            b=s.encode("utf-8","replace")
            if key in SCAFFOLD_FIELDS: return
            if key in CONTEXT_FIELDS:
                if s.lower() in PLACEHOLDERS: return
                f_full.write(b); f_ctx.write(b); return
            f_full.write(b)
            hh=hashlib.md5(b).hexdigest()
            if hh not in seen_gen: seen_gen.add(hh); f_train.write(b)
            else: f_dup.write(b)
        elif isinstance(obj,dict):
            for k,v in obj.items(): walk(v,k)
        elif isinstance(obj,list):
            for v in obj: walk(v,key)

    n=0; t0=time.time()
    with open(args.enriched) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except Exception: continue
            n+=1
            if n%10000==0:
                el=time.time()-t0
                print(f"\r[stream] {n:,} records  {el:.0f}s",end="",flush=True)
            src=rec.get("text","")
            if isinstance(src,str) and src.strip():
                b=src.encode("utf-8","replace"); f_full.write(b)
                hh=hashlib.md5(b).hexdigest()
                if hh not in seen_src: seen_src.add(hh); f_train.write(b)
            for tname,tval in rec.get("tasks",{}).items():
                walk(tval,None)
            uid=rec.get("uid")
            if uid is not None:
                if uid in recs_map: walk(recs_map[uid].get("recommendations",[]),None)
                if uid in states_map: walk(states_map[uid].get("states",[]),None)
                if uid in risks_map: walk(risks_map[uid].get("risks",[]),None)
    for fh in (f_full,f_train,f_ctx,f_dup): fh.close()
    print(f"\n[stream] done: {n:,} records",flush=True)
    for name in ["full","trainable","copied_ctx","dup_gen"]:
        p=os.path.join(args.outdir,name+".txt")
        print(f"  {name:11s} {os.path.getsize(p):>15,} B",flush=True)

# ---------------- STAGE 2: compress files one at a time ----------------
def compress_file(path, comp, ppmd_order, ppmd_mem_mb, chunk=1<<24):
    """Stream-compress a file; return compressed byte count. One model in RAM."""
    raw=os.path.getsize(path); comp_bytes=0
    if comp=="gzip":
        # gzip module has no streaming compressobj; use zlib with gzip header (wbits=31)
        c=zlib.compressobj(6, zlib.DEFLATED, 31)
    elif comp=="bzip2":
        c=bz2.BZ2Compressor(9)
    elif comp=="lzma":
        c=lzma.LZMACompressor(preset=6)
    elif comp=="ppmd":
        import pyppmd
        # streaming PPMd encoder at explicit high order + memory
        c=pyppmd.PpmdCompressor(max_order=ppmd_order, mem_size=ppmd_mem_mb<<20)
    else:
        raise ValueError(comp)
    with open(path,"rb") as f:
        while True:
            blk=f.read(chunk)
            if not blk: break
            comp_bytes+=len(c.compress(blk))
        comp_bytes+=len(c.flush())
    return raw, comp_bytes

def stage_compress(args):
    comps=["gzip","bzip2","lzma","ppmd"]
    streams=["full","trainable","copied_ctx","dup_gen"]
    out={"_meta":{"ppmd_order":args.ppmd_order,"ppmd_mem_mb":args.ppmd_mem_mb,
                  "note":"full-corpus, sequential, streaming; one model in RAM at a time"},
         "streams":{}, "raw_bytes":{}}
    for s in streams:
        p=os.path.join(args.outdir,s+".txt")
        if not os.path.exists(p): print(f"[skip] missing {p}"); continue
        out["streams"][s]={}
        for comp in comps:
            t0=time.time()
            try:
                raw,cb=compress_file(p,comp,args.ppmd_order,args.ppmd_mem_mb)
                ratio=round(cb/raw,4) if raw else None
                out["streams"][s][comp]={"raw_bytes":raw,"compressed_bytes":cb,
                                         "ratio":ratio,
                                         "reduction_pct":round(100*(1-cb/raw),2) if raw else None}
                out["raw_bytes"][s]=raw
                print(f"[{s}/{comp}] ratio={ratio}  ({time.time()-t0:.0f}s)",flush=True)
            except Exception as e:
                out["streams"][s][comp]={"error":str(e)}
                print(f"[{s}/{comp}] ERROR {e}",flush=True)
    # headline factors
    out["full_vs_trainable_factor"]={}
    for comp in comps:
        rf=out["streams"].get("full",{}).get(comp,{}).get("ratio")
        rt=out["streams"].get("trainable",{}).get(comp,{}).get("ratio")
        if rf and rt: out["full_vs_trainable_factor"][comp]=round(rt/rf,2)
    ff=out["raw_bytes"].get("full")
    if ff:
        out["copied_ctx_byte_fraction"]=round(out["raw_bytes"].get("copied_ctx",0)/ff,4)
        out["dup_gen_byte_fraction"]=round(out["raw_bytes"].get("dup_gen",0)/ff,4)
    json.dump(out,open(args.report,"w"),indent=2)
    print(f"\n[done] wrote {args.report}",flush=True)
    print("\nFULL vs TRAINABLE factor:")
    for c,v in out["full_vs_trainable_factor"].items(): print(f"  {c:6s} {v}x")

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    s1=sub.add_parser("stream"); s1.add_argument("--enriched",required=True)
    s1.add_argument("--risk-dir",default=None); s1.add_argument("--outdir",required=True)
    s2=sub.add_parser("compress"); s2.add_argument("--outdir",required=True)
    s2.add_argument("--report",required=True)
    s2.add_argument("--ppmd-order",type=int,default=16)
    s2.add_argument("--ppmd-mem-mb",type=int,default=2048)
    a=ap.parse_args()
    if a.cmd=="stream": stage_stream(a)
    else: stage_compress(a)

if __name__=="__main__": main()
