#!/usr/bin/env python3
"""
train_eval_ncbi.py
==================
Step 4+5 of the downstream experiment: train an encoder NER head on
NCBI-Disease and evaluate per rare/common/unseen slice, for each corpus
condition (A_raw / B_dedup / B1_ctxremoved), across multiple seeds.

THREE MODES (compute ladder -- start cheap):
  linear_probe   : freeze backbone, train only the token-classification head.
                   Fast (minutes-hours). Use this FIRST to see if a signal
                   exists before spending pretraining compute.
  full_finetune  : fine-tune the whole backbone on NCBI. Stronger signal.
  mlm_pretrain   : (separate entrypoint, see --mlm) continue MLM pretraining
                   on a condition corpus FIRST, then fine-tune on NCBI.

CONDITIONS come from Step 2 corpora. For linear_probe / full_finetune the
"condition" is which corpus the backbone was MLM-adapted on; if you have not
yet MLM-adapted (fast path), all conditions share the base backbone and the
comparison is trivially null -- so the meaningful A-vs-B contrast REQUIRES
the mlm_pretrain step to have produced condition-specific backbones. The
linear_probe mode is therefore for (a) pipeline validation and (b) the Q2
"does our corpus transfer at all" check once you DO have an adapted backbone.

OUTPUTS (per condition x seed):
  preds_<condition>_seed<k>.jsonl   per-test-mention prediction + gold + slice
  metrics_<condition>_seed<k>.json  entity-F1 overall + per slice
These feed Step 5 (aggregate_results.py) for bootstrap CIs + the A-vs-B
difference-in-differences.

Requires: transformers, datasets, torch, seqeval, numpy.

Usage (linear-probe read, base backbone, Q2 transfer check):
  python3 train_eval_ncbi.py \
      --backbone <hf-or-local-encoder> \
      --rare-slice /path/to/rare_slice/ncbi_test_tagged.jsonl \
      --mode linear_probe --seeds 1 2 3 \
      --out /path/to/results

Usage (full experiment, condition-specific backbones from MLM step):
  python3 train_eval_ncbi.py \
      --backbone /path/to/mlm_adapted/<condition> \
      --condition A_raw \
      --rare-slice .../ncbi_test_tagged.jsonl \
      --mode full_finetune --seeds 1 2 3 --out /path/to/results
"""
import argparse, json, os, sys, random
from pathlib import Path

def log(m): print(m, flush=True)

def set_seed(s):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_rare_tags(path):
    """uid-free: map normalised mention string -> slice bin, from Step 3."""
    bin_of = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for mb in rec.get("mention_bins", []):
                bin_of[mb["norm"]] = mb["bin"]
    return bin_of

def norm_mention(s):
    import re
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s); s = re.sub(r"[.,;:]+$", "", s)
    return s

def mentions_from_bio(tokens, tags, id2label=None):
    """Extract disease mention spans (as token-index ranges + text)."""
    spans = []; cur = []
    for i,(t,g) in enumerate(zip(tokens, tags)):
        gl = id2label[g] if id2label else g
        is_b = (gl == 1 or gl == "B" or gl == "B-Disease")
        is_i = (gl == 2 or gl == "I" or gl == "I-Disease")
        if is_b:
            if cur: spans.append(cur)
            cur = [i]
        elif is_i:
            cur.append(i)
        else:
            if cur: spans.append(cur); cur = []
    if cur: spans.append(cur)
    return [(s[0], s[-1], " ".join(tokens[s[0]:s[-1]+1])) for s in spans]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--condition", default="base")
    ap.add_argument("--rare-slice", required=True)
    ap.add_argument("--mode", choices=["linear_probe","full_finetune"], default="linear_probe")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 probe, 2e-5 finetune")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch, numpy as np
        from datasets import load_dataset
        from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                                  TrainingArguments, Trainer, DataCollatorForTokenClassification)
        from seqeval.metrics import f1_score, precision_score, recall_score
    except ImportError as e:
        log(f"[FATAL] missing dependency: {e}")
        log("  pip install --user torch transformers datasets seqeval numpy")
        sys.exit(2)

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    bin_of = load_rare_tags(args.rare_slice)
    log(f"[slice] loaded {len(bin_of):,} mention->bin tags")

    log("[data] loading NCBI-Disease ...")
    ds = load_dataset("ncbi_disease")
    label_list = ["O","B-Disease","I-Disease"]   # tags 0,1,2
    id2label = {0:"O",1:"B-Disease",2:"I-Disease"}; label2id = {v:k for k,v in id2label.items()}

    tok = AutoTokenizer.from_pretrained(args.backbone)

    def tokenize_align(ex):
        t = tok(ex["tokens"], truncation=True, is_split_into_words=True,
                max_length=args.max_len)
        labs = []
        for i, lab in enumerate(ex["ner_tags"]):
            pass
        word_ids = t.word_ids()
        prev = None; aligned = []
        for wid in word_ids:
            if wid is None: aligned.append(-100)
            elif wid != prev: aligned.append(ex["ner_tags"][wid])
            else:
                # inside same word: I- if was B/I else O
                aligned.append(ex["ner_tags"][wid] if ex["ner_tags"][wid]==2 else (2 if ex["ner_tags"][wid]==1 else 0))
            prev = wid
        t["labels"] = aligned
        return t

    tds = ds.map(tokenize_align, batched=False)
    collator = DataCollatorForTokenClassification(tok)
    lr = args.lr or (1e-3 if args.mode=="linear_probe" else 2e-5)

    all_seed_metrics = []
    for seed in args.seeds:
        set_seed(seed)
        log(f"\n[train] condition={args.condition} mode={args.mode} seed={seed} lr={lr}")
        model = AutoModelForTokenClassification.from_pretrained(
            args.backbone, num_labels=3, id2label=id2label, label2id=label2id)
        if args.mode == "linear_probe":
            for p in model.base_model.parameters(): p.requires_grad = False

        targs = TrainingArguments(
            output_dir=str(OUT/f"tmp_{args.condition}_s{seed}"),
            learning_rate=lr, per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch, num_train_epochs=args.epochs,
            eval_strategy="epoch", save_strategy="no", logging_steps=200,
            seed=seed, report_to=[])
        trainer = Trainer(model=model, args=targs,
                          train_dataset=tds["train"], eval_dataset=tds["validation"],
                          tokenizer=tok, data_collator=collator)
        trainer.train()

        # ---- predict on test, per-mention, per-slice ----
        pred = trainer.predict(tds["test"])
        logits = pred.predictions
        pred_ids = np.argmax(logits, axis=-1)

        # reconstruct word-level predictions per sentence, score per slice
        slice_gold = {"RARE":[], "COMMON":[], "UNSEEN":[], "ALL":[]}
        slice_pred = {"RARE":[], "COMMON":[], "UNSEEN":[], "ALL":[]}
        preds_out = []
        for si, ex in enumerate(ds["test"]):
            toks = ex["tokens"]; gold = ex["ner_tags"]
            # map subword preds back to word level
            t = tok(toks, truncation=True, is_split_into_words=True, max_length=args.max_len)
            wids = t.word_ids()
            word_pred = [0]*len(toks); seen=set()
            for j,wid in enumerate(wids):
                if wid is None or wid in seen: continue
                seen.add(wid)
                if j < pred_ids.shape[1]:
                    word_pred[wid] = int(pred_ids[si][j])
            gold_m = mentions_from_bio(toks, gold)
            for (a,b,mtext) in gold_m:
                nb = bin_of.get(norm_mention(mtext), "UNSEEN")
                # build BIO sequences for just this mention region for seqeval
                g_seq = [id2label[gold[k]] for k in range(a,b+1)]
                p_seq = [id2label[word_pred[k]] for k in range(a,b+1)]
                slice_gold[nb].append(g_seq); slice_pred[nb].append(p_seq)
                slice_gold["ALL"].append(g_seq); slice_pred["ALL"].append(p_seq)
                preds_out.append({"mention":mtext,"bin":nb,"gold":g_seq,"pred":p_seq})

        with open(OUT/f"preds_{args.condition}_seed{seed}.jsonl","w") as w:
            for r in preds_out: w.write(json.dumps(r)+"\n")

        metrics = {"condition":args.condition,"seed":seed,"mode":args.mode}
        for sl in ("ALL","RARE","COMMON","UNSEEN"):
            if slice_gold[sl]:
                metrics[sl] = {
                    "n_mentions": len(slice_gold[sl]),
                    "f1": round(float(f1_score(slice_gold[sl], slice_pred[sl])),4),
                    "precision": round(float(precision_score(slice_gold[sl], slice_pred[sl])),4),
                    "recall": round(float(recall_score(slice_gold[sl], slice_pred[sl])),4),
                }
            else:
                metrics[sl] = {"n_mentions":0}
        (OUT/f"metrics_{args.condition}_seed{seed}.json").write_text(json.dumps(metrics,indent=2))
        all_seed_metrics.append(metrics)
        log(f"  [seed {seed}] ALL F1={metrics['ALL'].get('f1')}  "
            f"RARE F1={metrics['RARE'].get('f1')}  COMMON F1={metrics['COMMON'].get('f1')}")

    # summary across seeds
    def mean_std(vals):
        vals=[v for v in vals if v is not None]
        if not vals: return None,None
        m=sum(vals)/len(vals)
        sd=(sum((x-m)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.0
        return round(m,4),round(sd,4)
    summary={"condition":args.condition,"mode":args.mode,"seeds":args.seeds,"per_slice":{}}
    for sl in ("ALL","RARE","COMMON","UNSEEN"):
        f1s=[m[sl].get("f1") for m in all_seed_metrics if m[sl].get("f1") is not None]
        mn,sd=mean_std(f1s)
        summary["per_slice"][sl]={"f1_mean":mn,"f1_std":sd,
                                  "n_mentions":all_seed_metrics[0][sl].get("n_mentions")}
    (OUT/f"summary_{args.condition}.json").write_text(json.dumps(summary,indent=2))
    log("\n=== SUMMARY ===")
    for sl,v in summary["per_slice"].items():
        log(f"  {sl:8s} F1 {v['f1_mean']} +/- {v['f1_std']}  (n={v['n_mentions']})")
    log(f"\n[done] wrote results to {OUT}")

if __name__ == "__main__":
    main()
