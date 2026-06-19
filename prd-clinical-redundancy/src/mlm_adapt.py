#!/usr/bin/env python3
"""
mlm_adapt.py
============
Step 4a: continued MLM pretraining ("domain adaptation") of a base encoder
on ONE condition corpus from Step 2 (A_raw / B_dedup / B1_ctxremoved).

Produces a condition-specific backbone. Run once per condition; then
train_eval_ncbi.py probes each adapted backbone on NCBI-Disease. The
A-vs-B difference on NCBI is the experiment's result.

EQUAL-BUDGET: point --corpus at the *.budget.txt files (all ~174.3M tokens)
so every condition is adapted on the same token budget -- this removes the
"raw just trained on more tokens" confound (pre-registered control).

Determinism: --seed fixes data order + init. Run multiple seeds for the
full experiment (each seed -> its own adapted backbone -> its own probe).

Usage (one condition, one seed):
  python3 mlm_adapt.py \
      --base thomas-sounack/BioClinical-ModernBERT-base \
      --corpus /path/pretrain_corpora/B_dedup.budget.txt \
      --condition B_dedup --seed 1 \
      --steps 5000 --out /path/mlm_adapted

  -> writes /path/mlm_adapted/B_dedup_seed1/  (a HF model dir)

Requires: transformers>=4.48, torch, datasets-free (reads .txt directly).
"""
import argparse, os, math, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus", required=True, help="a *.budget.txt condition corpus")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000,
                    help="total optimisation steps (equal across conditions)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-docs", type=int, default=0, help="cap docs (debug)")
    args = ap.parse_args()

    import torch, numpy as np
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              DataCollatorForLanguageModeling, TrainingArguments,
                              Trainer)
    from torch.utils.data import Dataset

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    outdir = Path(args.out) / f"{args.condition}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"[mlm] base={args.base} condition={args.condition} seed={args.seed}")
    print(f"[mlm] reading corpus {args.corpus} ...")

    # read documents (one per line from the Step-2 builder)
    docs = []
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                docs.append(line)
            if args.max_docs and len(docs) >= args.max_docs:
                break
    print(f"[mlm] {len(docs):,} documents")

    class TxtDS(Dataset):
        def __init__(self, docs, tok, max_len):
            self.docs = docs; self.tok = tok; self.max_len = max_len
        def __len__(self): return len(self.docs)
        def __getitem__(self, i):
            enc = self.tok(self.docs[i], truncation=True, max_length=self.max_len,
                           padding=False, return_special_tokens_mask=True)
            return enc

    ds = TxtDS(docs, tok, args.max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mlm_prob)
    model = AutoModelForMaskedLM.from_pretrained(args.base)

    targs = TrainingArguments(
        output_dir=str(outdir / "trainer"),
        overwrite_output_dir=True,
        max_steps=args.steps,                 # EQUAL steps across conditions
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
        dataloader_drop_last=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator, tokenizer=tok)
    print(f"[mlm] training for {args.steps} steps "
          f"(batch {args.batch} x grad-accum {args.grad_accum})")
    trainer.train()

    # save the adapted backbone (for AutoModelForTokenClassification later)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print(f"[done] adapted backbone -> {outdir}")
    (outdir / "adapt_manifest.json").write_text(
        __import__("json").dumps({
            "base": args.base, "condition": args.condition, "seed": args.seed,
            "corpus": args.corpus, "n_docs": len(docs), "steps": args.steps,
            "batch": args.batch, "grad_accum": args.grad_accum, "lr": args.lr,
            "max_len": args.max_len, "mlm_prob": args.mlm_prob,
            "note": "equal-step / equal-budget adaptation; compare across "
                    "conditions on NCBI via train_eval_ncbi.py",
        }, indent=2))

if __name__ == "__main__":
    main()
