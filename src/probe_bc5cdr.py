#!/usr/bin/env python3
"""
probe_bc5cdr.py
===============
Linear-probe a FROZEN adapted backbone on BC5CDR-Disease and report entity-F1
per frequency slice (RARE/COMMON/UNSEEN/ALL). Mirrors the NCBI probe protocol:
frozen encoder, single linear token-classification head, seqeval entity F1,
subword->word alignment on first subword.

Usage:
  python3 probe_bc5cdr.py \
      --backbone /scratch/.../mlm_adapted_40k/A_raw_seed1 \
      --data /scratch/SCWF00175/shared/datasets/bc5cdr_disease \
      --epochs 5 --lr 1e-3 --seed 1 \
      --out /scratch/.../results/bc5cdr_40k/metrics_A_raw_seed1.json
"""
import argparse, json, random
from pathlib import Path
import numpy as np

LABELS = ["O", "B-Disease", "I-Disease"]; L2I = {l:i for i,l in enumerate(LABELS)}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def align_labels(word_ids, word_labels):
    """First-subword alignment: label first subword of each word, -100 elsewhere."""
    out, prev = [], None
    for wid in word_ids:
        if wid is None: out.append(-100)
        elif wid != prev: out.append(word_labels[wid])
        else: out.append(-100)
        prev = wid
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModel
    from seqeval.metrics import f1_score, precision_score, recall_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}   [backbone] {a.backbone}")
    if dev == "cpu":
        print("[WARN] running on CPU -- on the cluster ensure a GPU node "
              "(login node cfl2 falls back to CPU silently)")

    tok = AutoTokenizer.from_pretrained(a.backbone, use_fast=True)
    encoder = AutoModel.from_pretrained(a.backbone).to(dev)
    for p in encoder.parameters():  # FROZEN backbone
        p.requires_grad = False
    encoder.eval()
    hidden = encoder.config.hidden_size

    head = torch.nn.Linear(hidden, len(LABELS)).to(dev)   # the only trainable part
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)

    D = Path(a.data)
    train = load_jsonl(D / "bc5cdr_train.jsonl")
    test_tagged = load_jsonl(D / "bc5cdr_test_tagged.jsonl")

    def encode_batch(rows):
        enc = tok([r["tokens"] for r in rows], is_split_into_words=True,
                  truncation=True, max_length=a.maxlen, padding=True,
                  return_tensors="pt")
        labels = []
        for i, r in enumerate(rows):
            labels.append(align_labels(enc.word_ids(i), r["ner_tags"]))
        # pad labels to enc length
        Ln = enc["input_ids"].shape[1]
        labels = [l + [-100]*(Ln-len(l)) for l in labels]
        return enc.to(dev), torch.tensor(labels).to(dev)

    # ---- train head ----
    for ep in range(a.epochs):
        random.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), a.bs):
            rows = train[i:i+a.bs]
            enc, labels = encode_batch(rows)
            with torch.no_grad():
                h = encoder(**{k:v for k,v in enc.items()}).last_hidden_state
            logits = head(h)
            loss = lossf(logits.view(-1, len(LABELS)), labels.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{a.epochs}  loss={tot/max(1,len(train)//a.bs):.4f}")

    # ---- evaluate, bucketed by slice ----
    # We score entity-level F1 overall, then per slice by restricting the
    # gold/pred mention set to mentions whose gold slice == target slice.
    head.eval()
    # collect per-sentence predicted + gold label sequences (in LABELS space)
    gold_seqs, pred_seqs, slice_seqs = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_tagged), a.bs):
            rows = test_tagged[i:i+a.bs]
            enc, _ = encode_batch(rows)
            h = encoder(**{k:v for k,v in enc.items()}).last_hidden_state
            logits = head(h); preds = logits.argmax(-1).cpu().numpy()
            for j, r in enumerate(rows):
                wids = enc.word_ids(j)
                # map first-subword predictions back to word level
                word_pred = {}
                for k, wid in enumerate(wids):
                    if wid is not None and wid not in word_pred:
                        word_pred[wid] = preds[j][k]
                n = len(r["tokens"])
                pg = [LABELS[r["ner_tags"][w]] for w in range(n)]
                pp = [LABELS[int(word_pred.get(w, 0))] for w in range(n)]
                gold_seqs.append(pg); pred_seqs.append(pp)

    # overall
    def scores(g, p):
        return (f1_score(g, p), precision_score(g, p), recall_score(g, p))
    f1_all, pr_all, rc_all = scores(gold_seqs, pred_seqs)

    # per-slice: build gold/pred sequences where only mentions of that slice
    # keep their B/I labels in GOLD; others set to O in BOTH gold and pred so
    # they don't count. Slice membership is per-gold-mention (from tagged file).
    def slice_filtered(target):
        gg, pp = [], []
        for r, gseq, pseq in zip(test_tagged, gold_seqs, pred_seqs):
            # determine, per token, whether its gold mention is in target slice
            keep = [False]*len(gseq)
            # walk gold mentions in order, matching r["slices"]
            mi = 0; k = 0
            while k < len(gseq):
                if gseq[k] == "B-Disease":
                    in_slice = (mi < len(r["slices"]) and r["slices"][mi]==target)
                    keep[k] = in_slice; k2 = k+1
                    while k2 < len(gseq) and gseq[k2]=="I-Disease":
                        keep[k2]=in_slice; k2+=1
                    mi += 1; k = k2
                else:
                    k += 1
            g2 = [gseq[i] if keep[i] else "O" for i in range(len(gseq))]
            p2 = [pseq[i] if keep[i] else "O" for i in range(len(pseq))]
            gg.append(g2); pp.append(p2)
        return scores(gg, pp)

    res = {"ALL":  {"f1":f1_all,"precision":pr_all,"recall":rc_all}}
    for sl in ["RARE","COMMON","UNSEEN"]:
        f1,pr,rc = slice_filtered(sl)
        res[sl] = {"f1":f1,"precision":pr,"recall":rc}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"backbone":a.backbone,"seed":a.seed,"slices":res}, open(a.out,"w"), indent=2)
    print(f"[done] {a.out}")
    for sl in ["RARE","COMMON","UNSEEN","ALL"]:
        print(f"  {sl:7s} F1={res[sl]['f1']:.4f}")

if __name__ == "__main__":
    main()
