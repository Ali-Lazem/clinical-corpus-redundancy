# Token-provenance analysis for LLM-generated clinical corpora

Code accompanying **"How much of an LLM-generated clinical corpus is actually
new? A production-scale measurement of content redundancy for provenance
classification"** ([authors], npj Digital Medicine, 2026). Preprint: [DOI].

This repository contains the provenance-classification tool, its validation
and robustness checks, and the figure-generation scripts. Applied to the
output of a multi-task clinical extraction pipeline (167,034 patient
narratives, 2.51 billion generated tokens across ten text-bearing channels),
the tool classifies every output token by provenance and shows that only
10.9% is trainable-unique content while 79.4% is redundant — raw token count
overstates information content by roughly ninefold.

## What this is

**Core tool**
- `redundancy_full_analysis.py` — the token-provenance classifier. Reads a
  pipeline's output and partitions every token into five provenance
  categories (unique source, unique generated, duplicated generated, copied
  context, scaffold), then reports the Trainable-Unique Ratio (TUR) and
  Context-Copy Ratio (CCR). Pipeline-agnostic: point it at any
  equivalently-structured corpus.

**Validation and robustness**
- `validate_classifier.py` — validates the field-to-category mapping on a
  random sample of records, confirming that copied-context fields are
  verbatim substrings of the source narrative and that generated fields are
  not. Reproduces the validation reported in the Methods.
- `near_dup_robustness.py` — MinHash near-duplicate robustness check
  (Jaccard > 0.85) on a sample of generated content, confirming the
  exact-match redundancy figures are a conservative lower bound.

**Figures**
- `make_all_figures.py` — regenerates all five paper figures in one pass
  (invokes `make_fig3.py` and `make_fig5_worked_example.py`).
- `make_fig3.py` — the context-copy mechanism schematic (Figure 3).
- `make_fig5_worked_example.py` — the single-record worked example (Figure 5),
  rendered from the measured field sizes of one real record.

**Data**
- `redundancy_full_v2_counts.json` — aggregate token counts per category per
  channel (counts only; no clinical text).

## What this is NOT

The extraction pipeline that produced the corpus, and the derived corpus
itself, are not included (see the paper's data-availability statement; the
corpus is reserved for a forthcoming dataset paper pending human evaluation).

## Install & run

```bash
pip install -r requirements.txt
python3 make_all_figures.py          # regenerate all five figures into figures/
```

Run the analysis on your own pipeline output:

```bash
python3 redundancy_full_analysis.py \
    --dir path/to/data \
    --hf-tokenizer path/to/tokenizer \
    --out path/to/output/redundancy_full_v2.json \
    --sample 1.0
```

The tokenizer is optional: with `transformers` installed it uses the exact
model-family tokenizer; otherwise it falls back to `tiktoken`, then to a
whitespace estimate. Token counts are tokenizer-dependent in absolute
magnitude but not in relative composition.

## Reproducing the paper's validation and robustness checks

```bash
# field-to-category mapping validation (Methods)
python3 validate_classifier.py \
    --enriched path/to/multitask_data_enriched.jsonl \
    --sample 200

# near-duplicate robustness (Results)
python3 near_dup_robustness.py \
    --dir path/to/output_dir \
    --sample 0.02 \
    --threshold 0.85
```

## Headline numbers (ten-channel, full 167,034-patient scale)

| Quantity | Value |
|---|---|
| Total output tokens | 2.51 B |
| Trainable-unique | 272.6 M (10.9%) |
| Redundant | 1.99 B (79.4%) |
| Scaffold | 244.6 M (9.7%) |
| TUR | 0.109 |
| CCR | 0.675 |
| Redundancy ratio (vs. source) | 19.1× |
| Overstatement (raw vs. trainable) | 9.2× (≈ ninefold) |

## Figures

1. Source-to-output flow (four-bucket split)
2. Two redundancy mechanisms per channel
3. The context-copy mechanism (schematic)
4. Per-channel token-provenance composition (100% stacked)
5. Worked example: context-copy redundancy in one real record

## Hardware requirements

The analysis tool, validation scripts, and figure scripts run on any machine
with Python 3.8+; no GPU required. The extraction pipeline that produced the
analysed corpus (not included here) was run on 4× NVIDIA H200 GPUs.

## License

See [LICENSE](LICENSE).
