# clinical-corpus-redundancy
Token-provenance analysis for LLM-generated clinical corpora

# Token-provenance analysis for LLM-generated clinical corpora

Code accompanying **"How much of an LLM-generated clinical corpus is actually
new? A production-scale measurement of content redundancy for provenance
classification"** ([authors], npj Digital Medicine, 2026). Preprint: [DOI].

This repository contains the provenance-classification tool and the
figure-generation scripts. Applied to the output of a multi-task clinical
extraction pipeline (167,034 patient narratives, 2.51 billion generated
tokens across ten text-bearing channels), the tool classifies every output
token by provenance and shows that only ~10.9% is trainable-unique content
while ~79.4% is redundant.

## What this is
- `redundancy_full_analysis.py` — the token-provenance classifier. Reads a
  pipeline's output and partitions every token into five provenance
  categories (unique source, unique generated, duplicated generated, copied
  context, scaffold), then reports the Trainable-Unique Ratio (TUR) and
  Context-Copy Ratio (CCR). Pipeline-agnostic: point it at any
  equivalently-structured corpus.
- `make_all_figures.py` / `make_fig3.py` — regenerate the paper's four figures
  from the embedded aggregate counts.
- `redundancy_full_v2_counts.json` — aggregate token counts per category per
  channel (counts only; no clinical text).

## What this is NOT
The extraction pipeline that produced the corpus, and the derived corpus
itself, are not included (see the paper's data-availability statement; the
corpus is reserved for a forthcoming dataset paper pending human evaluation).

## Install & run
```bash
pip install -r requirements.txt
python3 make_all_figures.py          # regenerate the four figures into figures/
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

## Headline numbers (ten-channel, full 167,034-patient scale)
| Quantity | Value |
|---|---|
| Total output tokens | 2.51 B |
| Trainable-unique | 272.6 M (10.9%) |
| Redundant | 1.99 B (79.4%) |
| Scaffold | 244.6 M (9.7%) |
| TUR | 0.109 |
| CCR | 0.675 |
| Redundancy ratio (vs. source) | 19.1x |

## Figures
1. Source-to-output flow (four-bucket split)
2. Two redundancy mechanisms per channel
3. The context-copy mechanism (schematic)
4. Per-channel token-provenance composition (100% stacked)

## Hardware requirements
The analysis tool and figure scripts run on any machine with Python 3.8+; no
GPU required. The extraction pipeline that produced the analysed corpus (not
included here) was run on 4x NVIDIA H200 GPUs.

## License
See [LICENSE](LICENSE).
