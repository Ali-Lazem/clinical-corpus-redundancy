# Provenance-based Redundancy Decomposition (PRD)

Code accompanying:

> **How much of an LLM-generated clinical corpus is actually new?
> A production-scale measurement of content redundancy for provenance classification**
> Ali H. Lazem, William J. Teahan. *(under review)*

This repository contains the token-provenance classifier (PRD), its
validation and robustness checks, the compression-based corroboration, and
the figure-generation scripts used in the paper. It does **not** contain the
extraction pipeline that produced the corpus, nor the derived corpus itself.
The source corpus (PMC-Patients) is publicly available (Zhao et al., 2023).

## What this code does

PRD classifies every token of a multi-task LLM extraction corpus into one of
five provenance categories — unique source, unique generated, duplicated
generated, copied context, and scaffold — and aggregates them into
corpus- and channel-level redundancy measures (the Trainable-Unique Ratio
and Context-Copy Ratio). An independent compression-based analysis
corroborates the decomposition without using the provenance labels.

## Repository layout

```
prd/
  redundancy_full_analysis_v2.py     # the PRD classifier (main tool)
  validate_classifier.py             # field-to-category mapping validation
  near_dup_robustness.py             # MinHash near-duplicate robustness check
compression/
  compression_redundancy.py          # aligned compression analysis (4 streams x 4 compressors)
  aggregate_compression_samples.py   # Monte-Carlo aggregation (mean +/- s.d.)
figures/
  make_sankey.py                     # Fig 1: source-to-output composition
  make_pertask.py                    # Fig 2: per-task two-mechanism breakdown
  make_schematic.py                  # Fig 3: per-patient narrative replication
  make_composition.py                # Fig 4: per-channel provenance composition
  make_worked_example.py             # Fig 5: single-record field-size breakdown
  make_compression_fig.py            # Fig: compression composite
  make_perchannel_complementarity.py # Fig: provenance vs compression per channel
  make_si_fig_s1.py                  # SI Fig S1: mechanism-level decomposition
example_usage.py                     # minimal end-to-end example
requirements.txt
LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The PRD classifier uses only the standard library
plus a tokenizer; the compression analysis additionally requires `pyppmd`.

## Usage

Run the provenance classifier over a pipeline's output directory:

```bash
python prd/redundancy_full_analysis_v2.py \
    --dir /path/to/pipeline/output \
    --hf-tokenizer <tokenizer> \
    --out report.json
```

Run the compression corroboration (Monte-Carlo over 10% subsamples):

```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python compression/compression_redundancy.py \
      --enriched /path/to/multitask_data_enriched.jsonl \
      --risk-dir /path/to/risk_files \
      --sample 0.1 --seed $s --workers 0 \
      --out compression_sample_$s.json
done
python compression/aggregate_compression_samples.py \
    --glob "compression_sample_*.json" --out compression_montecarlo.json
```

See `example_usage.py` for a minimal worked example.

## Reproducing the paper figures

Each script in `figures/` regenerates one figure from the analysis outputs.
See the header of each script for its required input.

## Citation

```bibtex
@article{lazem2026redundancy,
  title   = {How much of an LLM-generated clinical corpus is actually new?
             A production-scale measurement of content redundancy for
             provenance classification},
  author  = {Lazem, Ali H. and Teahan, William J.},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the MIT License. See `LICENSE`.
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

Code is licensed under MIT.
Papers is licensed under CC BY 4.0.
