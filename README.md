# Provenance-based Redundancy Decomposition (PRD)

### Measuring how much of an LLM-generated clinical corpus is actually new

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-in%20preparation-lightgrey.svg)](#citation)

Reference implementation for the study *“How much of an LLM-generated clinical
corpus is actually new? A production-scale measurement of content redundancy for
provenance classification”* (Lazem & Teahan).

A dual-LLM clinical extraction pipeline (generator **Llama-3.3-70B**, verifier
**MMed-Llama-3.1-70B**) was applied to the **167,034** patient narratives of
PMC-Patients, producing **2.51 billion** generated tokens across ten task
channels. This repository provides the tools to measure how much of that output
is genuinely informative versus redundant, to corroborate the measurement with
lossless compression, and to test its downstream consequence on a clinical
encoder.

---

## The finding in one table

| Component | Share | Composition |
| :-- | --: | :-- |
| **Trainable-unique** | **10.9 %** | unique source (4.2 %) + unique generated (6.7 %) |
| **Redundant** | **79.4 %** | copied context (67.5 %) + duplicated generation (11.9 %) |
| **Scaffold** | **9.7 %** | identifiers and enumerated metadata |

Raw token count overstates information content by roughly **ninefold**. Four
lossless compressors corroborate the split (the full corpus compresses
**2.7–4.7×** harder than its trainable-unique subset), and at equal token budget,
de-duplicating the corpus measurably improves a clinical encoder on two external
disease-NER benchmarks.

---

## What PRD does

Provenance-based Redundancy Decomposition classifies **every output token** by
where it came from, using a fixed field-to-category taxonomy applied across all
ten channels with a single corpus-wide hash set (so repetition is detected even
across channels):

```
                         ┌─ unique source        ┐
   trainable-unique  ────┤                       │
                         └─ unique generated     │
                                                 ├──  every token, classified
   redundant  ───────────┬─ copied context       │
                         └─ duplicated generation│
   scaffold  ────────────── identifiers/metadata ┘
```

Two reported ratios summarise a corpus:

- **TUR** (Trainable-Unique Ratio) — the informative fraction.
- **CCR** (Context-Copy Ratio) — the losslessly-removable fraction.

---

## Repository layout

| Path | Contents |
| :-- | :-- |
| **`src/`** | PRD classifier, compression corroboration, near-duplicate robustness, the full downstream pipeline, and the significance test |
| **`figures/`** | Scripts that regenerate every paper figure from the JSON in `results/` |
| **`scripts/`** | SLURM drivers and job scripts for the HPC adaptation and probing runs |
| **`results/`** | Released JSON outputs — every number in the paper traces to a file here |
| **`paper/`** | Manuscript and Supplementary LaTeX, with compiled figures in `paper/figures/` |
| **`data/`** | Notes on the corpora (regenerated, not shipped) |

Each folder contains its own `README.md` documenting every file.

---

## Quick start

```bash
pip install -r requirements.txt

# Regenerate the composition + compression figure from released outputs (no GPU):
python3 figures/make_compression_fig.py \
    --json results/compression_full.json \
    --prov results/redundancy_full_v2.json \
    --out  paper/figures/redundancy_fig_compression.pdf
```

Reproducing the full pipeline (corpus construction, encoder adaptation, probing)
requires GPU and the source corpus; see [`src/README.md`](src/README.md) and
[`scripts/README.md`](scripts/README.md).

---

## Reproducing each result

| Result | Command |
| :-- | :-- |
| Provenance decomposition | `python3 src/redundancy_full_analysis_v2.py --dir <output> --hf-tokenizer <llama3> --out results/redundancy_full_v2.json` |
| Compression corroboration | `python3 src/compression_analysis.py compress --streams <streams> --out results/compression_full.json` |
| Near-duplicate robustness | `python3 src/near_dup_robustness.py` |
| Downstream adaptation + probe | `bash scripts/run_all_40k.sh` then `python3 src/aggregate_results.py …` |
| Downstream significance | `python3 src/bootstrap_significance.py --results-dir results/full_40k` |

---

## Method scope

This repository covers the **corpus-redundancy study only**. The downstream
experiment uses a single encoder backbone (BioClinical ModernBERT-base) and two
disease-NER benchmarks (NCBI-Disease, BC5CDR-Disease); the de-duplication effect
is reported as a controlled, equal-budget, within-corpus comparison. The
generation pipeline that produced the corpus is described separately and is not
included here.

---

## Citation

```bibtex
@article{lazem2026redundancy,
  title   = {How much of an LLM-generated clinical corpus is actually new?
             A production-scale measurement of content redundancy for
             provenance classification},
  author  = {Lazem, Ali H. and Teahan, William J.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Computation performed on **Supercomputing Wales** (Falcon, project SCWF00175),
with support from the **Bangor eResearch Team**.
