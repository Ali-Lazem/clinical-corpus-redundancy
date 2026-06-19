# `results/` — released analysis outputs

Every table and figure in the paper is computed from these JSON files, so results
reproduce without rerunning the pipeline.

| File | Produced by | Contents |
|---|---|---|
| `redundancy_full_v2.json` | `redundancy_full_analysis_v2.py` | Global token composition (10.9% / 79.4% / 9.7%; TUR, CCR) and per-channel provenance |
| `compression_full.json` | `compression_analysis.py` | Full-corpus compression ratios for the four provenance streams under four compressors |
| `downstream_results.json` | `aggregate_results.py` | NCBI de-duplication gains, 10k depth |
| `downstream_results_20k.json` | `aggregate_results.py` | NCBI gains, 20k depth |
| `downstream_results_40k.json` | `aggregate_results.py` | NCBI gains, 40k (primary) depth |
| `bc5cdr_results.json` | `aggregate_bc5cdr.py` | BC5CDR de-duplication gains, all depths |
| `bc5cdr_stratification.json` | `prepare_bc5cdr.py` | BC5CDR rare/common/unseen tagging by corpus frequency |
