# `src/` — analysis and experiment code

## Provenance + compression (the core method)

| File | Purpose |
|---|---|
| `redundancy_full_analysis_v2.py` | PRD token classifier. Walks every string-valued field across all ten channels, assigns each to one of five provenance categories, and writes `redundancy_full_v2.json`. Uses one corpus-wide MD5 set, so duplicates are detected across channels. |
| `validate_classifier.py` | Sanity checks on the PRD classification (category coverage, placeholder handling). |
| `compression_analysis.py` | Full-corpus lossless-compression corroboration with gzip-6, bzip2-9, LZMA-6, and PPMd (order 16, 2048 MB). Two-stage (stream then compress) to stay within memory. Writes `compression_full.json`. |
| `near_dup_robustness.py` | MinHash near-duplicate check (Jaccard > 0.85) confirming the exact-match estimate is a conservative lower bound. |

## Downstream experiment (ModernBERT-base, two benchmarks)

| File | Purpose |
|---|---|
| `build_pretraining_corpora.py` | Builds the three equal-budget conditions (raw / de-duplicated / context-removed) and their `*.budget.txt` files (~174.3M tokens each). |
| `build_rare_slice.py` | Builds the NCBI rare/common/unseen stratification by corpus disease frequency. |
| `mlm_adapt.py` | Continued masked-language-model adaptation of the backbone on a budget corpus. |
| `train_eval_ncbi.py` | Trains a linear probe on NCBI-Disease and evaluates per slice. |
| `prepare_bc5cdr.py` | Builds the BC5CDR-Disease (disease-only) benchmark split. |
| `probe_bc5cdr.py` | Trains and evaluates the linear probe on BC5CDR-Disease. |
| `aggregate_results.py` | Aggregates NCBI per-(condition,seed) metrics into the paper's downstream JSON. |
| `aggregate_bc5cdr.py` | Same for BC5CDR. |
| `bootstrap_significance.py` | Mention-level bootstrap test of the de-duplication F1 gain: resamples test mentions (per-seed F1 averaged within each resample) and reports the gain, 95% CI, and two-sided p-value per slice. Runs on NCBI-Disease, for which per-mention predictions (`preds_*.jsonl`) are retained. |

All downstream conditions share one backbone, one token budget, and one probe
protocol, so the comparison isolates the corpus-redundancy property.

**Note on significance testing.** `bootstrap_significance.py` requires per-mention
prediction files (`preds_<cond>_seed<k>.jsonl`), which the NCBI probe
(`train_eval_ncbi.py`) writes. The BC5CDR probe retains aggregate metrics only;
to run the mention-level test on BC5CDR, regenerate its predictions with
prediction-saving enabled. The paper reports mention-level significance for
NCBI-Disease and the de-duplication gain and its direction for BC5CDR-Disease.
