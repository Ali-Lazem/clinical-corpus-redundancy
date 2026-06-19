# `data/` — corpora notes

The large source and generated corpora are **not** included in this repository:

- the full generated corpus (~10 GB, 2.51B tokens), and
- the three equal-budget condition files (`A_raw.budget.txt`, `B_dedup.budget.txt`, `B1_ctxremoved.budget.txt`, ~174.3M tokens each).

They are regenerated from the source by `../src/build_pretraining_corpora.py`, and
are available on request subject to the data terms of
[PMC-Patients](https://github.com/zhao-zy15/PMC-Patients).

The released JSON outputs in `../results/` contain every number needed to
reproduce the paper's tables and figures without the raw corpora.
