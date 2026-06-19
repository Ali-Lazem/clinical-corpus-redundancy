# `figures/` — figure-generation scripts

One standalone script per paper figure. Each reads the released numbers (either
embedded from `redundancy_full_v2.json`, or from the JSON in `../results/`) and
writes a PDF (and PNG) into `./figures/`. No GPU or source corpus required.

| Script | Paper figure | Output file |
| :-- | :-- | :-- |
| `make_sankey.py` | Fig. 2 — source-to-output flow | `redundancy_fig1_sankey.pdf` |
| `make_fig3_pertask.py` | Fig. 3 — two redundancy mechanisms per channel | `redundancy_fig2_pertask.pdf` |
| `make_fig_4.py` | Fig. 4 — context-copy schematic | `redundancy_fig3_schematic.pdf` |
| `make_fig5_composition.py` | Fig. 5 — per-channel provenance composition | `redundancy_fig4_composition.pdf` |
| `make_fig6_worked_example.py` | Fig. 6 — single-record worked example | `redundancy_fig5_worked_example.pdf` |
| `make_compression_fig.py` | Fig. 7 — composition + compression | `compression_fig.pdf` |
| `make_perchannel_complementarity.py` | Fig. 8 — provenance vs compression per channel | `redundancy_fig_perchannel_complementarity.pdf` |
| `make_twobench_grid.py` | Fig. 9 — downstream de-duplication grid | `downstream_grid.pdf` |
| `make_si_fig_s1.py` | SI Fig. S1 — mechanism decomposition | `fig_s1_mechanisms.png` |
| `make_loss_panel.py` | SI Fig. S2 — adaptation loss | `loss_panel.pdf` |

Figure 1 in the paper (the pipeline-overview diagram) is drawn directly in LaTeX
(TikZ) and has no generating script.

Example:

```bash
python3 make_compression_fig.py \
    --json ../results/compression_full.json \
    --prov ../results/redundancy_full_v2.json \
    --out  ../paper/figures/compression_fig.pdf
```

The provenance figures (`make_sankey.py`, `make_fig3_pertask.py`,
`make_fig5_composition.py`) embed the verified ten-channel counts from
`redundancy_full_v2.json`, so they reproduce the paper figures without an
external data file.
