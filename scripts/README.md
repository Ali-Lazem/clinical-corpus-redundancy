# `scripts/` — HPC driver scripts (SLURM)

Shell drivers that orchestrate the downstream experiment on Supercomputing Wales
(H200 GPUs). They enforce the equal-budget protocol: each condition is adapted on
its `*.budget.txt` corpus (~174.3M tokens) for a fixed step count.

| Script | Runs |
|---|---|
| `run_all.sh` | Adapt + probe at 10,000 steps (all conditions, all seeds) |
| `run_all_20k.sh` | Adapt + probe at 20,000 steps |
| `run_all_40k.sh` | Adapt + probe at 40,000 steps (primary depth) |
| `run_bc5cdr_probes.sh` | BC5CDR probing across the adapted backbones |

Paths and the SLURM account are set at the top of each script; adjust for your
environment before running.
