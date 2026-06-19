#!/bin/bash
# run_bc5cdr_probes.sh
# Probe all 27 existing adapted backbones on BC5CDR-Disease.
# No re-adaptation: reuses mlm_adapted{,_20k,_40k}/ from the NCBI experiment.
# Respects QOS by submitting one probe job at a time via the same self-chaining
# pattern as run_all.sh (probe jobs are short, so this is quick).

set -euo pipefail
SHARED=/scratch/SCWF00175/shared
DATA=$SHARED/datasets/bc5cdr_disease
CODE=$SHARED/code

CONDS=(A_raw B_dedup B1_ctxremoved)
SEEDS=(1 2 3)
# depth -> backbone dir
declare -A BBDIR=( [10k]=$SHARED/mlm_adapted [20k]=$SHARED/mlm_adapted_20k [40k]=$SHARED/mlm_adapted_40k )

submit_one () {
  local depth=$1 cond=$2 seed=$3
  local bb=${BBDIR[$depth]}/${cond}_seed${seed}
  local outdir=$SHARED/results/bc5cdr_${depth}
  local out=$outdir/metrics_${cond}_seed${seed}.json
  mkdir -p "$outdir"
  if [[ -f "$out" ]]; then echo "[skip] $out exists"; return; fi
  if [[ ! -d "$bb" ]]; then echo "[MISSING backbone] $bb"; return; fi
  echo "[probe] depth=$depth cond=$cond seed=$seed  backbone=$bb"
  sbatch --account=SCWF00175_w_teahan_171 --partition=gpu_h200 \
         --gres=gpu:2 --time=24:00:00 --mem=256G --cpus-per-task=32 --nodes=1 --ntasks-per-node=2\
         --job-name=bc5_${cond}_${depth}_s${seed} \
         --wrap="module load Miniforge3 CUDA/12.1.1; \
                 source $SHARED/envs/ai_pipeline_venv/bin/activate; \
                 python3 $CODE/probe_bc5cdr.py \
                   --backbone $bb --data $DATA \
                   --epochs 5 --lr 1e-3 --seed $seed \
                   --out $out"
}

# wait until < 1 of our jobs is running/pending (QOS-safe), then submit next
wait_slot () {
  while [[ $(squeue -u "$USER" -h | wc -l) -ge 1 ]]; do sleep 30; done
}

for depth in 10k 20k 40k; do
  for cond in "${CONDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      wait_slot
      submit_one "$depth" "$cond" "$seed"
      sleep 5
    done
  done
done
echo "[submitted all BC5CDR probes]"
