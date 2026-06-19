#!/bin/bash
# run_all.sh -- submit the 9 MLM adaptations + 9 NCBI probes ONE AT A TIME,
# respecting a MaxJobsPerUser=2 QOS. Run this from a LOGIN node (not sbatch);
# it loops, submitting the next job only when your queue has a free slot.
#
# Usage:  bash run_all.sh
#
# It tracks progress by checking for output backbone dirs / metrics files,
# so it is RESUMABLE -- if it dies, just run it again and it skips finished work.

USER=b.lhl23prg
MLM_OUT=/scratch/SCWF00175/shared/mlm_adapted_40k
EVAL_OUT=/scratch/SCWF00175/shared/results/full_40k
CORPORA=/scratch/SCWF00175/shared/datasets/pretrain_corpora
SLICE=/scratch/SCWF00175/shared/datasets/rare_slice_p25/ncbi_test_tagged.jsonl
STEPS=40000
MAXJOBS=2          # your QOS cap

CONDS=(A_raw B_dedup B1_ctxremoved)
SEEDS=(1 2 3)

# wait until you have < MAXJOBS jobs in the queue (leaves room to submit one)
wait_for_slot() {
    while true; do
        n=$(squeue -u $USER -h | wc -l)
        if [ "$n" -lt "$MAXJOBS" ]; then return 0; fi
        echo "  [$(date +%H:%M:%S)] queue full ($n/$MAXJOBS), waiting 60s..."
        sleep 60
    done
}

# ---------- PHASE 1: MLM adaptation (9 jobs) ----------
echo "=== PHASE 1: MLM adaptation ==="
for cond in "${CONDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [ -f "$MLM_OUT/${cond}_seed${seed}/config.json" ]; then
        echo "  SKIP $cond seed$seed (backbone exists)"; continue
    fi
    wait_for_slot
    echo "  submit MLM $cond seed$seed"
    sbatch --job-name=mlm_${cond}_s${seed} \
           --partition=gpu_h200 --account=SCWF00175_w_teahan_171 \
           --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=256G \
           --gres=gpu:1 --time=4:00:00 \
           --output=mlm_${cond}_s${seed}_%j.out \
           --wrap="module purge; module load Miniforge3; module load CUDA/12.1.1; \
                   source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate; \
                   cd /scratch/SCWF00175/shared/code/; \
                   export TOKENIZERS_PARALLELISM=false; export HF_HUB_OFFLINE=0; \
                   python3 mlm_adapt.py --base thomas-sounack/BioClinical-ModernBERT-base \
                     --corpus $CORPORA/${cond}.budget.txt --condition $cond --seed $seed \
                     --steps $STEPS --batch 32 --fp16 --out $MLM_OUT"
    sleep 3
  done
done

echo "  all MLM jobs submitted; waiting for the queue to fully drain..."
while [ "$(squeue -u $USER -h -n $(echo mlm_ | tr -d ' ') 2>/dev/null | wc -l)" -gt 0 ]; do
    # crude: wait until no mlm_ jobs remain
    remaining=$(squeue -u $USER -h | grep -c mlm_ || true)
    [ "$remaining" -eq 0 ] && break
    echo "  [$(date +%H:%M:%S)] $remaining MLM jobs still in queue..."
    sleep 120
done
echo "=== PHASE 1 complete: backbones in $MLM_OUT ==="
ls "$MLM_OUT"

# ---------- PHASE 2: NCBI probe (9 jobs) ----------
echo "=== PHASE 2: NCBI evaluation ==="
mkdir -p "$EVAL_OUT"
for cond in "${CONDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [ -f "$EVAL_OUT/metrics_${cond}_seed${seed}.json" ]; then
        echo "  SKIP eval $cond seed$seed (metrics exist)"; continue
    fi
    if [ ! -f "$MLM_OUT/${cond}_seed${seed}/config.json" ]; then
        echo "  WARN no backbone for $cond seed$seed; skipping"; continue
    fi
    wait_for_slot
    echo "  submit EVAL $cond seed$seed"
    sbatch --job-name=evl_${cond}_s${seed} \
           --partition=gpu_h200 --account=SCWF00175_w_teahan_171 \
           --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=256G \
           --gres=gpu:1 --time=4:00:00 \
           --output=evl_${cond}_s${seed}_%j.out \
           --wrap="module purge; module load Miniforge3; module load CUDA/12.1.1; \
                   source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate; \
                   cd /scratch/SCWF00175/shared/code/; \
                   export TOKENIZERS_PARALLELISM=false; export HF_HUB_OFFLINE=0; \
                   python3 train_eval_ncbi.py --backbone $MLM_OUT/${cond}_seed${seed} \
                     --condition $cond --rare-slice $SLICE \
                     --mode linear_probe --seeds $seed --out $EVAL_OUT"
    sleep 3
  done
done

echo "  all eval jobs submitted; waiting for queue to drain..."
while true; do
    remaining=$(squeue -u $USER -h | grep -c evl_ || true)
    [ "$remaining" -eq 0 ] && break
    echo "  [$(date +%H:%M:%S)] $remaining eval jobs still in queue..."
    sleep 60
done

echo "=== PHASE 2 complete: metrics in $EVAL_OUT ==="
ls "$EVAL_OUT"/metrics_*.json 2>/dev/null

# ---------- PHASE 3: aggregate ----------
echo "=== PHASE 3: aggregation ==="
python3 aggregate_results.py \
    --results-dir "$EVAL_OUT" \
    --conditions A_raw B_dedup B1_ctxremoved \
    --seeds 1 2 3 \
    --out /scratch/SCWF00175/shared/reports/downstream_results_40k.json

echo "=== ALL DONE ==="
