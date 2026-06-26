#!/bin/bash --login
#SBATCH --job-name=mlm_array
#SBATCH --partition=gpu_h200
#SBATCH --account=SCWF00175
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=mlm_arr_%A_%a.out
#SBATCH --array=0-8%2          # 9 tasks, MAX 2 running at once (respects QOS)

# ----- environment -----
module purge
module load Miniforge3
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code/
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0

# ----- map array index -> (condition, seed) -----
CONDS=(A_raw A_raw A_raw B_dedup B_dedup B_dedup B1_ctxremoved B1_ctxremoved B1_ctxremoved)
SEEDS=(1 2 3 1 2 3 1 2 3)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
STEPS=${STEPS:-10000}          # default 10k (override: STEPS=15000 sbatch ...)

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== array task $SLURM_ARRAY_TASK_ID : condition=$COND seed=$SEED steps=$STEPS ==="
date
python3 mlm_adapt.py \
    --base thomas-sounack/BioClinical-ModernBERT-base \
    --corpus /scratch/SCWF00175/shared/datasets/pretrain_corpora/${COND}.budget.txt \
    --condition "$COND" --seed "$SEED" \
    --steps "$STEPS" --batch 32 --fp16 \
    --out /scratch/SCWF00175/shared/mlm_adapted
date
echo "=== task $SLURM_ARRAY_TASK_ID done ==="
