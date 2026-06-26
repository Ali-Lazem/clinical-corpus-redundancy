#!/bin/bash --login
#SBATCH --job-name=eval_array
#SBATCH --partition=gpu_h200
#SBATCH --account=SCWF00175
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=eval_arr_%A_%a.out
#SBATCH --array=0-2%2           # 3 conditions, 2 at a time; each does 3 seeds

module purge
module load Miniforge3
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code/
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0

CONDS=(A_raw B_dedup B1_ctxremoved)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# probe all 3 seeds of this condition. Each seed has its own adapted backbone,
# so we run the eval once per seed pointing at that seed's backbone, writing
# metrics_<cond>_seed<k>.json that Step 5 aggregates.
for SEED in 1 2 3; do
  echo "=== eval condition=$COND seed=$SEED ==="
  python3 train_eval_ncbi.py \
      --backbone /scratch/SCWF00175/shared/mlm_adapted/${COND}_seed${SEED} \
      --condition "$COND" \
      --rare-slice /scratch/SCWF00175/shared/datasets/rare_slice_p25/ncbi_test_tagged.jsonl \
      --mode linear_probe --seeds $SEED \
      --out /scratch/SCWF00175/shared/results/full
done
echo "=== eval $COND done ==="
