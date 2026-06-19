#!/bin/bash --login
#SBATCH --job-name=encoder_mlm
#SBATCH --partition=gpu_h200
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=mlm_%j.out

# ----- environment -----
module purge
module load Miniforge3   # FIXED
module load CUDA/12.1.1

source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code/

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0

# ----- sanity check -----
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# ----- args -----
CORPUS=${1:-"/path/to/B_dedup.budget.txt"}
CONDITION=${2:-"B_dedup"}
SEED=${3:-1}
STEPS=${4:-5000}

echo "=== MLM adapt: condition=$CONDITION seed=$SEED steps=$STEPS ==="
date

python3 mlm_adapt.py \
    --base thomas-sounack/BioClinical-ModernBERT-base \
    --corpus "$CORPUS" \
    --condition "$CONDITION" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --batch 32 \
    --fp16 \
    --out /scratch/SCWF00175/shared/mlm_adapted

date
echo "=== done ==="
