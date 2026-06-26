#!/bin/bash --login
#SBATCH --job-name=ncbi_eval
#SBATCH --partition=gpu_h200
#SBATCH --account=SCWF00175
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=eval_%j.out

# ----- environment -----
module purge
module load Miniforge3
module load CUDA/12.1.1

source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code/

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0

# ----- sanity check -----
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# ----- execute evaluation -----
python3 train_eval_ncbi.py \
    --backbone /scratch/SCWF00175/shared/mlm_adapted/B_dedup_seed1 \
    --condition B_dedup \
    --rare-slice /scratch/SCWF00175/shared/datasets/rare_slice_p25/ncbi_test_tagged.jsonl \
    --mode linear_probe \
    --seeds 1 \
    --out /scratch/SCWF00175/shared/results/test_chain
