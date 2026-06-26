#!/bin/bash --login
#SBATCH --job-name=compress_redund
#SBATCH --output=compress_%j.out
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -A SCWF00175

module load Python/3.13.5-GCCcore-14.3.0
# make sure pyppmd is available to THIS python:
python3 -c "import pyppmd; print('pyppmd', pyppmd.__version__)" || pip install --user pyppmd

python3 compression_redundancy_v3_aligned_v2.py \
       --enriched /scratch/SCWF00175/shared/code/risk_v7/multitask_data_enriched.jsonl \
       --risk-dir /scratch/SCWF00175/shared/code/risk_v7 \
       --sample 1.0 --workers 4 --no-shuffle-control \
       --out /scratch/SCWF00175/shared/reports/compression_redundancy_v4.json
