#!/bin/bash
#SBATCH -J aic_v15_infer_best
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/inferbest_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/inferbest_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg
torchrun --nproc_per_node=4 infer_best_only.py

echo "Done: $(date)"
