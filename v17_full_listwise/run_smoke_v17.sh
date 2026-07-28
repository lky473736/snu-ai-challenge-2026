#!/bin/bash
#SBATCH -J aic_v17_smoke
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=00:40:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v17_full_listwise/logs/smoke_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v17_full_listwise/logs/smoke_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /data/gyuyeonlim/snu_ai_challenge/v17_full_listwise/logs
cd /data/gyuyeonlim/snu_ai_challenge/v17_full_listwise

accelerate launch --num_processes=4 --mixed_precision=bf16 smoke_test_v17.py

echo "Done: $(date)"
