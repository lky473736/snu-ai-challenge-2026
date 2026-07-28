#!/bin/bash
#SBATCH -J aic_hntv
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/train_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/train_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv

accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/train.py

echo "Done: $(date)"
