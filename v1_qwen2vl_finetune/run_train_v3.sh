#!/bin/bash
#SBATCH -J aic_hntv_v3
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/train_v3_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/train_v3_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv

# v3: MARGIN=1.2 + RANKING_WEIGHT=1.5 + LoRA r=64/alpha=128
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/train.py \
    --margin 1.2 \
    --ranking_weight 1.5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --ckpt_name best_v3

echo "Done: $(date)"
