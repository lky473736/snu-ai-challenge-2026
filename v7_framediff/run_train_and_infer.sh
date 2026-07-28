#!/bin/bash
#SBATCH -J aic_v7_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v7_framediff/logs/train_v7_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v7_framediff/logs/train_v7_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

# 후보당 이미지 4장->7장(diff 3장 추가)으로 늘어 VRAM 여유가 v6.5보다 줄어들 수 있음 -> 안전장치 유지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v7_framediff

echo "===================================================="
echo "STAGE 1: Training (Qwen3-VL-8B, 448px + frame-diff 이미지, WEIGHT_TEMP=6, live resampling, 10epoch)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v7 / best_v7_last, DDP 4-GPU)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
