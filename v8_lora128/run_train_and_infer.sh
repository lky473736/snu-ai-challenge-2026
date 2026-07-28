#!/bin/bash
#SBATCH -J aic_v8_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --dependency=afterany:226453
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v8_lora128/logs/train_v8_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v8_lora128/logs/train_v8_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

# 448px+minibatch=8 (v6와 동일 VRAM 프로파일) -> 메모리 단편화로 인한 불필요한 OOM 방지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v8_lora128

echo "===================================================="
echo "STAGE 1: Training (Qwen3-VL-8B, 448px, WEIGHT_TEMP=6, LIVE hard-negative resampling)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v8 / best_v8_last, DDP 4-GPU)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
