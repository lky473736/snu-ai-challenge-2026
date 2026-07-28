#!/bin/bash
#SBATCH -J aic_v6_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/train_v6_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/train_v6_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

# 512px+minibatch=8 = 여유 3.2GB로 빡빡함 -> 메모리 단편화로 인한 불필요한 OOM 방지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final

echo "===================================================="
echo "STAGE 1: Training (Qwen3-VL-8B, 512px, WEIGHT_TEMP=6)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v6 / best_v6_last, DDP 4-GPU)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
