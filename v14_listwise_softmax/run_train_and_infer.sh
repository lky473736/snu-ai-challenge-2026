#!/bin/bash
#SBATCH -J aic_v14_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax/logs/train_v14_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax/logs/train_v14_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v14_listwise_softmax

echo "===================================================="
echo "STAGE 1: Training (v8 레시피 100% 동일 + loss만 ListwiseSoftmaxLoss)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v14 / best_v14_last, DDP 4-GPU)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
