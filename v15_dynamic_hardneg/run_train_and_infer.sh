#!/bin/bash
#SBATCH -J aic_v15_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/train_v15_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/train_v15_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg

echo "===================================================="
echo "STAGE 1: Training (v14 + K=7 유지 + 동적 하드 네거티브 뱅크, TRAIN_MINIBATCH=8 검증된 최대치)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v15 / best_v15_last, DDP 4-GPU)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
