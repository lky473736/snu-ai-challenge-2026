#!/bin/bash
#SBATCH -J aic_v6_infer
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/infer_only_v6_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/infer_only_v6_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final

echo "===================================================="
echo "Inference ONLY (train은 이미 완료, 체크포인트 재사용, 448px로 train과 일치)"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
