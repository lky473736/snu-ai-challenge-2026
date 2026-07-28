#!/bin/bash
#SBATCH -J aic_v5_full
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/train_v5_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/train_v5_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise

echo "===================================================="
echo "STAGE 1: Training"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: Inference (best_v5 checkpoint, single GPU)"
echo "===================================================="
python inference.py --ckpt checkpoints/best_v5 --out submission_v5_best.csv

echo ""
echo "===================================================="
echo "STAGE 3: Inference (best_v5_last checkpoint, single GPU)"
echo "===================================================="
python inference.py --ckpt checkpoints/best_v5_last --out submission_v5_last.csv

echo "Done: $(date)"
