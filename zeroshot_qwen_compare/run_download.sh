#!/bin/bash
#SBATCH -J aic_qwen_dl
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/download_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/download_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare
python download_qwen25.py
python download_qwen3.py

echo "Done: $(date)"
