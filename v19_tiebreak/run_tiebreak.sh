#!/bin/bash
#SBATCH -J aic_v19_tiebreak
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v19_tiebreak/logs/tiebreak_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v19_tiebreak/logs/tiebreak_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p /data/gyuyeonlim/snu_ai_challenge/v19_tiebreak/logs
cd /data/gyuyeonlim/snu_ai_challenge/v19_tiebreak
python tiebreak_test.py

echo "Done: $(date)"
