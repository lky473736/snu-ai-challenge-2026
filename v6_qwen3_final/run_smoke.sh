#!/bin/bash
#SBATCH -J aic_v6_smoke
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=0:40:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/smoke_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/smoke_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final
python smoke_test_v6.py

echo "Done: $(date)"
