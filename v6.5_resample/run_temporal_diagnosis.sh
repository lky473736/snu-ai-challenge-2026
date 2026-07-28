#!/bin/bash
#SBATCH -J aic_temporal_diag
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v6.5_resample/logs/temporal_diag_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v6.5_resample/logs/temporal_diag_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /data/gyuyeonlim/snu_ai_challenge/v6.5_resample
python val_temporal_diagnosis.py

echo "Done: $(date)"
