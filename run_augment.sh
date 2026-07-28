#!/bin/bash
#SBATCH -J aic_augment
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/logs/augment_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/logs/augment_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

mkdir -p /data/gyuyeonlim/snu_ai_challenge/logs

torchrun --nproc_per_node=4 \
    /data/gyuyeonlim/snu_ai_challenge/augment_sentences_local.py

echo "Done: $(date)"
echo "결과: /data/gyuyeonlim/snu_ai_challenge/augmented_sentences.csv"
