#!/bin/bash
#SBATCH -J aic_v13_noun_cot
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v13_noun_grounding_cot/logs/prompt_ab_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v13_noun_grounding_cot/logs/prompt_ab_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /data/gyuyeonlim/snu_ai_challenge/v13_noun_grounding_cot
python prompt_ab_test.py

echo "Done: $(date)"
