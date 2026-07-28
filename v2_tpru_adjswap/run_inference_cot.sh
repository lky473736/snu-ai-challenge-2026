#!/bin/bash
#SBATCH -J aic_hntv_cot
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/infer_cot_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/infer_cot_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv

# Reasoning Prefix Injection: v1 체크포인트 + CoT
python src/inference_cot.py \
    --ckpt_name best_v1_score0.79057 \
    --max_reason_tokens 80 \
    --out_name submission_cot.csv

echo "Done: $(date)"
