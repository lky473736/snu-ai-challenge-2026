#!/bin/bash
#SBATCH -J aic_hntv_tta
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/infer_tta_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/infer_tta_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv

# v1 체크포인트 + TTA(horizontal flip) + 개선된 프롬프트
# 24 perms × 2 (orig + flip) = 48 forwards/sample → 약 130분 예상
python src/inference_tta.py \
    --ckpt_name best_v1_score0.79057 \
    --out /data/gyuyeonlim/hntv/data/submission_tta.csv

echo "Done: $(date)"
