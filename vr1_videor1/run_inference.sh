#!/bin/bash
#SBATCH -J aic_hntv_inf
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/infer_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/infer_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv
# 사용법: sbatch run_inference.sh [ckpt_name]
# 예: sbatch run_inference.sh best_v1_score0.79057
CKPT=${1:-best}
python src/inference.py --ckpt_name "$CKPT"

echo "Done: $(date)"
