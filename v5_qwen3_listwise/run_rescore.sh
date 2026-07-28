#!/bin/bash
#SBATCH -J aic_v5_rescore
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/rescore_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/rescore_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise

echo "### 작은 표본(20개)으로 정합성 먼저 확인 ###"
python rescore_val.py --n_samples 20

echo ""
echo "### 문제 없으면 전체 476개 ###"
python rescore_val.py

echo "Done: $(date)"
