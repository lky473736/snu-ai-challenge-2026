#!/bin/bash
#SBATCH -J aic_v5_smoke
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/smoke_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise/logs/smoke_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v5_qwen3_listwise

declare -a CONFIGS=("448 4" "560 4" "672 2" "560 8")
for cfg in "${CONFIGS[@]}"; do
    read size batch <<< "$cfg"
    echo "### size=$size batch=$batch ###"
    sed -i "s/^MAX_IMAGE_SIZE = .*/MAX_IMAGE_SIZE = $size/" config.py
    python smoke_test.py --size $size --batch $batch || echo "FAILED (size=$size batch=$batch)"
    echo ""
done

# 원래 값(672)으로 복원
sed -i "s/^MAX_IMAGE_SIZE = .*/MAX_IMAGE_SIZE = 672/" config.py

echo "Done: $(date)"
