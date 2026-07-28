#!/bin/bash
#SBATCH -J aic_dl_large
#SBATCH -p cpu-long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/download_large_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/download_large_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare
python download_large_models.py

echo "Done: $(date)"
