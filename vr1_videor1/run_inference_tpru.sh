#!/bin/bash
#SBATCH -J aic_hntv_tpru_inf
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/infer_tpru_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/infer_tpru_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap

CKPT=${1:-best_tpru}
python src/inference.py --ckpt_name "$CKPT"

echo "Done: $(date)"
