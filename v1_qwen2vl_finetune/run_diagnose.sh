#!/bin/bash
#SBATCH -J aic_hntv_diag
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/data/gyuyeonlim/hntv/hntv/logs/diag_%j.out
#SBATCH --error=/data/gyuyeonlim/hntv/hntv/logs/diag_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/hntv/hntv
python src/diagnose_val.py

echo "Done: $(date)"
