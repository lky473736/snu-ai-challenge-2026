#!/bin/bash
#SBATCH -J aic_v2.5_infer
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v2.5_tpru_ttapw/logs/infer_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v2.5_tpru_ttapw/logs/infer_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v2.5_tpru_ttapw

torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
