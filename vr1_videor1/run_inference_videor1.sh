#!/bin/bash
#SBATCH -J aic_vr1_inf
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/infer_vr1_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/infer_vr1_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/vr1_videor1

CKPT=${1:-best_videor1}
python src/inference.py --ckpt_name "$CKPT"

echo "Done: $(date)"
