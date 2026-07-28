#!/bin/bash
#SBATCH -J aic_hntv_v4
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/logs/train_v4_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/logs/train_v4_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair

# v4 redesign: all 23 perms (d=1..6 full coverage) + AdaptiveDistanceLoss + no_ordering x5 aug
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py \
    --lora_r 64 \
    --lora_alpha 128 \
    --lr 1e-5 \
    --ckpt_name best_v4

echo "Done: $(date)"
