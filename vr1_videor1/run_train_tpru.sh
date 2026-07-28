#!/bin/bash
#SBATCH -J aic_hntv_tpru
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/train_tpru_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/train_tpru_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap

# TPRU-7B + LoRA
# - base: TPRU-7B (Qwen2.5-VL, RL-pretrained for temporal reordering)
# - lr=1e-5 (절반): TPRU의 temporal 지식 보존
# - margin=1.2, ranking_weight=1.5 (v3 설정)
# - lora_r=32 (conservative)
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py \
    --margin 1.2 \
    --ranking_weight 1.5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lr 1e-5 \
    --ckpt_name best_tpru

echo "Done: $(date)"
