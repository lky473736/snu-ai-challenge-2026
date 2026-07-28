#!/bin/bash
#SBATCH -J aic_hntv_vr1
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/train_vr1_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/train_vr1_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/vr1_videor1

# Video-R1-7B + LoRA
# - base: Video-R1-7B (Qwen2.5-VL, T-GRPO temporal contrastive pre-training)
# - lr=1e-5: temporal RL 지식 보존
# - margin=1.2, ranking_weight=1.5 (v3/TPRU 설정)
# - lora_r=64, lora_alpha=128
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py \
    --margin 1.2 \
    --ranking_weight 1.5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lr 1e-5 \
    --ckpt_name best_videor1

echo "Done: $(date)"
