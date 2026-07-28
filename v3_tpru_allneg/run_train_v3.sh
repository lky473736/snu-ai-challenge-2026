#!/bin/bash
#SBATCH -J aic_hntv_v3
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v3_tpru_allneg/logs/train_v3_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v3_tpru_allneg/logs/train_v3_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v3_tpru_allneg

# v3: TPRU-7B + 23 negatives + ListNet + batch forward
# - base: TPRU-7B
# - all 23 wrong permutations as negatives (vs 3 adjacent swaps in v2)
# - ListNet loss (vs pairwise ranking in v2)
# - batched forward pass (6 samples at once)
# - num_workers=4 for faster data loading
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py \
    --margin 1.2 \
    --ranking_weight 1.5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lr 1e-5 \
    --ckpt_name best_v3

echo "Done: $(date)"
