#!/bin/bash
#SBATCH -J aic_combined_check
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/combined_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/combined_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "===================================================="
echo "STEP 1: 32B / 30B-A3B zero-shot 전체 val(399개) 재검증"
echo "===================================================="
cd /data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare
CUDA_VISIBLE_DEVICES=0 python eval_large_models.py --n_samples 399

echo ""
echo "===================================================="
echo "STEP 2: LoRA r=128 스모크 테스트"
echo "===================================================="
cd /data/gyuyeonlim/snu_ai_challenge/v8_lora128
CUDA_VISIBLE_DEVICES=0 python smoke_test_v8.py

echo ""
echo "Done: $(date)"
