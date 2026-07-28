#!/bin/bash
#SBATCH -J aic_v20_train
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v20_32b_qlora/logs/train_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v20_32b_qlora/logs/train_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /data/gyuyeonlim/snu_ai_challenge/v20_32b_qlora/logs
cd /data/gyuyeonlim/snu_ai_challenge/v20_32b_qlora

echo "===================================================="
echo "STAGE 1: v20 학습 — Qwen3-VL-32B QLoRA(4bit), v14 레시피, EPOCHS=3 (탐색)"
echo "  스모크 없이 바로 학습 — OOM 나면 train.py가 자동으로 TRAIN_MINIBATCH를 줄이고"
echo "  그 값을 계속 사용(§5-11 교훈: 실제 학습 루프 안에서 검증하는 게 더 정확함)"
echo "===================================================="
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py

echo ""
echo "===================================================="
echo "STAGE 2: v20 추론 — best_v20 체크포인트, 24-permutation 전수조사"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
