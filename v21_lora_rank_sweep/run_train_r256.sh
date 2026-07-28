#!/bin/bash
#SBATCH -J aic_v21_r256
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs/r256_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs/r256_%j.err

set -uo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs
cd /data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep

# 이미 완료된 스윕(job 236494)에서 r=256까지 안전함을 확인함 — 스윕 재실행 없이 바로 학습.
echo "########## STAGE 1: r=256 로 실제 학습(5epoch) ##########"
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py --lora_r 256
train_status=$?
if [ $train_status -ne 0 ]; then
  echo "학습이 비정상 종료(exit=$train_status)됨 — 추론 단계 건너뜀"
  echo "Done: $(date)"
  exit $train_status
fi

echo ""
echo "########## STAGE 2: 추론 — r=256 체크포인트로 24-permutation 전수조사 ##########"
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
