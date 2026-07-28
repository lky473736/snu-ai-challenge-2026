#!/bin/bash
#SBATCH -J aic_v21_ranksweep
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs/sweep_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs/sweep_%j.err

set -uo pipefail   # -e는 일부러 뺌 — OOM으로 인한 non-zero exit는 "정상적으로 예상된 결과"라 잡을 안 죽여야 함
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep/logs
cd /data/gyuyeonlim/snu_ai_challenge/v21_lora_rank_sweep

# 매 rank마다 완전히 새 프로세스(accelerate launch)로 검증 — idea.md §5-11 교훈
# (같은 프로세스/옵티마이저 재사용 시 상태 오염으로 오탐 위험) 그대로 적용.
run_probe() {
  local r=$1
  echo ""
  echo "=================================================="
  echo "  Testing r=$r  (started: $(date +%H:%M:%S))"
  echo "=================================================="
  accelerate launch --num_processes=4 --mixed_precision=bf16 smoke_test_rank.py --lora_r "$r"
  return $?
}

echo ""
echo "########## 1단계: 굵은 폭(32단위) 스윕으로 OOM 구간 찾기 ##########"
COARSE=(128 160 192 224 256 288 320)
last_ok=128
first_bad=""
for r in "${COARSE[@]}"; do
  if run_probe "$r"; then
    last_ok=$r
  else
    first_bad=$r
    break
  fi
done

final_ok=$last_ok

if [ -z "$first_bad" ]; then
  echo ""
  echo "=== 코스 스윕 전부 통과 — r=320까지 OK, 이걸로 바로 학습 진행 ==="
else
  echo ""
  echo "########## 2단계: 브래킷 [$last_ok(OK) ~ $first_bad(OOM)] 안에서 16단위 정밀 탐색(속도 우선) ##########"
  r=$((last_ok + 16))
  while [ "$r" -lt "$first_bad" ]; do
    if run_probe "$r"; then
      final_ok=$r
    else
      echo ""
      echo "=== 정밀 탐색 중 r=$r 에서 OOM (final_ok=$final_ok 유지) ==="
      break
    fi
    r=$((r + 16))
  done
fi

echo ""
echo "=================================================="
echo "최종 결론: r=$final_ok 까지는 안전(H100 4장, group_size=8, TRAIN_MINIBATCH=8 기준)"
echo "=================================================="

echo ""
echo "########## 3단계: r=$final_ok 로 실제 학습(3epoch) ##########"
accelerate launch --num_processes=4 --mixed_precision=bf16 src/train.py --lora_r "$final_ok"
train_status=$?
if [ $train_status -ne 0 ]; then
  echo "학습이 비정상 종료(exit=$train_status)됨 — 추론 단계 건너뜀"
  echo "Done: $(date)"
  exit $train_status
fi

echo ""
echo "########## 4단계: 추론 — r=$final_ok 체크포인트로 24-permutation 전수조사 ##########"
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
