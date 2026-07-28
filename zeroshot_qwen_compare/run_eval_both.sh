#!/bin/bash
#SBATCH -J aic_qwen_zs
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/eval_both_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/eval_both_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare

# Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct 순차 zero-shot 평가 (같은 job, 1개 GPU만 사용)
# v4의 val split(_val_raw.csv, SEED=42)에서 No_ordering 제외 399개로 TPRU-7B(26.3%) 기준과 직접 비교
python eval_both.py --n_samples 399

echo "Done: $(date)"
