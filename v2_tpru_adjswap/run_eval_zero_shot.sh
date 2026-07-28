#!/bin/bash
#SBATCH -J aic_tpru_zs
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/eval_zs_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap/logs/eval_zs_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v2_tpru_adjswap

# TPRU-7B zero-shot: LoRA 없이 base 모델만으로 val 평가
# v1 fine-tuned (0.5013) 대비 TPRU zero-shot이 얼마나 되는지 확인
python src/eval_zero_shot.py --n_samples 399

echo "Done: $(date)"
