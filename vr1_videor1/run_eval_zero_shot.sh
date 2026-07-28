#!/bin/bash
#SBATCH -J aic_vr1_zs
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/eval_zs_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/vr1_videor1/logs/eval_zs_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/vr1_videor1

# Video-R1-7B zero-shot: LoRA 없이 base 모델만으로 val 평가
# 비교: Qwen2-VL base 5.5% / TPRU zero-shot 26.3% / v1 fine-tuned 50.1%
python src/eval_zero_shot.py --n_samples 399

echo "Done: $(date)"
