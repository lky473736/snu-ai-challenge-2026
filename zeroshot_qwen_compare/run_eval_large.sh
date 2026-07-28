#!/bin/bash
#SBATCH -J aic_eval_large
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --dependency=afterok:226535
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/eval_large_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/eval_large_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export CUDA_VISIBLE_DEVICES=0

cd /data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare
python eval_large_models.py --n_samples 100

echo "Done: $(date)"
