#!/bin/bash
#SBATCH -J aic_v20_semifinal
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/semifinal_%j.out
#SBATCH --error=logs/semifinal_%j.err

set -uo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

torchrun --nproc_per_node=4 inference_semifinal.py

echo "Done: $(date)"
