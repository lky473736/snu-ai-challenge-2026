#!/bin/bash
#SBATCH -J aic_v11_infer
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v11_vision_lora/logs/infer_v11_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v11_vision_lora/logs/infer_v11_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v11_vision_lora

echo "===================================================="
echo "Inference only (best_v11=epoch3 val0.5924, best_v11_last=epoch4 val0.5735), DDP 4-GPU"
echo "===================================================="
torchrun --nproc_per_node=4 inference.py

echo "Done: $(date)"
