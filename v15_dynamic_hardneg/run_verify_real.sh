#!/bin/bash
#SBATCH -J aic_v15_verify_real
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/verifyreal_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg/logs/verifyreal_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Started: $(date)  N_EXTRA=${N_EXTRA_ARG}"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v15_dynamic_hardneg
accelerate launch --num_processes=1 --mixed_precision=bf16 verify_real_accelerate.py --n_extra "${N_EXTRA_ARG}" --n_cycles 3

echo "Done: $(date)"
