#!/bin/bash
#SBATCH -J aic_v4_base10
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/logs/test_baseline_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair/logs/test_baseline_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v4_tpru_adjpair
python test_baseline_same10.py --n_samples 10

echo "Done: $(date)"
