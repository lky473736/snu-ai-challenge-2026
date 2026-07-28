#!/bin/bash
#SBATCH -J aic_v6_vramchk
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/vramchk_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final/logs/vramchk_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

cd /data/gyuyeonlim/snu_ai_challenge/v6_qwen3_final

declare -a CONFIGS=("448 8" "448 16" "512 8" "512 16" "560 8" "560 12")
for cfg in "${CONFIGS[@]}"; do
    read size mb <<< "$cfg"
    echo "### size=$size minibatch=$mb ###"
    python check_vram_v6.py --size $size --minibatch $mb || echo "FAILED (size=$size minibatch=$mb)"
    echo ""
done

echo "Done: $(date)"
