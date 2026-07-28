#!/bin/bash
#SBATCH -J aic_v16_notebook
#SBATCH -A gpu
#SBATCH -p gpu-4farm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/v16_curriculum/logs/notebook_%j.out
#SBATCH --error=/data/gyuyeonlim/snu_ai_challenge/v16_curriculum/logs/notebook_%j.err

set -euo pipefail
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/gyuyeonlim/snu_ai_challenge/v16_curriculum

# --execute로 헤드리스 실행 + --inplace로 같은 파일에 출력/로그를 다시 저장
# (셀 실행 결과, print 로그, matplotlib 그림까지 전부 .ipynb 파일 안에 남음)
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=86400 \
    v16_curriculum.ipynb

echo "Done: $(date)"
