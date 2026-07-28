#!/bin/bash
#SBATCH -J check_gpu_util
#SBATCH -p cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=/data/gyuyeonlim/snu_ai_challenge/zeroshot_qwen_compare/logs/gpu_util_%j.out

ssh -o StrictHostKeyChecking=no node25 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv
