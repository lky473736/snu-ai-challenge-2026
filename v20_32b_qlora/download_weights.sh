#!/bin/bash
# 최종 제출 모델(v20) LoRA 가중치 다운로드
# Hugging Face Hub: https://huggingface.co/lky473736/snuaichallenge-v20-qwen3vl32b-qlora
# 실행: bash download_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoints/best_v20"
HF_REPO_ID="lky473736/snuaichallenge-v20-qwen3vl32b-qlora"

pip install -q "huggingface_hub>=0.24"

mkdir -p "$CKPT_DIR"
echo "다운로드 중: $HF_REPO_ID -> $CKPT_DIR"

python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$HF_REPO_ID",
    repo_type="model",
    local_dir="$CKPT_DIR",
)
EOF

echo "다운로드 완료: $CKPT_DIR"
ls -lh "$CKPT_DIR"
