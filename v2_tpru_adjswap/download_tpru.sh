#!/bin/bash
# TPRU-7B 다운로드 (Stephengzk/TPRU-7B, ~14GB)
# HuggingFace Hub에서 직접 다운로드
# 실행: bash download_tpru.sh

set -euo pipefail

source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate aichallenge

MODEL_DIR="/data/gyuyeonlim/models/TPRU-7B"
HF_MODEL_ID="Stephengzk/TPRU-7B"

mkdir -p "$MODEL_DIR"
echo "Downloading $HF_MODEL_ID → $MODEL_DIR"
echo "Started: $(date)"

python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$HF_MODEL_ID",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
EOF

echo "Download complete: $(date)"
echo "Files:"
ls -lh "$MODEL_DIR"
