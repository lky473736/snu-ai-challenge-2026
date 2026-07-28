#!/bin/bash
# 베이스 모델(Qwen/Qwen3-VL-32B-Instruct, 공개 오픈소스, bf16 원본) 다운로드
# 대회 규정(3.1): 최종 추론은 인터넷이 차단된 로컬 환경에서 실행 가능해야 하므로,
# 이 스크립트는 인터넷이 되는 상태에서 미리 한 번 실행해 두고, 실제 추론(inference.py)은
# 이 로컬 캐시만 사용하도록 config.py의 MODEL_PATH가 이 스크립트의 저장 위치를 가리킵니다.
# 실행: bash download_base_model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/base_model/Qwen3-VL-32B-Instruct"
HF_REPO_ID="Qwen/Qwen3-VL-32B-Instruct"

pip install -q "huggingface_hub>=0.24"

mkdir -p "$MODEL_DIR"
echo "다운로드 중 (약 65GB, 시간이 걸립니다): $HF_REPO_ID -> $MODEL_DIR"

python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$HF_REPO_ID",
    repo_type="model",
    local_dir="$MODEL_DIR",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
EOF

echo "다운로드 완료: $MODEL_DIR"
ls -lh "$MODEL_DIR"
echo ""
echo "이제부터 inference.py 실행 시 인터넷 연결이 필요 없습니다."
echo "(선택) 네트워크 호출을 코드 레벨에서도 완전히 차단하려면 실행 전 아래를 설정하세요:"
echo "  export HF_HUB_OFFLINE=1"
