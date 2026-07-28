"""
v6: Qwen3-VL-8B-Instruct + LoRA
zero-shot 32.83%(TPRU-7B 26.32%, 같은 태스크 형식 기준 검증됨)로 base 교체.
forward: 4장 이미지 + 텍스트 → Yes 토큰의 logit 반환
"""

import torch
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from config import (
    MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    VISION_LORA_ENABLE, VISION_LAST_N_BLOCKS, VISION_LORA_R, VISION_LORA_ALPHA,
    FREEZE_LLM_MERGE_FROM,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    _HAS_QWEN25 = True
except ImportError:
    _HAS_QWEN25 = False

try:
    from transformers import Qwen3VLForConditionalGeneration
    _HAS_QWEN3 = True
except ImportError:
    _HAS_QWEN3 = False

from transformers import Qwen2VLForConditionalGeneration


def _get_model_class(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path)
    model_type = getattr(cfg, "model_type", "")
    if _HAS_QWEN3 and model_type == "qwen3_vl":
        return Qwen3VLForConditionalGeneration
    if _HAS_QWEN25 and model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def load_model_and_processor(model_path: str = MODEL_PATH, lora_r: int = LORA_R, lora_alpha: int = LORA_ALPHA,
                              resume_from: str = None):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)

    ModelClass = _get_model_class(model_path)

    # FA2 → SDPA → eager 순서로 시도 (SDPA는 PyTorch 2.x 내장, FA2와 같은 효과)
    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            print(f"Loading model ({ModelClass.__name__}, bf16, attn={attn_impl})...")
            model = ModelClass.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            print(f"  -> {attn_impl} skipped: {e}")
    if model is None:
        raise RuntimeError("No attn_implementation worked")

    model.gradient_checkpointing_enable()

    if resume_from:
        # 저장된 LoRA 체크포인트에서 이어서 시작 (옵티마이저/스케줄러는 새로 초기화되는
        # "체크포인트 재출발" 방식 — job이 중간에 끊겼을 때를 대비한 것이지 true resume은 아님).
        # v11(freeze 전략)의 resume_from 체크포인트는 vision-only LoRA이므로, 반드시
        # v8 LLM LoRA를 먼저 병합한 모델 위에 얹어야 학습 시점과 동일한 조건이 됨.
        if FREEZE_LLM_MERGE_FROM:
            print(f"[freeze+resume] Merging v8 LoRA from {FREEZE_LLM_MERGE_FROM} into base weights")
            model = PeftModel.from_pretrained(model, FREEZE_LLM_MERGE_FROM, is_trainable=False)
            model = model.merge_and_unload()
        print(f"[resume] Loading LoRA weights from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        model.print_trainable_parameters()
        return model, processor

    if FREEZE_LLM_MERGE_FROM:
        # v11 freeze 전략: v8에서 이미 수렴된 LLM LoRA를 base 가중치에 영구 병합해서
        # LLM을 사실상 고정시키고, vision encoder에만 새 LoRA를 얹어 그것만 학습한다.
        print(f"[freeze] Merging v8 LoRA checkpoint from {FREEZE_LLM_MERGE_FROM} into base weights")
        model = PeftModel.from_pretrained(model, FREEZE_LLM_MERGE_FROM, is_trainable=False)
        model = model.merge_and_unload()

        if not VISION_LORA_ENABLE:
            raise ValueError("FREEZE_LLM_MERGE_FROM requires VISION_LORA_ENABLE=True (vision-only LoRA)")

        vis_block_ids = set()
        for n, _ in model.named_modules():
            if "visual.blocks." in n:
                vis_block_ids.add(int(n.split("visual.blocks.")[1].split(".")[0]))
        last_n = sorted(vis_block_ids)[-VISION_LAST_N_BLOCKS:]
        vision_targets = [
            n for n, _ in model.named_modules()
            if any(f"visual.blocks.{i}." in n for i in last_n)
            and n.split(".")[-1] in ("qkv", "proj", "linear_fc1", "linear_fc2")
        ]
        print(f"[vision LoRA] (LLM frozen/병합됨) 블록 {last_n} 대상, 타겟 모듈 {len(vision_targets)}개, "
              f"r={VISION_LORA_R} alpha={VISION_LORA_ALPHA}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=VISION_LORA_R,
            lora_alpha=VISION_LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=vision_targets,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, processor

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    rank_pattern, alpha_pattern = {}, {}

    if VISION_LORA_ENABLE:
        # vision encoder(27블록) 중 마지막 N블록의 attn.qkv/proj, mlp.linear_fc1/fc2만
        # 정확한 전체 경로로 target_modules에 추가. 나머지 블록은 계속 freeze.
        vis_block_ids = set()
        for n, _ in model.named_modules():
            if "visual.blocks." in n:
                vis_block_ids.add(int(n.split("visual.blocks.")[1].split(".")[0]))
        last_n = sorted(vis_block_ids)[-VISION_LAST_N_BLOCKS:]
        vision_targets = [
            n for n, _ in model.named_modules()
            if any(f"visual.blocks.{i}." in n for i in last_n)
            and n.split(".")[-1] in ("qkv", "proj", "linear_fc1", "linear_fc2")
        ]
        print(f"[vision LoRA] 블록 {last_n} 대상, 타겟 모듈 {len(vision_targets)}개, "
              f"r={VISION_LORA_R} alpha={VISION_LORA_ALPHA}")
        target_modules += vision_targets
        for t in vision_targets:
            rank_pattern[t] = VISION_LORA_R
            alpha_pattern[t] = VISION_LORA_ALPHA

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def get_yes_no_token_ids(processor):
    """Yes / No 토큰 ID 반환"""
    tok = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id  = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]
    return yes_id, no_id


def forward_logit(model, processor, inputs: dict, yes_id: int, no_id: int = None) -> torch.Tensor:
    """
    반환: (B,) — log P(Yes) - log P(No)  (log likelihood ratio)
    raw logit 대신 log odds ratio를 사용해 값 범위를 안정화 (-5 ~ +5)
    """
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()        # (B, vocab), fp32 for stability
    log_probs   = torch.log_softmax(last_logits, dim=-1)  # (B, vocab)
    score = log_probs[:, yes_id]
    if no_id is not None:
        score = score - log_probs[:, no_id]               # log P(Yes) - log P(No)
    # clamp: 모델이 확신할수록 score→±inf → ListNet log_softmax에서 inf-inf=NaN
    return score.clamp(-100.0, 100.0)
