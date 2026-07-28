"""
v6: Qwen3-VL-8B-Instruct + LoRA
zero-shot 32.83%(TPRU-7B 26.32%, 같은 태스크 형식 기준 검증됨)로 base 교체.
forward: 4장 이미지 + 텍스트 → Yes 토큰의 logit 반환
"""

import torch
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from config import MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT

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
        print(f"[resume] Loading LoRA weights from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
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
