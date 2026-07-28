"""
v20: Qwen3-VL-32B-Instruct + QLoRA(4bit) — v14 레시피(LoRA r/alpha, hard negative, loss)를
100% 유지하고 베이스 모델만 8B->32B QLoRA로 교체한 단일변수 실험.

로컬에 있는 건 bf16 원본(사전양자화 아님) — from_pretrained 시점에 BitsAndBytesConfig로
on-the-fly 4bit 양자화한다. myunhh 팀은 unsloth 사전양자화 체크포인트를 썼다가 vision skip
이름 불일치로 vision까지 4bit화되는 버그를 겪었는데(patch_prequant_vision_skip로 수정),
우리는 직접 skip 목록(LLM_INT8_SKIP_MODULES=["visual"])을 지정하는 on-the-fly 양자화라
그 버그 자체가 발생할 여지가 없음.
"""

import torch
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import (
    MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT, LLM_INT8_SKIP_MODULES,
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        llm_int8_skip_modules=LLM_INT8_SKIP_MODULES,
    )

    model = None
    errs = []
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            print(f"Loading model ({ModelClass.__name__}, 4bit-NF4, attn={attn_impl})...")
            model = ModelClass.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            errs.append(f"{attn_impl}: {e}")
            print(f"  -> {attn_impl} skipped: {e}")
    if model is None:
        raise RuntimeError("모델 로드 3가지 attn_implementation 전부 실패:\n" + "\n".join(errs))

    # QLoRA 표준 절차: k-bit 학습 준비(LayerNorm fp32 캐스팅, 그래디언트 체크포인팅 호환 설정 등).
    # myunhh가 로컬 4090(24GB)에서 OOM 났던 지점이 바로 여기(vision fp32 업캐스트) — 우리는
    # H100 80GB라 여유 충분할 것으로 예상되나, 스모크에서 실측 확인 필요.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False})

    if resume_from:
        print(f"[resume] Loading LoRA weights from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def get_yes_no_token_ids(processor):
    tok = processor.tokenizer
    yes_id = tok.convert_tokens_to_ids(tok.tokenize("Yes"))[-1]
    no_id = tok.convert_tokens_to_ids(tok.tokenize("No"))[-1]
    return yes_id, no_id


def forward_logit(model, processor, inputs: dict, yes_id: int, no_id: int = None) -> torch.Tensor:
    """반환: (B,) — log P(Yes) - log P(No)."""
    outputs = model(**inputs, use_cache=False, logits_to_keep=1)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id]
    if no_id is not None:
        score = score - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)
