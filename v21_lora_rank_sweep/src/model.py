"""
v21: Qwen3-VL-32B-Instruct + QLoRA(4bit), LoRA rank를 CLI로 받아 alpha를 항상 r*2.0(v14/v20
비율 유지)으로 자동 계산한다. 그 외(hard negative, loss, LR 등)는 v20과 100% 동일.
"""

import torch
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import (
    MODEL_PATH, LORA_ALPHA_RATIO, LORA_DROPOUT,
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


def load_model_and_processor(model_path: str = MODEL_PATH, lora_r: int = 128, lora_alpha: int = None,
                              resume_from: str = None):
    if lora_alpha is None:
        lora_alpha = int(round(lora_r * LORA_ALPHA_RATIO))

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
            print(f"Loading model ({ModelClass.__name__}, 4bit-NF4, r={lora_r}, alpha={lora_alpha}, attn={attn_impl})...")
            model = ModelClass.from_pretrained(
                model_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            errs.append(f"{attn_impl}: {e}")
            print(f"  -> {attn_impl} skipped: {e}")
    if model is None:
        raise RuntimeError("모델 로드 3가지 attn_implementation 전부 실패:\n" + "\n".join(errs))

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
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id]
    if no_id is not None:
        score = score - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)
