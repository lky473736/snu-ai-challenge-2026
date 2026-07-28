"""
Qwen2-VL-7B + LoRA (bfloat16, 102GB VRAM → QLoRA 불필요)
forward: 4장 이미지 + 텍스트 → Yes 토큰의 logit 반환
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from config import MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT


def load_model_and_processor(lora_r: int = LORA_R, lora_alpha: int = LORA_ALPHA):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model (bfloat16)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

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
    return score
