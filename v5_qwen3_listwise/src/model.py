"""
v5 — Qwen3-VL-8B-Instruct + LoRA, listwise 순열을 "3,1,2,4" 텍스트로 직접 생성(SFT)
hard-negative/score-head 없이 표준 causal LM cross-entropy만 사용.
"""

import torch
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from config import MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT


def _get_model_class(model_path: str):
    cfg = AutoConfig.from_pretrained(model_path)
    mt = getattr(cfg, "model_type", "")
    if mt == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def load_model_and_processor(model_path: str = MODEL_PATH, lora_r: int = LORA_R, lora_alpha: int = LORA_ALPHA):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    ModelClass = _get_model_class(model_path)

    model = None
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            print(f"Loading model ({ModelClass.__name__}, bf16, attn={attn_impl})...")
            model = ModelClass.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            print(f"  -> {attn_impl} skipped: {e}")
    if model is None:
        raise RuntimeError("No attn_implementation worked")

    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def get_digit_comma_ids(processor):
    """제약 디코딩용: digit 1~4, comma 토큰 id (전부 단일 토큰으로 확인됨)"""
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in [1, 2, 3, 4]}
    comma_id = tok.encode(",", add_special_tokens=False)[0]
    return digit_ids, comma_id


from transformers import LogitsProcessor


class PermutationConstraint(LogitsProcessor):
    """
    "d,d,d,d" 형식 강제: 짝수 스텝(0,2,4,6)=digit(아직 안 쓴 것만), 홀수 스텝(1,3,5)=comma 고정.
    4개 digit이 전부 소진되면 마지막 digit은 자동으로 유일 후보가 되어 항상 유효한 순열만 생성됨.
    """
    def __init__(self, prompt_len: int, digit_ids: dict, comma_id: int):
        self.prompt_len = prompt_len
        self.digit_ids = digit_ids
        self.comma_id = comma_id

    def __call__(self, input_ids, scores):
        batch = input_ids.shape[0]
        new_scores = torch.full_like(scores, float("-inf"))
        for b in range(batch):
            gen = input_ids[b, self.prompt_len:].tolist()
            step = len(gen)
            if step % 2 == 1:
                new_scores[b, self.comma_id] = 0.0
            else:
                used = {d for d, tid in self.digit_ids.items() if tid in gen}
                for d in (1, 2, 3, 4):
                    if d not in used:
                        tid = self.digit_ids[d]
                        new_scores[b, tid] = scores[b, tid]
        return new_scores


def generate_permutation(model, processor, images, sentence, digit_ids, comma_id, device, build_messages_fn):
    """배치=1, 제약 디코딩으로 항상 유효한 [rank(Input_1..4)] 반환"""
    from transformers import LogitsProcessorList
    msgs = build_messages_fn(images, sentence)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[images], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    constraint = PermutationConstraint(prompt_len, digit_ids, comma_id)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            logits_processor=LogitsProcessorList([constraint]),
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    gen = out_ids[0, prompt_len:].tolist()
    id_to_digit = {tid: d for d, tid in digit_ids.items()}
    ranks = [id_to_digit[g] for g in gen if g in id_to_digit]
    return ranks  # [Input_1의 순위, Input_2의 순위, Input_3의 순위, Input_4의 순위]
