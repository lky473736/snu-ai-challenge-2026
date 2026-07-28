"""
v18: Qwen3-VL-8B-Instruct + LoRA — Pairwise Bradley-Terry Temporal Ranking.
쌍별 질문(6개/샘플) 전부가 같은 4장 이미지를 보므로, vision encoder는 1번만 태우고
LLM 레이어만 배치로 통과시킨다(idea.md에 기록된 5-7절의 KV캐시 재사용 불가 문제와는
다른 지점 — vision encoder 출력만 재사용하는 것이라 M-RoPE와 무관).
"""

import torch
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from config import MODEL_PATH, LORA_R, LORA_ALPHA, LORA_DROPOUT
from src.dataset import PAIRS, build_messages_pair

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

    model = None
    errs = []
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            print(f"Loading model ({ModelClass.__name__}, bf16, attn={attn_impl})...")
            model = ModelClass.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            )
            print(f"  -> {attn_impl} OK")
            break
        except Exception as e:
            errs.append(f"{attn_impl}: {e}")
            print(f"  -> {attn_impl} skipped: {e}")
    if model is None:
        raise RuntimeError("모델 로드 3가지 attn_implementation 전부 실패:\n" + "\n".join(errs))

    model.gradient_checkpointing_enable()

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


def forward_logit(model, inputs: dict, yes_id: int, no_id: int) -> torch.Tensor:
    """느린 참조 경로(검증용). 반환: (B,) log P(Yes) - log P(No)."""
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :].float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    score = log_probs[:, yes_id] - log_probs[:, no_id]
    return score.clamp(-100.0, 100.0)


def _unwrap_qwen_causal_lm(model):
    """PeftModel으로 감싸져 있으면 벗겨서 Qwen3VLForConditionalGeneration(lm_head 보유)을 반환."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def pairwise_logits_fast(model, processor, base_imgs, sentence, device):
    """4장 이미지를 vision encoder에 1번만 태우고, 6개 쌍별 질문은 LLM 레이어만 배치로
    통과시켜 (6,) 로그오즈 텐서를 반환(PAIRS 순서). 미분 가능 — 학습에 그대로 씀.

    ⚠️ apply_chat_template(tokenize=False)만으로는 이미지 placeholder가 확장 안 된 마커
    1개로 남아있고, 실제 확장(image_grid_thw만큼 반복)은 processor.__call__ 내부에서만
    일어남(processing_qwen3_vl.py 직접 확인함) — 그래서 반드시 processor(text=...,
    images=...)를 통째로 호출해서 input_ids를 만들어야 함."""
    causal_lm = _unwrap_qwen_causal_lm(model)
    qwen = causal_lm.model  # Qwen3VLModel (vision + language_model)
    n_rows = len(PAIRS)

    texts = [processor.apply_chat_template(build_messages_pair(base_imgs, sentence, i, j),
                                            tokenize=False, add_generation_prompt=True)
             for (i, j) in PAIRS]
    # placeholder 확장이 필요해서 processor()를 통째로 호출(텍스트/토큰 목적) — pixel_values는
    # 버리고, vision encoder 입력은 아래에서 4장만 따로 깨끗하게 다시 처리해서 슬라이싱 실수 방지.
    # return_mm_token_type_ids=True: transformers>=4.57부터 get_rope_index()가 image_grid_thw
    # 대신 이 텐서(토큰별 0=text/1=image)를 요구하도록 시그니처가 바뀜 — processor가 직접 만들어주는
    # 걸 그대로 받아쓴다(수동 계산 시 groupby 로직을 잘못 재현할 위험 있음).
    full_inp = processor(text=texts, images=[base_imgs] * n_rows, return_tensors="pt", padding=True,
                          return_mm_token_type_ids=True)
    input_ids = full_inp["input_ids"].to(device)
    attention_mask = full_inp["attention_mask"].to(device)
    image_grid_thw = full_inp["image_grid_thw"].to(device)  # 24행(4장 x 6번 반복)
    mm_token_type_ids = full_inp["mm_token_type_ids"].to(device)

    img_out = processor.image_processor(images=base_imgs, return_tensors="pt")
    pixel_values = img_out["pixel_values"].to(device)
    unique_grid_thw = img_out["image_grid_thw"].to(device)
    # transformers>=4.57(설치판 5.12.1)부터 get_image_features()가 2-tuple이 아니라
    # BaseModelOutputWithDeepstackFeatures(dataclass)를 반환함 — 튜플 언팩 대신 속성으로 꺼낸다.
    # pooler_output은 get_image_features 내부에서 이미 이미지별로 split된 tuple로 재할당돼 있음.
    vision_out = qwen.get_image_features(pixel_values, unique_grid_thw)
    image_embeds_tuple = vision_out.pooler_output
    deepstack_image_embeds = vision_out.deepstack_features
    image_embeds_once = torch.cat(image_embeds_tuple, dim=0)  # (4장의 총 이미지토큰수, hidden)

    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    image_embeds_rep = image_embeds_once.repeat(n_rows, 1).to(inputs_embeds.dtype)
    image_mask, _ = qwen.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                image_features=image_embeds_rep)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_rep)

    # 구버전 시그니처: get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask=...)
    # 신버전(설치판 5.12.1) 시그니처: get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=,
    # video_grid_thw=, attention_mask=) — 위치 인자로 image_grid_thw를 2번째에 넣으면 mm_token_type_ids
    # 자리에 잘못 들어가 그룹핑(itertools.groupby)이 완전히 깨짐. 전부 키워드 인자로 명시해서
    # 향후 인자 순서가 또 바뀌어도 조용히 깨지지 않도록 한다.
    position_ids, _ = qwen.get_rope_index(
        input_ids, mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw, attention_mask=attention_mask,
    )

    visual_pos_masks = image_mask[..., 0]
    deepstack_visual_embeds_rep = [d.repeat(n_rows, 1) for d in deepstack_image_embeds]

    outputs = qwen.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds_rep,
    )
    hidden = outputs.last_hidden_state  # (n_rows, seq, hidden)

    seq_positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(n_rows, -1)
    masked_positions = torch.where(attention_mask.bool(), seq_positions, torch.full_like(seq_positions, -1))
    last_idx = masked_positions.max(dim=1).values
    last_hidden = hidden[torch.arange(n_rows, device=device), last_idx]  # (n_rows, hidden)

    last_logits = causal_lm.lm_head(last_hidden).float()
    log_probs = torch.log_softmax(last_logits, dim=-1)
    yes_id, no_id = get_yes_no_token_ids(processor)
    z = (log_probs[:, yes_id] - log_probs[:, no_id]).clamp(-100.0, 100.0)
    return z  # (6,), PAIRS 순서
