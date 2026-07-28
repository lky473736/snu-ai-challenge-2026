from transformers import AutoProcessor

MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tok = processor.tokenizer

for s in ["[3, 1, 2, 4]", "3,1,2,4", "1", "2", "3", "4", ",", " 1", " 2", " 3", " 4", "]"]:
    ids = tok.encode(s, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{s!r:20s} -> ids={ids}  pieces={pieces}")

print("\n--- chat template sanity check ---")
msgs = [{"role":"user","content":[{"type":"text","text":"hi"}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(text[-200:]))
