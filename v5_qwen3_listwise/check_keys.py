from transformers import AutoProcessor

MODEL_PATH = "/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("padding_side:", processor.tokenizer.padding_side)
print("pad_token:", processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)

from PIL import Image
img = Image.new("RGB", (300, 200), color=(120, 50, 200))
msgs = [{"role":"user","content":[{"type":"text","text":"Frame 1:"},{"type":"image","image":img},{"type":"text","text":"desc"}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = processor(text=[text], images=[[img]], return_tensors="pt")
print("keys:", list(inp.keys()))
for k, v in inp.items():
    print(k, getattr(v, "shape", v))
