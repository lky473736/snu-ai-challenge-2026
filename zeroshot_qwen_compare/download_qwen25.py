from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", local_dir="/data/gyuyeonlim/models/Qwen2.5-VL-7B-Instruct")
print("DONE", p)
