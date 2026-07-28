from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir="/data/gyuyeonlim/models/Qwen3-VL-8B-Instruct")
print("DONE", p)
