from huggingface_hub import snapshot_download

print("Qwen3-VL-30B-A3B-Instruct 다운로드 시작...")
snapshot_download('Qwen/Qwen3-VL-30B-A3B-Instruct', local_dir='/data/gyuyeonlim/models/Qwen3-VL-30B-A3B-Instruct')
print("Qwen3-VL-30B-A3B-Instruct 다운로드 완료")

print("Qwen3-VL-32B-Instruct 다운로드 시작...")
snapshot_download('Qwen/Qwen3-VL-32B-Instruct', local_dir='/data/gyuyeonlim/models/Qwen3-VL-32B-Instruct')
print("Qwen3-VL-32B-Instruct 다운로드 완료")
