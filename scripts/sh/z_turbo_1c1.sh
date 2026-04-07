#!/usr/bin/sh

# Run this from the root of cyllama
# ./scripts/sd/z_turbo_1c.sh

uv run python -m cyllama.sd txt2img \
	--diffusion-model models/z_image_turbo-Q6_K.gguf \
	--vae models/ae.safetensors \
	--llm models/Qwen3-4B-Q8_0.gguf \
	--offload-to-cpu \
	--vae-on-cpu \
	-H 1024 -W 512 \
	-p "a lovely cat"
