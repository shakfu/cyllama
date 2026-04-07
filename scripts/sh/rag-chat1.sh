
uv run cyllama rag \
	-m models/Qwen3-4B-Q8_0.gguf \
	-e models/bge-small-en-v1.5-q8_0.gguf \
	-f tests/media/corpus.txt
