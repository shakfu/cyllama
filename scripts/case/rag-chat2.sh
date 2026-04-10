#!/usr/bin/env sh

# Run this from the root of cyllama

uv run cyllama rag \
	-m models/Gemma-4-E4B-it-Q5_K_M.gguf \
	-e models/bge-small-en-v1.5-q8_0.gguf \
	-f tests/media/corpus.txt
