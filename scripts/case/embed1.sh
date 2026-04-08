#!/usr/bin/env sh

uv run cyllama embed \
	-m models/bge-small-en-v1.5-q8_0.gguf \
	-f tests/media/corpus.txt \
	--similarity "death and dying" \
	--threshold 0.5
