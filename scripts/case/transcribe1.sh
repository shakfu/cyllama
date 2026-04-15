#!/usr/bin/env sh

# Run this from the root of cyllama

uv run cyllama transcribe \
	-f tests/samples/jfk.wav \
	-m models/ggml-base.en.bin