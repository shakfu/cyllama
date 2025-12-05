# cyllama Makefile
VERSION := 0.1.18

export MACOSX_DEPLOYMENT_TARGET := 14.7

# Find system Python (python3 or python) - manage.py only uses stdlib
SYSTEM_PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

# Backend flags (can be overridden via environment variables)
# Default: Metal enabled on macOS only, all others disabled
ifeq ($(shell uname -s),Darwin)
    GGML_METAL ?= 1
    IS_MACOS := 1
else
    GGML_METAL ?= 0
    IS_MACOS := 0
endif
GGML_CUDA ?= 0
GGML_VULKAN ?= 0
GGML_SYCL ?= 0
GGML_HIP ?= 0
GGML_OPENCL ?= 0

# Export backend flags for manage.py and setup.py
export GGML_METAL GGML_CUDA GGML_VULKAN GGML_SYCL GGML_HIP GGML_OPENCL

# Paths
THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp
WHISPERCPP := $(THIRDPARTY)/whisper.cpp

# Models
MODEL := models/Llama-3.2-1B-Instruct-Q8_0.gguf
MODEL_RAG := models/all-MiniLM-L6-v2-Q5_K_S.gguf
MODEL_LLAVA := models/llava-llama-3-8b-v1_1-int4.gguf

# Library detection
WITH_DYLIB ?= 0
ifeq ($(WITH_DYLIB),1)
    LIBLAMMA := $(LLAMACPP)/lib/libllama.dylib
else
    LIBLAMMA := $(LLAMACPP)/lib/libllama.a
endif

# =============================================================================
# Primary targets
# =============================================================================
.PHONY: all build setup sync clean reset remake

all: build

$(LIBLAMMA):
	@$(SYSTEM_PYTHON) scripts/manage.py build --all --deps-only

setup: reset
	@$(SYSTEM_PYTHON) scripts/manage.py build --all --deps-only

sync: $(LIBLAMMA)
	@uv sync

build: sync
	@uv pip install -e .

remake: reset build test

# =============================================================================
# Wheel and distribution
# =============================================================================
.PHONY: wheel wheel-check dist build-wheel

wheel: $(LIBLAMMA)
	@uv build --wheel

dist: $(LIBLAMMA)
	@uv build

build-wheel: $(LIBLAMMA)
	@uv build --wheel
	@uv pip install dist/*.whl --force-reinstall

wheel-check:
	@uv run twine check dist/*.whl

# =============================================================================
# Testing
# =============================================================================
.PHONY: test coverage memray

test:
	@uv run pytest -s

coverage:
	@uv run pytest --cov=cyllama --cov-report html

memray:
	@uv run pytest --memray --native tests

# =============================================================================
# Development tools (via manage.py)
# =============================================================================
.PHONY: info bench profile bump

info:
	@$(SYSTEM_PYTHON) scripts/manage.py info

bench:
	@uv run python scripts/manage.py bench -m $(MODEL)

profile:
	@uv run python scripts/manage.py profile -m $(MODEL)

bump:
	@uv run python scripts/manage.py bump

# =============================================================================
# Cleaning
# =============================================================================
.PHONY: clean reset

clean:
	@$(SYSTEM_PYTHON) scripts/manage.py clean

reset:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset

# =============================================================================
# Model downloads
# =============================================================================
.PHONY: download download-all

$(MODEL):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

$(MODEL_RAG):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q5_K_S.gguf

$(MODEL_LLAVA):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-int4.gguf && \
		wget https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-mmproj-f16.gguf

download: $(MODEL)
	@echo "Model downloaded: $(MODEL)"

download-all: $(MODEL) $(MODEL_RAG) $(MODEL_LLAVA)
	@echo "All models downloaded"

# =============================================================================
# Backend-specific builds
# =============================================================================
.PHONY: show-backends build-cpu build-metal build-cuda build-vulkan build-sycl build-hip
.PHONY: wheel-cpu wheel-metal wheel-cuda wheel-vulkan wheel-hip wheel-sycl

show-backends:
	@echo "Current backend configuration:"
	@echo "  GGML_METAL:   $(GGML_METAL)"
	@echo "  GGML_CUDA:    $(GGML_CUDA)"
	@echo "  GGML_VULKAN:  $(GGML_VULKAN)"
	@echo "  GGML_SYCL:    $(GGML_SYCL)"
	@echo "  GGML_HIP:     $(GGML_HIP)"
	@echo "  GGML_OPENCL:  $(GGML_OPENCL)"

# Full backend builds: reset deps, rebuild with specific backend, install
build-cpu:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_METAL=0 GGML_CUDA=0 GGML_VULKAN=0 GGML_HIP=0 GGML_SYCL=0 \
		$(SYSTEM_PYTHON) scripts/manage.py build --all

build-metal:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_METAL=1 $(SYSTEM_PYTHON) scripts/manage.py build --all

build-cuda:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_CUDA=1 $(SYSTEM_PYTHON) scripts/manage.py build --all

build-vulkan:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_VULKAN=1 $(SYSTEM_PYTHON) scripts/manage.py build --all

build-sycl:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_SYCL=1 $(SYSTEM_PYTHON) scripts/manage.py build --all

build-hip:
	@$(SYSTEM_PYTHON) scripts/manage.py clean --reset
	@GGML_HIP=1 $(SYSTEM_PYTHON) scripts/manage.py build --all

# Wheel builds with specific backends (assumes clean state or matching deps)
wheel-cpu:
	@GGML_METAL=0 GGML_CUDA=0 GGML_VULKAN=0 GGML_HIP=0 GGML_SYCL=0 uv build --wheel

wheel-metal:
	@GGML_METAL=1 uv build --wheel

wheel-cuda:
	@GGML_CUDA=1 uv build --wheel

wheel-vulkan:
	@GGML_VULKAN=1 uv build --wheel

wheel-hip:
	@GGML_HIP=1 uv build --wheel

wheel-sycl:
	@GGML_SYCL=1 uv build --wheel

# =============================================================================
# CLI and server tests
# =============================================================================
.PHONY: cli test-cli test-chat test-server test-tts test-whisper

cli:
	@$(LLAMACPP)/bin/llama-cli -n 32 -no-cnv -lv 0 \
		-m $(MODEL) \
		-p "When did the french revolution begin?" \
		--no-display-prompt 2> /dev/null

test-cli:
	@uv run python -m cyllama.cli -m $(MODEL) \
		--no-cnv -c 32 \
		-p "When did the French Revolution start?"

test-chat:
	@uv run python -m cyllama.chat -m $(MODEL) -c 32 -ngl 99

test-server:
	@uv run python -m cyllama.llama.server \
		-m $(MODEL)

test-tts:
	@uv run python -m cyllama.tts \
		-m models/tts.gguf \
		-mv models/WavTokenizer-Large-75-F16.gguf \
		-p "Hello World"

test-whisper:
	@$(WHISPERCPP)/bin/whisper-cli -m models/ggml-base.en.bin -f tests/samples/jfk.wav

# =============================================================================
# macOS-only targets
# =============================================================================
ifeq ($(IS_MACOS),1)

LLAMACPP_LIBS := \
	$(LLAMACPP)/lib/libcommon.a \
	$(LLAMACPP)/lib/libllama.a \
	$(LLAMACPP)/lib/libggml-base.a \
	$(LLAMACPP)/lib/libggml.a \
	$(LLAMACPP)/lib/libggml-blas.a \
	$(LLAMACPP)/lib/libggml-cpu.a \
	$(LLAMACPP)/lib/libggml-metal.a \
	$(LLAMACPP)/lib/libmtmd.a

MACOS_FRAMEWORKS := -framework Foundation -framework Accelerate -framework Metal -framework MetalKit

.PHONY: test-llama-tts test-model test-llava test-lora

test-llama-tts:
	@$(LLAMACPP)/bin/llama-tts -m models/tts.gguf \
		-mv models/WavTokenizer-Large-75-F16.gguf \
		-p "Hello World"

test-model: $(MODEL)
	@$(LLAMACPP)/bin/llama-simple -m $(MODEL) -n 128 "Number of planets in our solar system"

test-llava: $(MODEL_LLAVA)
	@$(LLAMACPP)/bin/llama-llava-cli -m models/llava-llama-3-8b-v1_1-int4.gguf \
		--mmproj models/llava-llama-3-8b-v1_1-mmproj-f16.gguf \
		--image tests/media/dice.jpg -c 4096 -e \
		-p "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

test-lora:
	@$(LLAMACPP)/bin/llama-cli -c 2048 -n 64 \
		-p "What are your constraints?" \
		-m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
		--lora models/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf

endif

# =============================================================================
# Documentation
# =============================================================================
.PHONY: docs diff

docs:
	@make -C docs/book pdf VERSION=$(VERSION)

diff:
	@git diff thirdparty/llama.cpp/include > changes.diff
