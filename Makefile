# set path so `llama-cli` etc.. be in path
# export PATH := $(PWD)/thirdparty/llama.cpp/bin:$(PATH)
export MACOSX_DEPLOYMENT_TARGET := 14.7

# models
MODEL := models/Llama-3.2-1B-Instruct-Q8_0.gguf
MODEL_RAG := models/all-MiniLM-L6-v2-Q5_K_S.gguf
MODEL_LLAVA := models/llava-llama-3-8b-v1_1-int4.gguf
MODEL_LLAVA_MMPROG := models/llava-llama-3-8b-v1_1-mmproj-f16.gguf
WITH_DYLIB = 0

THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp
MIN_OSX_VER := -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)

ifeq ($(WITH_DYLIB),1)
	LIBLAMMA := $(LLAMACPP)/lib/libllama.dylib
	LLAMACPP_LIBS := \
		$(LLAMACPP)/lib/libcommon.dylib \
		$(LLAMACPP)/lib/libllama.dylib \
		$(LLAMACPP)/lib/libggml-base.dylib \
		$(LLAMACPP)/lib/libggml.dylib \
		$(LLAMACPP)/lib/libggml-blas.dylib\
		$(LLAMACPP)/lib/libggml-cpu.dylib \
		$(LLAMACPP)/lib/libggml-metal.dylib \
		$(LLAMACPP)/lib/libmtmd.dylib
else
	LIBLAMMA := $(LLAMACPP)/lib/libllama.a
	LLAMACPP_LIBS := \
		$(LLAMACPP)/lib/libcommon.a \
		$(LLAMACPP)/lib/libllama.a \
		$(LLAMACPP)/lib/libggml-base.a \
		$(LLAMACPP)/lib/libggml.a \
		$(LLAMACPP)/lib/libggml-blas.a\
		$(LLAMACPP)/lib/libggml-cpu.a \
		$(LLAMACPP)/lib/libggml-metal.a \
		$(LLAMACPP)/lib/libmtmd.a
endif


.PHONY: all build cmake clean reset setup setup_inplace \
		wheel bind header diff sync

all: build

$(LIBLAMMA):
	@scripts/setup.sh

setup: reset
	@scripts/setup.sh

build: $(LIBLAMMA)
	@uv run python setup.py build_ext --inplace
# 	@uv run python setup.py build_ext

diff:
	@git diff thirdparty/llama.cpp/include > changes.diff
	
sync: $(LIBLAMMA)
	uv sync

inplace: $(LIBLAMMA)
	uv run build_ext --inplace

wheel: $(LIBLAMMA)
	@echo "WITH_DYLIB=$(WITH_DYLIB)"
	uv build
ifeq ($(WITH_DYLIB),1)
	uv run delocate-wheel -v dist/*.whl 
endif

build/include:
	@scripts/header_utils.py --force-overwrite --output_dir build/include include

bind: build/include
	@rm -rf build/bind
	@make -f scripts/bind/bind.mk bind


.PHONY: test simple test_simple test_main test_retrieve test_model test_llava test_lora \
		test_platform coverage memray download download_all bump clean reset remake cli \
		test-cli test-chat

test: build
	uv run pytest -s

simple:
	@g++ -std=c++14 -o build/simple \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		$(LLAMACPP_LIBS) \
		build/llama.cpp/examples/simple/simple.cpp
	@./build/simple -m $(MODEL) -n 32 -ngl 99 \
		-p "When did the French Revolution start?"


test_simple:
	@g++ -std=c++14 -o build/test_simple \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		$(LLAMACPP_LIBS) \
		tests/test_simple.cpp
	@./build/test_simple -m $(MODEL) -n 32 -ngl 99 \
		-p "When did the French Revolution start?"

cli:
	@./bin/llama-cli -n 32 -no-cnv -lv 0 \
		-m $(MODEL) \
		-p "When did the french revolution begin?" \
		--no-display-prompt 2> /dev/null

test-chat:
	@python3 -m src.cyllama.chat -m $(MODEL) -c 32 -ngl 99


test-cli:
	@python3 -m src.cyllama.cli -m $(MODEL) \
		--no-cnv -c 32 \
		-p "When did the French Revolution start?" 

test_main:
	@g++ -std=c++14 -o build/main \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit -lcurl \
		$(LLAMACPP_LIBS) \
		build/llama.cpp/tools/main/main.cpp
	@./build/main -m $(MODEL) \
		-p "When did the French Revolution start?" -no-cnv -c 2048 -n 512

$(MODEL_LLAVA):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-int4.gguf &&
		wget https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-mmproj-f16.gguf


$(MODEL_RAG):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q5_K_S.gguf

test_retrieve: $(MODEL_RAG)
	@$(LLAMACPP)/bin/llama-retrieval --model $(MODEL_RAG) \
		--top-k 3 --context-file README.md \
		--context-file LICENSE \
		--chunk-size 100 \
		--chunk-separator .

$(MODEL):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

download: $(MODEL)
	@echo "minimal model downloaded to models directory"

download_all: $(MODEL) $(MODEL_RAG) $(MODEL_LLAVA)
	@echo "all tests models downloaded to models directory"

test_model: $(MODEL)
	@$(LLAMACPP)/bin/llama-simple -m $(MODEL) -n 128 "Number of planets in our solar system"

test_llava: $(MODEL_LLAVA)
	@$(LLAMACPP)/bin/llama-llava-cli -m models/llava-llama-3-8b-v1_1-int4.gguf \
		--mmproj models/llava-llama-3-8b-v1_1-mmproj-f16.gguf \
		--image tests/media/dice.jpg -c 4096 -e \
		-p "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

test_lora:
	@$(LLAMACPP)/bin/llama-cli -c 2048 -n 64 \
	-p "What are your constraints?" \
	-m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
	--lora models/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf

test_platform:
	@g++ -std=c++14 -o build/test_platform \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		$(LLAMACPP_LIBS) \
		tests/test_platform.cpp
	@./build/test_platform

test_platform_linux:
	@g++ -static -std=c++14 -fopenmp -o build/test_platform \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		tests/test_platform.cpp \
		$(LLAMACPP_LIBS) \
	@./build/test_platform

coverage:
	uv run pytest --cov=cyllama --cov-report html

memray:
	uv run pytest --memray --native tests

bump:
	@scripts/bump.sh

clean:
	@rm -rf build dist src/*.egg-info .pytest_cache .coverage

reset: clean
	@rm -rf bin
	@rm -rf thirdparty/llama.cpp/bin thirdparty/llama.cpp/lib
	@rm -rf thirdparty/whisper.cpp/bin thirdparty/whisper.cpp/lib
	@rm -rf thirdparty/stable-diffusion.cpp/bin thirdparty/stable-diffusion.cpp/lib

remake: reset build diff test
