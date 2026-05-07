# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)
export MACOSX_DEPLOYMENT_TARGET := 12

THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp

.PHONY: all build wheel clean test download

all: build

build:
	@bash scripts/setup.sh
	python setup.py build_ext --inplace

wheel:
	@python setup.py bdist_wheel

clean:
	@rm -rf build dist src/llama.cpp src/*.egg-inf thirdparty/llama.cpp/build o .pytest_cache .coverage

test: build
	@pytest

download:
	@python3 scripts/download_models.py
