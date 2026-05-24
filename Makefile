# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)

THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp

.PHONY: all build wheel clean test download

all: build

build:
	python setup.py build_ext --inplace

wheel:
	@python setup.py bdist_wheel

clean:
	@rm -rf build dist src/llama.cpp src/*.egg-inf thirdparty/llama.cpp/build o .pytest_cache .coverage

test: build
	@pytest

download:
	@python3 scripts/download_models.py
