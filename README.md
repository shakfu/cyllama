# cyllama - a cython wrapper of llama.cpp

This project provides a cython wrapper for @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) which is likely the most active open-source compiled LLM inference engine. It was spun-off from my earlier, now frozen, llama.cpp wrapper project, [llamalib](https://github.com/shakfu/llamalib)  which provided early stage, but functional, wrappers using [cython](https://github.com/cython/cython), [pybind11](https://github.com/pybind/pybind11), and [nanobind](https://github.com/wjakob/nanobind). Further development of `cyllama`, the cython wrapper from `llamalib`, will continue in this project.

Development goals are to:

- Stay up-to-date with bleeding-edge `llama.cpp` (last tag of stable build with llama.cpp `gguf-v0.17.1`)

- Produce a minimal, performant, compiled, thin python wrapper around the core `llama-cli` feature-set of `llama.cpp`.

- Maximize maintainability of code base by refactoring it into namespaced modules.

- Integrate and wrap features from related projects such as [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

- Learn about the internals of this popular C++/C LLM inference engine along the way. For me at least, this is definitely the most efficient way to learn about the underlying technologies.

Given that there is a fairly mature, well-maintained and performant ctypes-based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project and that LLM inference is gpu-driven rather than cpu-driven, this all may see quite redundant. Nonetheless, we anticipate some benefits to using a compiled cython-based wrapper instead of ctypes:

- Cython functions and extension classes can enforce strong type checking.

- Packaging benefits with respect to self-contained statically compiled extension modules, which include simpler compilation and reduced package size.

- There may be some performance improvements in the use of compiled wrappers over the use of ctypes.

- It may be possible to incorporate external optimizations more readily into compiled wrappers.

- It may be useful in case one wants to de-couple the python frontend and wrapper backends to existing frameworks: for example, to just replace the ctypes wrapper part in `llama-cpp-python` with compiled cython wrappers and contribute it back as a PR.

## Status

Development is done only on macOS to keep things simple, with intermittent testing to ensure it works on Linux. 

The following table provide an overview of the current wrapping/dev status:

| status                       | cyllama       |
| :--------------------------- | :-----------: |
| wrapper-type                 | cython        |
| wrap llama.h + other headers | yes           |
| wrap high-level simple-cli   | yes           |
| wrap low-level simple-cli    | yes           |
| wrap low-level llama-cli     | WIP           |
  
The initial milestone entailed creating a high-level wrapper of the `simple.cpp` llama.cpp example, followed by a low-level one. The next objective is to fully wrap the functionality of `llama-cli` which is ongoing (see: `cyllama.__init__.py`).

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!

## Wrapping Guidelines

As the intent is to provide a very thin wrapping layer and play to the strengths of the original c++ library as well as python, the approach to wrapping intentionally adopts the following guidelines:

- In general, key structs are implemented as cython extension classses with related functions implemented as methods of said classes.

- Be as consistent as possible with llama.cpp's naming of its api elements, except when it makes sense to shorten functions names which are used as methods.

- Minimize non-wrapper python code.

## Setup

To build `cyllama`:

1. A recent version of `python3` (testing on python 3.12)

2. Git clone the latest version of `cyllama`:

 ```sh
 git clone https://github.com/shakfu/cyllama.git
 cd cyllama
 ```

3. Install dependencies of `cython`, `setuptools`, and `pytest` for testing:

 ```sh
 pip install -r requirements.txt
 ```

4. Type `make` in the terminal.

This will:

1. Download and build `llama.cpp`
2. Install it into `bin`, `include`, and `lib` in the cloned `cyllama` folder
3. Build `cyllama`

## Testing

The `tests` directory in this repo provides extensive examples of using cyllama.

However, as a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). A good model to start and which is assumed by tests is [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf). `cyllama` expects models to be stored in a `models` folder in the cloned `cyllama` directory. So to create the `models` directory if doesn't exist and download this model, you can just type:

```sh
make download
```

This basically just does:

```sh
cd cyllama
mkdir models && cd models
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf 
```

Now you can test it using `llama-cli` or `llama-simple`:

```sh
bin/llama-cli -c 512 -n 32 -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
 -p "Is mathematics discovered or invented?"
```

You can also run the test suite with `pytest` by typing `pytest` or:

```sh
make test
```

If all tests pass, you can type `python3 -i scripts/start.py` or `ipython -i scripts/start.py` and explore the `cyllama` library with a pre-configured repl:

```python
>>> from cyllama import Llama
>>> llm = Llama('models/Llama-3.2-1B-Instruct-Q8_0.gguf')
>>> llm.ask("what is the age of the universe?")
'The estimated age of the universe is around 13.8 billion years'
```


## TODO

- [x] wrap llama-simple

- [ ] wrap llama-cli (WIP: see: `cyllama.__init__`)

- [ ] wrap [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

- [ ] wrap [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
