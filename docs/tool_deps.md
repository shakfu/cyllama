# tool dependencies

## llama-cli deps


- [ ] arg.h
- [x] common.h
- [ ] console.h
- [ ] log.h
- [x] sampling.h
- [x] llama.h
- [ ] chat.h



## batched-bench

Benchmark the batched decoding performance of `llama.cpp`

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

## cvector-generator

This example demonstrates how to generate a control vector using gguf models.

```c++
#include "ggml.h"
#include "gguf.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "pca.hpp"
#include "mean.hpp"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
```

## export-lora

Apply LORA adapters to base model and export the resulting model.

```c++
#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include "arg.h"
#include "common.h"
```

## gguf-split

CLI to split / merge GGUF files.

```c++
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
```

## imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

## llama-bench

Performance testing tool for llama.cpp.

```c++
#include "common.h"
#include "ggml.h"
#include "llama.h"
```

## main or llama-cli

This example program allows you to use various LLaMA language models easily and efficiently. It is specifically designed to work with the [llama.cpp](https://github.com/ggml-org/llama.cpp) project, which provides a plain C/C++ implementation with optional 4-bit quantization support for faster, lower memory inference, and is optimized for desktop CPUs. This program can be used to perform various inference tasks with LLaMA models, including generating text based on user-provided prompts and chat-like interactions with reverse prompts.

```c++
#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"
```

## mtmd

This directory provides multimodal capabilities for `llama.cpp`. Initially intended as a showcase for running LLaVA models, its scope has expanded significantly over time to include various other vision-capable models. As a result, LLaVA is no longer the only multimodal architecture supported.

```c++
#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
```

## perplexity

The `perplexity` example can be used to calculate the so-called perplexity value of a language model over a given text corpus.

Perplexity measures how well the model can predict the next token with lower values being better.

Note that perplexity is **not** directly comparable between models, especially if they use different tokenizers.

Also note that finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

## quantize

You can also use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to build your own quants without any setup.

```c++
#include "common.h"
#include "llama.h"
```

## run

The purpose of this example is to demonstrate a minimal usage of llama.cpp for running models.

```c++
#include "chat.h"
#include "common.h"
#include "json.hpp"
#include "llama-cpp.h"
#include "log.h"
```

## server

Fast, lightweight, pure C/C++ HTTP server based on [httplib](https://github.com/yhirose/cpp-httplib), [nlohmann::json](https://github.com/nlohmann/json) and **llama.cpp**.

```c++
#include "utils.hpp"
#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
```

## server

Fast, lightweight, pure C/C++ HTTP server based on [httplib](https://github.com/yhirose/cpp-httplib), [nlohmann::json](https://github.com/nlohmann/json) and **llama.cpp**.

```c++
#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
```

## tokenize

```c++
#include "common.h"
#include "llama.h"
```

## tts

This example demonstrates the Text To Speech feature.

```c++
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "json.hpp"
```

