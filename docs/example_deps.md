# example dependencies


## batched

The example demonstrates batched generation from a given prompt

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

# convert-llama2c-to-ggml

This example reads weights from project [llama2.c](https://github.com/karpathy/llama2.c) and saves them in ggml compatible format. The vocab that is available in `models/ggml-vocab.bin` is used by default.

```c++
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
#include "log.h"
```


# embedding

This example demonstrates generate high-dimensional embedding vector of a given text with llama.cpp.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```


# eval-callback

A simple example which demonstrates how to use callback during the inference.
It simply prints to the console all operations and tensor data.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
```

# gen-docs

This example reads weights from project [llama2.c](https://github.com/karpathy/llama2.c) and saves them in ggml compatible format. The vocab that is available in `models/ggml-vocab.bin` is used by default.

```c++
#include "arg.h"
#include "common.h"
```

# gguf

This example reads weights from project [llama2.c](https://github.com/karpathy/llama2.c) and saves them in ggml compatible format. The vocab that is available in `models/ggml-vocab.bin` is used by default.

```c++
#include "ggml.h"
#include "gguf.h"
```

# gguf-hash

CLI to hash GGUF files to detect difference on a per model and per tensor level.

```c++
#include "ggml.h"
#include "gguf.h"
```

# gritlm

Generative Representational Instruction Tuning (GRIT): a model which can generate embeddings as well as "normal" text generation depending on the instructions in the prompt.

```c++
#include "arg.h"
#include "common.h"
#include "llama.h"
```

# lookahead

Demonstration of lookahead decoding technique.

```c++
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
```

# lookup

Demonstration of Prompt Lookup Decoding.

```c++
#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-cache.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
```

# parallel

Simplified simulation of serving incoming requests in parallel.

```c++
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
```

# passkey

A passkey retrieval task is an evaluation method used to measure a language
models ability to recall information from long contexts.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

# retrieval

Demonstration of simple retrieval technique based on cosine similarity.

```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

# save-load-state

```c++
#include "arg.h"
#include "common.h"
#include "llama.h"
```

# simple

The purpose of this example is to demonstrate a minimal usage of llama.cpp for generating text with a given prompt.

```c++
#include "llama.h"
```

# simple-chat

The purpose of this example is to demonstrate a minimal usage of llama.cpp to create a simple chat program using the chat template from the GGUF file.

```c++
#include "llama.h"
```

# speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques.

```c++
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
```

# speculative-simple

Demonstration of basic greedy speculative decoding.

```c++
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
```

# training

This directory contains examples related to language model training using llama.cpp/GGML.

So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP.

Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory.

**For CPU training, compile llama.cpp without any additional backends such as CUDA.**

**For CUDA training, use the maximum number of GPU layers.**


```c++
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
```

