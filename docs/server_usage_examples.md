# Cyllama Server Usage Examples

Cyllama provides two server implementations, both offering the same OpenAI-compatible API:

## Server Types

### 1. EmbeddedServer (default)

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT), compiled as part of `make build`.
- Thread pool: requests beyond `--n-parallel` wait for a free slot.
- Streams chat completions as server-sent events with `"stream": true`.

### 2. PythonServer

- stdlib `http.server`; handles one request at a time.
- Streams with `"stream": true`; the connection closes at the end of the stream.

## Basic Usage

### Start Default Embedded Server

```bash
python -m cyllama.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

### Start the Python Server

```bash
python -m cyllama.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf --server-type python
```

### Streaming

Both servers send OpenAI `chat.completion.chunk` events, then `data: [DONE]`. Closing the connection stops generation and frees the slot.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

## Advanced Configuration

### Custom Host and Port

Both servers bind `127.0.0.1` by default. Binding any other address exposes the API to the network; set an API key first (see [Authentication and Limits](#authentication-and-limits)).

```bash
CYLLAMA_API_KEY=... python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080
```

### Authentication and Limits

With an API key set, every endpoint except `/health` requires `Authorization: Bearer <key>` and returns 401 otherwise. The server logs a warning when it binds a non-loopback address without a key.

- `CYLLAMA_API_KEY` environment variable, or `--api-key-file PATH`, which takes precedence. There is no `--api-key` flag: command-line arguments are visible to other local users.
- In Python: `ServerConfig(api_key=...)`.

Request bodies over `ServerConfig.max_body_bytes` (default 2 MiB) get 413. `EmbeddedServer` checks the key and `Content-Length` before it reads the body.

```bash
curl http://host:8080/v1/models -H "Authorization: Bearer $CYLLAMA_API_KEY"
```

There is no TLS. Across untrusted networks, put the server behind a TLS-terminating reverse proxy.

### Multiple Parallel Processing Slots

```bash
python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --server-type embedded \
    --n-parallel 4 \
    --ctx-size 2048
```

### GPU Configuration

```bash
python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --server-type embedded \
    --gpu-layers 32 \
    --ctx-size 4096
```

## API Endpoints

Both server implementations provide the same OpenAI-compatible API:

### Health Check

```bash
curl http://localhost:8080/health
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100
  }'
```

### Embeddings

The `/v1/embeddings` endpoint is available when the server is started with `embedding=True`. It accepts single strings or batches and returns OpenAI-compatible responses.

```bash
# Single text
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "nomic-embed-text"}'

# Batch input
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["first text", "second text"], "model": "nomic-embed-text"}'
```

Response format:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}
  ],
  "model": "nomic-embed-text",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

## Embedding Server Configuration

To serve embeddings, enable the `embedding` flag in `ServerConfig` and optionally specify a dedicated embedding model:

```python
from cyllama.llama.server.python import ServerConfig, PythonServer

config = ServerConfig(
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    embedding=True,

    # Optional: use a dedicated embedding model (defaults to model_path)
    embedding_model_path="models/bge-small-en-v1.5-q8_0.gguf",

    # Embedding-specific parameters (same options as the Embedder class)
    embedding_n_ctx=512,          # Context size (match model training)
    embedding_n_batch=512,        # Batch size
    embedding_n_gpu_layers=-1,    # GPU layers (-1 = all)
    embedding_pooling="mean",     # Pooling: mean, cls, last, none
    embedding_normalize=True,     # L2 normalize output vectors
)

with PythonServer(config) as server:
    # Server now handles both /v1/chat/completions and /v1/embeddings
    import time
    while True:
        time.sleep(1)
```

The same configuration works with `EmbeddedServer`:

```python
from cyllama.llama.server.embedded import EmbeddedServer

with EmbeddedServer(config) as server:
    server.wait_for_shutdown()
```

### Embedding Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding` | `False` | Enable `/v1/embeddings` endpoint |
| `embedding_model_path` | `None` | Path to embedding model (uses `model_path` if `None`) |
| `embedding_n_ctx` | `512` | Context size for embedding model |
| `embedding_n_batch` | `512` | Batch size for embedding model |
| `embedding_n_gpu_layers` | `-1` | GPU layers for embedding model (-1 = all) |
| `embedding_pooling` | `"mean"` | Pooling strategy: `mean`, `cls`, `last`, `none` |
| `embedding_normalize` | `True` | L2 normalize output embeddings |

## Comparison

| | EmbeddedServer | PythonServer |
|-|-|-|
| HTTP layer | cpp-httplib thread pool | stdlib `http.server` |
| Concurrent requests | queued onto `n_parallel` slots | one at a time |
| Streaming (`"stream": true`) | chunked SSE | SSE, connection closes at end |
| Requires compiled extension | yes | no |

`llama_decode`, sampling and tokenization release the GIL; the Python code between them was 0.07% of generation time in a CPU profile. Slots do not batch together, so throughput is bounded by memory bandwidth, not the GIL.

Each slot runs `ServerConfig.n_threads` threads, by default physical cores divided by `n_parallel`. Decode is memory-bound: with Llama-3.2-1B Q8_0 on a 16-core Ryzen 9 7940HX, 8 threads matched 16, and 32 threads (SMT siblings included) cut throughput by 45%.
