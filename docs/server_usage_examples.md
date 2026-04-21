# Cyllama Server Usage Examples

Cyllama provides two server implementations, both offering the same OpenAI-compatible API:

## Server Types

### 1. Mongoose C Server (EmbeddedServer)

- Native C networking via the Mongoose web server library

- High concurrency and low overhead

- Compiled as part of the standard `make build`

### 2. Python HTTP Server (PythonServer)

- Pure Python HTTP server (stdlib `http.server`)

- No compiled dependencies beyond the core Cython extensions

- Subject to Python GIL limitations

## Basic Usage

### Start Default Embedded Server

```bash
python -m cyllama.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

### Start High-Performance Mongoose Server

```bash
python -m cyllama.llama.server -m models/Llama-3.2-1B-Instruct-Q8_0.gguf --server-type mongoose
```

## Advanced Configuration

### Custom Host and Port

```bash
python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --server-type mongoose \
    --host 0.0.0.0 \
    --port 8080
```

### Multiple Parallel Processing Slots

```bash
python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --server-type mongoose \
    --n-parallel 4 \
    --ctx-size 2048
```

### GPU Configuration

```bash
python -m cyllama.llama.server \
    -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --server-type mongoose \
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

The same configuration works with `EmbeddedServer` (Mongoose):

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

## Performance Comparison

| Feature | EmbeddedServer (Mongoose) | PythonServer |
|---------|--------------------------|--------------|
| Networking | Native C | Python HTTP |
| Concurrency | High | GIL limited |
| Memory Usage | Lower | Higher |
| Startup Time | Fast | Fast |
| Best For | Production, high-throughput | Development, simplicity |

## When to Use Each

### Use EmbeddedServer (Mongoose) When

- Production deployments

- Multiple concurrent users

- High-throughput requirements

- Performance is critical

### Use PythonServer When

- Developing or testing

- Single-user applications

- Simplicity is preferred

- No special performance requirements
