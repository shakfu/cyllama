# RAG Pipeline

The RAG pipeline orchestrates the complete retrieval-augmented generation process: embedding queries, retrieving relevant documents, formatting prompts, and generating responses.

## High-Level RAG Class

The `RAG` class provides the simplest interface:

```python
from cyllama.rag import RAG, RAGConfig

# Initialize
rag = RAG(
    embedding_model="models/bge-small.gguf",
    generation_model="models/llama.gguf",
    chunk_size=512,          # Text splitting
    chunk_overlap=50,
    db_path=":memory:",      # Vector store location
    config=RAGConfig(        # Optional config
        top_k=5,
        temperature=0.7
    )
)

# Add documents
rag.add_texts([
    "Python was created by Guido van Rossum.",
    "Python emphasizes code readability.",
    "Python supports multiple programming paradigms."
])

# Or from files
rag.add_documents(["guide.md", "tutorial.txt"])

# Query
response = rag.query("Who created Python?")
print(response.text)

# Stream response
for chunk in rag.stream("Explain Python's philosophy"):
    print(chunk, end="", flush=True)

# Retrieve without generation
sources = rag.retrieve("Python creator")
for source in sources:
    print(f"[{source.score:.2f}] {source.text}")

# Direct vector search
results = rag.search("programming language", k=3)

# Clean up
rag.close()
```

## RAGConfig

Configure retrieval and generation parameters:

```python
from cyllama.rag import RAGConfig

config = RAGConfig(
    # Retrieval settings
    top_k=5,                      # Number of documents to retrieve
    similarity_threshold=0.5,     # Minimum similarity score

    # Generation settings
    max_tokens=512,               # Maximum response length
    temperature=0.7,              # Creativity (0.0 = deterministic)

    # Prompt formatting
    prompt_template="""Use the context to answer the question.

Context:
{context}

Question: {question}

Answer:""",
    context_separator="\n\n",     # Join retrieved documents
    include_metadata=False        # Include metadata in context
)
```

### Custom Prompt Templates

```python
# Simple template
config = RAGConfig(
    prompt_template="""Based on these facts:
{context}

Answer this: {question}"""
)

# With metadata
config = RAGConfig(
    include_metadata=True,
    prompt_template="""Sources:
{context}

Given the above sources, answer: {question}"""
)
```

### Chat-Template Generation

For chat-tuned models, route generation through the model's native chat template instead of the raw `Question:/Answer:` `prompt_template`. This sidesteps a class of bugs (paragraph paraphrase loops, leaked instruction-tuning artifacts, model re-roleplaying as the user) that can occur when chat-tuned models are fed bare completion prompts:

```python
config = RAGConfig(
    use_chat_template=True,
    system_prompt="Answer using only the provided context. Do not repeat yourself.",
)
```

When `use_chat_template=True`, the pipeline calls `generator.chat()` with a system message (from `system_prompt`) plus a user message containing the retrieved context. The raw `prompt_template` is ignored in this mode.

| Field | Default (library) | Default (`cyllama rag`) |
|-------|-------------------|--------------------------|
| `use_chat_template` | `False` | `True` (disable with `--no-chat-template`) |
| `system_prompt` | `None` (built-in instruction) | (same) |

### Reasoning-Block Stripping

Reasoning-tuned models (Qwen3, DeepSeek-R1, and similar) emit `<think>...</think>` blocks before their actual answer. On small `max_tokens` budgets the reasoning often consumes the entire budget, leaving no room for the answer the user wanted. `strip_think_blocks` removes these blocks from the streamed output:

```python
config = RAGConfig(strip_think_blocks=True)
```

The strip is implemented by `cyllama.rag.repetition.ThinkBlockStripper`, a stream-safe state machine that handles tags split across chunk boundaries, multiple blocks per stream, and unclosed blocks. It runs as the outermost filter in the pipeline so downstream filters (e.g. the repetition detector) see post-strip text.

| Field | Default (library) | Default (`cyllama rag`) |
|-------|-------------------|--------------------------|
| `strip_think_blocks` | `False` | `True` (expose with `--show-think`) |

### Repetition Detection

A streaming-level guard against the "model loops on the same paragraph" failure mode (notably hit on Qwen3-4B greedy decoding). `NGramRepetitionDetector` watches a rolling word window and stops generation when the same n-gram repeats too many times:

```python
config = RAGConfig(
    repetition_threshold=2,   # 0 disables; >=2 enables
    repetition_ngram=5,       # word-level n-gram length
    repetition_window=300,    # rolling window size
)
```

| Field | Default (library) | Default (`cyllama rag`) |
|-------|-------------------|--------------------------|
| `repetition_threshold` | `0` (disabled) | `2` (`--repetition-threshold N`, `0` disables) |
| `repetition_ngram` | `5` | `5` (`--repetition-ngram`) |
| `repetition_window` | `300` | `300` (`--repetition-window`) |

The library defaults are off so a bare `RAGConfig()` preserves historical behavior. The CLI opts in by default because that's where the bug surfaces.

## RAGResponse

Query responses include text, sources, and statistics:

```python
response = rag.query("What is Python?")

# Generated text
print(response.text)

# Original query
print(response.query)

# Retrieved sources
for source in response.sources:
    print(f"ID: {source.id}")
    print(f"Text: {source.text}")
    print(f"Score: {source.score}")
    print(f"Metadata: {source.metadata}")

# Generation stats (if available)
if response.stats:
    print(f"Tokens: {response.stats.generated_tokens}")
    print(f"Time: {response.stats.total_time}s")

# Serialize to dict
data = response.to_dict()
```

## RAGPipeline (Low-Level)

For more control, use `RAGPipeline` directly:

```python
from cyllama import LLM
from cyllama.rag import Embedder, VectorStore, RAGPipeline, RAGConfig

# Create components
embedder = Embedder("models/bge-small.gguf")
store = VectorStore(dimension=embedder.dimension)
llm = LLM("models/llama.gguf")

# Index documents
texts = ["Doc 1 content", "Doc 2 content"]
embeddings = embedder.embed_batch(texts)
store.add(embeddings, texts)

# Create pipeline
pipeline = RAGPipeline(
    embedder=embedder,
    store=store,
    generator=llm,
    config=RAGConfig(top_k=3)
)

# Query
response = pipeline.query("Your question?")
print(response.text)

# Retrieve only (no generation)
sources = pipeline.retrieve("Your question?")

# Stream
for chunk in pipeline.stream("Your question?"):
    print(chunk, end="")

# Override config for specific query
custom_config = RAGConfig(top_k=10, temperature=0.2)
response = pipeline.query("Question?", config=custom_config)

# Clean up
embedder.close()
store.close()
llm.close()
```

## Query Override

Override configuration per query:

```python
# Default config
rag = RAG(
    embedding_model="model.gguf",
    generation_model="model.gguf",
    config=RAGConfig(top_k=5, temperature=0.7)
)

# Override for specific query
precise_config = RAGConfig(top_k=10, temperature=0.1)
response = rag.query("Technical question?", config=precise_config)

# Creative query
creative_config = RAGConfig(top_k=3, temperature=0.9)
response = rag.query("Write a poem about...", config=creative_config)
```

## Complete Example

```python
from cyllama.rag import RAG, RAGConfig

# Custom configuration
config = RAGConfig(
    top_k=5,
    similarity_threshold=0.3,
    max_tokens=256,
    temperature=0.7,
    prompt_template="""You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
{context}

User Question: {question}

Answer:"""
)

# Initialize RAG
with RAG(
    embedding_model="models/bge-small.gguf",
    generation_model="models/llama.gguf",
    config=config
) as rag:
    # Build knowledge base
    rag.add_texts([
        "Python was created by Guido van Rossum in 1991.",
        "Python is known for its clear syntax and readability.",
        "Python supports object-oriented, functional, and procedural programming.",
        "The Python Package Index (PyPI) hosts thousands of third-party modules.",
    ])

    # Interactive query loop
    questions = [
        "Who created Python?",
        "What is Python known for?",
        "What programming paradigms does Python support?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        response = rag.query(question)
        print(f"A: {response.text}")
        print(f"   Sources: {len(response.sources)}")
```

## Corpus Deduplication

`RAG.add_texts` and `RAG.add_documents` md5-hash each input before indexing. Inputs whose hash is already in the store are silently skipped, so re-running the same indexing call (or a `cyllama rag --db PATH -f corpus.txt` re-run) is a true no-op on the indexing side rather than appending a duplicate copy.

Both methods return an `IndexResult` (a subclass of `list[int]` for backwards compatibility — `len(result)` and iteration still yield the newly inserted chunk IDs):

```python
result = rag.add_documents(["guide.md", "tutorial.md"])
print(f"Added {len(result)} new chunks")
print(f"Skipped (already indexed): {result.skipped_labels}")
```

If a file's basename matches an already-indexed source but its content hash differs, `add_documents` raises `ValueError` — this catches the "I edited the file but kept the name" case where a silent append would leave two versions in the index. The fix is to rebuild (`--rebuild` on the CLI) or rename the file.

## RAG Methods Summary

| Method | Description |
|--------|-------------|
| `add_texts(texts, metadata, split)` | Add text strings to knowledge base; returns `IndexResult` |
| `add_documents(paths, split)` | Load and add files; returns `IndexResult` |
| `add_document(document, split)` | Add single Document object |
| `query(question, config)` | Get RAGResponse with generated text |
| `stream(question, config)` | Stream response tokens |
| `retrieve(question, config)` | Get relevant sources only |
| `search(query, k, threshold)` | Direct vector search |
| `count` | Number of documents |
| `clear()` | Remove all documents |
| `close()` | Release resources |
