# Text Processing

cyllama provides utilities for splitting documents into chunks and loading various file formats.

## Text Splitters

### TextSplitter

Recursive character text splitting with configurable chunk size and overlap:

```python
from cyllama.rag import TextSplitter

splitter = TextSplitter(
    chunk_size=512,      # Target chunk size in characters
    chunk_overlap=50,    # Overlap between chunks
    separators=None,     # Use default separators
    keep_separator=True  # Keep separators in output
)

text = """
This is a long document that needs to be split into smaller chunks.
Each chunk will be embedded separately and stored in the vector database.

The splitter tries to maintain semantic coherence by splitting on
natural boundaries like paragraphs and sentences.
"""

chunks = splitter.split(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} chars")
```

#### Default Separators

The splitter uses a hierarchy of separators:

1. `\n\n` - Paragraph breaks

2. `\n` - Line breaks

3. `". "` - Sentences

4. `"! "` - Exclamations

5. `"? "` - Questions

6. `"; "` - Semicolons

7. `", "` - Commas

8. `" "` - Words

9. `""` - Characters

### TokenTextSplitter

Split based on token count using a custom tokenizer:

```python
from cyllama.rag import TokenTextSplitter

# Define tokenizer function: returns a token count
def my_tokenizer(text):
    return len(text.split())  # Simple word tokenizer

splitter = TokenTextSplitter(
    chunk_size=100,           # Tokens per chunk
    chunk_overlap=20,         # Token overlap
    tokenizer=my_tokenizer
)

chunks = splitter.split(long_text)
```

### MarkdownSplitter

Markdown-aware splitting that respects document structure:

````python
from cyllama.rag import MarkdownSplitter

splitter = MarkdownSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

markdown_text = """
# Introduction

This is the introduction section.

## Getting Started

Here's how to get started...

```python
code_block = "preserved"
```

- List item 1

- List item 2 """

chunks = splitter.split(markdown_text)
````

The MarkdownSplitter tries separators in this order:

1. Headers (`##` to `######`)

2. Code fences (```` ``` ````)

3. Horizontal rules and block quotes

4. List items (`-`, `*`, `+`, `1.`)

5. The generic separators: paragraph, line, sentence, word, character

It splits only when a section exceeds `chunk_size`. It does not parse frontmatter.

## Document Loaders

### TextLoader

Load plain text files:

```python
from cyllama.rag import TextLoader

loader = TextLoader()
documents = loader.load("document.txt")

for doc in documents:
    print(f"Text: {doc.text[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### MarkdownLoader

Load Markdown files with optional frontmatter parsing:

```python
from cyllama.rag import MarkdownLoader

loader = MarkdownLoader(parse_frontmatter=True)
documents = loader.load("README.md")

# Frontmatter becomes metadata
print(documents[0].metadata)
# {'title': 'My Doc', 'author': 'John', ...}
```

#### JSONLoader

Load JSON files with configurable text extraction:

```python
from cyllama.rag import JSONLoader

# Simple usage - extract 'content' field
loader = JSONLoader(text_key="content")
docs = loader.load("data.json")

# With jq-like filtering for nested data
loader = JSONLoader(
    text_key="body",
    jq_filter=".articles"
)
docs = loader.load("nested.json")
```

#### JSONLLoader

Load JSON Lines files with lazy loading:

```python
from cyllama.rag import JSONLLoader

# Regular loading
loader = JSONLLoader(text_key="content")
docs = loader.load("data.jsonl")

# Lazy loading for large files
for doc in loader.lazy_load("large.jsonl"):
    # Process one at a time
    process(doc)
```

#### PDFLoader

Load PDF files. Requires one backend: `pypdf`, `pymupdf`, `pdfminer.six`, or `docling`. The default `backend="auto"` picks the first installed one in that order.

```python
# Install: pip install pypdf  (or pymupdf, pdfminer.six, docling)

from cyllama.rag import PDFLoader

loader = PDFLoader()  # or PDFLoader(backend="docling")
documents = loader.load("document.pdf")

for doc in documents:
    print(f"Text: {doc.text[:200]}...")
    print(f"Source: {doc.metadata['source']}")
```

#### DirectoryLoader

Batch load files from directories:

```python
from cyllama.rag import DirectoryLoader

loader = DirectoryLoader(glob="**/*.md")  # Pattern to match

documents = loader.load("docs/")
print(f"Loaded {len(documents)} documents")
```

### Convenience Functions

#### load_document()

Auto-detect format and load:

```python
from cyllama.rag import load_document

# Automatically uses correct loader
docs = load_document("file.md")
docs = load_document("data.json", text_key="content")
docs = load_document("report.pdf")  # Requires a PDF backend
```

#### load_directory()

Load all matching files:

```python
from cyllama.rag import load_directory

docs = load_directory("docs/", glob="**/*.txt")
```

### Complete Example

```python
from cyllama.rag import (
    TextSplitter,
    DirectoryLoader,
    Embedder,
    SqliteVectorStore,
)

# Load documents
loader = DirectoryLoader(glob="**/*.md")
documents = loader.load("knowledge_base/")

# Split into chunks
splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
chunks = []
for doc in documents:
    doc_chunks = splitter.split(doc.text)
    chunks.extend(doc_chunks)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")

# Embed and store
embedder = Embedder("models/bge-small.gguf")

with SqliteVectorStore(dimension=embedder.dimension, db_path="kb.db") as store:
    embeddings = embedder.embed_batch(chunks)
    store.add(embeddings, chunks)
    print(f"Indexed {len(store)} chunks")

embedder.close()
```

### Data Types

#### Document

```python
from cyllama.rag import Document

doc = Document(
    text="Document content here",
    metadata={"source": "file.txt", "page": 1}
)
```

#### Chunk

```python
from cyllama.rag import Chunk

chunk = Chunk(
    text="Chunk content",
    metadata={"source": "file.txt"},
    source_id="file.txt",  # ID of the source document
    chunk_index=0          # Position of this chunk in the document
)
```

### Best Practices

1. **Chunk Size**: 256-1024 characters works well for most use cases

2. **Overlap**: 10-20% of chunk size helps maintain context

3. **Markdown**: Use `MarkdownSplitter` for structured documents

4. **Large Files**: Use lazy loading with `JSONLLoader.lazy_load()`

5. **Metadata**: Preserve source information for citation
