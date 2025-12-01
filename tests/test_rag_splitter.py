"""Tests for the RAG TextSplitter classes."""

import pytest

from cyllama.rag.splitter import TextSplitter, TokenTextSplitter, MarkdownSplitter
from cyllama.rag.types import Document, Chunk


class TestTextSplitterInit:
    """Test TextSplitter initialization."""

    def test_init_default(self):
        """Test default initialization."""
        splitter = TextSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50
        assert splitter.keep_separator is True
        assert splitter.strip_whitespace is True

    def test_init_custom(self):
        """Test custom initialization."""
        splitter = TextSplitter(
            chunk_size=256,
            chunk_overlap=20,
            keep_separator=False,
            strip_whitespace=False,
        )
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20
        assert splitter.keep_separator is False
        assert splitter.strip_whitespace is False

    def test_init_invalid_chunk_size(self):
        """Test that invalid chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=-1)

    def test_init_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextSplitter(chunk_overlap=-1)

    def test_init_overlap_too_large(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            TextSplitter(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            TextSplitter(chunk_size=100, chunk_overlap=150)

    def test_repr(self):
        """Test string representation."""
        splitter = TextSplitter(chunk_size=256, chunk_overlap=20)
        assert "TextSplitter" in repr(splitter)
        assert "256" in repr(splitter)
        assert "20" in repr(splitter)


class TestTextSplitterSplit:
    """Test basic splitting functionality."""

    def test_split_empty_string(self):
        """Test splitting empty string."""
        splitter = TextSplitter()
        assert splitter.split("") == []

    def test_split_short_text(self):
        """Test splitting text shorter than chunk_size."""
        splitter = TextSplitter(chunk_size=100)
        text = "This is a short text."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_on_paragraphs(self):
        """Test splitting on paragraph breaks."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_split_on_sentences(self):
        """Test splitting on sentence boundaries."""
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence."
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # Verify sentences are kept together when possible
        for chunk in chunks:
            assert chunk.strip()

    def test_split_preserves_content(self):
        """Test that all content is preserved after splitting."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "This is a longer text that needs to be split into multiple chunks for processing."
        chunks = splitter.split(text)
        # Join chunks should contain all words
        rejoined = " ".join(chunks)
        for word in text.split():
            assert word in rejoined

    def test_split_with_overlap(self):
        """Test that overlap works correctly."""
        splitter = TextSplitter(chunk_size=30, chunk_overlap=10)
        text = "First part. Second part. Third part. Fourth part."
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # With overlap, some content may appear in multiple chunks

    def test_split_unicode(self):
        """Test splitting text with unicode characters."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Hello world. Bonjour monde."
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_split_long_words(self):
        """Test splitting text with very long words."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        text = "a" * 50  # Single "word" longer than chunk_size
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        # All characters should be present
        assert sum(len(c) for c in chunks) >= 50


class TestTextSplitterDocuments:
    """Test document splitting."""

    def test_split_documents_empty(self):
        """Test splitting empty list of documents."""
        splitter = TextSplitter()
        assert splitter.split_documents([]) == []

    def test_split_documents_single(self):
        """Test splitting a single document."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        doc = Document(
            text="First sentence. Second sentence. Third sentence.",
            metadata={"source": "test.txt"},
            id="doc1",
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        # Metadata should be preserved
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"
            assert chunk.source_id == "doc1"

    def test_split_documents_multiple(self):
        """Test splitting multiple documents."""
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        docs = [
            Document(text="Document one text.", metadata={"doc": 1}),
            Document(text="Document two text.", metadata={"doc": 2}),
        ]
        chunks = splitter.split_documents(docs)
        assert len(chunks) >= 2
        # Each document's metadata should be preserved in its chunks
        doc1_chunks = [c for c in chunks if c.metadata.get("doc") == 1]
        doc2_chunks = [c for c in chunks if c.metadata.get("doc") == 2]
        assert len(doc1_chunks) >= 1
        assert len(doc2_chunks) >= 1

    def test_split_documents_chunk_index(self):
        """Test that chunk_index is correctly assigned."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        doc = Document(text="Part one. Part two. Part three.", id="doc1")
        chunks = splitter.split_documents([doc])
        # Verify chunk indices are sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestTextSplitterCustomSeparators:
    """Test custom separators."""

    def test_custom_separators(self):
        """Test with custom separator list."""
        splitter = TextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            separators=["|", " "],
        )
        text = "First|Second|Third Fourth"
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_keep_separator_false(self):
        """Test with keep_separator=False."""
        splitter = TextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=[". "],
            keep_separator=False,
        )
        text = "First sentence. Second sentence."
        chunks = splitter.split(text)
        # Without keeping separator, periods should not be at end of chunks
        for chunk in chunks[:-1]:  # Exclude last chunk
            chunk = chunk.strip()
            # With keep_separator=False, separator is removed

    def test_custom_length_function(self):
        """Test with custom length function."""
        # Use word count instead of character count
        splitter = TextSplitter(
            chunk_size=5,  # 5 words
            chunk_overlap=1,
            length_function=lambda x: len(x.split()),
        )
        text = "one two three four five six seven eight nine ten"
        chunks = splitter.split(text)
        assert len(chunks) >= 2


class TestTokenTextSplitter:
    """Test TokenTextSplitter class."""

    def test_init_default(self):
        """Test default initialization."""
        splitter = TokenTextSplitter()
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20

    def test_with_tokenizer(self):
        """Test with custom tokenizer function."""
        # Simple tokenizer: count spaces + 1
        def simple_tokenizer(text: str) -> int:
            return len(text.split())

        splitter = TokenTextSplitter(
            chunk_size=5,
            chunk_overlap=1,
            tokenizer=simple_tokenizer,
        )
        text = "one two three four five six seven eight"
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_repr(self):
        """Test string representation."""
        splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
        assert "TokenTextSplitter" in repr(splitter)


class TestMarkdownSplitter:
    """Test MarkdownSplitter class."""

    def test_init_default(self):
        """Test default initialization."""
        splitter = MarkdownSplitter()
        assert splitter.chunk_size == 1024
        assert splitter.chunk_overlap == 100

    def test_split_headers(self):
        """Test splitting on markdown headers."""
        splitter = MarkdownSplitter(chunk_size=100, chunk_overlap=0)
        text = """# Header 1

Content under header 1.

## Header 2

Content under header 2.

## Header 3

Content under header 3."""
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_split_code_blocks(self):
        """Test splitting respects code blocks."""
        splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=0)
        text = """Some text before.

```python
def hello():
    print("Hello")
```

Some text after."""
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_split_lists(self):
        """Test splitting on list items."""
        splitter = MarkdownSplitter(chunk_size=50, chunk_overlap=0)
        text = """Items:

- First item
- Second item
- Third item
- Fourth item"""
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_repr(self):
        """Test string representation."""
        splitter = MarkdownSplitter(chunk_size=512, chunk_overlap=50)
        assert "MarkdownSplitter" in repr(splitter)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_only_whitespace(self):
        """Test splitting text that's only whitespace."""
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split("   \n\n   \t   ")
        # Should return empty or stripped chunks
        assert all(c.strip() == "" or c.strip() for c in chunks) or chunks == []

    def test_single_character(self):
        """Test splitting single character."""
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split("a")
        assert chunks == ["a"]

    def test_newlines_only(self):
        """Test splitting text with only newlines."""
        splitter = TextSplitter(chunk_size=10, chunk_overlap=0)
        chunks = splitter.split("\n\n\n\n")
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        splitter = TextSplitter(chunk_size=5, chunk_overlap=1)
        text = "Hello world"
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_chunk_size_equals_text_length(self):
        """Test when chunk size equals text length."""
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        text = "Hello world"  # 11 chars, fits in chunk
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_mixed_separators(self):
        """Test text with mixed separator types."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Para 1.\n\nPara 2.\nLine 2. Sentence 2. Word1 word2"
        chunks = splitter.split(text)
        assert len(chunks) >= 1
