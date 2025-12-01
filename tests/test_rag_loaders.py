"""Tests for the RAG document loaders."""

import json
import tempfile
from pathlib import Path

import pytest

from cyllama.rag.loaders import (
    TextLoader,
    MarkdownLoader,
    JSONLoader,
    JSONLLoader,
    DirectoryLoader,
    PDFLoader,
    LoaderError,
    load_document,
    load_directory,
)
from cyllama.rag.types import Document


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTextLoader:
    """Test TextLoader class."""

    def test_load_text_file(self, temp_dir):
        """Test loading a plain text file."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello, world!")

        loader = TextLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert docs[0].text == "Hello, world!"
        assert docs[0].metadata["source"] == str(file_path)
        assert docs[0].metadata["filename"] == "test.txt"
        assert docs[0].metadata["filetype"] == "text"

    def test_load_with_encoding(self, temp_dir):
        """Test loading with specific encoding."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello", encoding="utf-8")

        loader = TextLoader(encoding="utf-8")
        docs = loader.load(file_path)
        assert docs[0].text == "Hello"

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = TextLoader()
        with pytest.raises(LoaderError, match="File not found"):
            loader.load("/nonexistent/path.txt")

    def test_load_directory_raises_error(self, temp_dir):
        """Test loading a directory raises error."""
        loader = TextLoader()
        with pytest.raises(LoaderError, match="Not a file"):
            loader.load(temp_dir)

    def test_load_multiline(self, temp_dir):
        """Test loading multiline text."""
        file_path = temp_dir / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3"
        file_path.write_text(content)

        loader = TextLoader()
        docs = loader.load(file_path)
        assert docs[0].text == content

    def test_load_many(self, temp_dir):
        """Test loading multiple files."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        loader = TextLoader()
        docs = loader.load_many([file1, file2])
        assert len(docs) == 2
        texts = [d.text for d in docs]
        assert "Content 1" in texts
        assert "Content 2" in texts


class TestMarkdownLoader:
    """Test MarkdownLoader class."""

    def test_load_markdown_file(self, temp_dir):
        """Test loading a markdown file."""
        file_path = temp_dir / "test.md"
        file_path.write_text("# Header\n\nContent here.")

        loader = MarkdownLoader()
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert "# Header" in docs[0].text
        assert docs[0].metadata["filetype"] == "markdown"

    def test_strip_frontmatter(self, temp_dir):
        """Test stripping YAML frontmatter."""
        file_path = temp_dir / "test.md"
        content = """---
title: Test Document
author: Test Author
---

# Header

Content here."""
        file_path.write_text(content)

        loader = MarkdownLoader(strip_frontmatter=True, parse_frontmatter=True)
        docs = loader.load(file_path)

        assert "---" not in docs[0].text
        assert "# Header" in docs[0].text
        assert docs[0].metadata.get("title") == "Test Document"
        assert docs[0].metadata.get("author") == "Test Author"

    def test_keep_frontmatter(self, temp_dir):
        """Test keeping YAML frontmatter."""
        file_path = temp_dir / "test.md"
        content = """---
title: Test
---

Content."""
        file_path.write_text(content)

        loader = MarkdownLoader(strip_frontmatter=False)
        docs = loader.load(file_path)

        assert "---" in docs[0].text

    def test_frontmatter_types(self, temp_dir):
        """Test parsing different types in frontmatter."""
        file_path = temp_dir / "test.md"
        content = """---
title: "Quoted String"
count: 42
rating: 3.14
active: true
disabled: false
---

Content."""
        file_path.write_text(content)

        loader = MarkdownLoader()
        docs = loader.load(file_path)

        assert docs[0].metadata.get("title") == "Quoted String"
        assert docs[0].metadata.get("count") == 42
        assert docs[0].metadata.get("rating") == 3.14
        assert docs[0].metadata.get("active") is True
        assert docs[0].metadata.get("disabled") is False


class TestJSONLoader:
    """Test JSONLoader class."""

    def test_load_single_object(self, temp_dir):
        """Test loading JSON with single object."""
        file_path = temp_dir / "test.json"
        data = {"text": "Hello world", "author": "Test"}
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="text")
        docs = loader.load(file_path)

        assert len(docs) == 1
        assert docs[0].text == "Hello world"
        assert docs[0].metadata.get("author") == "Test"

    def test_load_array(self, temp_dir):
        """Test loading JSON array."""
        file_path = temp_dir / "test.json"
        data = [
            {"text": "First item", "id": 1},
            {"text": "Second item", "id": 2},
        ]
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="text")
        docs = loader.load(file_path)

        assert len(docs) == 2
        assert docs[0].text == "First item"
        assert docs[1].text == "Second item"

    def test_custom_text_key(self, temp_dir):
        """Test with custom text key."""
        file_path = temp_dir / "test.json"
        data = {"content": "The content", "title": "Title"}
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="content")
        docs = loader.load(file_path)

        assert docs[0].text == "The content"

    def test_metadata_keys(self, temp_dir):
        """Test selecting specific metadata keys."""
        file_path = temp_dir / "test.json"
        data = {"text": "Content", "author": "Alice", "date": "2024-01-01", "extra": "ignored"}
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="text", metadata_keys=["author", "date"])
        docs = loader.load(file_path)

        assert docs[0].metadata.get("author") == "Alice"
        assert docs[0].metadata.get("date") == "2024-01-01"
        assert "extra" not in docs[0].metadata

    def test_missing_text_key(self, temp_dir):
        """Test error when text key is missing."""
        file_path = temp_dir / "test.json"
        data = {"content": "Hello"}
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="text")
        with pytest.raises(LoaderError, match="Text key 'text' not found"):
            loader.load(file_path)

    def test_invalid_json(self, temp_dir):
        """Test error for invalid JSON."""
        file_path = temp_dir / "test.json"
        file_path.write_text("not valid json {")

        loader = JSONLoader()
        with pytest.raises(LoaderError, match="Invalid JSON"):
            loader.load(file_path)

    def test_jq_filter_nested(self, temp_dir):
        """Test jq-like filter for nested data."""
        file_path = temp_dir / "test.json"
        data = {"data": {"items": [{"text": "Item 1"}, {"text": "Item 2"}]}}
        file_path.write_text(json.dumps(data))

        loader = JSONLoader(text_key="text", jq_filter=".data.items")
        docs = loader.load(file_path)

        assert len(docs) == 2


class TestJSONLLoader:
    """Test JSONLLoader class."""

    def test_load_jsonl(self, temp_dir):
        """Test loading JSONL file."""
        file_path = temp_dir / "test.jsonl"
        lines = [
            json.dumps({"text": "Line 1", "id": 1}),
            json.dumps({"text": "Line 2", "id": 2}),
            json.dumps({"text": "Line 3", "id": 3}),
        ]
        file_path.write_text("\n".join(lines))

        loader = JSONLLoader(text_key="text")
        docs = loader.load(file_path)

        assert len(docs) == 3
        assert docs[0].text == "Line 1"
        assert docs[1].text == "Line 2"
        assert docs[2].text == "Line 3"

    def test_lazy_load(self, temp_dir):
        """Test lazy loading."""
        file_path = temp_dir / "test.jsonl"
        lines = [json.dumps({"text": f"Line {i}"}) for i in range(5)]
        file_path.write_text("\n".join(lines))

        loader = JSONLLoader(text_key="text")
        docs_iter = loader.lazy_load(file_path)

        # Should be an iterator
        doc1 = next(docs_iter)
        assert doc1.text == "Line 0"

    def test_skip_empty_lines(self, temp_dir):
        """Test that empty lines are skipped."""
        file_path = temp_dir / "test.jsonl"
        content = '{"text": "First"}\n\n{"text": "Second"}\n   \n{"text": "Third"}'
        file_path.write_text(content)

        loader = JSONLLoader(text_key="text")
        docs = loader.load(file_path)

        assert len(docs) == 3

    def test_line_numbers_in_metadata(self, temp_dir):
        """Test that line numbers are included in metadata."""
        file_path = temp_dir / "test.jsonl"
        lines = [json.dumps({"text": f"Line {i}"}) for i in range(3)]
        file_path.write_text("\n".join(lines))

        loader = JSONLLoader(text_key="text")
        docs = loader.load(file_path)

        assert docs[0].metadata.get("line_number") == 1
        assert docs[1].metadata.get("line_number") == 2
        assert docs[2].metadata.get("line_number") == 3


class TestDirectoryLoader:
    """Test DirectoryLoader class."""

    def test_load_directory(self, temp_dir):
        """Test loading all files from directory."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")
        (temp_dir / "file3.md").write_text("# Markdown")

        loader = DirectoryLoader()
        docs = loader.load(temp_dir)

        assert len(docs) == 3

    def test_glob_pattern(self, temp_dir):
        """Test with glob pattern."""
        (temp_dir / "doc1.txt").write_text("Text 1")
        (temp_dir / "doc2.txt").write_text("Text 2")
        (temp_dir / "doc3.md").write_text("Markdown")

        loader = DirectoryLoader(glob="*.txt")
        docs = loader.load(temp_dir)

        assert len(docs) == 2
        assert all("txt" in d.metadata["filename"] for d in docs)

    def test_recursive_loading(self, temp_dir):
        """Test recursive directory loading."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        loader = DirectoryLoader(glob="**/*.txt", recursive=True)
        docs = loader.load(temp_dir)

        assert len(docs) == 2

    def test_exclude_pattern(self, temp_dir):
        """Test excluding files by pattern."""
        (temp_dir / "include.txt").write_text("Include")
        (temp_dir / "exclude.txt").write_text("Exclude")
        (temp_dir / "data.json").write_text('{"text": "JSON"}')

        loader = DirectoryLoader(exclude=["exclude.txt"])
        docs = loader.load(temp_dir)

        filenames = [d.metadata["filename"] for d in docs]
        assert "exclude.txt" not in filenames

    def test_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        loader = DirectoryLoader()
        with pytest.raises(LoaderError, match="Directory not found"):
            loader.load("/nonexistent/path")

    def test_file_instead_of_directory(self, temp_dir):
        """Test error when path is file not directory."""
        file_path = temp_dir / "file.txt"
        file_path.write_text("Content")

        loader = DirectoryLoader()
        with pytest.raises(LoaderError, match="Not a directory"):
            loader.load(file_path)

    def test_custom_loader_mapping(self, temp_dir):
        """Test with custom loader mapping."""
        (temp_dir / "data.json").write_text('{"content": "Custom"}')

        custom_json_loader = JSONLoader(text_key="content")
        loader = DirectoryLoader(loader_mapping={".json": custom_json_loader})
        docs = loader.load(temp_dir)

        assert len(docs) == 1
        assert docs[0].text == "Custom"


class TestPDFLoader:
    """Test PDFLoader class."""

    def test_missing_docling(self, temp_dir):
        """Test error when docling is not installed."""
        file_path = temp_dir / "test.pdf"
        file_path.write_bytes(b"%PDF-1.4 fake pdf")

        loader = PDFLoader()
        # This should raise an error about docling not being installed
        # (unless docling is actually installed in the test environment)
        try:
            loader.load(file_path)
        except LoaderError as e:
            assert "docling" in str(e).lower()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_document_txt(self, temp_dir):
        """Test load_document with text file."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Hello")

        docs = load_document(file_path)
        assert len(docs) == 1
        assert docs[0].text == "Hello"

    def test_load_document_md(self, temp_dir):
        """Test load_document with markdown file."""
        file_path = temp_dir / "test.md"
        file_path.write_text("# Title\n\nContent")

        docs = load_document(file_path)
        assert len(docs) == 1

    def test_load_document_json(self, temp_dir):
        """Test load_document with JSON file."""
        file_path = temp_dir / "test.json"
        file_path.write_text('{"text": "JSON content"}')

        docs = load_document(file_path)
        assert len(docs) == 1
        assert docs[0].text == "JSON content"

    def test_load_document_unsupported(self, temp_dir):
        """Test load_document with unsupported file type."""
        file_path = temp_dir / "test.xyz"
        file_path.write_text("Content")

        with pytest.raises(LoaderError, match="Unsupported file type"):
            load_document(file_path)

    def test_load_directory_function(self, temp_dir):
        """Test load_directory convenience function."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        docs = load_directory(temp_dir, glob="*.txt")
        assert len(docs) == 2


class TestDocumentMetadata:
    """Test metadata handling across loaders."""

    def test_document_id_assignment(self, temp_dir):
        """Test that document IDs are assigned."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Content")

        loader = TextLoader()
        docs = loader.load(file_path)

        assert docs[0].id == str(file_path)

    def test_json_item_index(self, temp_dir):
        """Test that JSON array items get index in metadata."""
        file_path = temp_dir / "test.json"
        data = [{"text": "A"}, {"text": "B"}]
        file_path.write_text(json.dumps(data))

        loader = JSONLoader()
        docs = loader.load(file_path)

        assert docs[0].metadata.get("item_index") == 0
        assert docs[1].metadata.get("item_index") == 1

    def test_source_in_metadata(self, temp_dir):
        """Test that source path is in metadata."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Content")

        loader = TextLoader()
        docs = loader.load(file_path)

        assert docs[0].metadata["source"] == str(file_path)
