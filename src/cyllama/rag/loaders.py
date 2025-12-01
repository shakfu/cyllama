"""Document loaders for RAG pipelines.

Provides utilities to load documents from various file formats including
plain text, Markdown, JSON, and optionally PDF (via docling).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from .types import Document


class LoaderError(Exception):
    """Exception raised for document loading errors."""

    pass


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    Document loaders read files and convert them to Document objects
    with text content and metadata.
    """

    @abstractmethod
    def load(self, path: str | Path) -> list[Document]:
        """Load documents from a file.

        Args:
            path: Path to the file to load

        Returns:
            List of Document objects

        Raises:
            LoaderError: If file cannot be loaded
        """
        pass

    def load_many(self, paths: list[str | Path]) -> list[Document]:
        """Load documents from multiple files.

        Args:
            paths: List of file paths to load

        Returns:
            List of all loaded documents
        """
        documents = []
        for path in paths:
            documents.extend(self.load(path))
        return documents

    def lazy_load(self, path: str | Path) -> Iterator[Document]:
        """Lazily load documents from a file.

        Default implementation just wraps load(). Subclasses can override
        for memory-efficient streaming of large files.

        Args:
            path: Path to the file to load

        Yields:
            Document objects
        """
        yield from self.load(path)

    def _validate_path(self, path: str | Path) -> Path:
        """Validate that path exists and is a file.

        Args:
            path: Path to validate

        Returns:
            Path object

        Raises:
            LoaderError: If path doesn't exist or isn't a file
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(f"File not found: {path}")
        if not path.is_file():
            raise LoaderError(f"Not a file: {path}")
        return path


class TextLoader(BaseLoader):
    """Load plain text files.

    Example:
        >>> loader = TextLoader()
        >>> docs = loader.load("document.txt")
        >>> print(docs[0].text)
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
    ):
        """Initialize text loader.

        Args:
            encoding: Text encoding (default: utf-8)
            errors: How to handle encoding errors ('strict', 'ignore', 'replace')
        """
        self.encoding = encoding
        self.errors = errors

    def load(self, path: str | Path) -> list[Document]:
        """Load a text file as a single document.

        Args:
            path: Path to text file

        Returns:
            List containing one Document

        Raises:
            LoaderError: If file cannot be read
        """
        path = self._validate_path(path)
        try:
            text = path.read_text(encoding=self.encoding, errors=self.errors)
        except UnicodeDecodeError as e:
            raise LoaderError(f"Failed to decode {path}: {e}") from e
        except OSError as e:
            raise LoaderError(f"Failed to read {path}: {e}") from e

        return [
            Document(
                text=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "filetype": "text",
                },
                id=str(path),
            )
        ]


class MarkdownLoader(TextLoader):
    """Load Markdown files.

    Optionally strips YAML frontmatter and extracts it as metadata.

    Example:
        >>> loader = MarkdownLoader(strip_frontmatter=True)
        >>> docs = loader.load("README.md")
        >>> print(docs[0].metadata.get("title"))
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        strip_frontmatter: bool = True,
        parse_frontmatter: bool = True,
    ):
        """Initialize Markdown loader.

        Args:
            encoding: Text encoding
            strip_frontmatter: Whether to remove YAML frontmatter from text
            parse_frontmatter: Whether to parse frontmatter as metadata
        """
        super().__init__(encoding=encoding)
        self.strip_frontmatter = strip_frontmatter
        self.parse_frontmatter = parse_frontmatter

    def load(self, path: str | Path) -> list[Document]:
        """Load a Markdown file.

        Args:
            path: Path to Markdown file

        Returns:
            List containing one Document
        """
        docs = super().load(path)
        doc = docs[0]

        text = doc.text
        metadata = doc.metadata.copy()
        metadata["filetype"] = "markdown"

        # Handle frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()

                if self.parse_frontmatter:
                    frontmatter = self._parse_yaml_frontmatter(frontmatter_text)
                    metadata.update(frontmatter)

                if self.strip_frontmatter:
                    text = parts[2].strip()

        return [
            Document(
                text=text,
                metadata=metadata,
                id=doc.id,
            )
        ]

    def _parse_yaml_frontmatter(self, text: str) -> dict[str, Any]:
        """Parse YAML frontmatter text.

        Uses a simple parser to avoid requiring PyYAML dependency.

        Args:
            text: YAML frontmatter text

        Returns:
            Parsed key-value pairs
        """
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()

                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Try to parse as number or boolean
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False
                else:
                    try:
                        result[key] = int(value)
                    except ValueError:
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = value

        return result


class JSONLoader(BaseLoader):
    """Load JSON files.

    Can load JSON files with various structures:
    - Single object with text field
    - Array of objects with text fields
    - Nested structures using JSONPath-like keys

    Example:
        >>> loader = JSONLoader(text_key="content")
        >>> docs = loader.load("data.json")

        >>> # Load from array of objects
        >>> loader = JSONLoader(text_key="body", metadata_keys=["title", "author"])
        >>> docs = loader.load("articles.json")
    """

    def __init__(
        self,
        text_key: str = "text",
        metadata_keys: list[str] | None = None,
        jq_filter: str | None = None,
        encoding: str = "utf-8",
    ):
        """Initialize JSON loader.

        Args:
            text_key: Key containing the text content
            metadata_keys: Keys to extract as metadata (default: all except text_key)
            jq_filter: Simple path to extract (e.g., ".data.items" or "[*]")
            encoding: Text encoding for reading file
        """
        self.text_key = text_key
        self.metadata_keys = metadata_keys
        self.jq_filter = jq_filter
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        """Load a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of Documents

        Raises:
            LoaderError: If file cannot be parsed or text_key not found
        """
        path = self._validate_path(path)

        try:
            text = path.read_text(encoding=self.encoding)
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in {path}: {e}") from e
        except OSError as e:
            raise LoaderError(f"Failed to read {path}: {e}") from e

        # Apply jq-like filter
        if self.jq_filter:
            data = self._apply_filter(data, self.jq_filter)

        # Handle different data structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            raise LoaderError(f"Unexpected JSON structure in {path}: {type(data)}")

        documents = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            # Extract text
            if self.text_key not in item:
                raise LoaderError(
                    f"Text key '{self.text_key}' not found in item {i} of {path}"
                )
            text_content = str(item[self.text_key])

            # Extract metadata
            metadata = {
                "source": str(path),
                "filename": path.name,
                "filetype": "json",
                "item_index": i,
            }

            if self.metadata_keys:
                for key in self.metadata_keys:
                    if key in item:
                        metadata[key] = item[key]
            else:
                # Include all keys except text_key
                for key, value in item.items():
                    if key != self.text_key:
                        metadata[key] = value

            documents.append(
                Document(
                    text=text_content,
                    metadata=metadata,
                    id=f"{path}:{i}",
                )
            )

        return documents

    def _apply_filter(self, data: Any, filter_path: str) -> Any:
        """Apply a simple jq-like filter to data.

        Supports:
        - ".key" - access dict key
        - "[*]" - flatten array
        - ".key1.key2" - nested access

        Args:
            data: JSON data
            filter_path: Filter path

        Returns:
            Filtered data
        """
        if not filter_path or filter_path == ".":
            return data

        # Remove leading dot
        if filter_path.startswith("."):
            filter_path = filter_path[1:]

        parts = filter_path.split(".")
        result = data

        for part in parts:
            if not part:
                continue

            if part == "[*]":
                # Flatten array
                if isinstance(result, list):
                    continue
                else:
                    return result

            if isinstance(result, dict):
                if part in result:
                    result = result[part]
                else:
                    raise LoaderError(f"Key '{part}' not found in JSON data")
            elif isinstance(result, list):
                # Try to extract from all items
                result = [
                    item.get(part) for item in result if isinstance(item, dict)
                ]
            else:
                raise LoaderError(f"Cannot access '{part}' on {type(result)}")

        return result


class JSONLLoader(BaseLoader):
    """Load JSON Lines (JSONL) files.

    Each line is a separate JSON object, converted to a Document.

    Example:
        >>> loader = JSONLLoader(text_key="content")
        >>> docs = loader.load("data.jsonl")
    """

    def __init__(
        self,
        text_key: str = "text",
        metadata_keys: list[str] | None = None,
        encoding: str = "utf-8",
    ):
        """Initialize JSONL loader.

        Args:
            text_key: Key containing the text content
            metadata_keys: Keys to extract as metadata
            encoding: Text encoding
        """
        self.text_key = text_key
        self.metadata_keys = metadata_keys
        self.encoding = encoding

    def load(self, path: str | Path) -> list[Document]:
        """Load a JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of Documents
        """
        return list(self.lazy_load(path))

    def lazy_load(self, path: str | Path) -> Iterator[Document]:
        """Lazily load documents from JSONL file.

        Memory-efficient for large files.

        Args:
            path: Path to JSONL file

        Yields:
            Document objects
        """
        path = self._validate_path(path)

        try:
            with open(path, "r", encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise LoaderError(
                            f"Invalid JSON on line {line_num} of {path}: {e}"
                        ) from e

                    if not isinstance(item, dict):
                        continue

                    if self.text_key not in item:
                        raise LoaderError(
                            f"Text key '{self.text_key}' not found on line {line_num} of {path}"
                        )

                    text_content = str(item[self.text_key])

                    metadata = {
                        "source": str(path),
                        "filename": path.name,
                        "filetype": "jsonl",
                        "line_number": line_num,
                    }

                    if self.metadata_keys:
                        for key in self.metadata_keys:
                            if key in item:
                                metadata[key] = item[key]
                    else:
                        for key, value in item.items():
                            if key != self.text_key:
                                metadata[key] = value

                    yield Document(
                        text=text_content,
                        metadata=metadata,
                        id=f"{path}:{line_num}",
                    )

        except OSError as e:
            raise LoaderError(f"Failed to read {path}: {e}") from e


class DirectoryLoader(BaseLoader):
    """Load all matching documents from a directory.

    Automatically selects the appropriate loader based on file extension.

    Example:
        >>> loader = DirectoryLoader(glob="**/*.md")
        >>> docs = loader.load("docs/")

        >>> # With custom loaders
        >>> loader = DirectoryLoader(
        ...     glob="*.json",
        ...     loader_mapping={".json": JSONLoader(text_key="content")}
        ... )
        >>> docs = loader.load("data/")
    """

    DEFAULT_LOADERS: dict[str, BaseLoader] = {
        ".txt": TextLoader(),
        ".md": MarkdownLoader(),
        ".markdown": MarkdownLoader(),
        ".json": JSONLoader(),
        ".jsonl": JSONLLoader(),
    }

    def __init__(
        self,
        glob: str = "**/*",
        loader_mapping: dict[str, BaseLoader] | None = None,
        recursive: bool = True,
        exclude: list[str] | None = None,
    ):
        """Initialize directory loader.

        Args:
            glob: Glob pattern for matching files (default: all files)
            loader_mapping: Map of extension to loader (merged with defaults)
            recursive: Whether to search recursively
            exclude: List of glob patterns to exclude
        """
        self.glob_pattern = glob
        self.recursive = recursive
        self.exclude = exclude or []

        self.loaders = self.DEFAULT_LOADERS.copy()
        if loader_mapping:
            self.loaders.update(loader_mapping)

    def load(self, path: str | Path) -> list[Document]:
        """Load all matching documents from a directory.

        Args:
            path: Path to directory

        Returns:
            List of all loaded documents
        """
        return list(self.lazy_load(path))

    def lazy_load(self, path: str | Path) -> Iterator[Document]:
        """Lazily load documents from directory.

        Args:
            path: Path to directory

        Yields:
            Document objects
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(f"Directory not found: {path}")
        if not path.is_dir():
            raise LoaderError(f"Not a directory: {path}")

        # Get matching files
        if self.recursive:
            files = list(path.glob(self.glob_pattern))
        else:
            files = list(path.glob(self.glob_pattern.lstrip("*/")))

        # Filter out excluded patterns
        for exclude_pattern in self.exclude:
            excluded = set(path.glob(exclude_pattern))
            files = [f for f in files if f not in excluded]

        # Sort for consistent ordering
        files.sort()

        for file_path in files:
            if not file_path.is_file():
                continue

            # Get appropriate loader
            suffix = file_path.suffix.lower()
            loader = self.loaders.get(suffix)

            if loader is None:
                # Skip unsupported file types
                continue

            try:
                yield from loader.lazy_load(file_path)
            except LoaderError:
                # Re-raise loader errors
                raise
            except Exception as e:
                raise LoaderError(f"Failed to load {file_path}: {e}") from e


class PDFLoader(BaseLoader):
    """Load PDF files using docling.

    Requires docling to be installed: `pip install docling`

    Example:
        >>> loader = PDFLoader()
        >>> docs = loader.load("document.pdf")

        >>> # With OCR enabled
        >>> loader = PDFLoader(ocr=True)
        >>> docs = loader.load("scanned.pdf")
    """

    def __init__(
        self,
        ocr: bool = False,
        extract_images: bool = False,
    ):
        """Initialize PDF loader.

        Args:
            ocr: Whether to use OCR for scanned documents
            extract_images: Whether to extract and include image descriptions
        """
        self.ocr = ocr
        self.extract_images = extract_images
        self._docling = None

    def _get_docling(self):
        """Lazy import of docling."""
        if self._docling is None:
            try:
                import docling
                self._docling = docling
            except ImportError:
                raise LoaderError(
                    "docling is required for PDF loading. "
                    "Install it with: pip install docling"
                )
        return self._docling

    def load(self, path: str | Path) -> list[Document]:
        """Load a PDF file.

        Args:
            path: Path to PDF file

        Returns:
            List of Documents (one per page or entire document)

        Raises:
            LoaderError: If docling not installed or file cannot be parsed
        """
        path = self._validate_path(path)

        try:
            docling = self._get_docling()
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(path))

            # Get the markdown export of the document
            text = result.document.export_to_markdown()

            return [
                Document(
                    text=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "filetype": "pdf",
                    },
                    id=str(path),
                )
            ]

        except ImportError:
            raise LoaderError(
                "docling is required for PDF loading. "
                "Install it with: pip install docling"
            )
        except Exception as e:
            raise LoaderError(f"Failed to parse PDF {path}: {e}") from e


def load_document(path: str | Path, **kwargs) -> list[Document]:
    """Load a document using the appropriate loader based on file extension.

    Convenience function that automatically selects the right loader.

    Args:
        path: Path to document
        **kwargs: Additional arguments passed to the loader

    Returns:
        List of Documents

    Raises:
        LoaderError: If file type is unsupported or loading fails
    """
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".json": JSONLoader,
        ".jsonl": JSONLLoader,
        ".pdf": PDFLoader,
    }

    loader_class = loaders.get(suffix)
    if loader_class is None:
        raise LoaderError(f"Unsupported file type: {suffix}")

    loader = loader_class(**kwargs)
    return loader.load(path)


def load_directory(
    path: str | Path,
    glob: str = "**/*",
    **kwargs,
) -> list[Document]:
    """Load all matching documents from a directory.

    Convenience function for directory loading.

    Args:
        path: Path to directory
        glob: Glob pattern for matching files
        **kwargs: Additional arguments passed to DirectoryLoader

    Returns:
        List of all loaded Documents
    """
    loader = DirectoryLoader(glob=glob, **kwargs)
    return loader.load(path)
