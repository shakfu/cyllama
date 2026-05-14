"""Document loaders for RAG pipelines.

Provides utilities to load documents from various file formats including
plain text, Markdown, JSON, and optionally PDF (via a pluggable backend system).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, NamedTuple

from .types import Document


class PageText(NamedTuple):
    """Single unit of extracted PDF text.

    page_no is 1-indexed when the backend reports per-page output;
    None means the backend returned the whole document as one blob.
    """

    page_no: int | None
    text: str


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

    _VALID_ERROR_HANDLERS = frozenset(
        {
            "strict",
            "ignore",
            "replace",
            "xmlcharrefreplace",
            "backslashreplace",
            "namereplace",
            "surrogateescape",
            "surrogatepass",
        }
    )

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
        if errors not in self._VALID_ERROR_HANDLERS:
            raise ValueError(
                f"Invalid errors parameter: {errors!r}. Must be one of: {', '.join(sorted(self._VALID_ERROR_HANDLERS))}"
            )
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
                        result[key] = int(value)  # type: ignore[assignment]
                    except ValueError:
                        try:
                            result[key] = float(value)  # type: ignore[assignment]
                        except ValueError:
                            result[key] = value  # type: ignore[assignment]

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
                raise LoaderError(f"Text key '{self.text_key}' not found in item {i} of {path}")
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
                result = [item.get(part) for item in result if isinstance(item, dict)]
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
                        raise LoaderError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

                    if not isinstance(item, dict):
                        continue

                    if self.text_key not in item:
                        raise LoaderError(f"Text key '{self.text_key}' not found on line {line_num} of {path}")

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


class PDFBackend(ABC):
    """Abstract base class for PDF extraction backends.

    Implementations lazy-import their underlying library inside extract()
    so that none of them become hard dependencies of cyllama.

    Public extension contract (stable):
        Subclass ``PDFBackend`` and set four class-level attributes:

        * ``name`` (str): unique identifier, used as ``PDFLoader(backend=name)``
        * ``install_hint`` (str): one-line install command shown in error
          messages when the backend is requested but not available.
        * ``capabilities`` (frozenset[str]): tags from the open vocabulary
          ``{"per_page", "ocr", "tables", "images", "layout", "markdown"}``
          (extras are allowed -- callers filter by string match).
        * ``_probe_import`` (method): perform the lazy import of the
          underlying library. Raise any exception on failure;
          :meth:`is_available` will catch it.

        Implement :meth:`extract` to return ``list[PageText]``. Whole-document
        backends should return a single entry with ``page_no=None``.

        Register the class with :func:`register_pdf_backend` so it is
        selectable by name and (optionally) participates in ``"auto"``
        probing. Do not mutate the module-level ``_PDF_BACKENDS`` /
        ``_PDF_BACKEND_PRIORITY`` dicts directly -- those are private and
        their structure may change. ``register_pdf_backend`` is the
        supported surface.
    """

    name: str = ""
    install_hint: str = ""
    #: Capability tags advertised by the backend. Common values:
    #: "per_page", "ocr", "tables", "images", "layout", "markdown".
    capabilities: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def extract(self, path: Path, **options: Any) -> list[PageText]:
        """Extract text from a PDF file.

        Returns a list of PageText entries. Whole-document backends should
        return a single entry with page_no=None.
        """
        ...

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the backend's underlying library is importable."""
        try:
            cls()._probe_import()
            return True
        except Exception:
            return False

    def _probe_import(self) -> None:
        """Override in subclasses to import the underlying library."""
        raise NotImplementedError


class DoclingBackend(PDFBackend):
    """PDF backend using docling. Whole-document markdown extraction with
    layout/table awareness. Heaviest dependency, highest quality."""

    name = "docling"
    install_hint = "pip install docling"
    capabilities = frozenset({"ocr", "tables", "images", "layout", "markdown"})

    def _probe_import(self) -> None:
        import docling  # noqa: F401

    def extract(self, path: Path, **options: Any) -> list[PageText]:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        text = result.document.export_to_markdown()
        return [PageText(page_no=None, text=text)]


class PypdfBackend(PDFBackend):
    """PDF backend using pypdf. Lightweight, pure-Python, per-page text."""

    name = "pypdf"
    install_hint = "pip install pypdf"
    capabilities = frozenset({"per_page"})

    def _probe_import(self) -> None:
        import pypdf  # noqa: F401

    def extract(self, path: Path, **options: Any) -> list[PageText]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [PageText(page_no=i + 1, text=page.extract_text() or "") for i, page in enumerate(reader.pages)]


class PymupdfBackend(PDFBackend):
    """PDF backend using PyMuPDF (fitz). Fast, per-page text, native deps."""

    name = "pymupdf"
    install_hint = "pip install pymupdf"
    capabilities = frozenset({"per_page", "tables", "images"})

    def _probe_import(self) -> None:
        import fitz  # noqa: F401

    def extract(self, path: Path, **options: Any) -> list[PageText]:
        import fitz

        doc = fitz.open(str(path))
        try:
            return [PageText(page_no=i + 1, text=doc.load_page(i).get_text()) for i in range(doc.page_count)]
        finally:
            doc.close()


class PdfminerBackend(PDFBackend):
    """PDF backend using pdfminer.six. Pure-Python, whole-document text."""

    name = "pdfminer"
    install_hint = "pip install pdfminer.six"
    capabilities = frozenset({"layout"})

    def _probe_import(self) -> None:
        import pdfminer.high_level  # noqa: F401

    def extract(self, path: Path, **options: Any) -> list[PageText]:
        from pdfminer.high_level import extract_text

        return [PageText(page_no=None, text=extract_text(str(path)))]


# Backend registry. Order in _PDF_BACKEND_PRIORITY drives "auto" selection:
# lightest-first, so users who installed only pypdf get pypdf, while users
# who explicitly installed docling can opt in via backend="docling".
_PDF_BACKENDS: dict[str, type[PDFBackend]] = {
    "docling": DoclingBackend,
    "pypdf": PypdfBackend,
    "pymupdf": PymupdfBackend,
    "pdfminer": PdfminerBackend,
}
_PDF_BACKEND_PRIORITY: list[str] = ["pypdf", "pymupdf", "pdfminer", "docling"]


def register_pdf_backend(name: str, backend_cls: type[PDFBackend], priority: int | None = None) -> None:
    """Register a custom PDF backend. (Stable public API.)

    This is the sole supported way to extend PDF parsing without modifying
    cyllama. Downstream applications (e.g., cyllama-desktop) should call
    this from their own setup code rather than monkey-patching the
    ``_PDF_BACKENDS`` / ``_PDF_BACKEND_PRIORITY`` module-level dicts, whose
    structure is not part of the public API.

    Re-registering an existing ``name`` replaces the prior class; this is
    intentional so applications can override built-in backends (e.g., swap
    in a customized DoclingBackend with non-default converter options).

    Args:
        name: Backend identifier. Must be unique within the process; passed
            as ``PDFLoader(backend=name)``.
        backend_cls: ``PDFBackend`` subclass. See :class:`PDFBackend` for
            the required class attributes and methods.
        priority: Optional 0-based insertion index in the ``"auto"`` probe
            order. ``priority=0`` makes the backend probed first.
            If ``None`` (default), the backend is registered but excluded
            from ``"auto"`` and must be selected explicitly by name -- the
            right choice for application-specific or experimental backends
            that shouldn't change ``PDFLoader()``'s default behavior.

    Example:
        >>> class MyBackend(PDFBackend):
        ...     name = "my-backend"
        ...     install_hint = "pip install my-lib"
        ...     capabilities = frozenset({"per_page", "ocr"})
        ...     def _probe_import(self): import my_lib  # noqa
        ...     def extract(self, path, **opts):
        ...         return [PageText(page_no=1, text="...")]
        >>> register_pdf_backend("my-backend", MyBackend)
        >>> PDFLoader(backend="my-backend").load("doc.pdf")
    """
    _PDF_BACKENDS[name] = backend_cls
    if priority is not None:
        # Remove if already present, then insert at requested position.
        if name in _PDF_BACKEND_PRIORITY:
            _PDF_BACKEND_PRIORITY.remove(name)
        _PDF_BACKEND_PRIORITY.insert(priority, name)


def available_pdf_backends(require: Iterable[str] = ()) -> list[str]:
    """Return names of installed PDF backends.

    Args:
        require: Optional capability tags the backend must advertise
            (e.g. {"ocr"}). Backends missing any required capability are
            excluded.
    """
    req = frozenset(require)
    return [n for n, cls in _PDF_BACKENDS.items() if cls.is_available() and req.issubset(cls.capabilities)]


def pdf_backend_info(name: str) -> dict[str, Any]:
    """Return metadata about a registered PDF backend."""
    cls = _PDF_BACKENDS.get(name)
    if cls is None:
        raise LoaderError(f"Unknown PDF backend: {name!r}")
    return {
        "name": cls.name,
        "install_hint": cls.install_hint,
        "capabilities": sorted(cls.capabilities),
        "available": cls.is_available(),
    }


class PDFLoader(BaseLoader):
    """Load PDF files via a pluggable backend.

    Example:
        >>> loader = PDFLoader()                       # auto-select backend
        >>> loader = PDFLoader(backend="pypdf")        # explicit lightweight
        >>> loader = PDFLoader(backend="docling")      # explicit high-quality
        >>> loader = PDFLoader(per_page=True)          # one Document per page
    """

    def __init__(
        self,
        backend: str | PDFBackend = "auto",
        per_page: bool = False,
        require: Iterable[str] = (),
        **backend_options: Any,
    ):
        """Initialize PDF loader.

        Args:
            backend: Backend name ("docling", "pypdf", "pymupdf", "pdfminer"),
                "auto" to probe installed backends in priority order, or an
                already-instantiated PDFBackend.
            per_page: If True, emit one Document per page (when the backend
                supports it). If False, concatenate into a single Document.
            require: Capability tags the backend must advertise (e.g. {"ocr"}).
                With backend="auto", filters the probe list. With an explicit
                backend, raises if the backend lacks any required capability —
                preventing silent "you asked for OCR but got a backend that
                can't do it".
            **backend_options: Forwarded to the backend's extract() call.
        """
        self.required_caps = frozenset(require)
        self._backend = self._resolve_backend(backend, self.required_caps)
        self.per_page = per_page
        self.backend_options = backend_options

    @staticmethod
    def _resolve_backend(backend: str | PDFBackend, require: frozenset[str]) -> PDFBackend:
        if isinstance(backend, PDFBackend):
            missing = require - backend.capabilities
            if missing:
                raise LoaderError(f"PDF backend {backend.name!r} lacks required capabilities: {sorted(missing)}")
            return backend
        if backend == "auto":
            for name in _PDF_BACKEND_PRIORITY:
                cls = _PDF_BACKENDS.get(name)
                if cls is not None and cls.is_available() and require.issubset(cls.capabilities):
                    return cls()
            qualifying = [
                f"{n} ({_PDF_BACKENDS[n].install_hint})"
                for n in _PDF_BACKEND_PRIORITY
                if n in _PDF_BACKENDS and require.issubset(_PDF_BACKENDS[n].capabilities)
            ]
            if not qualifying:
                raise LoaderError(f"No registered PDF backend advertises capabilities {sorted(require)}")
            raise LoaderError(
                f"No installed PDF backend satisfies capabilities "
                f"{sorted(require) or 'PDF parsing'}. "
                f"Install one of: {', '.join(qualifying)}"
            )
        cls = _PDF_BACKENDS.get(backend)
        if cls is None:
            raise LoaderError(f"Unknown PDF backend: {backend!r}. Known: {sorted(_PDF_BACKENDS)}")
        if not cls.is_available():
            raise LoaderError(f"PDF backend {backend!r} is not installed. Install it with: {cls.install_hint}")
        missing = require - cls.capabilities
        if missing:
            raise LoaderError(f"PDF backend {backend!r} lacks required capabilities: {sorted(missing)}")
        return cls()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def load(self, path: str | Path) -> list[Document]:
        """Load a PDF file using the configured backend."""
        path = self._validate_path(path)

        try:
            pages = self._backend.extract(path, **self.backend_options)
        except LoaderError:
            raise
        except Exception as e:
            raise LoaderError(f"Failed to parse PDF {path} with backend {self._backend.name!r}: {e}") from e

        base_meta = {
            "source": str(path),
            "filename": path.name,
            "filetype": "pdf",
            "backend": self._backend.name,
        }

        if self.per_page and any(p.page_no is not None for p in pages):
            return [
                Document(
                    text=p.text,
                    metadata={**base_meta, "page": p.page_no},
                    id=f"{path}#p{p.page_no}",
                )
                for p in pages
            ]

        text = "\n\n".join(p.text for p in pages if p.text)
        return [Document(text=text, metadata=base_meta, id=str(path))]


def load_document(path: str | Path, **kwargs: Any) -> list[Document]:
    """Load a document using the appropriate loader based on file extension.

    Convenience function that selects the loader class by extension and
    forwards ``**kwargs`` to its constructor. Unknown kwargs are NOT
    silently swallowed -- they are passed to the loader's ``__init__`` and
    will raise ``TypeError`` if the loader does not accept them. This is
    intentional: it surfaces format/option mismatches loudly rather than
    masking them.

    PDF backend selection:
        For ``.pdf`` files this function does forward ``backend``,
        ``require``, ``per_page`` and any backend-specific options to
        :class:`PDFLoader`. Example::

            load_document("doc.pdf", backend="docling", per_page=True)

        However ``load_document`` is format-coupled: passing ``backend=``
        on a non-PDF path will ``TypeError`` because text/JSON/etc.
        loaders do not (currently) accept a ``backend`` kwarg. For full
        control over PDF backend selection -- especially in code that
        might be called with mixed file types -- instantiate
        :class:`PDFLoader` directly::

            loader = PDFLoader(backend="docling", require={"ocr"})
            docs = loader.load(path)

        The signature of ``load_document`` may grow first-class
        ``backend`` / ``require`` kwargs once a second file format adopts
        the backend pattern (so the kwargs have a well-defined cross-format
        meaning). Until then, prefer ``PDFLoader`` for PDF-specific tuning.

    Args:
        path: Path to document
        **kwargs: Forwarded to the loader class selected by extension.

    Returns:
        List of Documents

    Raises:
        LoaderError: If file type is unsupported or loading fails
        TypeError: If a kwarg is not accepted by the selected loader
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
    return list(loader.load(path))


def load_directory(
    path: str | Path,
    glob: str = "**/*",
    **kwargs: Any,
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
