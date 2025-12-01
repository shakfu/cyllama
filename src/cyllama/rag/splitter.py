"""Text splitting utilities for RAG pipelines."""

from __future__ import annotations

import re
from typing import Callable

from .types import Chunk, Document


class TextSplitter:
    """Split text into chunks for embedding.

    Uses recursive character splitting with configurable separators.
    Attempts to split on natural boundaries (paragraphs, sentences, words)
    while respecting the target chunk size.

    Example:
        >>> splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        >>> chunks = splitter.split("Long document text...")
        >>> print(len(chunks))

        >>> # Split documents preserving metadata
        >>> docs = [Document(text="...", metadata={"source": "file.txt"})]
        >>> chunks = splitter.split_documents(docs)
    """

    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamation
        "? ",    # Question
        "; ",    # Semicolon
        ", ",    # Comma
        " ",     # Word breaks
        "",      # Character level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize text splitter.

        Args:
            chunk_size: Target chunk size in characters (or tokens if using
                        custom length_function)
            chunk_overlap: Number of characters to overlap between chunks.
                           Helps maintain context across chunk boundaries.
            separators: Hierarchy of separators to split on.
                        Default: ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            length_function: Function to measure text length. Defaults to len().
                             Can be customized for token-based splitting.
            keep_separator: Whether to keep separators in the output chunks.
            strip_whitespace: Whether to strip whitespace from chunk edges.

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0 or
                        chunk_overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS.copy()
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

    def split(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        return self._split_text(text, self.separators)

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks, preserving metadata.

        Each chunk inherits metadata from its source document and gets
        additional chunk_index and source_id fields.

        Args:
            documents: List of documents to split

        Returns:
            List of chunks with preserved metadata
        """
        chunks = []
        for doc in documents:
            text_chunks = self.split(doc.text)
            for i, chunk_text in enumerate(text_chunks):
                chunk = Chunk(
                    text=chunk_text,
                    metadata=doc.metadata.copy(),
                    source_id=doc.id,
                    chunk_index=i,
                )
                chunks.append(chunk)
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks
        """
        final_chunks = []

        # Get the best separator for this level
        separator = separators[-1] if separators else ""
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split the text
        splits = self._split_by_separator(text, separator)

        # Process splits
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Merge any accumulated good splits
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                # Recursively split the large piece
                if new_separators:
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # No more separators, force split by character
                    final_chunks.extend(self._force_split(split))

        # Merge remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by separator, optionally keeping the separator.

        Args:
            text: Text to split
            separator: Separator to split on

        Returns:
            List of splits
        """
        if not separator:
            # Split into individual characters
            return list(text)

        if self.keep_separator:
            # Use regex to keep separator with the preceding text
            pattern = f"({re.escape(separator)})"
            splits = re.split(pattern, text)

            # Combine separators with preceding text
            result = []
            for i, split in enumerate(splits):
                if i % 2 == 0:  # Text part
                    if split:
                        result.append(split)
                else:  # Separator part
                    if result:
                        result[-1] += split
                    else:
                        result.append(split)
            return result
        else:
            return text.split(separator)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits into chunks respecting size limits.

        Args:
            splits: List of text pieces to merge
            separator: Separator used (for rejoining)

        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            # Check if adding this split would exceed the limit
            separator_length = self.length_function(separator) if current_chunk else 0
            if current_length + separator_length + split_length > self.chunk_size:
                # Save current chunk if non-empty
                if current_chunk:
                    chunk_text = self._join_and_strip(current_chunk, separator)
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Start new chunk with overlap
                    current_chunk, current_length = self._get_overlap_start(
                        current_chunk, separator
                    )

            current_chunk.append(split)
            current_length += separator_length + split_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = self._join_and_strip(current_chunk, separator)
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _get_overlap_start(
        self, chunks: list[str], separator: str
    ) -> tuple[list[str], int]:
        """Get the overlap portion from the end of chunks.

        Args:
            chunks: Current accumulated chunks
            separator: Separator for length calculation

        Returns:
            Tuple of (overlap_chunks, overlap_length)
        """
        if self.chunk_overlap == 0:
            return [], 0

        overlap_chunks = []
        overlap_length = 0

        # Work backwards to get overlap
        for chunk in reversed(chunks):
            chunk_len = self.length_function(chunk)
            sep_len = self.length_function(separator) if overlap_chunks else 0

            if overlap_length + sep_len + chunk_len > self.chunk_overlap:
                break

            overlap_chunks.insert(0, chunk)
            overlap_length += sep_len + chunk_len

        return overlap_chunks, overlap_length

    def _force_split(self, text: str) -> list[str]:
        """Force split text that can't be split by any separator.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Adjust for overlap on subsequent chunks
            if start > 0:
                start = max(0, start - self.chunk_overlap)

            chunk = text[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            start = end

        return chunks

    def _join_and_strip(self, parts: list[str], separator: str) -> str:
        """Join parts and optionally strip whitespace.

        Args:
            parts: Text parts to join
            separator: Separator (not used for joining since separators are kept)

        Returns:
            Joined text
        """
        # Parts already have separators attached when keep_separator=True
        text = "".join(parts)
        if self.strip_whitespace:
            text = text.strip()
        return text

    def __repr__(self) -> str:
        return (
            f"TextSplitter(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


class TokenTextSplitter(TextSplitter):
    """Text splitter that uses token count instead of character count.

    Requires a tokenizer function that converts text to token count.
    Useful for ensuring chunks fit within model context windows.

    Example:
        >>> from cyllama import LlamaVocab
        >>> vocab = LlamaVocab("model.gguf")
        >>> splitter = TokenTextSplitter(
        ...     chunk_size=256,
        ...     chunk_overlap=20,
        ...     tokenizer=lambda t: len(vocab.tokenize(t))
        ... )
        >>> chunks = splitter.split("Long text...")
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        tokenizer: Callable[[str], int] | None = None,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """Initialize token-based text splitter.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer: Function that returns token count for text.
                       If None, uses character count (same as TextSplitter)
            separators: Hierarchy of separators to split on
            keep_separator: Whether to keep separators in output
            strip_whitespace: Whether to strip whitespace from chunks
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=tokenizer or len,
            keep_separator=keep_separator,
            strip_whitespace=strip_whitespace,
        )
        self._tokenizer = tokenizer

    def __repr__(self) -> str:
        return (
            f"TokenTextSplitter(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


class MarkdownSplitter(TextSplitter):
    """Text splitter optimized for Markdown documents.

    Uses Markdown-aware separators to split on headers, code blocks,
    and other structural elements before falling back to generic separators.

    Example:
        >>> splitter = MarkdownSplitter(chunk_size=1024)
        >>> chunks = splitter.split(markdown_text)
    """

    MARKDOWN_SEPARATORS = [
        # Headers (highest priority)
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n##### ",
        "\n###### ",
        # Code blocks
        "\n```\n",
        "\n```",
        # Horizontal rules
        "\n---\n",
        "\n***\n",
        "\n___\n",
        # Block quotes
        "\n> ",
        # Lists
        "\n- ",
        "\n* ",
        "\n+ ",
        # Numbered lists (simplified)
        "\n1. ",
        # Standard text separators
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        **kwargs,
    ):
        """Initialize Markdown-aware text splitter.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: Custom separators (defaults to MARKDOWN_SEPARATORS)
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or self.MARKDOWN_SEPARATORS.copy(),
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MarkdownSplitter(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
