"""VectorStore class using sqlite-vector for similarity search."""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

from .types import SearchResult


class VectorStoreError(Exception):
    """Exception raised for VectorStore errors."""

    pass


class VectorStore:
    """SQLite-based vector store using sqlite-vector extension.

    VectorStore provides high-performance vector similarity search using the
    sqlite-vector extension. It supports multiple distance metrics and can
    handle large datasets efficiently through quantization.

    Example:
        >>> with VectorStore(dimension=384) as store:
        ...     store.add([[0.1, 0.2, ...]], ["Hello world"])
        ...     results = store.search([0.1, 0.2, ...], k=5)
        ...     print(results[0].text)

        >>> # Persistent storage
        >>> store = VectorStore(dimension=384, db_path="vectors.db")
        >>> store.add(embeddings, texts, metadata=[{"source": "doc1"}])
        >>> store.close()

        >>> # Re-open existing store
        >>> store = VectorStore.open("vectors.db")
    """

    # Path to sqlite-vector extension (without file extension)
    # The extension is built to: src/cyllama/rag/vector.{dylib,so,dll}
    EXTENSION_PATH = Path(__file__).parent / "vector"

    # Valid distance metrics
    VALID_METRICS = {"cosine", "l2", "dot", "l1", "squared_l2"}

    # Valid vector types (bfloat16 excluded - not supported by sqlite-vector)
    VALID_VECTOR_TYPES = {"float32", "float16", "int8", "uint8"}

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        table_name: str = "embeddings",
        metric: str = "cosine",
        vector_type: str = "float32",
    ):
        """Initialize vector store with sqlite-vector.

        Args:
            dimension: Embedding dimension (must match your embeddings)
            db_path: SQLite database path (":memory:" for in-memory)
            table_name: Name of the embeddings table
            metric: Distance metric: "cosine", "l2", "dot", "l1", "squared_l2"
            vector_type: Vector storage type: "float32", "float16", "int8", "uint8", "bfloat16"

        Raises:
            VectorStoreError: If extension cannot be loaded or invalid parameters
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {self.VALID_METRICS}"
            )

        vector_type_lower = vector_type.lower()
        if vector_type_lower not in self.VALID_VECTOR_TYPES:
            raise ValueError(
                f"Invalid vector_type: {vector_type}. "
                f"Must be one of: {self.VALID_VECTOR_TYPES}"
            )

        self.dimension = dimension
        self.db_path = db_path
        self.table_name = table_name
        self.metric = metric_lower
        self.vector_type = vector_type_lower
        self._quantized = False
        self._closed = False

        # Connect to database
        try:
            self.conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to connect to database: {e}") from e

        # Load sqlite-vector extension
        self._load_extension()

        # Create table and initialize vector search
        self._init_table()

    def _load_extension(self) -> None:
        """Load the sqlite-vector extension."""
        try:
            self.conn.enable_load_extension(True)
            # SQLite load_extension expects path without extension
            ext_path = str(self.EXTENSION_PATH)
            self.conn.load_extension(ext_path)
        except sqlite3.OperationalError as e:
            # Check if extension file exists
            ext_file = self._get_extension_file()
            if not ext_file.exists():
                raise VectorStoreError(
                    f"sqlite-vector extension not found at {ext_file}. "
                    "Run 'scripts/setup.sh' or 'python scripts/manage.py build --sqlite-vector' "
                    "to build it."
                ) from e
            raise VectorStoreError(f"Failed to load sqlite-vector extension: {e}") from e

    def _get_extension_file(self) -> Path:
        """Get the platform-specific extension file path."""
        if sys.platform == "darwin":
            return self.EXTENSION_PATH.with_suffix(".dylib")
        elif sys.platform == "win32":
            return self.EXTENSION_PATH.with_suffix(".dll")
        else:
            return self.EXTENSION_PATH.with_suffix(".so")

    def _init_table(self) -> None:
        """Create table and initialize vector search."""
        # Create table if not exists
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        """)

        # Map metric names to sqlite-vector distance names
        distance_map = {
            "cosine": "COSINE",
            "l2": "L2",
            "squared_l2": "SQUARED_L2",
            "dot": "DOT",
            "l1": "L1",
        }
        distance = distance_map[self.metric]

        # Map vector type names
        type_map = {
            "float32": "FLOAT32",
            "float16": "FLOAT16",
            "int8": "INT8",
            "uint8": "UINT8",
        }
        vtype = type_map[self.vector_type]

        # Initialize vector extension for this table
        try:
            self.conn.execute(f"""
                SELECT vector_init('{self.table_name}', 'embedding',
                    'dimension={self.dimension},type={vtype},distance={distance}')
            """)
            self.conn.commit()
        except sqlite3.OperationalError as e:
            raise VectorStoreError(f"Failed to initialize vector search: {e}") from e

    def _encode_vector(self, vector: list[float]) -> bytes:
        """Encode vector as binary BLOB (Float32).

        Args:
            vector: Vector to encode

        Returns:
            Binary representation of the vector
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
        return struct.pack(f"{len(vector)}f", *vector)

    def _decode_vector(self, blob: bytes) -> list[float]:
        """Decode binary BLOB back to vector.

        Args:
            blob: Binary blob to decode

        Returns:
            Vector as list of floats
        """
        return list(struct.unpack(f"{self.dimension}f", blob))

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add embeddings with associated texts and metadata.

        Args:
            embeddings: List of embedding vectors
            texts: List of text strings (must match embeddings length)
            metadata: Optional list of metadata dicts

        Returns:
            List of generated IDs for the added items

        Raises:
            ValueError: If lengths don't match or vectors have wrong dimension
        """
        self._check_closed()

        if len(embeddings) != len(texts):
            raise ValueError(
                f"embeddings and texts must have same length: "
                f"{len(embeddings)} vs {len(texts)}"
            )

        if metadata is None:
            metadata = [{}] * len(embeddings)
        elif len(metadata) != len(embeddings):
            raise ValueError(
                f"metadata must have same length as embeddings: "
                f"{len(metadata)} vs {len(embeddings)}"
            )

        ids = []
        cursor = self.conn.cursor()
        for emb, text, meta in zip(embeddings, texts, metadata):
            blob = self._encode_vector(emb)
            cursor.execute(
                f"INSERT INTO {self.table_name} (text, embedding, metadata) VALUES (?, ?, ?)",
                (text, blob, json.dumps(meta) if meta else None),
            )
            ids.append(cursor.lastrowid)
        self.conn.commit()

        # Invalidate quantization on new data
        self._quantized = False
        return ids

    def add_one(
        self,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a single embedding.

        Args:
            embedding: Embedding vector
            text: Associated text
            metadata: Optional metadata dict

        Returns:
            Generated ID
        """
        ids = self.add([embedding], [text], [metadata] if metadata else None)
        return ids[0]

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Find k most similar embeddings.

        Uses vector_full_scan() for small datasets or
        vector_quantize_scan() for quantized large datasets.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold (results below are filtered)

        Returns:
            List of SearchResult(id, text, score, metadata) ordered by similarity
        """
        self._check_closed()

        query_blob = self._encode_vector(query_embedding)

        # Use quantized search if available, otherwise full scan
        scan_fn = "vector_quantize_scan" if self._quantized else "vector_full_scan"

        try:
            cursor = self.conn.execute(
                f"""
                SELECT e.id, e.text, e.metadata, v.distance
                FROM {self.table_name} AS e
                JOIN {scan_fn}('{self.table_name}', 'embedding', ?, ?) AS v
                    ON e.id = v.rowid
            """,
                (query_blob, k),
            )
        except sqlite3.OperationalError as e:
            raise VectorStoreError(f"Search failed: {e}") from e

        results = []
        for row in cursor:
            id_, text, meta_json, distance = row

            # Convert distance to similarity score
            if self.metric == "cosine":
                # Cosine distance is 1 - similarity, so similarity = 1 - distance
                score = 1.0 - distance
            elif self.metric == "dot":
                # Dot product: higher is more similar, negate distance
                score = -distance
            else:
                # L2, L1, etc: lower distance = higher similarity
                score = -distance

            if threshold is not None and score < threshold:
                continue

            results.append(
                SearchResult(
                    id=str(id_),
                    text=text,
                    score=score,
                    metadata=json.loads(meta_json) if meta_json else {},
                )
            )

        return results

    def get(self, id: str | int) -> SearchResult | None:
        """Get a single embedding by ID.

        Args:
            id: The embedding ID

        Returns:
            SearchResult or None if not found
        """
        self._check_closed()

        cursor = self.conn.execute(
            f"SELECT id, text, metadata FROM {self.table_name} WHERE id = ?",
            (int(id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        id_, text, meta_json = row
        return SearchResult(
            id=str(id_),
            text=text,
            score=1.0,  # Perfect match
            metadata=json.loads(meta_json) if meta_json else {},
        )

    def get_vector(self, id: str | int) -> list[float] | None:
        """Get the embedding vector for an ID.

        Args:
            id: The embedding ID

        Returns:
            Embedding vector or None if not found
        """
        self._check_closed()

        cursor = self.conn.execute(
            f"SELECT embedding FROM {self.table_name} WHERE id = ?",
            (int(id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._decode_vector(row[0])

    def delete(self, ids: list[str | int]) -> int:
        """Delete embeddings by ID.

        Args:
            ids: List of IDs to delete

        Returns:
            Number of rows deleted
        """
        self._check_closed()

        if not ids:
            return 0

        placeholders = ",".join("?" * len(ids))
        cursor = self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            [int(id_) for id_ in ids],
        )
        self.conn.commit()
        self._quantized = False  # Invalidate quantization
        return cursor.rowcount

    def clear(self) -> int:
        """Delete all embeddings.

        Returns:
            Number of rows deleted
        """
        self._check_closed()

        cursor = self.conn.execute(f"DELETE FROM {self.table_name}")
        self.conn.commit()
        self._quantized = False
        return cursor.rowcount

    def quantize(self, max_memory: str = "30MB") -> int:
        """Quantize vectors for faster approximate search.

        Call this after bulk inserts for datasets >10k vectors.
        Quantized search provides >0.95 recall with 4-5x speedup.

        Args:
            max_memory: Maximum memory for quantization (e.g., "30MB", "100MB")

        Returns:
            Number of quantized rows
        """
        self._check_closed()

        try:
            cursor = self.conn.execute(f"""
                SELECT vector_quantize('{self.table_name}', 'embedding', 'max_memory={max_memory}')
            """)
            count = cursor.fetchone()[0]
            self._quantized = True
            return count
        except sqlite3.OperationalError as e:
            raise VectorStoreError(f"Quantization failed: {e}") from e

    def preload_quantization(self) -> None:
        """Load quantized data into memory for 4-5x speedup.

        Call this after quantize() to preload data into memory.
        """
        self._check_closed()

        try:
            self.conn.execute(f"""
                SELECT vector_quantize_preload('{self.table_name}', 'embedding')
            """)
        except sqlite3.OperationalError as e:
            raise VectorStoreError(f"Preload failed: {e}") from e

    @property
    def is_quantized(self) -> bool:
        """Whether the store has been quantized."""
        return self._quantized

    def _check_closed(self) -> None:
        """Raise error if store is closed."""
        if self._closed:
            raise VectorStoreError("VectorStore is closed")

    def close(self) -> None:
        """Close the database connection."""
        if not self._closed:
            self.conn.close()
            self._closed = True

    @classmethod
    def open(
        cls,
        db_path: str,
        table_name: str = "embeddings",
    ) -> "VectorStore":
        """Open existing vector store from disk.

        Args:
            db_path: Path to SQLite database
            table_name: Name of the embeddings table

        Returns:
            VectorStore instance

        Raises:
            VectorStoreError: If database doesn't exist or table not found
        """
        if not Path(db_path).exists():
            raise VectorStoreError(f"Database not found: {db_path}")

        # Connect to read metadata
        conn = sqlite3.connect(db_path)

        try:
            # Load extension to read vector metadata
            conn.enable_load_extension(True)
            conn.load_extension(str(cls.EXTENSION_PATH))

            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if cursor.fetchone() is None:
                raise VectorStoreError(f"Table '{table_name}' not found in {db_path}")

            # Get a sample embedding to determine dimension
            cursor = conn.execute(f"SELECT embedding FROM {table_name} LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                raise VectorStoreError(f"Table '{table_name}' is empty, cannot determine dimension")

            # Decode to get dimension
            blob = row[0]
            dimension = len(blob) // 4  # float32 = 4 bytes

        finally:
            conn.close()

        # Create store with detected dimension
        # Note: metric and vector_type default to cosine/float32
        # The actual values are stored in sqlite-vector's internal state
        return cls(
            dimension=dimension,
            db_path=db_path,
            table_name=table_name,
            metric="cosine",
            vector_type="float32",
        )

    def __len__(self) -> int:
        """Return number of stored embeddings."""
        self._check_closed()
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def __contains__(self, id: str | int) -> bool:
        """Check if an ID exists in the store."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT 1 FROM {self.table_name} WHERE id = ?",
            (int(id),),
        )
        return cursor.fetchone() is not None

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} vectors"
        return (
            f"VectorStore(dimension={self.dimension}, db_path={self.db_path!r}, "
            f"table_name={self.table_name!r}, metric={self.metric!r}, "
            f"status={status})"
        )
