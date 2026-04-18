"""VectorStore class using sqlite-vector for similarity search."""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

from .types import SearchResult

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Validate that a table name is a safe SQL identifier."""
    if not _VALID_TABLE_NAME.match(name):
        raise ValueError(
            f"Invalid table name: {name!r}. "
            "Must start with a letter or underscore and contain only alphanumerics and underscores."
        )


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
        embedding_model_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize vector store with sqlite-vector.

        Args:
            dimension: Embedding dimension (must match your embeddings)
            db_path: SQLite database path (":memory:" for in-memory)
            table_name: Name of the embeddings table
            metric: Distance metric: "cosine", "l2", "dot", "l1", "squared_l2"
            vector_type: Vector storage type: "float32", "float16", "int8", "uint8", "bfloat16"
            embedding_model_path: Optional path to the embedding model GGUF
                used to populate this store. If provided, the model's
                basename and file size are recorded in the metadata table
                so reopens against the same DB can verify the user is
                using the same (or at least size-compatible) embedding
                model. Mismatch raises ``VectorStoreError`` with a
                friendly message rather than silently producing garbage.
            chunk_size: Optional chunk size used to populate this store.
                If provided, recorded in metadata so reopens can detect
                a chunking-config mismatch.
            chunk_overlap: Optional chunk overlap used to populate this
                store. Same purpose as ``chunk_size``.

        Raises:
            VectorStoreError: If extension cannot be loaded, invalid
                parameters, or the database already exists with
                incompatible metadata (different dimension, metric,
                vector_type, embedding model, or chunking config).
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        _validate_table_name(table_name)

        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of: {self.VALID_METRICS}")

        vector_type_lower = vector_type.lower()
        if vector_type_lower not in self.VALID_VECTOR_TYPES:
            raise ValueError(f"Invalid vector_type: {vector_type}. Must be one of: {self.VALID_VECTOR_TYPES}")

        self.dimension = dimension
        self.db_path = db_path
        self.table_name = table_name
        self.metric = metric_lower
        self.vector_type = vector_type_lower
        self._quantized = False
        self._closed = False

        # Compute the embedding-model fingerprint up front so we can
        # both verify it on reopen AND store it on first init. We use
        # basename + file size in bytes as the fingerprint:
        # - basename catches the obvious "user switched models" case
        # - file size catches "user re-quantized to a different bit
        #   width" without requiring a multi-GB content hash
        # Together they detect the realistic mismatch cases without
        # being expensive to compute.
        self._embedding_model_path = embedding_model_path
        self._embedding_model_basename: str | None = None
        self._embedding_model_size_bytes: int | None = None
        if embedding_model_path is not None:
            import os as _os

            self._embedding_model_basename = _os.path.basename(embedding_model_path)
            try:
                self._embedding_model_size_bytes = _os.path.getsize(embedding_model_path)
            except OSError:
                # Path doesn't exist or isn't readable -- store basename
                # only. The compatibility check will be looser but
                # we shouldn't refuse to construct the store.
                self._embedding_model_size_bytes = None

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Connect to database
        try:
            self.conn = sqlite3.connect(db_path, timeout=10)
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to connect to database: {e}") from e

        # Load sqlite-vector extension
        self._load_extension()

        # Create table and initialize vector search. This is the point
        # where, on reopen of an existing populated DB, the metadata
        # compatibility check fires and may raise.
        self._init_table()

    def _load_extension(self) -> None:
        """Load the sqlite-vector extension."""
        try:
            if not hasattr(self.conn, "enable_load_extension"):
                raise VectorStoreError(
                    "Python was built without SQLite extension loading support. "
                    "Rebuild Python with --enable-loadable-sqlite-extensions."
                )
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
        """Create table and initialize vector search.

        On a fresh DB this writes the configuration to the
        ``{table_name}_meta`` table. On reopen of an existing
        populated DB this reads the stored configuration and
        verifies it matches what the caller passed in -- mismatch
        raises ``VectorStoreError`` rather than silently corrupting
        the index by mixing two configurations.
        """
        # Create the data table if it doesn't already exist. CREATE
        # TABLE IF NOT EXISTS makes the reopen case a no-op.
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        """)

        meta_table = f"{self.table_name}_meta"
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {meta_table} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Source-deduplication table. Records (content_hash, label,
        # chunk_count, indexed_at) for every source that has ever been
        # added to this store, so future add() calls can detect a
        # source that was already indexed and skip it instead of
        # producing duplicate chunks. The hash is the user's choice
        # (typically md5 of the file bytes for add_documents or
        # md5 of the text bytes for add_texts); VectorStore just
        # stores and queries the values.
        sources_table = f"{self.table_name}_sources"
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {sources_table} (
                content_hash TEXT PRIMARY KEY,
                source_label TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL
            )
        """)
        # Secondary index on the label so the "same basename,
        # different content" check is fast even for large source lists.
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {sources_table}_label_idx
            ON {sources_table}(source_label)
        """)

        # Read whatever metadata is already in the DB. On a fresh DB
        # this returns an empty dict and we fall through to writing
        # ours. On a reopen this returns the stored config and we
        # validate before writing anything.
        stored = {row[0]: row[1] for row in self.conn.execute(f"SELECT key, value FROM {meta_table}")}

        if stored:
            self._verify_compatibility(stored)

        # Either fresh DB (stored was empty) or compatible reopen
        # (verify_compatibility returned without raising). Write/update
        # the metadata fields. INSERT OR REPLACE is correct here
        # because all the values we're about to write either match
        # what's already there (compatible reopen) or are new (fresh
        # DB / new metadata fields added in a later cyllama version).
        from datetime import datetime, timezone

        try:
            from .. import __version__ as _cyllama_version
        except ImportError:  # pragma: no cover - extremely defensive
            _cyllama_version = "unknown"

        meta_to_write = {
            "metric": self.metric,
            "vector_type": self.vector_type,
            "dimension": str(self.dimension),
            "cyllama_version": str(_cyllama_version),
        }
        # Only write fingerprints if the caller provided them. We
        # don't want to overwrite a populated DB's fingerprint with
        # NULL just because someone called the constructor without
        # the new params. The fields are individually optional.
        if self._embedding_model_basename is not None:
            meta_to_write["embedding_model_basename"] = self._embedding_model_basename
        if self._embedding_model_size_bytes is not None:
            meta_to_write["embedding_model_size_bytes"] = str(self._embedding_model_size_bytes)
        if self._chunk_size is not None:
            meta_to_write["chunk_size"] = str(self._chunk_size)
        if self._chunk_overlap is not None:
            meta_to_write["chunk_overlap"] = str(self._chunk_overlap)
        # Set created_at only on a fresh DB so it remains the original
        # creation timestamp on reopens. Use UTC for portability.
        if "created_at" not in stored:
            meta_to_write["created_at"] = datetime.now(timezone.utc).isoformat()

        for key, value in meta_to_write.items():
            self.conn.execute(
                f"INSERT OR REPLACE INTO {meta_table} (key, value) VALUES (?, ?)",
                (key, value),
            )
        self.conn.commit()

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

    def _verify_compatibility(self, stored: dict[str, str]) -> None:
        """Verify the caller's configuration matches what's already in
        the database, raising ``VectorStoreError`` on mismatch.

        Called only when reopening an existing populated DB. The
        comparison rules:

        * **Hard mismatches** (raise immediately) -- dimension, metric,
          vector_type. These would corrupt the index if the caller
          tried to add new vectors with the wrong shape or compare
          old vectors with the wrong distance.
        * **Soft mismatches** (raise unless the caller passed
          ``None`` for the field) -- ``embedding_model_basename``,
          ``embedding_model_size_bytes``, ``chunk_size``,
          ``chunk_overlap``. These would silently produce bad results
          (different embedder vectors aren't comparable; different
          chunking produces inconsistent retrieval) but the caller
          can opt out of the check by simply not passing the field
          to the constructor.

        The error message for each case names the stored value, the
        attempted value, and a one-line hint about how to fix.
        """
        # Hard checks: these always fire because the caller always
        # passes them (or constructor defaults).
        stored_dimension = stored.get("dimension")
        if stored_dimension is not None and int(stored_dimension) != self.dimension:
            raise VectorStoreError(
                f"VectorStore at {self.db_path!r} was created with "
                f"dimension={stored_dimension} but the caller is opening "
                f"it with dimension={self.dimension}. This usually means "
                f"you switched to a different embedding model. Either "
                f"use the original embedding model, point at a different "
                f"--db, or pass --rebuild to recreate the index from "
                f"scratch."
            )

        stored_metric = stored.get("metric")
        if stored_metric is not None and stored_metric != self.metric:
            raise VectorStoreError(
                f"VectorStore at {self.db_path!r} was created with "
                f"metric={stored_metric!r} but the caller is opening "
                f"it with metric={self.metric!r}. Distance metrics are "
                f"not interchangeable -- vectors stored under one metric "
                f"would produce wrong rankings under another. Either "
                f"use the original metric or pass --rebuild to recreate."
            )

        stored_vector_type = stored.get("vector_type")
        if stored_vector_type is not None and stored_vector_type != self.vector_type:
            raise VectorStoreError(
                f"VectorStore at {self.db_path!r} was created with "
                f"vector_type={stored_vector_type!r} but the caller is "
                f"opening it with vector_type={self.vector_type!r}. "
                f"Vector storage formats are not interchangeable -- "
                f"the encoded BLOBs are different sizes. Either use "
                f"the original vector_type or pass --rebuild to recreate."
            )

        # Soft checks: only fire if the caller actually provided the
        # field. A caller that doesn't pass embedding_model_path /
        # chunk_size / chunk_overlap explicitly opts out of the check.
        if self._embedding_model_basename is not None:
            stored_basename = stored.get("embedding_model_basename")
            if stored_basename is not None and stored_basename != self._embedding_model_basename:
                raise VectorStoreError(
                    f"VectorStore at {self.db_path!r} was created with "
                    f"embedding model {stored_basename!r} but the caller "
                    f"is opening it with {self._embedding_model_basename!r}. "
                    f"Vectors from two different embedding models live "
                    f"in different vector spaces and cannot be compared. "
                    f"Either use the original embedding model or pass "
                    f"--rebuild to recreate the index."
                )

        if self._embedding_model_size_bytes is not None:
            stored_size = stored.get("embedding_model_size_bytes")
            if stored_size is not None and int(stored_size) != self._embedding_model_size_bytes:
                raise VectorStoreError(
                    f"VectorStore at {self.db_path!r} was created with "
                    f"an embedding model file of {stored_size} bytes, "
                    f"but the current embedding model file is "
                    f"{self._embedding_model_size_bytes} bytes. The "
                    f"basename matches but the file size differs, which "
                    f"usually means a different quantization (e.g. q4 vs "
                    f"q8) or a different release of the same model. The "
                    f"vectors are not necessarily compatible. Either use "
                    f"the original file, or pass --rebuild to recreate "
                    f"the index against the new file."
                )

        if self._chunk_size is not None:
            stored_chunk_size = stored.get("chunk_size")
            if stored_chunk_size is not None and int(stored_chunk_size) != self._chunk_size:
                raise VectorStoreError(
                    f"VectorStore at {self.db_path!r} was created with "
                    f"chunk_size={stored_chunk_size} but the caller is "
                    f"opening it with chunk_size={self._chunk_size}. "
                    f"Mixing chunk sizes in one index produces inconsistent "
                    f"retrieval. Either use the original chunk_size or "
                    f"pass --rebuild to recreate the index."
                )

        if self._chunk_overlap is not None:
            stored_chunk_overlap = stored.get("chunk_overlap")
            if stored_chunk_overlap is not None and int(stored_chunk_overlap) != self._chunk_overlap:
                raise VectorStoreError(
                    f"VectorStore at {self.db_path!r} was created with "
                    f"chunk_overlap={stored_chunk_overlap} but the caller "
                    f"is opening it with chunk_overlap={self._chunk_overlap}. "
                    f"Mixing chunk overlaps in one index produces "
                    f"inconsistent retrieval. Either use the original "
                    f"chunk_overlap or pass --rebuild to recreate the index."
                )

    def _encode_vector(self, vector: list[float]) -> bytes:
        """Encode vector as binary BLOB (Float32).

        Args:
            vector: Vector to encode

        Returns:
            Binary representation of the vector
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        return struct.pack(f"{len(vector)}f", *vector)

    def _decode_vector(self, blob: bytes) -> list[float]:
        """Decode binary BLOB back to vector.

        Args:
            blob: Binary blob to decode

        Returns:
            Vector as list of floats
        """
        return list(struct.unpack(f"{self.dimension}f", blob))

    # ------------------------------------------------------------------
    # Source-deduplication queries (the records are written by add()
    # when the source_hash kwarg is passed; these methods read them
    # back so the higher-level RAG layer can decide whether to skip
    # an already-indexed source).
    # ------------------------------------------------------------------

    def is_source_indexed(self, content_hash: str) -> bool:
        """Return True if a source with this content hash has already
        been added to the store via ``add(source_hash=...)``."""
        self._check_closed()
        sources_table = f"{self.table_name}_sources"
        cursor = self.conn.execute(
            f"SELECT 1 FROM {sources_table} WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return cursor.fetchone() is not None

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        """Look up a source by its label (e.g. file basename) and
        return the stored row, or None if no source with that label
        has been indexed.

        Used by the RAG layer to detect "same basename, different
        content" collisions: if a label is already in the table but
        the caller's content hash differs, we know the file content
        changed and the user needs to either rename or rebuild.
        """
        self._check_closed()
        sources_table = f"{self.table_name}_sources"
        cursor = self.conn.execute(
            f"SELECT content_hash, source_label, chunk_count, indexed_at "
            f"FROM {sources_table} WHERE source_label = ? LIMIT 1",
            (source_label,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "content_hash": row[0],
            "source_label": row[1],
            "chunk_count": row[2],
            "indexed_at": row[3],
        }

    def list_sources(self) -> list[dict[str, Any]]:
        """Return all source records ordered by indexing time.

        Useful for ``cyllama rag --db PATH --list-sources``-style
        diagnostics and for tests that want to assert what's in the
        dedup table without poking at the schema directly.
        """
        self._check_closed()
        sources_table = f"{self.table_name}_sources"
        cursor = self.conn.execute(
            f"SELECT content_hash, source_label, chunk_count, indexed_at FROM {sources_table} ORDER BY indexed_at"
        )
        return [
            {
                "content_hash": row[0],
                "source_label": row[1],
                "chunk_count": row[2],
                "indexed_at": row[3],
            }
            for row in cursor.fetchall()
        ]

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> list[int]:
        """Add embeddings with associated texts and metadata.

        Args:
            embeddings: List of embedding vectors
            texts: List of text strings (must match embeddings length)
            metadata: Optional list of metadata dicts
            source_hash: Optional content hash (typically md5 hex digest)
                identifying the source these chunks came from. When
                provided, an entry is recorded in the
                ``{table_name}_sources`` table in the same commit as
                the chunk inserts, so the chunks-and-source pair is
                atomically visible. Callers use this together with
                :meth:`is_source_indexed` to skip re-indexing a source
                whose content hasn't changed.
            source_label: Optional human-readable label for the source
                (e.g. file basename). Required when ``source_hash`` is
                provided. Stored as-is for display in
                :meth:`list_sources` and used to detect "same basename,
                different content" collisions one layer up.

        Returns:
            List of generated IDs for the added items

        Raises:
            ValueError: If lengths don't match, vectors have wrong
                dimension, or ``source_hash`` is provided without
                ``source_label``.
        """
        self._check_closed()

        if len(embeddings) != len(texts):
            raise ValueError(f"embeddings and texts must have same length: {len(embeddings)} vs {len(texts)}")

        if source_hash is not None and source_label is None:
            raise ValueError("source_hash requires source_label")

        if metadata is None:
            metadata = [{} for _ in range(len(embeddings))]
        elif len(metadata) != len(embeddings):
            raise ValueError(f"metadata must have same length as embeddings: {len(metadata)} vs {len(embeddings)}")

        # Pre-validate metadata serializability to give clear errors
        for i, meta in enumerate(metadata):
            if meta:
                try:
                    json.dumps(meta)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Metadata at index {i} is not JSON-serializable: {e}") from e

        # Use `with self.conn:` to get transactional semantics: commit
        # on successful exit, rollback on any exception. This is what
        # makes chunks-and-source atomic -- if the source INSERT
        # raises (e.g. PRIMARY KEY violation on a duplicate hash),
        # the chunk inserts above are also rolled back. Without this,
        # a failed source insert would leave the store with orphaned
        # chunks that the dedup table doesn't know about.
        ids: list[int] = []
        try:
            with self.conn:
                cursor = self.conn.cursor()
                for emb, text, meta in zip(embeddings, texts, metadata):
                    blob = self._encode_vector(emb)
                    cursor.execute(
                        f"INSERT INTO {self.table_name} (text, embedding, metadata) VALUES (?, ?, ?)",
                        (text, blob, json.dumps(meta) if meta else None),
                    )
                    ids.append(cursor.lastrowid or 0)

                if source_hash is not None:
                    from datetime import datetime, timezone

                    sources_table = f"{self.table_name}_sources"
                    cursor.execute(
                        f"INSERT INTO {sources_table} "
                        f"(content_hash, source_label, chunk_count, indexed_at) "
                        f"VALUES (?, ?, ?, ?)",
                        (
                            source_hash,
                            source_label,
                            len(embeddings),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except sqlite3.IntegrityError:
            # The `with self.conn:` block already rolled back, but
            # the local `ids` list is now misleading because the
            # chunks aren't actually in the DB. Clear it so the
            # caller doesn't see phantom IDs.
            ids = []
            raise

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
            count: int = cursor.fetchone()[0]
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

        _validate_table_name(table_name)

        # Connect to read metadata
        conn = sqlite3.connect(db_path, timeout=10)

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

            # Read stored configuration from metadata table
            meta_table = f"{table_name}_meta"
            metric = "cosine"
            vector_type = "float32"
            dimension = None

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (meta_table,),
            )
            if cursor.fetchone() is not None:
                # Read from metadata table
                for row in conn.execute(f"SELECT key, value FROM {meta_table}"):
                    if row[0] == "metric":
                        metric = row[1]
                    elif row[0] == "vector_type":
                        vector_type = row[1]
                    elif row[0] == "dimension":
                        dimension = int(row[1])

            # Fall back to inferring dimension from embedding data
            if dimension is None:
                cursor = conn.execute(f"SELECT embedding FROM {table_name} LIMIT 1")
                row = cursor.fetchone()
                if row is None:
                    raise VectorStoreError(f"Table '{table_name}' is empty, cannot determine dimension")
                blob = row[0]
                dimension = len(blob) // 4  # float32 = 4 bytes

        finally:
            conn.close()

        return cls(
            dimension=dimension,
            db_path=db_path,
            table_name=table_name,
            metric=metric,
            vector_type=vector_type,
        )

    def __len__(self) -> int:
        """Return number of stored embeddings."""
        self._check_closed()
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count: int = cursor.fetchone()[0]
        return count

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

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} vectors"
        return (
            f"VectorStore(dimension={self.dimension}, db_path={self.db_path!r}, "
            f"table_name={self.table_name!r}, metric={self.metric!r}, "
            f"status={status})"
        )
