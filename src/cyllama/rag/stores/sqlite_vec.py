"""sqlite-vec adapter for :class:`~cyllama.rag.types.VectorStoreProtocol`.

An MIT/Apache-2.0 licensed alternative to the vendored ``sqlite-vector``
backend behind :class:`~cyllama.rag.store.SqliteVectorStore`. Ships
behind an optional dependency::

    pip install sqlite-vec

Example:
    >>> from cyllama.rag import RAG
    >>> from cyllama.rag.stores import SqliteVecStore
    >>> store = SqliteVecStore(dimension=384, db_path="vectors.db")
    >>> rag = RAG(embedding_model=..., generation_model=..., store=store)

Schema
------

``sqlite-vec`` exposes vectors through a ``vec0`` *virtual* table, and
SQLite forbids triggers on virtual tables. So rather than putting the
chunk rows inside ``vec0``, this adapter keeps a plain base table and
hangs a ``vec0`` sidecar off it, keyed by the base table's ``id``:

* ``{table_name}``          -- ``id``, ``text``, ``metadata`` (plain table)
* ``{table_name}_vec``      -- ``vec0`` virtual table holding the vectors
* ``{table_name}_meta``     -- stored ``dimension`` / ``metric`` / ``vector_type``
* ``{table_name}_sources``  -- source-dedup records

Keeping the base table plain is what lets callers layer FTS5 triggers
(or any other trigger) over the chunk rows the way
:class:`~cyllama.rag.advanced.HybridStore` does on the default backend.

Differences from :class:`~cyllama.rag.store.SqliteVectorStore`
--------------------------------------------------------------

* No ``dot`` metric and no ``uint8`` vector type -- ``vec0`` offers
  cosine / L2 / L1 and float32 / int8. (``vec0`` also parses a
  ``float16`` column type, but as of sqlite-vec 0.1.9 it stores those
  columns as float32 -- ``vec_type()`` reports ``float32`` and each
  element still occupies 4 bytes -- so this adapter does not offer it
  rather than promise a halving that doesn't happen.)
* ``squared_l2`` is served by asking ``vec0`` for L2 and squaring the
  result (monotonic, so the ordering is identical).
* No ``quantize()`` / ``preload_quantization()``. A ``vec0`` table is
  either exhaustive or built with an ANN index at CREATE time; there is
  no runtime quantization step to call.
* ``vec0`` returns SQL NULL for the cosine distance to a zero-norm
  vector (undefined); such rows are ranked last rather than dropped.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import SearchResult, VectorStoreProtocol

# vec0 has no dot-product metric and no squared-L2 metric. squared_l2 is
# served by requesting L2 and squaring the distance in Python.
_METRIC_TO_VEC0 = {
    "cosine": "cosine",
    "l2": "L2",
    "squared_l2": "L2",
    "l1": "L1",
}

# vec0 element types. Each entry is (vec0 column type, struct format
# code, SQL expression wrapping the bound blob).
#
# Unlike sqlite-vector -- which accepted a float32 blob for every column
# type and converted internally -- vec0 validates the blob against the
# column's element type. A bare BLOB is always read as float32, so an
# int8 vector has to be tagged with vec_int8() on both insert and query.
_VECTOR_TYPE_TO_VEC0 = {
    "float32": ("float", "f", "?"),
    "int8": ("int8", "b", "vec_int8(?)"),
}

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Reject table names that can't be safely interpolated into SQL.

    Table and column names can't be bound as parameters, so they are
    interpolated. Restricting them to plain identifiers keeps that safe.
    """
    if not _VALID_TABLE_NAME.match(name):
        raise ValueError(
            f"Invalid table name: {name!r}. Must be a valid SQL identifier "
            "(letters, digits and underscores, not starting with a digit)."
        )


def _resolve_extension_path(extension_path: str | Path | None) -> str:
    """Return the loadable-extension path, without its file suffix.

    An explicit ``extension_path`` wins so callers can point at a build
    of their own. Otherwise fall back to the ``sqlite-vec`` PyPI
    package, which ships prebuilt binaries for the common platforms.
    """
    if extension_path is not None:
        return str(Path(extension_path).with_suffix(""))
    try:
        import sqlite_vec
    except ImportError as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "sqlite-vec is required for SqliteVecStore. Install with: pip install sqlite-vec "
            "(or pass extension_path=... to point at your own build of the extension)."
        ) from e
    return str(sqlite_vec.loadable_path())


class SqliteVecStore(VectorStoreProtocol):
    """sqlite-vec backed :class:`VectorStoreProtocol` implementation.

    Args:
        dimension: Embedding dimension. Must match the embedder.
        db_path: SQLite database path; ``":memory:"`` for ephemeral.
        table_name: Base table name; the sidecar tables derive from it.
        metric: One of ``cosine``, ``l2``, ``squared_l2``, ``l1``.
        vector_type: One of ``float32``, ``int8``.
        extension_path: Optional path to a ``vec0`` loadable extension,
            with or without its file suffix. Defaults to the copy
            shipped by the ``sqlite-vec`` PyPI package.

    Raises:
        ValueError: On an invalid dimension, table name, metric or
            vector type.
        ImportError: When neither ``extension_path`` nor the
            ``sqlite-vec`` package is available.
        VectorStoreError: When the extension can't be loaded, or when
            reopening a database whose stored configuration conflicts
            with the arguments passed here.
    """

    VALID_METRICS = frozenset(_METRIC_TO_VEC0.keys())
    VALID_VECTOR_TYPES = frozenset(_VECTOR_TYPE_TO_VEC0.keys())

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        table_name: str = "embeddings",
        metric: str = "cosine",
        vector_type: str = "float32",
        *,
        extension_path: str | Path | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        _validate_table_name(table_name)

        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric!r}. Must be one of: {sorted(self.VALID_METRICS)}")

        vector_type_lower = vector_type.lower()
        if vector_type_lower not in self.VALID_VECTOR_TYPES:
            raise ValueError(f"Invalid vector_type: {vector_type!r}. Must be one of: {sorted(self.VALID_VECTOR_TYPES)}")

        self.dimension = dimension
        self.db_path = db_path
        self.table_name = table_name
        self.metric = metric_lower
        self.vector_type = vector_type_lower
        self._closed = False

        # Imported lazily so `import cyllama.rag` stays free of the
        # dependency on the default (sqlite-vector) backend's error type.
        from ..store import VectorStoreError

        self._error = VectorStoreError

        self._extension_path = _resolve_extension_path(extension_path)

        try:
            self.conn = sqlite3.connect(db_path, timeout=10)
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to connect to database: {e}") from e

        self._load_extension()
        self._init_tables()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def vec_table(self) -> str:
        """Name of the ``vec0`` sidecar table holding the vectors."""
        return f"{self.table_name}_vec"

    @property
    def meta_table(self) -> str:
        """Name of the table holding this store's stored configuration."""
        return f"{self.table_name}_meta"

    @property
    def sources_table(self) -> str:
        """Name of the table holding source-dedup records."""
        return f"{self.table_name}_sources"

    def _load_extension(self) -> None:
        if not hasattr(self.conn, "enable_load_extension"):
            raise self._error(
                "Python was built without SQLite extension loading support. "
                "Rebuild Python with --enable-loadable-sqlite-extensions."
            )
        try:
            self.conn.enable_load_extension(True)
            # SQLite derives the entrypoint symbol from the filename,
            # but the symbol is named for the upstream artifact (vec0),
            # so name it explicitly instead.
            # The entrypoint kwarg landed in Python 3.12 (which cyllama
            # already requires) but mypy's stdlib stubs don't carry it.
            self.conn.load_extension(  # type: ignore[call-arg]
                self._extension_path, entrypoint="sqlite3_vec_init"
            )
        except sqlite3.OperationalError as e:
            raise self._error(f"Failed to load sqlite-vec extension from {self._extension_path!r}: {e}") from e
        finally:
            self.conn.enable_load_extension(False)

    def _init_tables(self) -> None:
        vec0_type = _VECTOR_TYPE_TO_VEC0[self.vector_type][0]
        distance = _METRIC_TO_VEC0[self.metric]

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.meta_table} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.sources_table} (
                content_hash TEXT PRIMARY KEY,
                source_label TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL
            )
        """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.sources_table}_label_idx
            ON {self.sources_table}(source_label)
        """)

        stored = {row[0]: row[1] for row in self.conn.execute(f"SELECT key, value FROM {self.meta_table}")}
        if stored:
            self._verify_compatibility(stored)

        for key, value in (
            ("dimension", str(self.dimension)),
            ("metric", self.metric),
            ("vector_type", self.vector_type),
        ):
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.meta_table} (key, value) VALUES (?, ?)",
                (key, value),
            )
        self.conn.commit()

        try:
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.vec_table} USING vec0(
                    embedding {vec0_type}[{self.dimension}] distance_metric={distance}
                )
            """)
            self.conn.commit()
        except sqlite3.OperationalError as e:
            raise self._error(f"Failed to initialize vec0 table: {e}") from e

    def _verify_compatibility(self, stored: dict[str, str]) -> None:
        """Refuse to reopen a database under a conflicting configuration.

        Mixing dimensions, metrics or element types in one ``vec0``
        table would silently corrupt the index, so these are hard
        errors rather than warnings.
        """
        checks = (
            ("dimension", str(self.dimension)),
            ("metric", self.metric),
            ("vector_type", self.vector_type),
        )
        for key, current in checks:
            previous = stored.get(key)
            if previous is not None and previous != current:
                raise self._error(
                    f"Database {self.db_path!r} was created with {key}={previous!r} "
                    f"but is being opened with {key}={current!r}. Reopen with the "
                    f"original {key} or recreate the store."
                )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_vector(self, vector: list[float]) -> bytes:
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        _, code, _ = _VECTOR_TYPE_TO_VEC0[self.vector_type]
        if code == "b":
            return struct.pack(f"{len(vector)}b", *(int(round(x)) for x in vector))
        return struct.pack(f"{len(vector)}{code}", *vector)

    def _decode_vector(self, blob: bytes) -> list[float]:
        _, code, _ = _VECTOR_TYPE_TO_VEC0[self.vector_type]
        return [float(x) for x in struct.unpack(f"{self.dimension}{code}", blob)]

    @property
    def _vector_expr(self) -> str:
        """SQL expression that tags a bound blob with its element type."""
        return _VECTOR_TYPE_TO_VEC0[self.vector_type][2]

    def _check_closed(self) -> None:
        if self._closed:
            raise self._error("SqliteVecStore is closed")

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> list[int]:
        """Insert chunks and their vectors; return the assigned IDs.

        The base row, the sidecar vector and the source-dedup record are
        written in one transaction, so a failed source insert (e.g. a
        duplicate ``source_hash``) rolls the chunks back with it.
        """
        self._check_closed()

        if len(embeddings) != len(texts):
            raise ValueError(f"embeddings and texts must have same length: {len(embeddings)} vs {len(texts)}")
        if source_hash is not None and source_label is None:
            raise ValueError("source_hash requires source_label")

        if metadata is None:
            metadata = [{} for _ in embeddings]
        elif len(metadata) != len(embeddings):
            raise ValueError(f"metadata must have same length as embeddings: {len(metadata)} vs {len(embeddings)}")

        for i, meta in enumerate(metadata):
            if meta:
                try:
                    json.dumps(meta)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Metadata at index {i} is not JSON-serializable: {e}") from e

        ids: list[int] = []
        try:
            with self.conn:
                cursor = self.conn.cursor()
                for emb, text, meta in zip(embeddings, texts, metadata):
                    blob = self._encode_vector(emb)
                    cursor.execute(
                        f"INSERT INTO {self.table_name} (text, metadata) VALUES (?, ?)",
                        (text, json.dumps(meta) if meta else None),
                    )
                    rowid = cursor.lastrowid or 0
                    cursor.execute(
                        f"INSERT INTO {self.vec_table} (rowid, embedding) VALUES (?, {self._vector_expr})",
                        (rowid, blob),
                    )
                    ids.append(rowid)

                if source_hash is not None:
                    cursor.execute(
                        f"INSERT INTO {self.sources_table} "
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
            # The transaction rolled back, so the local `ids` no longer
            # correspond to anything in the database. Clear them rather
            # than hand the caller phantom IDs.
            ids = []
            raise

        return ids

    def add_one(
        self,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a single embedding and return its ID."""
        return self.add([embedding], [text], [metadata] if metadata else None)[0]

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Return the ``k`` nearest chunks, best match first."""
        self._check_closed()

        query_blob = self._encode_vector(query_embedding)

        try:
            cursor = self.conn.execute(
                f"""
                SELECT e.id, e.text, e.metadata, v.distance
                FROM {self.table_name} AS e
                JOIN (
                    SELECT rowid, distance FROM {self.vec_table}
                    WHERE embedding MATCH {self._vector_expr} AND k = ?
                ) AS v ON e.id = v.rowid
                -- SQLite sorts NULL first; vec0 reports NULL for the
                -- cosine distance to a zero-norm vector, and those rows
                -- belong last, not first.
                ORDER BY v.distance IS NULL, v.distance
                """,
                (query_blob, k),
            )
        except sqlite3.OperationalError as e:
            raise self._error(f"Search failed: {e}") from e

        results: list[SearchResult] = []
        for id_, text, meta_json, distance in cursor:
            if distance is None:
                # Cosine distance to a zero-norm vector is undefined and
                # vec0 reports NULL. Rank the row last instead of
                # letting it crash the score arithmetic.
                distance = 1.0 if self.metric == "cosine" else float("inf")
            if self.metric == "squared_l2":
                distance = distance * distance

            if self.metric == "cosine":
                score = 1.0 - distance
            else:
                # L1/L2/squared_L2: smaller distance is a better match.
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

    def is_source_indexed(self, content_hash: str) -> bool:
        """Return True if a source with this content hash was added."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT 1 FROM {self.sources_table} WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return cursor.fetchone() is not None

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        """Look up an indexed source by its human-readable label."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT content_hash, source_label, chunk_count, indexed_at "
            f"FROM {self.sources_table} WHERE source_label = ? LIMIT 1",
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
        """Return every indexed source, oldest first."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT content_hash, source_label, chunk_count, indexed_at FROM {self.sources_table} ORDER BY indexed_at"
        )
        return [
            {
                "content_hash": row[0],
                "source_label": row[1],
                "chunk_count": row[2],
                "indexed_at": row[3],
            }
            for row in cursor
        ]

    def get(self, id: str | int) -> SearchResult | None:
        """Fetch a single chunk by ID, or None when it doesn't exist."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT id, text, metadata FROM {self.table_name} WHERE id = ?",
            (int(id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return SearchResult(
            id=str(row[0]),
            text=row[1],
            score=1.0,
            metadata=json.loads(row[2]) if row[2] else {},
        )

    def get_vector(self, id: str | int) -> list[float] | None:
        """Fetch a stored embedding by ID, or None when it doesn't exist."""
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT embedding FROM {self.vec_table} WHERE rowid = ?",
            (int(id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._decode_vector(row[0])

    def delete(self, ids: list[str | int]) -> int:
        """Delete chunks by ID; return the number of rows removed."""
        self._check_closed()
        if not ids:
            return 0
        int_ids = [int(id_) for id_ in ids]
        placeholders = ",".join("?" * len(int_ids))
        cursor = self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            int_ids,
        )
        self.conn.execute(
            f"DELETE FROM {self.vec_table} WHERE rowid IN ({placeholders})",
            int_ids,
        )
        self.conn.commit()
        return cursor.rowcount

    def clear(self) -> int:
        """Remove every chunk, vector and source record."""
        self._check_closed()
        cursor = self.conn.execute(f"DELETE FROM {self.table_name}")
        self.conn.execute(f"DELETE FROM {self.vec_table}")
        self.conn.execute(f"DELETE FROM {self.sources_table}")
        self.conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __len__(self) -> int:
        self._check_closed()
        count: int = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        return count

    def __contains__(self, id: str | int) -> bool:
        self._check_closed()
        cursor = self.conn.execute(
            f"SELECT 1 FROM {self.table_name} WHERE id = ?",
            (int(id),),
        )
        return cursor.fetchone() is not None

    def __enter__(self) -> "SqliteVecStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} vectors"
        return (
            f"SqliteVecStore(dimension={self.dimension}, "
            f"db_path={self.db_path!r}, table_name={self.table_name!r}, "
            f"metric={self.metric!r}, vector_type={self.vector_type!r}, status={status})"
        )
