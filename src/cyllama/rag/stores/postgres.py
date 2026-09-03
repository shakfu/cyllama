"""pgvector adapter for :class:`~cyllama.rag.types.VectorStoreProtocol`.

Ships behind two optional dependencies::

    pip install "psycopg[binary]" pgvector

Example:
    >>> from cyllama.rag import RAG
    >>> from cyllama.rag.stores import PgVectorStore
    >>> store = PgVectorStore(dimension=384, dsn="postgresql://localhost/rag")
    >>> rag = RAG(embedding_model=..., generation_model=..., store=store)

Unlike the other adapters, pgvector has no in-process mode: it is a
PostgreSQL extension, so a reachable server with ``CREATE EXTENSION
vector`` available is a hard requirement. Pass either ``dsn=`` (a libpq
connection string or URI) or ``conn=`` (a ``psycopg`` connection the
caller owns and closes).

Schema
------

* ``{table_name}``          -- ``id``, ``text``, ``embedding vector(N)``, ``metadata JSONB``
* ``{table_name}_meta``     -- stored ``dimension`` / ``metric``
* ``{table_name}_sources``  -- source-dedup records

Metadata is stored as native ``JSONB``, so arbitrary JSON-serializable
values round-trip with no encoding tricks -- no scalar-only restriction
of the kind Chroma imposes.

Distance metrics
----------------

pgvector supplies an operator per metric, so this adapter covers the
full set the default :class:`~cyllama.rag.store.SqliteVectorStore`
offers -- including the ``dot`` that sqlite-vec lacks:

==============  ==========  ====================================
``metric``      Operator    Notes
==============  ==========  ====================================
``cosine``      ``<=>``     cosine distance
``l2``          ``<->``     euclidean distance
``squared_l2``  ``<->``     squared in Python; ordering identical
``dot``         ``<#>``     negative inner product
``l1``          ``<+>``     requires pgvector >= 0.7.0
==============  ==========  ====================================

``l1`` is checked against the server's installed pgvector version at
construction, since ``<+>`` simply does not exist on older ones and the
failure would otherwise surface as an opaque "operator does not exist"
at query time.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ..types import SearchResult, VectorStoreProtocol

# Operator per metric, plus whether the score is derived by negating the
# distance (true for every metric where smaller means closer) and the
# minimum pgvector version providing the operator.
_METRIC_TO_OPERATOR = {
    "cosine": ("<=>", (0, 5, 0)),
    "l2": ("<->", (0, 5, 0)),
    "squared_l2": ("<->", (0, 5, 0)),
    "dot": ("<#>", (0, 5, 0)),
    "l1": ("<+>", (0, 7, 0)),
}

# HNSW / IVFFlat operator classes, keyed by metric. squared_l2 shares
# l2's index -- it is the same ordering.
_METRIC_TO_OPCLASS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "squared_l2": "vector_l2_ops",
    "dot": "vector_ip_ops",
    "l1": "vector_l1_ops",
}

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Reject table names that aren't plain identifiers.

    Identifiers are quoted with ``psycopg.sql.Identifier`` everywhere
    they're used, so this is belt-and-braces -- but it turns a
    surprising quoted-identifier table into an early, clear error.
    """
    if not _VALID_TABLE_NAME.match(name):
        raise ValueError(
            f"Invalid table name: {name!r}. Must be a valid SQL identifier "
            "(letters, digits and underscores, not starting with a digit)."
        )


def _require_psycopg() -> tuple[Any, Any, Any, Any]:
    try:
        import psycopg
        from psycopg import sql
        from psycopg.types.json import Jsonb
    except ImportError as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError('psycopg is required for PgVectorStore. Install with: pip install "psycopg[binary]"') from e
    try:
        from pgvector.psycopg import register_vector
    except ImportError as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError("pgvector is required for PgVectorStore. Install with: pip install pgvector") from e
    return psycopg, sql, Jsonb, register_vector


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse ``'0.8.1'`` into ``(0, 8, 1)``, ignoring any suffix."""
    parts: list[int] = []
    for chunk in text.split("."):
        match = re.match(r"\d+", chunk)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


class PgVectorStore(VectorStoreProtocol):
    """pgvector-backed :class:`VectorStoreProtocol` implementation.

    Args:
        dimension: Embedding dimension. Becomes the ``vector(N)`` column
            width, and is verified against the column on reopen.
        dsn: libpq connection string or URI. Mutually exclusive with
            ``conn``.
        conn: An open ``psycopg.Connection`` the caller owns; it is not
            closed by :meth:`close`. Mutually exclusive with ``dsn``.
        table_name: Base table name; the sidecar tables derive from it.
        metric: One of ``cosine``, ``l2``, ``squared_l2``, ``dot``,
            ``l1``.
        **connect_kwargs: Forwarded to ``psycopg.connect`` when ``dsn``
            is given.

    Raises:
        ValueError: On an invalid dimension, table name or metric, or
            when neither/both of ``dsn`` and ``conn`` are given.
        ImportError: When ``psycopg`` or ``pgvector`` isn't installed.
        VectorStoreError: When the ``vector`` extension is unavailable,
            the server's pgvector is too old for the chosen metric, or
            an existing table conflicts with the arguments passed here.
    """

    VALID_METRICS = frozenset(_METRIC_TO_OPERATOR.keys())

    def __init__(
        self,
        dimension: int,
        dsn: str | None = None,
        *,
        conn: Any = None,
        table_name: str = "embeddings",
        metric: str = "cosine",
        **connect_kwargs: Any,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        _validate_table_name(table_name)

        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric!r}. Must be one of: {sorted(self.VALID_METRICS)}")

        if (dsn is None) == (conn is None):
            raise ValueError("Pass exactly one of: dsn, conn")

        psycopg, sql, Jsonb, register_vector = _require_psycopg()
        self._sql = sql
        self._jsonb = Jsonb

        self.dimension = dimension
        self.table_name = table_name
        self.metric = metric_lower
        self._closed = False

        from ..store import VectorStoreError

        self._error = VectorStoreError

        if conn is not None:
            self.conn = conn
            self._owns_conn = False
        else:
            try:
                self.conn = psycopg.connect(dsn, **connect_kwargs)
            except psycopg.Error as e:
                raise VectorStoreError(f"Failed to connect to PostgreSQL: {e}") from e
            self._owns_conn = True

        try:
            self._install_extension()
            register_vector(self.conn)
            self._check_metric_supported()
            self._init_tables()
        except Exception:
            if self._owns_conn:
                self.conn.close()
            raise

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------

    @property
    def meta_table(self) -> str:
        """Name of the table holding this store's stored configuration."""
        return f"{self.table_name}_meta"

    @property
    def sources_table(self) -> str:
        """Name of the table holding source-dedup records."""
        return f"{self.table_name}_sources"

    def _ident(self, name: str) -> Any:
        return self._sql.Identifier(name)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _install_extension(self) -> None:
        try:
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise self._error(
                "Could not enable the pgvector extension (CREATE EXTENSION vector). "
                "Install it server-side and grant the connecting role permission to "
                f"create it, or pre-create it in the target database: {e}"
            ) from e

    def _pgvector_version(self) -> tuple[int, ...]:
        row = self.conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        return _parse_version(row[0]) if row else ()

    def _check_metric_supported(self) -> None:
        """Fail early when the server's pgvector predates the operator.

        Without this the ``l1`` metric would construct fine and then die
        at first query with ``operator does not exist: vector <+>
        vector``, which points at nothing actionable.
        """
        operator, minimum = _METRIC_TO_OPERATOR[self.metric]
        installed = self._pgvector_version()
        if installed and installed < minimum:
            want = ".".join(str(n) for n in minimum)
            have = ".".join(str(n) for n in installed)
            raise self._error(
                f"metric={self.metric!r} needs the {operator} operator, added in "
                f"pgvector {want}, but the server has pgvector {have}. Upgrade the "
                f"extension or choose another metric."
            )

    def _init_tables(self) -> None:
        sql = self._sql
        try:
            self.conn.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id       BIGSERIAL PRIMARY KEY,
                        text     TEXT NOT NULL,
                        embedding VECTOR({dim}) NOT NULL,
                        metadata JSONB
                    )
                """).format(table=self._ident(self.table_name), dim=sql.Literal(self.dimension))
            )
            self.conn.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """).format(table=self._ident(self.meta_table))
            )
            self.conn.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        content_hash TEXT PRIMARY KEY,
                        source_label TEXT NOT NULL,
                        chunk_count  INTEGER NOT NULL,
                        indexed_at   TEXT NOT NULL
                    )
                """).format(table=self._ident(self.sources_table))
            )
            self.conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (source_label)").format(
                    idx=self._ident(f"{self.sources_table}_label_idx"),
                    table=self._ident(self.sources_table),
                )
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise self._error(f"Failed to create tables: {e}") from e

        stored = {
            row[0]: row[1]
            for row in self.conn.execute(
                sql.SQL("SELECT key, value FROM {table}").format(table=self._ident(self.meta_table))
            )
        }
        if stored:
            self._verify_compatibility(stored)
        self._verify_column_dimension()

        for key, value in (("dimension", str(self.dimension)), ("metric", self.metric)):
            self.conn.execute(
                sql.SQL(
                    "INSERT INTO {table} (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ).format(table=self._ident(self.meta_table)),
                (key, value),
            )
        self.conn.commit()

    def _verify_compatibility(self, stored: dict[str, str]) -> None:
        for key, current in (("dimension", str(self.dimension)), ("metric", self.metric)):
            previous = stored.get(key)
            if previous is not None and previous != current:
                raise self._error(
                    f"Table {self.table_name!r} was created with {key}={previous!r} "
                    f"but is being opened with {key}={current!r}. Reopen with the "
                    f"original {key} or use a different table_name."
                )

    def _verify_column_dimension(self) -> None:
        """Cross-check the declared width of the ``embedding`` column.

        The meta table can be absent (a table created by an older
        version, or by hand), so read the truth out of the catalog:
        for a ``vector`` column, ``atttypmod`` is the dimension.
        """
        row = self.conn.execute(
            "SELECT a.atttypmod FROM pg_attribute a "
            "WHERE a.attrelid = %s::regclass AND a.attname = 'embedding' AND NOT a.attisdropped",
            (self.table_name,),
        ).fetchone()
        if row is None or row[0] is None or row[0] < 0:
            return
        if row[0] != self.dimension:
            raise self._error(
                f"Table {self.table_name!r} has an embedding column of "
                f"vector({row[0]}) but the store was opened with "
                f"dimension={self.dimension}."
            )

    def _check_closed(self) -> None:
        if self._closed:
            raise self._error("PgVectorStore is closed")

    def _vector(self, values: list[float]) -> Any:
        from pgvector import Vector

        if len(values) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(values)}")
        return Vector(list(values))

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
        """Insert chunks; return their assigned IDs.

        The chunk rows and the source-dedup record are written in one
        transaction, so a duplicate ``source_hash`` rolls the chunks
        back with it.
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

        if not embeddings:
            return []

        vectors = [self._vector(e) for e in embeddings]

        sql = self._sql
        insert = sql.SQL("INSERT INTO {table} (text, embedding, metadata) VALUES (%s, %s, %s) RETURNING id").format(
            table=self._ident(self.table_name)
        )
        ids: list[int] = []
        try:
            with self.conn.transaction():
                for vector, text, meta in zip(vectors, texts, metadata):
                    row = self.conn.execute(insert, (text, vector, self._jsonb(meta) if meta else None)).fetchone()
                    ids.append(int(row[0]))

                if source_hash is not None:
                    self.conn.execute(
                        sql.SQL(
                            "INSERT INTO {table} (content_hash, source_label, chunk_count, indexed_at) "
                            "VALUES (%s, %s, %s, %s)"
                        ).format(table=self._ident(self.sources_table)),
                        (
                            source_hash,
                            source_label,
                            len(embeddings),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except Exception:
            # The transaction rolled back, so the IDs describe rows that
            # no longer exist. Don't hand the caller phantoms.
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

        query = self._vector(query_embedding)
        operator = _METRIC_TO_OPERATOR[self.metric][0]
        sql = self._sql

        statement = sql.SQL(
            "SELECT id, text, metadata, embedding {op} %s AS distance FROM {table} ORDER BY distance LIMIT %s"
        ).format(op=sql.SQL(operator), table=self._ident(self.table_name))

        try:
            rows = self.conn.execute(statement, (query, k)).fetchall()
        except Exception as e:
            self.conn.rollback()
            raise self._error(f"Search failed: {e}") from e

        results: list[SearchResult] = []
        for id_, text, meta, distance in rows:
            distance = float(distance)
            if self.metric == "squared_l2":
                distance = distance * distance

            if self.metric == "cosine":
                score = 1.0 - distance
            else:
                # dot: <#> is the *negative* inner product, so negating
                # recovers the dot product itself -- and for l1/l2 it
                # gives the usual smaller-is-better-becomes-larger score.
                score = -distance

            if threshold is not None and score < threshold:
                continue

            results.append(
                SearchResult(
                    id=str(id_),
                    text=text,
                    score=score,
                    metadata=meta or {},
                )
            )
        return results

    def is_source_indexed(self, content_hash: str) -> bool:
        """Return True if a source with this content hash was added."""
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL("SELECT 1 FROM {table} WHERE content_hash = %s LIMIT 1").format(
                table=self._ident(self.sources_table)
            ),
            (content_hash,),
        ).fetchone()
        return row is not None

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        """Look up an indexed source by its human-readable label."""
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL(
                "SELECT content_hash, source_label, chunk_count, indexed_at "
                "FROM {table} WHERE source_label = %s LIMIT 1"
            ).format(table=self._ident(self.sources_table)),
            (source_label,),
        ).fetchone()
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
        rows = self.conn.execute(
            self._sql.SQL(
                "SELECT content_hash, source_label, chunk_count, indexed_at FROM {table} ORDER BY indexed_at"
            ).format(table=self._ident(self.sources_table))
        ).fetchall()
        return [
            {
                "content_hash": r[0],
                "source_label": r[1],
                "chunk_count": r[2],
                "indexed_at": r[3],
            }
            for r in rows
        ]

    def get(self, id: str | int) -> SearchResult | None:
        """Fetch a single chunk by ID, or None when it doesn't exist."""
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL("SELECT id, text, metadata FROM {table} WHERE id = %s").format(
                table=self._ident(self.table_name)
            ),
            (int(id),),
        ).fetchone()
        if row is None:
            return None
        return SearchResult(id=str(row[0]), text=row[1], score=1.0, metadata=row[2] or {})

    def get_vector(self, id: str | int) -> list[float] | None:
        """Fetch a stored embedding by ID, or None when it doesn't exist."""
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL("SELECT embedding FROM {table} WHERE id = %s").format(table=self._ident(self.table_name)),
            (int(id),),
        ).fetchone()
        if row is None:
            return None
        # register_vector() decodes the column into a pgvector Vector,
        # which exposes to_list() rather than being iterable itself.
        # Fall back to plain iteration for callers who registered a
        # numpy-returning adapter instead.
        value = row[0]
        to_list = getattr(value, "to_list", None)
        return [float(x) for x in (to_list() if callable(to_list) else value)]

    def delete(self, ids: list[str | int]) -> int:
        """Delete chunks by ID; return the number of rows removed."""
        self._check_closed()
        if not ids:
            return 0
        cursor = self.conn.execute(
            self._sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(table=self._ident(self.table_name)),
            ([int(i) for i in ids],),
        )
        self.conn.commit()
        return int(cursor.rowcount)

    def clear(self) -> int:
        """Remove every chunk and source record; return chunks removed."""
        self._check_closed()
        cursor = self.conn.execute(self._sql.SQL("DELETE FROM {table}").format(table=self._ident(self.table_name)))
        count = int(cursor.rowcount)
        self.conn.execute(self._sql.SQL("DELETE FROM {table}").format(table=self._ident(self.sources_table)))
        self.conn.commit()
        return count

    def close(self) -> None:
        """Close the connection if this store opened it. Idempotent."""
        if self._closed:
            return
        if self._owns_conn:
            self.conn.close()
        self._closed = True

    # ------------------------------------------------------------------
    # pgvector-specific extras (not part of VectorStoreProtocol)
    # ------------------------------------------------------------------

    def create_index(self, method: str = "hnsw", **options: Any) -> None:
        """Build an ANN index over the embedding column.

        This is pgvector's answer to
        :meth:`~cyllama.rag.store.SqliteVectorStore.quantize`: exact
        search needs no index, but past a few tens of thousands of rows
        an HNSW or IVFFlat index turns the scan into a graph or list
        probe. The operator class is chosen from the store's ``metric``,
        so the index actually serves the queries :meth:`search` issues.

        Args:
            method: ``"hnsw"`` (default) or ``"ivfflat"``.
            **options: Index build parameters passed through to
                ``WITH (...)`` -- e.g. ``m=16, ef_construction=64`` for
                HNSW, or ``lists=100`` for IVFFlat.

        Note:
            Building an index takes a table lock and can be slow on a
            large table. pgvector's own guidance applies: create it
            after bulk loading, not before.
        """
        self._check_closed()
        method_lower = method.lower()
        if method_lower not in {"hnsw", "ivfflat"}:
            raise ValueError(f"Invalid index method: {method!r}. Must be 'hnsw' or 'ivfflat'.")

        sql = self._sql
        statement = sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} USING {method} (embedding {opclass})").format(
            idx=self._ident(f"{self.table_name}_{method_lower}_idx"),
            table=self._ident(self.table_name),
            method=sql.SQL(method_lower),
            opclass=sql.SQL(_METRIC_TO_OPCLASS[self.metric]),
        )
        if options:
            statement = sql.SQL("{stmt} WITH ({opts})").format(
                stmt=statement,
                opts=sql.SQL(", ").join(
                    sql.SQL("{k} = {v}").format(k=sql.SQL(str(key)), v=sql.Literal(value))
                    for key, value in options.items()
                ),
            )
        try:
            self.conn.execute(statement)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise self._error(f"Failed to create {method_lower} index: {e}") from e

    def drop_index(self, method: str = "hnsw") -> None:
        """Drop the index :meth:`create_index` built, if present."""
        self._check_closed()
        self.conn.execute(
            self._sql.SQL("DROP INDEX IF EXISTS {idx}").format(
                idx=self._ident(f"{self.table_name}_{method.lower()}_idx")
            )
        )
        self.conn.commit()

    @property
    def pgvector_version(self) -> str:
        """The pgvector extension version reported by the server."""
        self._check_closed()
        row = self.conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        return str(row[0]) if row else "unknown"

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL("SELECT COUNT(*) FROM {table}").format(table=self._ident(self.table_name))
        ).fetchone()
        return int(row[0])

    def __contains__(self, id: str | int) -> bool:
        self._check_closed()
        row = self.conn.execute(
            self._sql.SQL("SELECT 1 FROM {table} WHERE id = %s").format(table=self._ident(self.table_name)),
            (int(id),),
        ).fetchone()
        return row is not None

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} vectors"
        return (
            f"PgVectorStore(dimension={self.dimension}, "
            f"table_name={self.table_name!r}, metric={self.metric!r}, status={status})"
        )
