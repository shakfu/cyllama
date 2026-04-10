"""Tests for the corpus-deduplication logic in RAG.add_documents and
RAG.add_texts.

These tests don't load real LLM/embedder models. They construct a
RAG instance via `__new__` so the model-loading side of `__init__`
is skipped, then wire a stub embedder, real `TextSplitter`, and
real `VectorStore` (which uses sqlite-vector and is fast against
``:memory:``). This lets us assert the dedup decisions and the
``IndexResult`` shape without paying the cost of loading any model.

The dedup contract being tested:

1. Re-running ``add_documents`` with the same files is a no-op:
   `IndexResult.skipped_labels` lists the basenames, the chunk store
   is unchanged, and the returned IDs list is empty.
2. Same for ``add_texts`` with the same text strings.
3. A modified file (same basename, different content) raises a
   `ValueError` with a clear message pointing the user at
   ``--rebuild`` or rename.
4. A new file alongside already-indexed files is indexed only once;
   the others are skipped.
5. ``IndexResult`` is a list-of-int subclass so existing callers
   doing ``len(rag.add_documents(...))`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cyllama.rag import IndexResult, VectorStore
from cyllama.rag.rag import RAG, _content_hash
from cyllama.rag.splitter import TextSplitter


# Skip the entire module on platforms where sqlite-vector isn't built
def _extension_available() -> bool:
    import sqlite3
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return False
    ext_path = Path(__file__).parent.parent / "src" / "cyllama" / "rag" / "vector"
    if sys.platform == "darwin":
        return ext_path.with_suffix(".dylib").exists()
    if sys.platform == "win32":
        return ext_path.with_suffix(".dll").exists()
    return ext_path.with_suffix(".so").exists()


pytestmark = pytest.mark.skipif(
    not _extension_available(),
    reason="sqlite-vector extension not built",
)


class _StubEmbedder:
    """Deterministic stub embedder. Returns a constant vector for
    every input so we don't pay the cost of loading a real GGUF
    embedder. Dimension matches the VectorStore the tests construct.
    """

    dimension = 3

    def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3]] * len(texts)


@pytest.fixture
def rag_with_stub_embedder(tmp_path):
    """Yield a RAG instance with a stub embedder, real splitter, and
    fresh in-memory VectorStore. Skips RAG.__init__ so no model
    loading happens."""
    rag = RAG.__new__(RAG)
    rag._closed = False
    rag.embedder = _StubEmbedder()
    rag.splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
    rag.store = VectorStore(dimension=3, db_path=":memory:")
    yield rag
    rag.store.close()


# ---------------------------------------------------------------------------
# IndexResult shape
# ---------------------------------------------------------------------------


class TestIndexResult:
    """Pin the backward-compat contract on the return type of
    ``add_documents`` / ``add_texts``."""

    def test_is_a_list_subclass(self):
        result = IndexResult([1, 2, 3], ["skipped.txt"])
        assert isinstance(result, list)

    def test_len_returns_added_id_count(self):
        result = IndexResult([1, 2, 3], ["skipped.txt"])
        assert len(result) == 3

    def test_iteration_yields_added_ids(self):
        result = IndexResult([10, 20, 30], [])
        assert list(result) == [10, 20, 30]

    def test_indexing_works(self):
        result = IndexResult([10, 20, 30], [])
        assert result[0] == 10
        assert result[-1] == 30

    def test_skipped_labels_is_separate_attribute(self):
        result = IndexResult([1], ["a.txt", "b.txt"])
        assert result.skipped_labels == ["a.txt", "b.txt"]
        # Adding a label should not affect the list payload
        assert list(result) == [1]

    def test_empty_result_is_falsy_via_list_semantics(self):
        result = IndexResult([], ["all-skipped.txt"])
        assert len(result) == 0
        assert not result  # bool([]) is False
        # But the dedup info is still accessible
        assert result.skipped_labels == ["all-skipped.txt"]


# ---------------------------------------------------------------------------
# add_documents dedup
# ---------------------------------------------------------------------------


class TestAddDocumentsDedup:
    def test_first_add_indexes_all_files(self, rag_with_stub_embedder, tmp_path):
        """Baseline: a fresh DB with no prior sources indexes all
        provided files."""
        rag = rag_with_stub_embedder
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Content of file A")
        f2.write_text("Content of file B")

        result = rag.add_documents([f1, f2])
        assert len(result) > 0  # at least one chunk per file
        assert result.skipped_labels == []
        assert len(rag.store.list_sources()) == 2

    def test_second_add_skips_unchanged_files(self, rag_with_stub_embedder, tmp_path):
        """Re-running add_documents with the same files must be a
        no-op on the indexing side. The returned IDs list is empty
        and skipped_labels names every input file."""
        rag = rag_with_stub_embedder
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Content A")
        f2.write_text("Content B")

        rag.add_documents([f1, f2])
        chunks_after_first = len(rag.store)

        result = rag.add_documents([f1, f2])
        assert len(result) == 0
        assert sorted(result.skipped_labels) == ["a.txt", "b.txt"]
        # The store didn't grow
        assert len(rag.store) == chunks_after_first

    def test_partial_skip_indexes_only_new_files(self, rag_with_stub_embedder, tmp_path):
        """A mix of already-indexed and new files indexes only the
        new ones; the others appear in skipped_labels."""
        rag = rag_with_stub_embedder
        f1 = tmp_path / "old.txt"
        f1.write_text("Already there")
        rag.add_documents([f1])

        f2 = tmp_path / "new.txt"
        f2.write_text("Brand new content")
        result = rag.add_documents([f1, f2])

        assert len(result) > 0
        assert result.skipped_labels == ["old.txt"]
        # Two sources total in the dedup table
        assert len(rag.store.list_sources()) == 2

    def test_modified_file_raises_value_error(self, rag_with_stub_embedder, tmp_path):
        """Same basename, different content -> raise. The error
        message must mention the file and tell the user how to
        proceed."""
        rag = rag_with_stub_embedder
        f1 = tmp_path / "a.txt"
        f1.write_text("Original content")
        rag.add_documents([f1])

        # Modify the file
        f1.write_text("MODIFIED CONTENT")

        with pytest.raises(ValueError, match="different content hash"):
            rag.add_documents([f1])

    def test_modified_file_error_mentions_rebuild(self, rag_with_stub_embedder, tmp_path):
        """The error message should explicitly mention --rebuild as
        the recovery path so users don't have to guess."""
        rag = rag_with_stub_embedder
        f1 = tmp_path / "a.txt"
        f1.write_text("v1")
        rag.add_documents([f1])
        f1.write_text("v2")

        with pytest.raises(ValueError) as excinfo:
            rag.add_documents([f1])
        assert "rebuild" in str(excinfo.value).lower()

    def test_unreadable_file_raises_value_error(self, rag_with_stub_embedder, tmp_path):
        """A path that doesn't exist should raise ValueError from
        the hash step (before the loader sees it)."""
        rag = rag_with_stub_embedder
        with pytest.raises(ValueError, match="Could not read"):
            rag.add_documents([tmp_path / "does_not_exist.txt"])

    def test_dedup_persists_across_close_reopen(self, tmp_path):
        """The whole point of dedup is for persistent stores: after
        close+reopen, re-running add_documents with the same files
        must skip them. This is the user-visible behaviour for
        ``cyllama rag --db PATH`` re-runs."""
        db_path = str(tmp_path / "store.db")
        f1 = tmp_path / "a.txt"
        f1.write_text("Persistent content")

        # Session 1: index
        rag1 = RAG.__new__(RAG)
        rag1._closed = False
        rag1.embedder = _StubEmbedder()
        rag1.splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        rag1.store = VectorStore(dimension=3, db_path=db_path)
        rag1.add_documents([f1])
        rag1.store.close()

        # Session 2: re-add the same file -> should skip
        rag2 = RAG.__new__(RAG)
        rag2._closed = False
        rag2.embedder = _StubEmbedder()
        rag2.splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        rag2.store = VectorStore(dimension=3, db_path=db_path)
        result = rag2.add_documents([f1])
        assert len(result) == 0
        assert result.skipped_labels == ["a.txt"]
        rag2.store.close()


# ---------------------------------------------------------------------------
# add_texts dedup
# ---------------------------------------------------------------------------


class TestAddTextsDedup:
    def test_first_add_indexes_all_texts(self, rag_with_stub_embedder):
        rag = rag_with_stub_embedder
        result = rag.add_texts(["alpha", "beta"])
        assert len(result) > 0
        assert result.skipped_labels == []

    def test_duplicate_text_skipped(self, rag_with_stub_embedder):
        rag = rag_with_stub_embedder
        rag.add_texts(["the same text"])
        result = rag.add_texts(["the same text"])
        assert len(result) == 0
        assert len(result.skipped_labels) == 1
        # Label is "text:<8 hex chars>"
        assert result.skipped_labels[0].startswith("text:")
        assert len(result.skipped_labels[0]) == len("text:") + 8

    def test_partial_skip_in_text_batch(self, rag_with_stub_embedder):
        rag = rag_with_stub_embedder
        rag.add_texts(["existing"])
        result = rag.add_texts(["existing", "new"])
        # One added, one skipped
        assert len(result) > 0
        assert len(result.skipped_labels) == 1

    def test_different_texts_have_different_labels(self, rag_with_stub_embedder):
        """The text:<hash> label must be deterministic per content,
        so two different texts produce two different labels and
        don't collide on the dedup primary key."""
        rag = rag_with_stub_embedder
        rag.add_texts(["text one", "text two"])
        sources = rag.store.list_sources()
        assert len(sources) == 2
        labels = {s["source_label"] for s in sources}
        assert len(labels) == 2  # all distinct


# ---------------------------------------------------------------------------
# _content_hash helper
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_md5_hex_format(self):
        h = _content_hash(b"hello")
        assert len(h) == 32  # md5 hex digest is 32 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert _content_hash(b"hello") == _content_hash(b"hello")

    def test_different_inputs_produce_different_hashes(self):
        assert _content_hash(b"a") != _content_hash(b"b")

    def test_known_md5_value(self):
        # md5("") == d41d8cd98f00b204e9800998ecf8427e
        assert _content_hash(b"") == "d41d8cd98f00b204e9800998ecf8427e"
