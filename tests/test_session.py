"""Tests for session storage backends."""

import os
import tempfile
import time
import pytest

from cyllama.agents.session import (
    Message,
    ToolCallRecord,
    Permission,
    Session,
    MemorySessionStore,
    FileSessionStore,
    SqliteSessionStore,
    create_session_store,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        msg = Message(role="assistant", content="Hi", metadata={"tokens": 10})
        assert msg.metadata["tokens"] == 10


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_tool_call_creation(self):
        tc = ToolCallRecord(
            id="tc_1",
            name="search",
            arguments={"query": "test"},
            status="pending"
        )
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.status == "pending"
        assert tc.result is None

    def test_tool_call_with_result(self):
        tc = ToolCallRecord(
            id="tc_2",
            name="calc",
            arguments={"x": 1},
            status="completed",
            result="42"
        )
        assert tc.result == "42"


class TestPermission:
    """Tests for Permission dataclass."""

    def test_permission_creation(self):
        perm = Permission(tool_name="shell", kind="allow_always")
        assert perm.tool_name == "shell"
        assert perm.kind == "allow_always"
        assert perm.timestamp > 0

    def test_hash_key(self):
        perm = Permission(tool_name="test_tool", kind="allow_once")
        key = perm.hash_key()
        assert len(key) == 16
        assert key.isalnum()


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        session = Session(id="sess_1")
        assert session.id == "sess_1"
        assert session.mode_id is None
        assert session.messages == []
        assert session.tool_calls == []
        assert session.permissions == []

    def test_add_message(self):
        session = Session(id="sess_1")
        msg = session.add_message("user", "Hello")

        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"

    def test_add_message_with_metadata(self):
        session = Session(id="sess_1")
        msg = session.add_message("assistant", "Hi", tokens=5)

        assert session.messages[0].metadata["tokens"] == 5

    def test_add_tool_call(self):
        session = Session(id="sess_1")
        tc = session.add_tool_call("tc_1", "search", {"q": "test"})

        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].name == "search"
        assert session.tool_calls[0].status == "pending"

    def test_get_tool_call(self):
        session = Session(id="sess_1")
        session.add_tool_call("tc_1", "a", {})
        session.add_tool_call("tc_2", "b", {})

        tc = session.get_tool_call("tc_2")
        assert tc is not None
        assert tc.name == "b"

        tc_none = session.get_tool_call("tc_999")
        assert tc_none is None

    def test_add_permission(self):
        session = Session(id="sess_1")
        perm = session.add_permission("shell", "allow_always")

        assert len(session.permissions) == 1
        assert session.permissions[0].tool_name == "shell"

    def test_add_permission_replaces_existing(self):
        session = Session(id="sess_1")
        session.add_permission("shell", "reject_always")
        session.add_permission("shell", "allow_always")

        assert len(session.permissions) == 1
        assert session.permissions[0].kind == "allow_always"

    def test_get_permission(self):
        session = Session(id="sess_1")
        session.add_permission("shell", "allow_always")

        perm = session.get_permission("shell")
        assert perm is not None
        assert perm.kind == "allow_always"

        perm_none = session.get_permission("other")
        assert perm_none is None

    def test_to_dict_and_from_dict(self):
        session = Session(id="sess_1", mode_id="code")
        session.add_message("user", "Hello")
        session.add_tool_call("tc_1", "test", {"x": 1})
        session.add_permission("shell", "allow_always")

        d = session.to_dict()
        restored = Session.from_dict(d)

        assert restored.id == "sess_1"
        assert restored.mode_id == "code"
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Hello"
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "test"
        assert len(restored.permissions) == 1


class TestMemorySessionStore:
    """Tests for in-memory session store."""

    def test_save_and_load(self):
        store = MemorySessionStore()
        session = Session(id="mem_1")
        session.add_message("user", "test")

        store.save(session)
        loaded = store.load("mem_1")

        assert loaded is not None
        assert loaded.id == "mem_1"
        assert len(loaded.messages) == 1

    def test_load_nonexistent(self):
        store = MemorySessionStore()
        loaded = store.load("nonexistent")
        assert loaded is None

    def test_delete(self):
        store = MemorySessionStore()
        session = Session(id="del_1")
        store.save(session)

        assert store.delete("del_1") is True
        assert store.load("del_1") is None
        assert store.delete("del_1") is False

    def test_list_sessions(self):
        store = MemorySessionStore()
        store.save(Session(id="a"))
        store.save(Session(id="b"))
        store.save(Session(id="c"))

        sessions = store.list_sessions()
        assert set(sessions) == {"a", "b", "c"}

    def test_exists(self):
        store = MemorySessionStore()
        store.save(Session(id="exists"))

        assert store.exists("exists") is True
        assert store.exists("nope") is False


class TestFileSessionStore:
    """Tests for file-based session store."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            session = Session(id="file_1")
            session.add_message("user", "hello file")

            store.save(session)
            loaded = store.load("file_1")

            assert loaded is not None
            assert loaded.id == "file_1"
            assert loaded.messages[0].content == "hello file"

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            store.save(Session(id="test_file"))

            # Check file exists
            files = os.listdir(tmpdir)
            assert "test_file.json" in files

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            store.save(Session(id="to_delete"))

            assert store.exists("to_delete")
            assert store.delete("to_delete")
            assert not store.exists("to_delete")

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            store.save(Session(id="x"))
            store.save(Session(id="y"))

            sessions = store.list_sessions()
            assert set(sessions) == {"x", "y"}


class TestSqliteSessionStore:
    """Tests for SQLite session store."""

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SqliteSessionStore(db_path)
            session = Session(id="sql_1")
            session.add_message("user", "hello sqlite")

            store.save(session)
            loaded = store.load("sql_1")

            assert loaded is not None
            assert loaded.id == "sql_1"
            assert loaded.messages[0].content == "hello sqlite"
        finally:
            os.unlink(db_path)

    def test_update_existing(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SqliteSessionStore(db_path)
            session = Session(id="update_1")
            session.add_message("user", "first")
            store.save(session)

            # Update
            session.add_message("assistant", "second")
            store.save(session)

            loaded = store.load("update_1")
            assert len(loaded.messages) == 2
        finally:
            os.unlink(db_path)

    def test_delete(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SqliteSessionStore(db_path)
            store.save(Session(id="to_del"))

            assert store.exists("to_del")
            assert store.delete("to_del")
            assert not store.exists("to_del")
        finally:
            os.unlink(db_path)

    def test_list_sessions(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SqliteSessionStore(db_path)
            store.save(Session(id="p"))
            store.save(Session(id="q"))
            store.save(Session(id="r"))

            sessions = store.list_sessions()
            assert set(sessions) == {"p", "q", "r"}
        finally:
            os.unlink(db_path)


class TestCreateSessionStore:
    """Tests for session store factory."""

    def test_create_memory_store(self):
        store = create_session_store("memory")
        assert isinstance(store, MemorySessionStore)

    def test_create_file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_session_store("file", tmpdir)
            assert isinstance(store, FileSessionStore)

    def test_create_sqlite_store(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = create_session_store("sqlite", db_path)
            assert isinstance(store, SqliteSessionStore)
        finally:
            os.unlink(db_path)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown storage type"):
            create_session_store("unknown")
