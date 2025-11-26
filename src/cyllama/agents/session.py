"""
Session storage backends for ACP agent.

Supports in-memory, file-based (JSON), and SQLite storage.
"""

import json
import logging
import os
import sqlite3
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in a session conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """Record of a tool call in a session."""
    id: str
    name: str
    arguments: Dict[str, Any]
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Permission:
    """A cached permission decision."""
    tool_name: str
    kind: str  # "allow_always", "reject_always"
    timestamp: float = field(default_factory=time.time)

    def hash_key(self) -> str:
        """Generate a hash key for this permission."""
        return hashlib.sha256(self.tool_name.encode()).hexdigest()[:16]


@dataclass
class Session:
    """An ACP session."""
    id: str
    mode_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the session."""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def add_tool_call(self, id: str, name: str, arguments: Dict[str, Any]) -> ToolCallRecord:
        """Record a tool call."""
        record = ToolCallRecord(id=id, name=name, arguments=arguments, status="pending")
        self.tool_calls.append(record)
        self.updated_at = time.time()
        return record

    def get_tool_call(self, id: str) -> Optional[ToolCallRecord]:
        """Get a tool call by ID."""
        for tc in self.tool_calls:
            if tc.id == id:
                return tc
        return None

    def add_permission(self, tool_name: str, kind: str) -> Permission:
        """Add a cached permission."""
        perm = Permission(tool_name=tool_name, kind=kind)
        # Remove any existing permission for this tool
        self.permissions = [p for p in self.permissions if p.tool_name != tool_name]
        self.permissions.append(perm)
        self.updated_at = time.time()
        return perm

    def get_permission(self, tool_name: str) -> Optional[Permission]:
        """Get cached permission for a tool."""
        for perm in self.permissions:
            if perm.tool_name == tool_name:
                return perm
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "mode_id": self.mode_id,
            "messages": [asdict(m) for m in self.messages],
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "permissions": [asdict(p) for p in self.permissions],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        return cls(
            id=d["id"],
            mode_id=d.get("mode_id"),
            messages=[Message(**m) for m in d.get("messages", [])],
            tool_calls=[ToolCallRecord(**tc) for tc in d.get("tool_calls", [])],
            permissions=[Permission(**p) for p in d.get("permissions", [])],
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )


class SessionStore(ABC):
    """Abstract base class for session storage."""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Save a session."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        pass

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass


class MemorySessionStore(SessionStore):
    """In-memory session storage (default)."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def load(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


class FileSessionStore(SessionStore):
    """File-based session storage (JSON files)."""

    def __init__(self, directory: str):
        self._directory = directory
        self._lock = threading.Lock()

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        # Sanitize session ID for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return os.path.join(self._directory, f"{safe_id}.json")

    def save(self, session: Session) -> None:
        path = self._session_path(session.id)
        with self._lock:
            with open(path, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
        logger.debug("Saved session %s to %s", session.id, path)

    def load(self, session_id: str) -> Optional[Session]:
        path = self._session_path(session_id)
        with self._lock:
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return Session.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to load session %s: %s", session_id, e)
                return None

    def delete(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False

    def list_sessions(self) -> List[str]:
        with self._lock:
            sessions = []
            for filename in os.listdir(self._directory):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self._directory, filename), "r") as f:
                            data = json.load(f)
                            sessions.append(data["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            return sessions

    def exists(self, session_id: str) -> bool:
        return os.path.exists(self._session_path(session_id))


class SqliteSessionStore(SessionStore):
    """SQLite-based session storage."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(updated_at DESC)
                """)
                conn.commit()
            finally:
                conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def save(self, session: Session) -> None:
        data = json.dumps(session.to_dict())
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions (id, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session.id, data, session.created_at, session.updated_at))
                conn.commit()
            finally:
                conn.close()
        logger.debug("Saved session %s to SQLite", session.id)

    def load(self, session_id: str) -> Optional[Session]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT data FROM sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                data = json.loads(row[0])
                return Session.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to load session %s: %s", session_id, e)
                return None
            finally:
                conn.close()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def list_sessions(self) -> List[str]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT id FROM sessions ORDER BY updated_at DESC"
                )
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()

    def exists(self, session_id: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (session_id,)
                )
                return cursor.fetchone() is not None
            finally:
                conn.close()


def create_session_store(
    storage_type: str = "memory",
    path: Optional[str] = None
) -> SessionStore:
    """
    Create a session store based on configuration.

    Args:
        storage_type: "memory", "file", or "sqlite"
        path: Path for file/sqlite storage

    Returns:
        SessionStore instance
    """
    if storage_type == "memory":
        return MemorySessionStore()
    elif storage_type == "file":
        if not path:
            path = os.path.expanduser("~/.cyllama/sessions")
        return FileSessionStore(path)
    elif storage_type == "sqlite":
        if not path:
            path = os.path.expanduser("~/.cyllama/sessions.db")
        return SqliteSessionStore(path)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
