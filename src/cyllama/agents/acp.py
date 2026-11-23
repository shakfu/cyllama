"""
Agent Client Protocol (ACP) implementation.

Provides an ACP agent that editors (Zed, Neovim, etc.) spawn and talk to with
JSON-RPC over stdio. Message shapes follow the ACP schema, protocol version 1:
https://agentclientprotocol.com/protocol/schema

Supported: ``initialize``, ``session/new``, ``session/load`` (history replay),
``session/prompt`` with text and embedded-resource blocks, ``session/cancel``,
``session/set_mode``, and the ``session/update`` kinds ``agent_message_chunk``,
``agent_thought_chunk``, ``user_message_chunk``, ``tool_call`` and
``tool_call_update``. Tools that act on the editor (``fs/*``, ``terminal/*``)
are offered only when the client advertises them, and writes and commands
ask for ``session/request_permission`` first.

Not supported: images and audio in prompts, plans, HTTP MCP servers, and
authentication methods. No test runs against a real ACP client; the shapes
are checked against the schema by unit tests.
"""

import logging
import os
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from .jsonrpc import JsonRpcServer, StdioTransport
from .mcp import McpClient, McpServerConfig, McpTransportType
from .react import ReActAgent
from .session import Session, create_session_store
from .tools import Tool
from .types import AgentEvent, AgentProtocol, EventType

logger = logging.getLogger(__name__)

# The ACP protocol version this agent implements (an integer in the spec).
ACP_PROTOCOL_VERSION = 1


def _package_version() -> str:
    # Lazy: cyllama/__init__ imports the agents package. Not importlib.metadata: GPU wheels use other dist names.
    from .. import __version__

    return __version__


# ACP ToolKind of the editor-backed tools; a UI hint for the client.
_TOOL_KINDS = {"read_file": "read", "write_file": "edit", "edit_file": "edit", "run_command": "execute"}

# Characters of earlier turns given to the inner agent with each prompt.
HISTORY_CHARS = 8000

# Bytes of terminal output the client keeps for one command.
TERMINAL_OUTPUT_LIMIT = 64 * 1024


class StopReason(Enum):
    """Why a prompt turn ended (``session/prompt`` result)."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class ToolCallStatus(Enum):
    """Status of a tool call."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ContentBlock:
    """ACP content block."""

    type: str  # "text", "image", "audio", "resource"
    text: Optional[str] = None
    data: Optional[str] = None  # base64 for binary
    mime_type: Optional[str] = None
    uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.data is not None:
            result["data"] = self.data
        if self.mime_type is not None:
            result["mimeType"] = self.mime_type
        if self.uri is not None:
            result["uri"] = self.uri
        return result

    @classmethod
    def from_text(cls, text: str) -> "ContentBlock":
        """Build a text content block.

        Renamed from ``text`` (the original method shadowed the ``text``
        dataclass field, which strict typecheck rejects).
        """
        return cls(type="text", text=text)


@dataclass
class ToolCallUpdate:
    """Fields of a ``tool_call`` or ``tool_call_update`` session update."""

    tool_call_id: str
    status: ToolCallStatus
    title: Optional[str] = None
    kind: Optional[str] = None
    raw_input: Optional[Dict[str, Any]] = None
    content: Optional[List[ContentBlock]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"toolCallId": self.tool_call_id, "status": self.status.value}
        if self.title is not None:
            result["title"] = self.title
        if self.kind is not None:
            result["kind"] = self.kind
        if self.raw_input is not None:
            result["rawInput"] = self.raw_input
        if self.content is not None:
            result["content"] = [{"type": "content", "content": c.to_dict()} for c in self.content]
        return result


@dataclass
class SessionUpdate:
    """A ``session/update`` notification: ``{sessionId, update: {sessionUpdate: kind, ...}}``."""

    session_id: str
    kind: str  # the sessionUpdate discriminator, e.g. "agent_message_chunk"
    fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"sessionId": self.session_id, "update": {"sessionUpdate": self.kind, **self.fields}}

    @classmethod
    def text(cls, session_id: str, kind: str, text: str) -> "SessionUpdate":
        """A message or thought chunk holding one text block."""
        return cls(session_id, kind, {"content": ContentBlock.from_text(text).to_dict()})

    @classmethod
    def tool_call(cls, session_id: str, call: ToolCallUpdate, first: bool) -> "SessionUpdate":
        """``tool_call`` when ``first``, else ``tool_call_update``."""
        return cls(session_id, "tool_call" if first else "tool_call_update", call.to_dict())


def _prompt_text(blocks: List[Dict[str, Any]]) -> str:
    """Text of a prompt: text blocks, embedded-resource text, and resource-link references."""
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "resource":
            res = block.get("resource") or {}
            if "text" in res:
                parts.append(f"[{res.get('uri', 'resource')}]\n{res['text']}")
        elif block.get("type") == "resource_link":
            parts.append(f"[{block.get('name') or block.get('uri', 'resource')}: {block.get('uri', '')}]")
    return "\n".join(p for p in parts if p)


def _mcp_configs(servers: List[Dict[str, Any]]) -> List[McpServerConfig]:
    """McpServerConfigs for the stdio servers of a ``session/new`` or ``session/load`` request."""
    configs = []
    for s in servers or []:
        if s.get("type", "stdio") != "stdio" or not s.get("command"):
            logger.warning("Skipping unsupported MCP server %r (only stdio is supported)", s.get("name"))
            continue
        env = {e["name"]: e["value"] for e in s.get("env", []) if "name" in e and "value" in e}
        configs.append(
            McpServerConfig(
                name=s.get("name", s["command"]),
                transport=McpTransportType.STDIO,
                command=s["command"],
                args=list(s.get("args", [])),
                env=env or None,
            )
        )
    return configs


class ACPAgent:
    """
    ACP-compliant agent that can be spawned by editors.

    Wraps an inner agent (ReActAgent by default) and exposes it via the
    Agent Client Protocol over stdio.
    """

    STEP_MAX_TOKENS = 2048  # per ReAct step

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Tool]] = None,
        inner_agent_type: str = "react",
        mcp_servers: Optional[List[McpServerConfig]] = None,
        session_storage: str = "memory",
        session_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initialize ACP agent.

        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            inner_agent_type: Type of inner agent ("react" or "constrained")
            mcp_servers: MCP servers to connect at initialize, in addition to
                those a client names in ``session/new``
            session_storage: Session storage type ("memory", "file", "sqlite")
            session_path: Path for file/sqlite session storage
            system_prompt: Custom system prompt
            max_iterations: Maximum iterations for inner agent
            verbose: Enable verbose output
        """
        self.llm = llm
        self.tools = tools or []
        self.inner_agent_type = inner_agent_type
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose

        self._mcp_client: Optional[McpClient] = McpClient(mcp_servers) if mcp_servers else None
        self._session_store = create_session_store(session_storage, session_path)
        self._server: Optional[JsonRpcServer] = None

        # What the client said it supports in initialize (clientCapabilities).
        self._client_fs_read = False
        self._client_fs_write = False
        self._client_terminal = False

        # Current processing state
        self._current_session_id: Optional[str] = None
        self._current_tool_call_id: Optional[str] = None
        self._cancelled = False

        self._modes = {
            "default": {"id": "default", "name": "Default", "description": "Standard assistant mode"},
        }

    # === Inner agent ===

    def _create_inner_agent(self, session: Session) -> AgentProtocol:
        """Create an inner agent with the built-in, MCP and editor-backed tools."""
        all_tools = list(self.tools)
        if self._mcp_client and self._mcp_client.is_connected:
            all_tools.extend(self._mcp_client.get_tools_for_agent())
        all_tools.extend(self._create_acp_tools())

        agent: Any
        if self.inner_agent_type == "constrained":
            from .constrained import ConstrainedAgent

            agent = ConstrainedAgent(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
        else:
            agent = ReActAgent(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
        # write_file carries a whole file in one action; 512 tokens cut it off mid-call
        agent.generation_config.max_tokens = self.STEP_MAX_TOKENS
        return cast(AgentProtocol, agent)

    @staticmethod
    def _with_history(session: Session, task: str) -> str:
        """``task`` preceded by the session's earlier turns, newest kept when over HISTORY_CHARS.

        The inner agents take one task at a time, so earlier turns go in as text.
        """
        lines = []
        for msg in session.messages[:-1]:  # the last message is this task
            who = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{who}: {msg.content}")
        history = "\n".join(lines)
        if not history:
            return task
        if len(history) > HISTORY_CHARS:
            history = "..." + history[-HISTORY_CHARS:]
        return f"Conversation so far:\n{history}\n\nCurrent request:\n{task}"

    def _create_acp_tools(self) -> List[Tool]:
        """Tools that delegate to the editor, for the capabilities the client advertised."""
        tools = []

        if self._client_fs_read:

            def read_file(path: str) -> str:
                """Read a file from the editor's workspace."""
                return self._acp_read_file(path)

            tools.append(
                Tool(
                    name="read_file",
                    description="Read the contents of a file from the workspace",
                    func=read_file,
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
                        "required": ["path"],
                    },
                )
            )

        # Writes and commands ask the editor first: the model may be acting on
        # injected text. Reads are not gated; their result only reaches the model.
        if self._client_fs_write:

            def write_file(path: str, content: str) -> str:
                """Write content to a file in the editor's workspace."""
                if not self._request_permission("write_file", {"path": path}):
                    return "Error: permission denied by the user"
                return self._acp_write_file(path, content)

            tools.append(
                Tool(
                    name="write_file",
                    description="Create a new file, or replace a whole file. To change part of an existing file, "
                    "use edit_file",
                    func=write_file,
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to write"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                )
            )

        if self._client_fs_read and self._client_fs_write:

            def edit_file(path: str, old_text: str, new_text: str) -> str:
                """Replace one exact occurrence of old_text with new_text in a file."""
                return self._acp_edit_file(path, old_text, new_text)

            tools.append(
                Tool(
                    name="edit_file",
                    description="Change part of an existing file: replace old_text, which must occur exactly once, "
                    "with new_text. Copy old_text exactly from read_file output; use an empty new_text to delete it",
                    func=edit_file,
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to edit"},
                            "old_text": {"type": "string", "description": "Exact text to replace"},
                            "new_text": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                )
            )

        if self._client_terminal:

            def run_command(command: str) -> str:
                """Execute a shell command via the editor's terminal."""
                if not self._request_permission("run_command", {"command": command}):
                    return "Error: permission denied by the user"
                return self._acp_run_command(command)

            tools.append(
                Tool(
                    name="run_command",
                    description="Execute a shell command in the terminal",
                    func=run_command,
                    parameters={
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "Command to execute"}},
                        "required": ["command"],
                    },
                )
            )

        return tools

    # === ACP File/Terminal Operations (delegate to editor) ===

    def _session_cwd(self) -> str:
        session = self._session_store.load(self._current_session_id) if self._current_session_id else None
        return str((session.metadata.get("cwd") if session else None) or os.getcwd())

    def _abs_path(self, path: str) -> str:
        """``path`` made absolute against the session's cwd; ACP requires absolute paths."""
        return os.path.normpath(path if os.path.isabs(path) else os.path.join(self._session_cwd(), path))

    def _acp_read_text(self, path: str) -> str:
        """Read a file through the editor; raise on failure."""
        if not self._server:
            raise RuntimeError("ACP server not running")
        response = self._server.send_request(
            "fs/read_text_file",
            {"sessionId": self._current_session_id, "path": self._abs_path(path)},
            timeout=30.0,
        )
        if response.is_error:
            assert response.error is not None
            raise RuntimeError(response.error.message)
        return str((response.result or {}).get("content", ""))

    def _acp_read_file(self, path: str) -> str:
        """Request file read from editor via ACP; errors come back as text for the model."""
        if not self._server:
            raise RuntimeError("ACP server not running")
        try:
            return self._acp_read_text(path)
        except TimeoutError:
            return "Error: File read timed out"
        except Exception as e:
            return f"Error reading file: {e}"

    def _acp_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace the one occurrence of ``old_text`` in ``path``, after asking the user."""
        if not old_text:
            return "Error: old_text is empty; copy the exact text to replace from the file"
        try:
            content = self._acp_read_text(path)
        except TimeoutError:
            return "Error: File read timed out"
        except Exception as e:
            return f"Error reading file: {e}"
        # Checked before asking: a call that cannot apply should not reach the user
        count = content.count(old_text)
        if count == 0:
            return f"Error: old_text not found in {path}; read the file and copy the text exactly"
        if count > 1:
            return f"Error: old_text occurs {count} times in {path}; include surrounding lines so it occurs once"
        if not self._request_permission("edit_file", {"path": path, "old_text": old_text, "new_text": new_text}):
            return "Error: permission denied by the user"
        result = self._acp_write_file(path, content.replace(old_text, new_text, 1))
        return result if result.startswith("Error") else f"Edited {path}"

    def _acp_write_file(self, path: str, content: str) -> str:
        """Request file write from editor via ACP."""
        if not self._server:
            raise RuntimeError("ACP server not running")
        try:
            response = self._server.send_request(
                "fs/write_text_file",
                {"sessionId": self._current_session_id, "path": self._abs_path(path), "content": content},
                timeout=30.0,
            )
            if response.is_error:
                assert response.error is not None
                return f"Error writing file: {response.error.message}"
            return "File written successfully"
        except TimeoutError:
            return "Error: File write timed out"
        except Exception as e:
            return f"Error writing file: {e}"

    def _acp_run_command(self, command: str, timeout: float = 120.0) -> str:
        """Run ``command`` in a shell in an editor terminal; return its output and exit status."""
        if not self._server:
            raise RuntimeError("ACP server not running")
        sid = self._current_session_id
        shell, flag = ("cmd", "/c") if os.name == "nt" else ("/bin/sh", "-c")
        try:
            create = self._server.send_request(
                "terminal/create",
                {
                    "sessionId": sid,
                    "command": shell,
                    "args": [flag, command],
                    "cwd": self._session_cwd(),
                    "outputByteLimit": TERMINAL_OUTPUT_LIMIT,
                },
                timeout=10.0,
            )
            if create.is_error:
                assert create.error is not None
                return f"Error creating terminal: {create.error.message}"
            terminal_id = (create.result or {}).get("terminalId")
            if not terminal_id:
                return "Error: No terminal ID returned"
        except TimeoutError:
            return "Error: terminal/create timed out"
        except Exception as e:
            return f"Error running command: {e}"

        ids = {"sessionId": sid, "terminalId": terminal_id}
        try:
            try:
                self._server.send_request("terminal/wait_for_exit", ids, timeout=timeout)
            except TimeoutError:
                self._server.send_request("terminal/kill", ids, timeout=5.0)
            output = self._server.send_request("terminal/output", ids, timeout=10.0)
            if output.is_error:
                assert output.error is not None
                return f"Error getting output: {output.error.message}"
            result = output.result or {}
            text = str(result.get("output", ""))
            if result.get("truncated"):
                text = "[earlier output truncated]\n" + text
            status = result.get("exitStatus") or {}
            if status.get("signal"):
                text += f"\n[terminated by signal {status['signal']}]"
            elif status.get("exitCode") not in (None, 0):
                text += f"\n[exit code {status['exitCode']}]"
            return text
        except Exception as e:
            return f"Error running command: {e}"
        finally:
            try:
                self._server.send_request("terminal/release", ids, timeout=5.0)
            except Exception as e:
                logger.warning("terminal/release failed: %s", e)

    def _request_permission(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Request permission from the editor for a tool call.

        Checks cached permissions first.
        """
        if not self._server or not self._current_session_id:
            return True  # Allow if not in ACP context

        session = self._session_store.load(self._current_session_id)
        if session:
            perm = session.get_permission(tool_name)
            if perm:
                if perm.kind == "allow_always":
                    return True
                elif perm.kind == "reject_always":
                    return False

        try:
            # Shape per the ACP schema (RequestPermissionRequest / PermissionOption):
            # https://agentclientprotocol.com/protocol/tool-calls
            response = self._server.send_request(
                "session/request_permission",
                {
                    "sessionId": self._current_session_id,
                    "toolCall": {
                        "toolCallId": self._current_tool_call_id or f"call_{uuid.uuid4().hex[:12]}",
                        "title": tool_name,
                        "kind": _TOOL_KINDS.get(tool_name, "other"),
                        "rawInput": arguments,
                    },
                    "options": [
                        {"optionId": "allow_once", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "allow_always", "name": "Always Allow", "kind": "allow_always"},
                        {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
                        {"optionId": "reject_always", "name": "Always Deny", "kind": "reject_always"},
                    ],
                },
                timeout=300.0,  # Long timeout for user interaction
            )

            if response.is_error:
                assert response.error is not None
                logger.warning("Permission request failed: %s", response.error.message)
                return False

            # result: {"outcome": {"outcome": "selected", "optionId": ...}} or {"outcome": {"outcome": "cancelled"}}.
            # Anything else is not approval.
            outcome = (response.result or {}).get("outcome")
            if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
                return False

            option_id = outcome.get("optionId", "")

            # Cache "always" permissions
            if option_id in ("allow_always", "reject_always") and session:
                session.add_permission(tool_name, option_id)
                self._session_store.save(session)

            return option_id in ("allow_once", "allow_always")

        except TimeoutError:
            logger.warning("Permission request timed out")
            return False
        except Exception as e:
            logger.error("Permission request error: %s", e)
            return False

    # === ACP Method Handlers ===

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {}) or {}
        logger.info("ACP client connected: %s %s", client_info.get("name", "unknown"), client_info.get("version", ""))

        caps = params.get("clientCapabilities", {}) or {}
        fs = caps.get("fs", {}) or {}
        self._client_fs_read = bool(fs.get("readTextFile"))
        self._client_fs_write = bool(fs.get("writeTextFile"))
        self._client_terminal = bool(caps.get("terminal"))

        if self._mcp_client:
            try:
                self._mcp_client.connect_all()
            except Exception as e:
                logger.error("Failed to connect MCP servers: %s", e)

        return {
            # Version 1 is the only one implemented; a client that cannot speak it disconnects.
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
                "mcpCapabilities": {"http": False, "sse": False},
            },
            "agentInfo": {"name": "cyllama", "version": _package_version()},
            "authMethods": [],
        }

    def _handle_authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle authenticate request; no method is advertised, so any call succeeds."""
        return {}

    def _connect_session_mcp(self, servers: List[Dict[str, Any]]) -> None:
        """Connect the MCP servers a session/new or session/load request names."""
        for config in _mcp_configs(servers):
            if self._mcp_client is None:
                self._mcp_client = McpClient([])
            try:
                self._mcp_client.add_server(config)
                self._mcp_client._connect_server(config)
                self._mcp_client._connected = True
            except Exception as e:
                logger.error("Failed to connect MCP server %r: %s", config.name, e)

    def _handle_session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/new request."""
        session_id = f"sess_{uuid.uuid4().hex}"
        session = Session(id=session_id, mode_id=params.get("modeId", "default"))
        if params.get("cwd"):
            session.metadata["cwd"] = params["cwd"]
        self._session_store.save(session)
        self._connect_session_mcp(params.get("mcpServers", []))
        logger.info("Created new session: %s", session_id)
        return {"sessionId": session_id}

    def _handle_session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/load: replay the conversation as session/update notifications, then reply."""
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")

        session = self._session_store.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        if params.get("cwd"):
            session.metadata["cwd"] = params["cwd"]
            self._session_store.save(session)
        self._connect_session_mcp(params.get("mcpServers", []))

        for msg in session.messages:
            kind = "user_message_chunk" if msg.role == "user" else "agent_message_chunk"
            self._send_update(SessionUpdate.text(session_id, kind, msg.content))

        logger.info("Loaded session: %s", session_id)
        return {}

    def _handle_session_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/set_mode request."""
        session_id = params.get("sessionId")
        mode_id = params.get("modeId")

        if not session_id or not mode_id:
            raise ValueError("sessionId and modeId are required")

        session = self._session_store.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if mode_id not in self._modes:
            raise ValueError(f"Unknown mode: {mode_id}")

        session.mode_id = mode_id
        self._session_store.save(session)
        return {}

    def _handle_session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/prompt request."""
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")

        session = self._session_store.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        task = _prompt_text(params.get("prompt", []) or [])
        if not task:
            raise ValueError("No text content provided")

        session.add_message("user", task)
        self._session_store.save(session)

        self._current_session_id = session_id
        self._cancelled = False
        try:
            stop_reason = self._process_prompt(session, task)
        finally:
            self._current_session_id = None
            self._current_tool_call_id = None

        return {"stopReason": stop_reason.value}

    def _process_prompt(self, session: Session, task: str) -> StopReason:
        """Run the inner agent on ``task`` and stream its events as session updates."""
        inner_agent = self._create_inner_agent(session)
        prefix = f"call_{uuid.uuid4().hex[:8]}"
        n_calls = 0

        for event in inner_agent.stream(self._with_history(session, task)):
            if self._cancelled:
                return StopReason.CANCELLED

            if event.type == EventType.ACTION:
                n_calls += 1
                self._current_tool_call_id = f"{prefix}_{n_calls}"
            if event.type == EventType.ERROR and "error_type" in event.metadata:
                # A failed tool call: the agent continues, and its observation marks the call failed
                continue
            update = self._event_to_update(session.id, event, self._current_tool_call_id)
            if update:
                self._send_update(update)

            if event.type == EventType.ANSWER:
                session.add_message("assistant", event.content)
                self._session_store.save(session)
                return StopReason.END_TURN
            if event.type == EventType.ERROR:
                session.add_message("assistant", f"Error: {event.content}")
                self._session_store.save(session)
                if "maximum iterations" in event.content:
                    return StopReason.MAX_TURN_REQUESTS
                return StopReason.END_TURN

        return StopReason.CANCELLED if self._cancelled else StopReason.END_TURN

    def _event_to_update(
        self, session_id: str, event: AgentEvent, tool_call_id: Optional[str] = None
    ) -> Optional[SessionUpdate]:
        """Convert an AgentEvent to an ACP session update."""
        if event.type == EventType.THOUGHT:
            return SessionUpdate.text(session_id, "agent_thought_chunk", event.content)

        if event.type == EventType.ACTION:
            # None for an action that failed to parse; a null title made Zed drop the call
            tool_name = event.metadata.get("tool_name") or event.content.split("(", 1)[0].strip()[:60] or "tool call"
            call = ToolCallUpdate(
                tool_call_id=tool_call_id or "call_0",
                status=ToolCallStatus.IN_PROGRESS,
                title=tool_name,
                kind=_TOOL_KINDS.get(tool_name, "other"),
                raw_input=event.metadata.get("tool_args", {}),
            )
            return SessionUpdate.tool_call(session_id, call, first=True)

        if event.type == EventType.OBSERVATION:
            failed = event.content.startswith("Error")
            call = ToolCallUpdate(
                tool_call_id=tool_call_id or "call_0",
                status=ToolCallStatus.FAILED if failed else ToolCallStatus.COMPLETED,
                content=[ContentBlock.from_text(event.content)],
            )
            return SessionUpdate.tool_call(session_id, call, first=False)

        if event.type == EventType.ANSWER:
            return SessionUpdate.text(session_id, "agent_message_chunk", event.content)

        if event.type == EventType.ERROR:
            return SessionUpdate.text(session_id, "agent_message_chunk", f"Error: {event.content}")

        return None

    def _send_update(self, update: SessionUpdate) -> None:
        """Send a session/update notification.

        Synchronous, so every update reaches the client before the response
        to the request that produced it, as ACP requires.
        """
        if self._server:
            self._server.send_notification("session/update", update.to_dict())

    def _handle_session_cancel(self, params: Dict[str, Any]) -> None:
        """Handle session/cancel: stop the prompt turn, aborting generation in flight."""
        session_id = params.get("sessionId")
        if session_id == self._current_session_id:
            self._cancelled = True
            cancel = getattr(self.llm, "cancel", None)
            if callable(cancel):
                cancel()
            logger.info("Cancelling session: %s", session_id)

    # === Server Lifecycle ===

    def serve(
        self,
        input_stream: Optional[Any] = None,
        output_stream: Optional[Any] = None,
    ) -> None:
        """
        Start the ACP server and process requests.

        This blocks until EOF on input or the server is stopped.

        Args:
            input_stream: Input stream (default: stdin)
            output_stream: Output stream (default: stdout)
        """
        transport = StdioTransport(
            input_stream=input_stream or sys.stdin,
            output_stream=output_stream or sys.stdout,
        )
        self._server = JsonRpcServer(transport)

        self._server.register("initialize", self._handle_initialize)
        self._server.register("authenticate", self._handle_authenticate)
        self._server.register("session/new", self._handle_session_new)
        self._server.register("session/load", self._handle_session_load)
        self._server.register("session/set_mode", self._handle_session_set_mode)
        self._server.register("session/prompt", self._handle_session_prompt)
        self._server.register("session/cancel", self._handle_session_cancel)

        logger.info("ACP server starting")

        try:
            self._server.serve()
        finally:
            if self._mcp_client:
                self._mcp_client.disconnect_all()
            logger.info("ACP server stopped")

    def stop(self) -> None:
        """Stop the ACP server."""
        if self._server:
            self._server.stop()


def serve_acp(
    llm: Any,
    tools: Optional[List[Tool]] = None,
    mcp_servers: Optional[List[McpServerConfig]] = None,
    session_storage: str = "memory",
    session_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Convenience function to start an ACP server.

    Args:
        llm: Language model instance
        tools: List of tools
        mcp_servers: MCP server configurations
        session_storage: Session storage type
        session_path: Path for session storage
        **kwargs: Additional arguments for ACPAgent
    """
    agent = ACPAgent(
        llm=llm,
        tools=tools,
        mcp_servers=mcp_servers,
        session_storage=session_storage,
        session_path=session_path,
        **kwargs,
    )
    agent.serve()
