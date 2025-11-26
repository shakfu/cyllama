"""
Agent Client Protocol (ACP) implementation.

Provides an ACP-compliant agent that can be spawned by editors (Zed, Neovim, etc.)
and communicates using JSON-RPC over stdio.
"""

import json
import logging
import sys
import threading
import uuid
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .tools import Tool
from .react import ReActAgent, AgentEvent, EventType
from .jsonrpc import (
    JsonRpcServer,
    StdioTransport,
    JsonRpcError,
    ErrorCode,
    AsyncBridge,
)
from .mcp import McpClient, McpServerConfig
from .session import (
    Session,
    SessionStore,
    create_session_store,
    Message,
    ToolCallRecord,
    Permission,
)

logger = logging.getLogger(__name__)

# ACP Protocol Version
ACP_PROTOCOL_VERSION = "2025-01-01"


class StopReason(Enum):
    """Reasons why an agent stopped processing."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"
    REFUSAL = "refusal"
    ERROR = "error"


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

    def to_dict(self) -> dict:
        result = {"type": self.type}
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
    def text(cls, text: str) -> "ContentBlock":
        return cls(type="text", text=text)


@dataclass
class ToolCallUpdate:
    """Update for a tool call in progress."""
    id: str
    name: str
    status: ToolCallStatus
    arguments: Optional[Dict[str, Any]] = None
    content: Optional[List[ContentBlock]] = None

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
        }
        if self.arguments is not None:
            result["arguments"] = self.arguments
        if self.content is not None:
            result["content"] = [c.to_dict() for c in self.content]
        return result


@dataclass
class SessionUpdate:
    """ACP session update notification."""
    session_id: str
    content: Optional[List[ContentBlock]] = None
    tool_calls: Optional[List[ToolCallUpdate]] = None
    stop_reason: Optional[StopReason] = None

    def to_dict(self) -> dict:
        result = {"sessionId": self.session_id}
        if self.content is not None:
            result["content"] = [c.to_dict() for c in self.content]
        if self.tool_calls is not None:
            result["toolCalls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.stop_reason is not None:
            result["stopReason"] = self.stop_reason.value
        return result


class ACPAgent:
    """
    ACP-compliant agent that can be spawned by editors.

    Wraps an inner agent (ReActAgent by default) and exposes it via the
    Agent Client Protocol over stdio.
    """

    VERSION = "0.1.10"

    def __init__(
        self,
        llm,
        tools: Optional[List[Tool]] = None,
        inner_agent_type: str = "react",
        mcp_servers: Optional[List[McpServerConfig]] = None,
        session_storage: str = "memory",
        session_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize ACP agent.

        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            inner_agent_type: Type of inner agent ("react" or "constrained")
            mcp_servers: List of MCP server configurations
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

        # MCP client
        self._mcp_client: Optional[McpClient] = None
        if mcp_servers:
            self._mcp_client = McpClient(mcp_servers)

        # Session storage
        self._session_store = create_session_store(session_storage, session_path)

        # JSON-RPC server (initialized in serve())
        self._server: Optional[JsonRpcServer] = None
        self._async_bridge: Optional[AsyncBridge] = None

        # Current processing state
        self._current_session_id: Optional[str] = None
        self._cancelled = False

        # Available modes
        self._modes = {
            "default": {"id": "default", "name": "Default", "description": "Standard assistant mode"},
        }

    def _create_inner_agent(self, session: Session) -> ReActAgent:
        """Create an inner agent for processing prompts."""
        # Combine built-in tools with MCP tools
        all_tools = list(self.tools)
        if self._mcp_client and self._mcp_client.is_connected:
            all_tools.extend(self._mcp_client.get_tools_for_agent())

        # Add ACP-specific tools (file operations delegate to editor)
        all_tools.extend(self._create_acp_tools())

        if self.inner_agent_type == "constrained":
            from .constrained import ConstrainedAgent
            return ConstrainedAgent(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
        else:
            return ReActAgent(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )

    def _create_acp_tools(self) -> List[Tool]:
        """Create ACP-backed tools that delegate to the editor."""
        tools = []

        # File read tool
        def read_file(path: str) -> str:
            """Read a file from the editor's workspace."""
            return self._acp_read_file(path)

        tools.append(Tool(
            name="read_file",
            description="Read the contents of a file from the workspace",
            func=read_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["path"]
            }
        ))

        # File write tool
        def write_file(path: str, content: str) -> str:
            """Write content to a file in the editor's workspace."""
            return self._acp_write_file(path, content)

        tools.append(Tool(
            name="write_file",
            description="Write content to a file in the workspace",
            func=write_file,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        ))

        # Terminal/shell tool
        def run_command(command: str) -> str:
            """Execute a shell command via the editor's terminal."""
            return self._acp_run_command(command)

        tools.append(Tool(
            name="run_command",
            description="Execute a shell command in the terminal",
            func=run_command,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"}
                },
                "required": ["command"]
            }
        ))

        return tools

    # === ACP File/Terminal Operations (delegate to editor) ===

    def _acp_read_file(self, path: str) -> str:
        """Request file read from editor via ACP."""
        if not self._server:
            raise RuntimeError("ACP server not running")

        try:
            response = self._server.send_request("fs/read_text_file", {"path": path}, timeout=30.0)
            if response.is_error:
                return f"Error reading file: {response.error.message}"
            return response.result.get("contents", "")
        except TimeoutError:
            return "Error: File read timed out"
        except Exception as e:
            return f"Error reading file: {e}"

    def _acp_write_file(self, path: str, content: str) -> str:
        """Request file write from editor via ACP."""
        if not self._server:
            raise RuntimeError("ACP server not running")

        try:
            response = self._server.send_request(
                "fs/write_text_file",
                {"path": path, "contents": content},
                timeout=30.0
            )
            if response.is_error:
                return f"Error writing file: {response.error.message}"
            return "File written successfully"
        except TimeoutError:
            return "Error: File write timed out"
        except Exception as e:
            return f"Error writing file: {e}"

    def _acp_run_command(self, command: str) -> str:
        """Request command execution from editor via ACP."""
        if not self._server:
            raise RuntimeError("ACP server not running")

        try:
            # Create terminal
            create_resp = self._server.send_request(
                "terminal/create",
                {"command": command},
                timeout=10.0
            )
            if create_resp.is_error:
                return f"Error creating terminal: {create_resp.error.message}"

            terminal_id = create_resp.result.get("terminalId")
            if not terminal_id:
                return "Error: No terminal ID returned"

            # Wait for exit
            self._server.send_request(
                "terminal/wait_for_exit",
                {"terminalId": terminal_id},
                timeout=120.0
            )

            # Get output
            output_resp = self._server.send_request(
                "terminal/output",
                {"terminalId": terminal_id},
                timeout=10.0
            )

            # Release terminal
            self._server.send_request(
                "terminal/release",
                {"terminalId": terminal_id},
                timeout=5.0
            )

            if output_resp.is_error:
                return f"Error getting output: {output_resp.error.message}"

            return output_resp.result.get("output", "")

        except TimeoutError:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error running command: {e}"

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
            response = self._server.send_request(
                "session/request_permission",
                {
                    "sessionId": self._current_session_id,
                    "toolCall": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                    "options": [
                        {"id": "allow_once", "kind": "allow_once", "label": "Allow"},
                        {"id": "allow_always", "kind": "allow_always", "label": "Always Allow"},
                        {"id": "reject_once", "kind": "reject_once", "label": "Deny"},
                        {"id": "reject_always", "kind": "reject_always", "label": "Always Deny"},
                    ]
                },
                timeout=300.0  # Long timeout for user interaction
            )

            if response.is_error:
                logger.warning("Permission request failed: %s", response.error.message)
                return False

            outcome = response.result.get("outcome", {})
            if outcome.get("kind") == "cancelled":
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

    def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {})
        logger.info("ACP client connected: %s %s",
                   client_info.get("name", "unknown"),
                   client_info.get("version", ""))

        # Connect to MCP servers
        if self._mcp_client:
            try:
                self._mcp_client.connect_all()
            except Exception as e:
                logger.error("Failed to connect MCP servers: %s", e)

        capabilities = {
            "prompts": {
                "text": True,
                "image": False,
                "audio": False,
            },
            "tools": True,
            "permissions": True,
            "sessions": {
                "load": True,
            },
            "fileSystem": {
                "read": True,
                "write": True,
            },
            "terminal": True,
        }

        if self._mcp_client:
            capabilities["mcp"] = self._mcp_client.get_capabilities()

        return {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "capabilities": capabilities,
            "agentInfo": {
                "name": "cyllama",
                "version": self.VERSION,
            },
            "modes": list(self._modes.values()),
        }

    def _handle_authenticate(self, params: dict) -> dict:
        """Handle authenticate request."""
        # Currently no authentication required
        return {"authenticated": True}

    def _handle_session_new(self, params: dict) -> dict:
        """Handle session/new request."""
        session_id = str(uuid.uuid4())
        mode_id = params.get("modeId", "default")

        session = Session(id=session_id, mode_id=mode_id)
        self._session_store.save(session)

        logger.info("Created new session: %s", session_id)

        return {
            "sessionId": session_id,
            "mode": self._modes.get(mode_id, self._modes["default"]),
        }

    def _handle_session_load(self, params: dict) -> dict:
        """Handle session/load request."""
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")

        session = self._session_store.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        logger.info("Loaded session: %s", session_id)

        return {
            "sessionId": session_id,
            "mode": self._modes.get(session.mode_id, self._modes["default"]),
        }

    def _handle_session_set_mode(self, params: dict) -> dict:
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

        return {"mode": self._modes[mode_id]}

    def _handle_session_prompt(self, params: dict) -> dict:
        """Handle session/prompt request."""
        session_id = params.get("sessionId")
        content = params.get("content", [])

        if not session_id:
            raise ValueError("sessionId is required")

        session = self._session_store.load(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Extract text from content blocks
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        task = "\n".join(text_parts)
        if not task:
            raise ValueError("No text content provided")

        # Record user message
        session.add_message("user", task)
        self._session_store.save(session)

        # Process with inner agent
        self._current_session_id = session_id
        self._cancelled = False

        try:
            stop_reason = self._process_prompt(session, task)
        finally:
            self._current_session_id = None

        return {"stopReason": stop_reason.value}

    def _process_prompt(self, session: Session, task: str) -> StopReason:
        """Process a prompt and stream updates."""
        inner_agent = self._create_inner_agent(session)

        tool_call_counter = 0

        for event in inner_agent.stream(task):
            if self._cancelled:
                return StopReason.CANCELLED

            update = self._event_to_update(session.id, event, tool_call_counter)
            if update:
                if event.type == EventType.ACTION:
                    tool_call_counter += 1
                self._send_update(update)

            # Record to session
            if event.type == EventType.ANSWER:
                session.add_message("assistant", event.content)
                self._session_store.save(session)
                return StopReason.END_TURN
            elif event.type == EventType.ERROR:
                session.add_message("assistant", f"Error: {event.content}")
                self._session_store.save(session)
                return StopReason.ERROR

        return StopReason.END_TURN

    def _event_to_update(
        self,
        session_id: str,
        event: AgentEvent,
        tool_call_index: int
    ) -> Optional[SessionUpdate]:
        """Convert an AgentEvent to an ACP SessionUpdate."""
        if event.type == EventType.THOUGHT:
            return SessionUpdate(
                session_id=session_id,
                content=[ContentBlock.text(f"Thinking: {event.content}")]
            )

        elif event.type == EventType.ACTION:
            tool_name = event.metadata.get("tool_name", "unknown")
            tool_args = event.metadata.get("tool_args", {})

            return SessionUpdate(
                session_id=session_id,
                tool_calls=[ToolCallUpdate(
                    id=f"tc_{tool_call_index}",
                    name=tool_name,
                    status=ToolCallStatus.IN_PROGRESS,
                    arguments=tool_args,
                )]
            )

        elif event.type == EventType.OBSERVATION:
            tool_name = event.metadata.get("tool_name", "unknown")

            return SessionUpdate(
                session_id=session_id,
                tool_calls=[ToolCallUpdate(
                    id=f"tc_{tool_call_index - 1}",  # Previous tool call
                    name=tool_name,
                    status=ToolCallStatus.COMPLETED,
                    content=[ContentBlock.text(event.content)]
                )]
            )

        elif event.type == EventType.ANSWER:
            return SessionUpdate(
                session_id=session_id,
                content=[ContentBlock.text(event.content)],
                stop_reason=StopReason.END_TURN
            )

        elif event.type == EventType.ERROR:
            return SessionUpdate(
                session_id=session_id,
                content=[ContentBlock.text(f"Error: {event.content}")],
                stop_reason=StopReason.ERROR
            )

        return None

    def _send_update(self, update: SessionUpdate) -> None:
        """Send a session update notification."""
        if self._async_bridge:
            self._async_bridge.send_notification("session/update", update.to_dict())
        elif self._server:
            self._server.send_notification("session/update", update.to_dict())

    def _handle_session_cancel(self, params: dict) -> None:
        """Handle session/cancel notification."""
        session_id = params.get("sessionId")
        if session_id == self._current_session_id:
            self._cancelled = True
            logger.info("Cancelling session: %s", session_id)

    # === Server Lifecycle ===

    def serve(
        self,
        input_stream=None,
        output_stream=None,
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

        # Set up async bridge for sending notifications from sync agent
        self._async_bridge = AsyncBridge(self._server)
        self._async_bridge.start()

        # Register ACP method handlers
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
            self._async_bridge.stop()
            if self._mcp_client:
                self._mcp_client.disconnect_all()
            logger.info("ACP server stopped")

    def stop(self) -> None:
        """Stop the ACP server."""
        if self._server:
            self._server.stop()


def serve_acp(
    llm,
    tools: Optional[List[Tool]] = None,
    mcp_servers: Optional[List[McpServerConfig]] = None,
    session_storage: str = "memory",
    session_path: Optional[str] = None,
    **kwargs
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
        **kwargs
    )
    agent.serve()
