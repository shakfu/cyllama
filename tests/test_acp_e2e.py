"""End-to-end ACP: a scripted client drives the real agent process over stdio.

The model is scripted (tests/acp_scripted.py) and the editor is this test: it
answers the agent's fs, permission and terminal requests from an in-memory
workspace. No model and no editor are needed, so every protocol path runs
deterministically. tests/test_acp.py covers the handlers in-process.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

LAUNCHER = Path(__file__).with_name("acp_scripted.py")
CWD = "/work"
TIMEOUT = 30.0


class Client:
    """Plays the editor: sends requests, answers the agent's, records updates."""

    def __init__(self, tmp_path: Path, replies: List[str], *args: str, files: Optional[Dict[str, str]] = None):
        script = tmp_path / f"script_{time.monotonic_ns()}.txt"
        script.write_text("\n---\n".join(replies), encoding="utf-8")
        self.files: Dict[str, str] = dict(files or {})  # absolute path -> content
        self.allow = True
        self.agent_requests: List[str] = []
        self.updates: List[Dict[str, Any]] = []
        self.terminal_commands: List[List[str]] = []
        self._next_id = 0
        self._messages: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self.proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), str(script), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.strip():
                self._messages.put(json.loads(line))
        self._messages.put(None)

    def _write(self, msg: Dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def send(self, method: str, params: Dict[str, Any]) -> int:
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        return self._next_id

    def wait(self, request_id: int) -> Dict[str, Any]:
        """Serve the agent until the response to ``request_id`` arrives; return it."""
        deadline = time.monotonic() + TIMEOUT
        while True:
            msg = self._messages.get(timeout=max(0.1, deadline - time.monotonic()))
            if msg is None:
                raise AssertionError("agent exited")
            if "method" not in msg:
                if msg.get("id") == request_id:
                    return msg
                continue
            if msg["method"] == "session/update":
                self.updates.append(msg["params"]["update"])
            elif "id" in msg:
                self.agent_requests.append(msg["method"])
                self._answer(msg)

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        response = self.wait(self.send(method, params))
        assert "error" not in response, response
        return response["result"]

    def _answer(self, msg: Dict[str, Any]) -> None:
        method, params = msg["method"], msg["params"]
        result: Any = {}
        error = None
        if method == "fs/read_text_file":
            if params["path"] in self.files:
                result = {"content": self.files[params["path"]]}
            else:
                error = {"code": -32002, "message": f"Resource not found: {params['path']}"}
        elif method == "fs/write_text_file":
            self.files[params["path"]] = params["content"]
            result = None
        elif method == "session/request_permission":
            kind = "allow_once" if self.allow else "reject_once"
            option = next(o for o in params["options"] if o["kind"] == kind)
            result = {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
        elif method == "terminal/create":
            self.terminal_commands.append([params["command"], *params.get("args", [])])
            result = {"terminalId": "term-1"}
        elif method == "terminal/wait_for_exit":
            result = {"exitCode": 0, "signal": None}
        elif method == "terminal/output":
            result = {"output": "hello\n", "truncated": False, "exitStatus": {"exitCode": 0, "signal": None}}
        if error:
            self._write({"jsonrpc": "2.0", "id": msg["id"], "error": error})
        else:
            self._write({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    def start(self, fs: bool = True, terminal: bool = True) -> str:
        caps = {"fs": {"readTextFile": fs, "writeTextFile": fs}, "terminal": terminal}
        init = self.call("initialize", {"protocolVersion": 1, "clientCapabilities": caps})
        assert init["protocolVersion"] == 1
        return str(self.call("session/new", {"cwd": CWD, "mcpServers": []})["sessionId"])

    def prompt(self, sid: str, text: str) -> str:
        result = self.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": text}]})
        return str(result["stopReason"])

    def kinds(self) -> List[str]:
        return [u["sessionUpdate"] for u in self.updates]

    def tool_results(self) -> List[Dict[str, Any]]:
        return [u for u in self.updates if u["sessionUpdate"] == "tool_call_update"]

    def reply(self) -> str:
        return "".join(u["content"]["text"] for u in self.updates if u["sessionUpdate"] == "agent_message_chunk")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def client(tmp_path):
    clients: List[Client] = []

    def make(replies: List[str], *args: str, **kw: Any) -> Client:
        c = Client(tmp_path, replies, *args, **kw)
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.close()


TODO = f"{CWD}/TODO.md"


def test_read_then_edit(client):
    c = client(
        [
            'Action: read_file({"path": "TODO.md"})',
            'Action: edit_file({"path": "TODO.md", "old_text": "- [x] done\\n", "new_text": ""})',
            "Answer: Removed the completed item.",
        ],
        files={TODO: "- [x] done\n- [ ] open\n"},
    )
    sid = c.start()
    assert c.prompt(sid, "remove completed items") == "end_turn"
    assert c.files[TODO] == "- [ ] open\n"  # only old_text changed
    assert c.agent_requests == [
        "fs/read_text_file",
        "fs/read_text_file",
        "session/request_permission",
        "fs/write_text_file",
    ]  # the read is not gated; the edit reads, asks, then writes
    calls = [u for u in c.updates if u["sessionUpdate"] == "tool_call"]
    assert [(u["title"], u["kind"]) for u in calls] == [("read_file", "read"), ("edit_file", "edit")]
    assert [u["status"] for u in c.tool_results()] == ["completed", "completed"]
    assert c.reply() == "Removed the completed item."


def test_write_file_refuses_existing_and_creates_new(client):
    c = client(
        [
            'Action: write_file({"path": "TODO.md", "content": "gone"})',
            'Action: write_file({"path": "NEW.md", "content": "fresh"})',
            "Answer: done",
        ],
        files={TODO: "keep me\n"},
    )
    sid = c.start()
    assert c.prompt(sid, "write") == "end_turn"
    assert c.files[TODO] == "keep me\n"
    assert c.files[f"{CWD}/NEW.md"] == "fresh"
    first, second = c.tool_results()
    assert first["status"] == "failed" and "use edit_file" in first["content"][0]["content"]["text"]
    assert second["status"] == "completed"
    # the refused write asked nothing; the new file asked once
    assert c.agent_requests.count("session/request_permission") == 1


def test_denied_edit_writes_nothing(client):
    c = client(
        ['Action: edit_file({"path": "TODO.md", "old_text": "a", "new_text": "b"})', "Answer: ok"],
        files={TODO: "a\n"},
    )
    c.allow = False
    sid = c.start()
    assert c.prompt(sid, "edit") == "end_turn"
    assert c.files[TODO] == "a\n"
    assert "fs/write_text_file" not in c.agent_requests
    assert "permission denied" in c.tool_results()[0]["content"][0]["content"]["text"]


def test_run_command_uses_terminal(client):
    c = client(['Action: run_command({"command": "echo hello"})', "Answer: ran it"])
    sid = c.start()
    assert c.prompt(sid, "run") == "end_turn"
    assert c.agent_requests == [
        "session/request_permission",
        "terminal/create",
        "terminal/wait_for_exit",
        "terminal/output",
        "terminal/release",
    ]
    assert c.terminal_commands[0][-1] == "echo hello"
    assert c.tool_results()[0]["content"][0]["content"]["text"].startswith("hello")


def test_malformed_action_does_not_end_turn(client):
    # Zed showed "Tool call not found" and ended the turn on this
    c = client(['Action: write_file({"path": "a.md", "content": "cut off', "Answer: recovered"])
    sid = c.start()
    assert c.prompt(sid, "write") == "end_turn"
    call = next(u for u in c.updates if u["sessionUpdate"] == "tool_call")
    assert call["title"] == "write_file"
    assert c.tool_results()[0]["status"] == "failed"
    assert c.reply() == "recovered"


def test_no_editor_capabilities_means_no_tools(client):
    c = client(['Action: read_file({"path": "TODO.md"})', "Answer: no tools"])
    sid = c.start(fs=False, terminal=False)
    assert c.prompt(sid, "read") == "end_turn"
    assert c.agent_requests == []  # the unknown tool fails inside the agent
    assert c.tool_results()[0]["status"] == "failed"


def test_cancel_stops_the_turn(client):
    c = client(["Answer: too late"], "--delay", "20")
    sid = c.start()
    request_id = c.send("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "slow"}]})
    time.sleep(0.5)
    started = time.monotonic()
    c.notify("session/cancel", {"sessionId": sid})
    response = c.wait(request_id)
    assert response["result"]["stopReason"] == "cancelled"
    assert time.monotonic() - started < 10  # not the 20 s delay
    assert "too late" not in c.reply()


def test_session_load_replays_after_restart(client, tmp_path):
    db = str(tmp_path / "sessions.db")
    first = client(["Answer: hello there"], "--session-storage", "sqlite", "--session-path", db)
    sid = first.start()
    assert first.prompt(sid, "hi") == "end_turn"
    first.close()

    second = client([], "--session-storage", "sqlite", "--session-path", db)
    second.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    assert second.call("session/load", {"sessionId": sid, "cwd": CWD, "mcpServers": []}) == {}
    replay = [(u["sessionUpdate"], u["content"]["text"]) for u in second.updates]
    assert replay == [("user_message_chunk", "hi"), ("agent_message_chunk", "hello there")]
