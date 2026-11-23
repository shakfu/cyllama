"""Run the ACP agent with a scripted model, to test the protocol without a real one.

Test-only. The agent is the real ``ACPAgent`` with its ReAct inner agent; only the
model is replaced. Each call to the model returns the next reply from SCRIPT, in
order, across all prompts. Replies are separated by lines holding only ``---``.
After the last reply, the model answers ``Answer: (script exhausted)``.

Example script::

    Action: read_file({"path": "TODO.md"})
    ---
    Action: edit_file({"path": "TODO.md", "old_text": "- [x] done\\n", "new_text": ""})
    ---
    Answer: Removed the completed item.

Usage::

    python tests/acp_scripted.py SCRIPT [--delay SECONDS]
        [--session-storage {memory,file,sqlite} --session-path PATH]

``--delay`` makes each model call wait, so a client can cancel during it.

Zed (``settings.json``)::

    "agent_servers": {
      "cyllama-scripted": {
        "type": "custom",
        "command": "/path/to/cyllama/.venv/bin/python",
        "args": ["/path/to/cyllama/tests/acp_scripted.py", "/path/to/script.txt"],
        "env": {}
      }
    }
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional


def parse_script(text: str) -> List[str]:
    """Split ``text`` into replies at lines holding only ``---``; drop empty replies."""
    return [r.strip() for r in re.split(r"^---[ \t]*$", text, flags=re.MULTILINE) if r.strip()]


class ScriptedLLM:
    """Stands in for ``LLM``: returns the scripted replies in order, and supports cancel."""

    def __init__(self, replies: List[str], delay: float = 0.0) -> None:
        self._replies = list(replies)
        self._delay = delay
        self._cancelled = threading.Event()
        self.prompts: List[str] = []

    def __call__(self, prompt: str, config: Optional[Any] = None, stream: bool = False) -> str:
        self.prompts.append(prompt)
        self._cancelled.clear()
        if self._delay and self._cancelled.wait(self._delay):
            # A real model returns its partial output; a thought yields an event, so ACP sees the cancel.
            return "Thought: cancelled"
        return self._replies.pop(0) if self._replies else "Answer: (script exhausted)"

    def cancel(self) -> None:
        self._cancelled.set()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", type=Path, help="Replies separated by lines holding only ---")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds each model call waits")
    parser.add_argument("--session-storage", choices=["memory", "file", "sqlite"], default="memory")
    parser.add_argument("--session-path", help="Path for file/sqlite session storage")
    args = parser.parse_args(argv)

    # stdout carries JSON-RPC; logs go to stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")

    from cyllama.agents.acp import ACPAgent

    llm = ScriptedLLM(parse_script(args.script.read_text(encoding="utf-8")), delay=args.delay)
    agent = ACPAgent(llm=llm, session_storage=args.session_storage, session_path=args.session_path)
    agent.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
