"""Shared loop-detection logic for ReAct-style agents.

Detects two failure modes in agent loops:

1. **action_repeat**: the agent emits the exact same action string N times in
   a row (the model is stuck and re-issuing identical calls).
2. **tool_repeat**: the agent calls the same tool N times in a row even with
   different arguments (likely thrashing through a parameter space).

Both ReActAgent and ConstrainedAgent previously duplicated this logic; this
module is the single source of truth.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

LoopKind = Literal["action_repeat", "tool_repeat"]


@dataclass
class LoopDetection:
    kind: LoopKind
    # Last action string (for action_repeat) or tool name (for tool_repeat).
    value: str
    # Threshold that triggered the detection.
    threshold: int


def detect_loop(
    recent_actions: List[str],
    recent_tools: List[str],
    max_consecutive_same_action: int,
    max_consecutive_same_tool: int,
) -> Optional[LoopDetection]:
    """Return a LoopDetection if a stuck-loop pattern is found, else None.

    Caller is responsible for maintaining the `recent_actions` and
    `recent_tools` lists (append-only sliding windows are fine).
    """
    if len(recent_actions) >= max_consecutive_same_action:
        last_n = recent_actions[-max_consecutive_same_action:]
        if all(a == last_n[0] for a in last_n):
            return LoopDetection("action_repeat", last_n[0], max_consecutive_same_action)

    if len(recent_tools) >= max_consecutive_same_tool:
        last_n_tools = recent_tools[-max_consecutive_same_tool:]
        if all(t == last_n_tools[0] for t in last_n_tools):
            return LoopDetection("tool_repeat", last_n_tools[0], max_consecutive_same_tool)

    return None


def format_loop_error(det: LoopDetection) -> str:
    """Human-readable error message for a loop detection."""
    if det.kind == "action_repeat":
        return f"Loop detected: same action repeated {det.threshold} times: {det.value}"
    # tool_repeat
    return f"Loop detected: tool '{det.value}' called {det.threshold} times consecutively"


__all__ = ["LoopDetection", "LoopKind", "detect_loop", "format_loop_error"]
