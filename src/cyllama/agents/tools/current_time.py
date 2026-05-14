"""``current_time`` -- zero-arg lookup with an optional, validated string
parameter. Demonstrates returning a structured result (dict) and graceful
failure on bad input.
"""

from datetime import datetime as _datetime, timezone as _timezone
from typing import Any, Dict

from .core import tool


@tool
def current_time(timezone: str = "UTC") -> Dict[str, str]:
    """Return the current date and time.

    Args:
        timezone: IANA timezone name (e.g. ``"UTC"``, ``"America/New_York"``,
            ``"Europe/Berlin"``). Defaults to ``"UTC"``. Must be a name the
            host's tz database recognises; unknown names raise a clear
            error instead of silently falling back.

    Returns:
        A dict with ``iso`` (ISO 8601 timestamp including UTC offset),
        ``timezone`` (the resolved zone name), and ``unix`` (seconds
        since the epoch, as a string for deterministic JSON encoding).
    """
    # zoneinfo is stdlib on 3.9+. We import lazily so the cost is only
    # paid by users who actually call the tool.
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if timezone.upper() == "UTC":
        tz: Any = _timezone.utc
        resolved = "UTC"
    else:
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as e:
            # Re-raise as a value error so the agent's tool-error handling
            # treats it as a bad argument rather than an internal crash.
            raise ValueError(f"unknown timezone: {timezone!r}") from e
        resolved = timezone

    now = _datetime.now(tz)
    return {
        "iso": now.isoformat(timespec="seconds"),
        "timezone": resolved,
        "unix": str(int(now.timestamp())),
    }
