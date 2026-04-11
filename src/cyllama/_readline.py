"""Readline history setup for cyllama's interactive REPLs.

This module exists to enable up/down-arrow history cycling, in-line
editing (Ctrl-A / Ctrl-E / Ctrl-R / etc.), and persistent history files
for the interactive modes of ``cyllama rag``, ``cyllama chat``, and any
other future REPL-style command.

The actual mechanism is the standard library's :mod:`readline` module:
just importing it transparently upgrades Python's built-in :func:`input`
to use libreadline (or libedit on macOS) for line editing, with no
other code changes required. This module's only job is to wire up
history file persistence and to gracefully no-op on platforms where
``readline`` isn't available.

Why a dedicated module rather than inlining the few lines into each
caller: the helper is shared between :mod:`cyllama.__main__` (which
hosts ``cmd_rag``'s interactive loop) and :mod:`cyllama.llama.chat`
(which has its own ``chat_loop``), so a single source of truth avoids
the two paths drifting apart.
"""

from __future__ import annotations

import atexit
import os


def setup_history(
    history_path: str,
    max_entries: int = 1000,
) -> bool:
    """Enable readline + persistent history for the calling REPL.

    After this call returns, Python's built-in :func:`input` will:

    * cycle through prior entries with the up/down arrow keys,
    * support basic line editing (left/right arrows, Ctrl-A / Ctrl-E,
      Ctrl-R reverse search, etc.),
    * load history from ``history_path`` on startup, and
    * write the (possibly truncated) history back to ``history_path``
      on interpreter shutdown via an :mod:`atexit` handler.

    The function is idempotent in the sense that calling it twice with
    the same path is harmless: the second call re-reads the same file
    and registers a second :mod:`atexit` handler. Callers should
    typically only call it once per process.

    Args:
        history_path: Path to the history file. Tildes are expanded
            via :func:`os.path.expanduser`. The parent directory is
            created if it doesn't exist.
        max_entries: Maximum number of entries kept in memory and
            written to disk. Older entries are dropped. Default 1000,
            which is the same default the standard CPython interactive
            shell uses.

    Returns:
        ``True`` if readline was successfully enabled (the common
        case on macOS and Linux). ``False`` on platforms where the
        :mod:`readline` module is not available -- notably Windows,
        where users can install ``pyreadline3`` to get the same
        behaviour, but cyllama does not require it. Returning ``False``
        means the calling REPL still works, just without arrow-key
        history.
    """
    try:
        import readline
    except ImportError:
        # Windows without pyreadline3 lands here. The REPL still
        # functions; users just don't get history navigation.
        return False

    history_path = os.path.expanduser(history_path)

    # Ensure the parent directory exists. We don't fail on permission
    # errors here -- if the user can't create the directory we just
    # skip persistence and the in-memory history still works for the
    # current session.
    parent = os.path.dirname(history_path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass

    # Load any existing history. Missing-file is the normal first-run
    # case; corrupted-file is rare but we handle it the same way
    # (drop and start fresh) rather than crashing the REPL on startup.
    try:
        readline.read_history_file(history_path)
    except (FileNotFoundError, OSError):
        pass

    readline.set_history_length(max_entries)

    # Persist history on exit. We register one atexit handler per call;
    # in practice each command only calls setup_history once so this is
    # fine. The handler captures history_path by closure so it stays
    # correct even if the caller mutates its own variables later.
    def _save_history(path: str = history_path) -> None:
        try:
            readline.write_history_file(path)
        except OSError:
            # Disk full, permission lost mid-session, etc. Silently
            # drop -- losing history is much less bad than crashing
            # the user's REPL on exit.
            pass

    atexit.register(_save_history)
    return True


def history_path_for(command: str) -> str:
    """Return the canonical history file path for a cyllama subcommand.

    Cyllama's REPLs use one history file per command so the rag history
    and the chat history don't overwrite each other. The path follows
    the conventional ``~/.{tool}_{command}_history`` pattern that
    Python's built-in REPL, ``ipython``, and most CLI tools use, so
    users can easily find and edit it.

    Args:
        command: Subcommand name, e.g. ``"rag"`` or ``"chat"``.

    Returns:
        Absolute path (tilde-expanded) to the history file.
    """
    return os.path.expanduser(f"~/.cyllama_{command}_history")


__all__ = ["setup_history", "history_path_for"]
