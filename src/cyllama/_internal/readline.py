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
    atexit.register(save_history, history_path)
    return True


def save_history(history_path: str) -> bool:
    """Write the in-memory readline history to ``history_path``.

    This is the function the :func:`setup_history` atexit handler calls,
    exposed publicly so callers can flush history mid-session if they
    want. It's also the right entry point for any code that needs to
    "snapshot" history outside the normal exit path.

    The function transparently applies the libedit magic-header
    workaround (see :func:`_patch_libedit_history_header`), so a write
    here is guaranteed to round-trip through ``read_history_file`` on
    the same machine -- including the case where the in-memory history
    exceeds the stifled length and libedit's own writer would otherwise
    produce an unreadable file.

    Args:
        history_path: Destination path for the history file.

    Returns:
        ``True`` if the file was written successfully, ``False`` if
        readline is unavailable or the write failed (e.g. disk full,
        permission lost). Failure is silent because losing history is
        much less bad than crashing the user's REPL on exit.
    """
    try:
        import readline
    except ImportError:
        return False
    try:
        readline.write_history_file(history_path)
    except OSError:
        return False
    _patch_libedit_history_header(history_path)
    return True


def _patch_libedit_history_header(path: str) -> None:
    """Restore libedit's ``_HiStOrY_V2_`` magic header if it was dropped.

    libedit (used as the readline backend by uv-prebuilt CPython on
    Linux and by the system Python on macOS) has an asymmetric
    history-file codec: ``write_history_file`` writes the
    ``_HiStOrY_V2_`` magic header on the *first* line of a fresh file,
    but **omits the header whenever truncation kicks in** (i.e. whenever
    the in-memory history exceeds the value passed to
    :func:`readline.set_history_length`). The next call to
    ``read_history_file`` then sees a header-less file, rejects it, and
    raises ``OSError(EINVAL)``.

    The net effect on cyllama users on libedit-backed Pythons: once
    their REPL accumulates enough entries to trigger truncation, the
    *entire* history file becomes unreadable on the next session, and
    they silently start over. This workaround prepends the magic header
    when libedit dropped it, so the round-trip survives.

    GNU readline (used by Debian/Ubuntu system Python) does not have
    this bug; the function is a no-op there.
    """
    try:
        import readline
    except ImportError:
        return
    # ``readline.backend`` is "editline" for libedit and "readline" for
    # GNU readline. The attribute exists on Python 3.13+.
    if getattr(readline, "backend", None) != "editline":
        return
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError:
        return
    if not content or content.startswith(b"_HiStOrY_V2_\n"):
        return
    try:
        with open(path, "wb") as f:
            f.write(b"_HiStOrY_V2_\n" + content)
    except OSError:
        # Same rationale as the write failure above: losing history
        # is preferable to crashing the REPL on exit.
        pass


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


__all__ = ["setup_history", "save_history", "history_path_for"]
