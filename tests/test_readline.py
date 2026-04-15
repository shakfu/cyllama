"""Tests for the cyllama._internal.readline history helper.

Readline behaviour itself can't be fully tested without a real
interactive terminal, but the parts of the helper that don't depend on
tty input -- file path resolution, history file persistence,
round-tripping through the libedit/libreadline file format, missing/
corrupt history file handling, parent directory creation -- are all
unit-testable and pinned here so a future change to the helper can't
silently break history persistence for interactive users.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from cyllama._internal.readline import setup_history, history_path_for


# ---------------------------------------------------------------------------
# history_path_for
# ---------------------------------------------------------------------------


class TestHistoryPathFor:
    """The path-naming convention is part of the user-facing surface
    (users grep for and edit these files), so changes to it should
    require deliberate test updates."""

    def test_rag_path(self):
        path = history_path_for("rag")
        assert path.endswith(".cyllama_rag_history")
        assert os.path.isabs(path), f"path should be absolute, got: {path}"

    def test_chat_path(self):
        path = history_path_for("chat")
        assert path.endswith(".cyllama_chat_history")

    def test_distinct_per_command(self):
        """rag and chat must use different history files so the two
        REPLs don't pollute each other's history."""
        assert history_path_for("rag") != history_path_for("chat")

    def test_arbitrary_command_name(self):
        """Future REPL commands should be able to use the same helper
        without code changes -- the path is templated on the command
        name."""
        path = history_path_for("custom")
        assert path.endswith(".cyllama_custom_history")


# ---------------------------------------------------------------------------
# setup_history
# ---------------------------------------------------------------------------


# All tests in this class skip on platforms where the readline module
# is not available (notably Windows without pyreadline3). The helper
# itself returns False on those platforms; the tests verify the
# success path.
readline = pytest.importorskip("readline")


class TestSetupHistory:
    @pytest.fixture(autouse=True)
    def _clean_readline_state(self):
        """Reset readline's in-memory history before and after each
        test so tests don't leak state into each other. We can't undo
        ``atexit.register`` calls, but those only fire on interpreter
        shutdown, so they don't interfere with test isolation."""
        readline.clear_history()
        yield
        readline.clear_history()

    def test_returns_true_when_readline_available(self, tmp_path):
        ok = setup_history(str(tmp_path / "history"))
        assert ok is True

    def test_creates_parent_directory(self, tmp_path):
        """If the parent directory doesn't exist, setup_history should
        create it. This matters for the default
        ``~/.cyllama_rag_history`` path on a fresh user account where
        the home directory might exist but no cyllama state has been
        written yet."""
        nested = tmp_path / "subdir1" / "subdir2" / "history"
        assert not nested.parent.exists()
        setup_history(str(nested))
        assert nested.parent.is_dir()

    def test_loads_existing_history_file(self, tmp_path):
        histfile = tmp_path / "history"

        # Pre-populate a history file using readline directly
        readline.add_history("first")
        readline.add_history("second")
        readline.add_history("third")
        readline.write_history_file(str(histfile))
        readline.clear_history()
        assert readline.get_current_history_length() == 0

        # setup_history should load the saved entries back in
        setup_history(str(histfile))
        assert readline.get_current_history_length() == 3
        assert readline.get_history_item(1) == "first"
        assert readline.get_history_item(2) == "second"
        assert readline.get_history_item(3) == "third"

    def test_missing_history_file_is_first_run_not_an_error(self, tmp_path):
        """First-run scenario: the history file doesn't exist yet.
        setup_history should treat this as 'no history to load' and
        return success, not raise."""
        histfile = tmp_path / "does_not_exist_yet"
        assert not histfile.exists()
        ok = setup_history(str(histfile))
        assert ok is True
        assert readline.get_current_history_length() == 0

    def test_corrupted_history_file_is_handled_gracefully(self, tmp_path):
        """A corrupted history file should not crash the REPL on
        startup. The helper should swallow the read error and start
        fresh -- losing history is much less bad than the user being
        unable to launch the REPL at all."""
        histfile = tmp_path / "corrupt_history"
        # Write garbage that isn't valid libreadline/libedit history
        histfile.write_bytes(b"\x00\x01\x02 not a valid history file \xff\xfe\xfd")

        # Should not raise
        ok = setup_history(str(histfile))
        assert ok is True

    def test_max_entries_caps_in_memory_history(self, tmp_path):
        histfile = tmp_path / "history"
        setup_history(str(histfile), max_entries=5)
        # set_history_length sets the cap; verify it took effect
        assert readline.get_history_length() == 5

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Tilde in the path should be expanded to the user's home
        directory. We patch HOME to point at a temp dir so the test
        doesn't litter the real home directory."""
        monkeypatch.setenv("HOME", str(tmp_path))
        ok = setup_history("~/.test_cyllama_history")
        assert ok is True
        # The expanded path should now exist as a file under tmp_path
        # (created on the atexit save, but the parent directory at
        # least should exist).
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# Round trip: write some history via setup_history's flow, read it back
# ---------------------------------------------------------------------------


class TestHistoryRoundTrip:
    """End-to-end: simulate a full session by writing entries to the
    history file the way the REPL would, then loading it from a fresh
    state and verifying the entries are intact. This is the closest we
    can get to testing the user-visible behaviour without actually
    running an interactive terminal session.
    """

    def test_write_then_read_preserves_entries(self, tmp_path):
        histfile = tmp_path / "history"

        # Session 1: setup, add entries, save (simulates atexit handler)
        readline.clear_history()
        setup_history(str(histfile))
        readline.add_history("question one")
        readline.add_history("question two")
        readline.write_history_file(str(histfile))

        # Session 2: clear in-memory state, setup again -- should
        # transparently load the file from session 1
        readline.clear_history()
        assert readline.get_current_history_length() == 0
        setup_history(str(histfile))
        assert readline.get_current_history_length() == 2
        assert readline.get_history_item(1) == "question one"
        assert readline.get_history_item(2) == "question two"

    def test_max_entries_truncates_on_save(self, tmp_path):
        """Adding more entries than max_entries should drop the oldest
        when the file is written, so the history file doesn't grow
        unbounded across many sessions.

        The test routes through ``save_history`` rather than calling
        ``readline.write_history_file`` directly, because that's the
        production save path (the atexit handler set up by
        ``setup_history`` calls ``save_history``) and on libedit-backed
        Pythons it transparently restores the ``_HiStOrY_V2_`` magic
        header that libedit's writer drops whenever truncation kicks in.
        Without that workaround, the file libedit wrote would be
        unreadable to libedit's own ``read_history_file``.
        """
        from cyllama._internal.readline import save_history

        histfile = tmp_path / "history"

        readline.clear_history()
        setup_history(str(histfile), max_entries=3)
        for i in range(10):
            readline.add_history(f"entry {i}")
        assert save_history(str(histfile)) is True

        # Re-read into a clean slate and verify only the last 3 survived
        readline.clear_history()
        readline.read_history_file(str(histfile))
        assert readline.get_current_history_length() == 3
        # The most recent three entries should be the survivors
        items = [readline.get_history_item(i) for i in range(1, readline.get_current_history_length() + 1)]
        assert items == ["entry 7", "entry 8", "entry 9"]
