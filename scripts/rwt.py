#!/usr/bin/env python3
"""Self-contained smoke-test runner for built cyllama wheels.

``--venv`` names the environment under test and every subprocess runs that
interpreter directly. Without it the script falls back to ``uv run``, which
re-syncs whichever project owns the cwd -- so from the cyllama checkout it
would build the extension from source and test that, never the wheel.

``--cuda`` (and ``--cpu`` / ``--metal`` / ``--vulkan`` / ``--rocm`` /
``--sycl``) names the backend and points ``--venv`` at ``.venv-<backend>``; an
explicit ``--venv`` wins. Without one the backend is detected from what the
venv has installed. ``--metal`` and ``--cpu`` install the same ``cyllama``
distribution -- CI builds it with Metal on macOS and without it elsewhere --
so a bare ``cyllama`` in a venv is reported as ``metal`` on macOS.

``install`` is the only subcommand that writes to the venv: ``--wheel`` says
what to put there -- a local wheel or a requirement for the index, told apart
by shape -- and creating the venv is part of it. Every test target expects an
environment that already has cyllama in it.

``test`` takes one target -- ``test-all``, ``test-gen-all``, ``test-sd-3`` --
named identically to the generated Makefile rules; ``list tests`` prints them.

``run`` is ``install``, ``test`` and ``clean`` in one invocation, stopping at
the first step that fails and taking the options of all three. It is the whole
cycle for one backend, so a wheel can be checked on a machine that has nothing
installed yet without three commands that must agree on which venv they mean.
``--fast`` swaps ``test-all`` for ``test-gen-1``, ``test-gen-2`` and
``test-sd-3`` -- the same shape of coverage without the image cases that
dominate the wall clock, or the one case whose model may not be downloadable.

The script is organised as a handful of collaborating objects rather than
module state: :class:`Paths` resolves the directory layout, :class:`Env` owns
the environment under test (venv, backend, subprocesses), :class:`ModelRegistry`
knows where models and data assets come from, :class:`TestSuite` holds the test
cases, and :class:`Cli` wires them to argparse.

Examples:
    # create .venv-cuda and install the latest cyllama-cuda12 from the index;
    # the backend names the distribution, so --wheel is not needed here
    python rwt.py install --cuda
    python rwt.py install --metal          # macOS: the plain `cyllama` wheel

    # --wheel is only for pinning a version or naming a local artifact
    python rwt.py install --cuda --wheel cyllama-cuda12==0.4.2
    python rwt.py install --vulkan --wheel dist/cyllama_vulkan-0.4.3-cp312-abi3-win_amd64.whl

    # install, test everything, then remove the venv again -- one command
    python rwt.py run --cuda
    python rwt.py run --cuda --fast    # a short cycle instead of everything
    python rwt.py run --vulkan test-sd-all --timeout 900

    # run everything, one family, or one case
    python rwt.py test --cuda test-all
    python rwt.py test --cuda test-rag-all
    python rwt.py test --cuda test-sd-3 --timeout 600

    # against a venv somewhere else; the backend is detected from what is
    # installed, so no --cuda/--vulkan/... is needed
    python rwt.py test --venv /tmp/wheel-check test-all

    # show the matrix without downloading or running anything
    python rwt.py test --cuda test-all --dry-run

    # environment, registry and target listings
    python rwt.py info --cuda
    python rwt.py list
    python rwt.py download all --models-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata as md
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_NAME = Path(__file__).name
# The `project` column in the shared run history; see RunLog.
PROJECT = "cyllama"


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


class ModelSourceUnavailable(RuntimeError):
    """Raised when a model has no configured source and isn't on disk."""


# ---------------------------------------------------------------------------
# image checks
# ---------------------------------------------------------------------------


def read_png(path: Path) -> tuple[int, int, int, bytes]:
    """Decode an 8-bit, non-interlaced PNG to (width, height, channels, pixels).

    Stdlib only: the script must run standalone, and the venv under test has no
    image library. That covers what stb_image_write produces.

    Raises:
        OSError: the file cannot be read.
        ValueError, zlib.error: the file is not a PNG this can decode.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    header: tuple[int, ...] | None = None
    idat = bytearray()
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise ValueError("no IHDR chunk")
    width, height, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"unsupported PNG (bit depth {depth}, color type {color}, interlace {interlace})")

    raw = zlib.decompress(idat)
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise ValueError(f"image data is {len(raw)} bytes, expected {height * (stride + 1)}")
    pixels = bytearray()
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1) + 1
        ftype = raw[start - 1]
        row = bytearray(raw[start : start + stride])
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + (left + prev[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = row[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"row {y} has unknown filter type {ftype}")
        pixels += row
        prev = row
    return width, height, channels, bytes(pixels)


def channel_stddevs(pixels: bytes, channels: int) -> list[float]:
    """Population standard deviation of each channel of interleaved 8-bit pixels."""
    result = []
    for ch in range(channels):
        hist = Counter(pixels[ch::channels])
        n = sum(hist.values())
        mean = sum(v * k for v, k in hist.items()) / n
        result.append(math.sqrt(sum(k * (v - mean) ** 2 for v, k in hist.items()) / n))
    return result


# ---------------------------------------------------------------------------
# run history (keep identical across cyllama/inferna rwt.py and chimera rat.py)
# ---------------------------------------------------------------------------


class RunLog:
    """Run history in one SQLite database shared by rwt.py and rat.py.

    Every project writes to the same file, so `runs diff` compares any two runs:
    two versions, two backends, or two projects. Rows are written as each case
    ends; a run with no `finished_at` was interrupted. A database error disables
    recording with a warning and never fails the test run.
    """

    SCHEMA_VERSION = 1
    SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY,
    session         TEXT NOT NULL,  -- shared by the test steps of one `run`
    project         TEXT NOT NULL,
    target          TEXT NOT NULL,
    backend         TEXT NOT NULL,
    version         TEXT,
    artifact        TEXT,           -- distribution name, or binary path
    artifact_sha256 TEXT,           -- wheel RECORD, or binary
    git_commit      TEXT,
    git_dirty       INTEGER,
    host            TEXT NOT NULL,
    platform        TEXT NOT NULL,
    argv            TEXT NOT NULL,  -- JSON
    extra           TEXT,           -- JSON, project-specific
    started_at      TEXT NOT NULL,  -- UTC ISO 8601
    finished_at     TEXT,
    seconds         REAL,
    rc              INTEGER
);
CREATE TABLE IF NOT EXISTS cases (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    family  TEXT NOT NULL,
    n       TEXT NOT NULL,
    status  TEXT NOT NULL,  -- pass | fail | timeout | skip
    rc      INTEGER,
    seconds REAL NOT NULL,
    detail  TEXT
);
CREATE TABLE IF NOT EXISTS outputs (
    id      INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    bytes   INTEGER NOT NULL,
    sha256  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,  -- e.g. tokens_per_second
    value   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_key ON runs(project, backend, target, id);
CREATE INDEX IF NOT EXISTS cases_by_run ON cases(run_id);
CREATE INDEX IF NOT EXISTS outputs_by_case ON outputs(case_id);
CREATE INDEX IF NOT EXISTS metrics_by_case ON metrics(case_id);
"""

    def __init__(self, project: str, path: Path | None = None) -> None:
        self.project = project
        self.path = path or self.default_path()
        self.enabled = True
        self.session = uuid.uuid4().hex[:12]
        self.run_id: int | None = None
        self._db: sqlite3.Connection | None = None
        self._started = 0.0

    @staticmethod
    def default_path() -> Path:
        """``$RUNS_DB``, else ``~/config/runs/db.sqlite``.

        Not under ``~/.config``: snap-packaged browsers cannot read hidden
        directories, so a report written beside the database would not open.
        """
        return Path(os.environ.get("RUNS_DB") or "~/config/runs/db.sqlite").expanduser()

    # -- plumbing -----------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            # WAL lets a run in one project write while another project's run reads.
            db.execute("PRAGMA journal_mode = WAL")
            found = db.execute("PRAGMA user_version").fetchone()[0]
            if found > self.SCHEMA_VERSION:
                db.close()
                raise sqlite3.DatabaseError(f"{self.path} has schema {found}; this script knows {self.SCHEMA_VERSION}")
            db.executescript(self.SCHEMA)
            db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            self._db = db
        return self._db

    def _tx(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run `fn` in one transaction; on any database error, stop recording."""
        if not self.enabled:
            return None
        try:
            db = self.connect()
            with db:
                return fn(db)
        except (sqlite3.Error, OSError) as e:
            print(f"warning: run history disabled ({self.path}): {e}", file=sys.stderr)
            self.enabled = False
            return None

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def git_state(root: Path) -> tuple[str | None, bool | None]:
        """(HEAD commit, has uncommitted changes) of `root`; (None, None) outside git."""
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            )
            if head.returncode != 0:
                return None, None
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        return head.stdout.strip(), bool(status.stdout.strip())

    # -- recording ----------------------------------------------------------

    def start(
        self,
        target: str,
        backend: str,
        root: Path,
        version: str | None = None,
        artifact: str | None = None,
        artifact_sha256: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        commit, dirty = self.git_state(root)
        self._started = time.monotonic()
        row = (
            self.session,
            self.project,
            target,
            backend,
            version,
            artifact,
            artifact_sha256,
            commit,
            None if dirty is None else int(dirty),
            platform.node(),
            f"{sys.platform}-{platform.machine()}",
            json.dumps(sys.argv[1:]),
            json.dumps(extra or {}, sort_keys=True),
            self.now(),
        )
        self.run_id = self._tx(
            lambda db: (
                db.execute(
                    "INSERT INTO runs (session, project, target, backend, version, artifact, artifact_sha256,"
                    " git_commit, git_dirty, host, platform, argv, extra, started_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                ).lastrowid
            )
        )

    @staticmethod
    def status(rc: int, skipped: str | None) -> str:
        if skipped is not None:
            return "skip"
        if rc == 124:  # Env.run's timeout code
            return "timeout"
        return "pass" if rc == 0 else "fail"

    def case(
        self,
        family: str,
        n: str,
        rc: int,
        seconds: float,
        skipped: str | None = None,
        outputs: Sequence[Path] = (),
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Record one case, its `metrics`, and the size and sha256 of each output it left on disk."""
        if self.run_id is None:
            return
        run_id = self.run_id
        files = [(p.name, p.stat().st_size, self.sha256(p)) for p in outputs if p.is_file()]

        def write(db: sqlite3.Connection) -> None:
            case_id = db.execute(
                "INSERT INTO cases (run_id, family, n, status, rc, seconds, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, family, n, self.status(rc, skipped), None if skipped else rc, seconds, skipped),
            ).lastrowid
            db.executemany(
                "INSERT INTO outputs (case_id, name, bytes, sha256) VALUES (?, ?, ?, ?)",
                [(case_id, *f) for f in files],
            )
            db.executemany(
                "INSERT INTO metrics (case_id, name, value) VALUES (?, ?, ?)",
                [(case_id, k, v) for k, v in sorted((metrics or {}).items())],
            )

        self._tx(write)

    def finish(self, rc: int) -> None:
        if self.run_id is None:
            return
        row = (self.now(), time.monotonic() - self._started, rc, self.run_id)
        self._tx(lambda db: db.execute("UPDATE runs SET finished_at = ?, seconds = ?, rc = ? WHERE id = ?", row))
        self.run_id = None

    # -- reporting ----------------------------------------------------------

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        return self.connect().execute(sql, params).fetchall()

    def print_list(self, limit: int, backend: str | None = None, all_projects: bool = False) -> int:
        rows = self._query(
            "SELECT r.*, COUNT(c.id) AS ran, COALESCE(SUM(c.status = 'pass'), 0) AS passed"
            " FROM runs r LEFT JOIN cases c ON c.run_id = r.id"
            " WHERE (? OR r.project = ?) AND (? IS NULL OR r.backend = ?)"
            " GROUP BY r.id ORDER BY r.id DESC LIMIT ?",
            (all_projects, self.project, backend, backend, limit),
        )
        if not rows:
            print(f"no runs recorded in {self.path}")
            return 0
        print(
            f"{'id':>5}  {'started (UTC)':<19}  {'project':<8}  {'backend':<7}  {'version':<12}  "
            f"{'target':<16}  {'passed':>6}  {'secs':>7}  rc"
        )
        for r in reversed(rows):
            secs = f"{r['seconds']:.1f}" if r["seconds"] is not None else "-"
            rc = "-" if r["rc"] is None else str(r["rc"])
            print(
                f"{r['id']:>5}  {r['started_at'][:19]:<19}  {r['project']:<8}  {r['backend']:<7}  "
                f"{(r['version'] or '?'):<12}  {r['target']:<16}  {r['passed']:>3}/{r['ran']:<2}  {secs:>7}  {rc}"
            )
        return 0

    def _resolve_pair(self, a: int | None, b: int | None, backend: str | None) -> tuple[sqlite3.Row, sqlite3.Row]:
        """Runs `a` and `b`. Missing `b` is this project's latest finished run;
        missing `a` is the finished run before `b` with the same project, backend
        and target."""

        def one(sql: str, params: Sequence[Any], what: str) -> sqlite3.Row:
            rows = self._query(sql, params)
            if not rows:
                raise LookupError(f"no {what} in {self.path}")
            return rows[0]

        if b is None:
            run_b = one(
                "SELECT * FROM runs WHERE project = ? AND (? IS NULL OR backend = ?) AND finished_at IS NOT NULL"
                " ORDER BY id DESC LIMIT 1",
                (self.project, backend, backend),
                f"finished {self.project} run",
            )
        else:
            run_b = one("SELECT * FROM runs WHERE id = ?", (b,), f"run {b}")
        if a is None:
            run_a = one(
                "SELECT * FROM runs WHERE project = ? AND backend = ? AND target = ? AND id < ?"
                " AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1",
                (run_b["project"], run_b["backend"], run_b["target"], run_b["id"]),
                f"earlier {run_b['project']} {run_b['backend']} {run_b['target']} run to compare run {run_b['id']} with",
            )
        else:
            run_a = one("SELECT * FROM runs WHERE id = ?", (a,), f"run {a}")
        return run_a, run_b

    def _cases(
        self, run_id: int
    ) -> dict[tuple[str, str], tuple[sqlite3.Row, dict[str, sqlite3.Row], dict[str, float]]]:
        cases = self._query("SELECT * FROM cases WHERE run_id = ? ORDER BY id", (run_id,))
        result = {}
        for c in cases:
            outs = self._query("SELECT * FROM outputs WHERE case_id = ?", (c["id"],))
            mets = self._query("SELECT name, value FROM metrics WHERE case_id = ?", (c["id"],))
            result[(c["family"], c["n"])] = (c, {o["name"]: o for o in outs}, {m["name"]: m["value"] for m in mets})
        return result

    def print_diff(self, a: int | None = None, b: int | None = None, backend: str | None = None) -> int:
        try:
            run_a, run_b = self._resolve_pair(a, b, backend)
        except LookupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        def short(value: Any, n: int = 12) -> str:
            return "-" if value is None else str(value)[:n]

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        def commit(r: sqlite3.Row) -> str:
            return short(r["git_commit"], 10) + ("+dirty" if r["git_dirty"] else "")

        fields: list[tuple[str, Callable[[sqlite3.Row], str]]] = [
            ("run", lambda r: str(r["id"])),
            ("started", lambda r: r["started_at"][:19]),
            ("project", lambda r: r["project"]),
            ("backend", lambda r: r["backend"]),
            ("target", lambda r: r["target"]),
            ("version", lambda r: short(r["version"], 30)),
            ("artifact", lambda r: short(r["artifact_sha256"])),
            ("commit", commit),
            ("host", lambda r: r["host"]),
            ("seconds", lambda r: secs(r["seconds"])),
            ("rc", lambda r: short(r["rc"])),
        ]
        for label, get in fields:
            va, vb = get(run_a), get(run_b)
            mark = "*" if va != vb and label not in ("run", "started") else " "
            print(f"{mark} {label:<9}{va:<32}{vb}")

        print(f"\n  {'case':<14}{'A':<9}{'B':<9}{'secs A':>8}{'secs B':>8}{'delta':>9}  tok/s, outputs")
        for row in self._diff_rows(run_a["id"], run_b["id"]):
            mark = " " if row["status_a"] == row["status_b"] else "*"
            print(
                f"{mark} {row['case']:<14}{row['status_a']:<9}{row['status_b']:<9}"
                f"{secs(row['secs_a']):>8}{secs(row['secs_b']):>8}{row['delta']:>9}  {', '.join(row['notes'])}".rstrip()
            )
        return 0

    # rwt.py's tokens/s is end to end; rat.py's generation rate excludes prompt
    # time. Different names keep the two from being compared.
    RATES: tuple[tuple[str, str], ...] = (
        ("tokens_per_second", "tok/s"),
        ("generation_tokens_per_second", "gen tok/s"),
    )

    def _diff_rows(self, id_a: int, id_b: int) -> list[dict[str, Any]]:
        """One row per case of runs `id_a` and `id_b`: statuses, seconds, notes."""

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        cases_a, cases_b = self._cases(id_a), self._cases(id_b)
        rows = []
        for key in [*cases_a, *(k for k in cases_b if k not in cases_a)]:
            ca, outs_a, mets_a = cases_a.get(key, (None, {}, {}))
            cb, outs_b, mets_b = cases_b.get(key, (None, {}, {}))
            sa = ca["seconds"] if ca is not None else None
            sb = cb["seconds"] if cb is not None else None
            notes = []
            for metric, label in self.RATES:
                ta, tb = mets_a.get(metric), mets_b.get(metric)
                if ta is not None or tb is not None:
                    change = f" ({100 * (tb - ta) / ta:+.1f}%)" if ta and tb is not None else ""
                    notes.append(f"{label} {secs(ta)} -> {secs(tb)}{change}")
            for name in sorted({*outs_a, *outs_b}):
                oa, ob = outs_a.get(name), outs_b.get(name)
                if oa is None or ob is None:
                    notes.append(f"{name} {'new' if oa is None else 'missing'}")
                elif oa["sha256"] != ob["sha256"]:
                    notes.append(f"{name} changed ({oa['bytes']} -> {ob['bytes']} bytes)")
                else:
                    notes.append(f"{name} identical")
            rows.append(
                {
                    "case": " ".join(key),
                    "status_a": ca["status"] if ca is not None else "-",
                    "status_b": cb["status"] if cb is not None else "-",
                    "secs_a": sa,
                    "secs_b": sb,
                    "delta": f"{100 * (sb - sa) / sa:+.1f}%" if sa and sb is not None else "",
                    "notes": notes,
                }
            )
        return rows

    # -- html report --------------------------------------------------------

    REPORT_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --border: #e2e1dc;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --series: #2a78d6; --grid: #e8e7e3;
  --good: #006300; --critical: #b52f2f;
  --b-cuda: #2a78d6; --b-vulkan: #eb6834; --b-cpu: #1baf7a; --b-metal: #eda100; --b-rocm: #e87ba4; --b-sycl: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #232322; --border: #3a3a37;
    --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
    --series: #3987e5; --grid: #2e2e2c;
    --good: #4fbf4f; --critical: #e66767;
    --b-cuda: #3987e5; --b-vulkan: #d95926; --b-cpu: #199e70; --b-metal: #c98500; --b-rocm: #d55181; --b-sycl: #008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #232322; --border: #3a3a37;
  --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
  --series: #3987e5; --grid: #2e2e2c;
  --good: #4fbf4f; --critical: #e66767;
  --b-cuda: #3987e5; --b-vulkan: #d95926; --b-cpu: #199e70; --b-metal: #c98500; --b-rocm: #d55181; --b-sycl: #008300;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--surface); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 40px 0 4px; padding-top: 16px; border-top: 1px solid var(--border); }
h3 { font-size: 14px; margin: 20px 0 8px; color: var(--text-2); font-weight: 600; }
.meta { color: var(--text-2); margin: 0 0 16px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; }
td.num, th.num { text-align: right; }
td.notes { white-space: normal; min-width: 240px; color: var(--text-2); }
.pass { color: var(--good); } .fail, .timeout { color: var(--critical); font-weight: 600; }
.skip, .none { color: var(--text-3); }
.changed { background: var(--surface-2); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
figure { margin: 0; padding: 10px 12px 6px; background: var(--surface-2); border-radius: 8px; }
figcaption { font-size: 13px; font-weight: 600; }
figcaption span { color: var(--text-2); font-weight: 400; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 6px 0 10px; font-size: 12px; color: var(--text-2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.cmp-row { display: grid; grid-template-columns: 90px 1fr; gap: 10px; padding: 5px 0; border-top: 1px solid var(--border); }
.cmp-case { font-size: 13px; padding-top: 1px; }
.cmp-bar { display: flex; align-items: center; gap: 6px; height: 16px; font-size: 11px;
  color: var(--text-2); font-variant-numeric: tabular-nums; }
.cmp-bar + .cmp-bar { margin-top: 2px; }
.regression, .now-failing { color: var(--critical); font-weight: 600; }
.improvement, .now-passing { color: var(--good); font-weight: 600; }
.within-noise { color: var(--text-3); }
.scroll + .meta { margin-top: 12px; }
tr.flag td { background: var(--surface-2); }
.cmp-bar .fill { height: 10px; border-radius: 0 3px 3px 0; min-width: 2px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg .axis { fill: var(--text-3); font-size: 10px; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .release { stroke: var(--text-3); stroke-width: 1; stroke-dasharray: 3 3; }
svg .line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; }
svg .dot { fill: var(--series); stroke: var(--surface-2); stroke-width: 2; }
svg .hit { fill: transparent; cursor: default; }
svg .hit:hover + .dot, svg .dot.on { r: 6; }
#tip { position: fixed; pointer-events: none; display: none; z-index: 10; padding: 6px 8px;
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  font-size: 12px; white-space: pre; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
"""

    REPORT_JS = """
const tip = document.getElementById("tip");
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest(".hit");
  if (!t) return;
  tip.textContent = t.dataset.tip;
  tip.style.display = "block";
});
document.addEventListener("mousemove", (e) => {
  if (tip.style.display !== "block") return;
  const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (e.clientY + 14) + "px";
});
document.addEventListener("mouseout", (e) => {
  if (e.target.closest(".hit")) tip.style.display = "none";
});
"""

    @staticmethod
    def _svg_trend(points: list[tuple[str, float | None, str, str]], unit: str) -> str:
        """Line chart of one measure over runs; each point is (x label, value, tooltip,
        version). A None value (a failed or skipped case) breaks the line rather than
        plotting a time that measures nothing. A dashed line marks each version change."""
        w, h, left, right, top, bottom = 300, 140, 40, 8, 18, 20
        values = [p[1] for p in points if p[1] is not None]
        # Two gridline steps, each 1, 2, 2.5 or 5 times a power of ten.
        raw = max(max(values), 1e-9) / 2
        mag = 10 ** math.floor(math.log10(raw))
        step = next(f * mag for f in (1, 2, 2.5, 5, 10) if f * mag >= raw)
        top_value = 2 * step
        n = len(points)

        def x(i: int) -> float:
            return left + (w - left - right) * (i / (n - 1) if n > 1 else 0.5)

        def y(v: float) -> float:
            return top + (h - top - bottom) * (1 - v / top_value)

        parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(unit)} per run">']
        for v in (0, step, top_value):
            parts.append(f'<line class="gridline" x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            parts.append(f'<text class="axis" x="{left - 6}" y="{y(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
        for i in range(1, n):
            if points[i][3] != points[i - 1][3]:
                xv = (x(i - 1) + x(i)) / 2
                parts.append(f'<line class="release" x1="{xv:.1f}" x2="{xv:.1f}" y1="{top - 4}" y2="{h - bottom}"/>')
                parts.append(
                    f'<text class="axis" x="{xv + 3:.1f}" y="{top - 6}">{html.escape(points[i][3][:16])}</text>'
                )
        for i in {0, n - 1}:
            parts.append(
                f'<text class="axis" x="{x(i):.1f}" y="{h - 4}" text-anchor="middle">{html.escape(points[i][0])}</text>'
            )
        segment: list[str] = []
        for i, (_, v, _, _) in enumerate([*points, ("", None, "", "")]):
            if v is not None:
                segment.append(f"{x(i):.1f},{y(v):.1f}")
            elif segment:
                if len(segment) > 1:
                    parts.append(f'<polyline class="line" points="{" ".join(segment)}"/>')
                segment = []
        for i, (_, v, tip, _) in enumerate(points):
            if v is not None:
                parts.append(
                    f'<circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="11" data-tip="{html.escape(tip)}"/>'
                )
                parts.append(f'<circle class="dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4"/>')
        parts.append("</svg>")
        return "".join(parts)

    # A version is a regression (or improvement) on a measure when its median moves
    # by at least this much AND lands outside the range of the previous version's
    # runs. The second condition keeps one noisy run from deciding the verdict.
    REGRESSION_PCT = 10.0

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        """Numeric parts of `version`, for ordering: "0.10.1" sorts above "0.9.3".
        Pre-release suffixes are not understood ("0.6.0rc1" sorts above "0.6.0")."""
        return tuple(int(part) for part in re.findall(r"\d+", version))

    def _version_rows(self, group: list[sqlite3.Row]) -> tuple[str, str, int, int, list[dict[str, Any]]] | None:
        """Compare the newest version in `group` (finished runs of one project,
        backend and target) with the next version below it.

        Versions are ordered by number, not by when they were tested, so a
        baseline recorded after the release it precedes still compares the
        right way round. Returns (previous, current, runs of previous, runs of
        current, rows), or None when the group has one version. Each row is one
        case and measure.
        """
        versions = sorted({r["version"] or "?" for r in group}, key=self._version_key)
        if len(versions) < 2:
            return None
        previous, current = versions[-2], versions[-1]
        runs_prev = [r for r in group if (r["version"] or "?") == previous]
        runs_cur = [r for r in group if (r["version"] or "?") == current]
        cases_prev = [self._cases(r["id"]) for r in runs_prev]
        cases_cur = [self._cases(r["id"]) for r in runs_cur]
        keys = list(dict.fromkeys(k for c in [*cases_cur, *cases_prev] for k in c))

        # (label, metric or None for seconds, higher is better)
        measures: list[tuple[str, str | None, bool]] = [
            ("seconds", None, False),
            *((label, metric, True) for metric, label in self.RATES),
        ]
        rows: list[dict[str, Any]] = []
        for key in keys:
            entries_prev = [c[key] for c in cases_prev if key in c]
            entries_cur = [c[key] for c in cases_cur if key in c]
            passed_prev = any(e[0]["status"] == "pass" for e in entries_prev)
            passed_cur = any(e[0]["status"] == "pass" for e in entries_cur)
            hashes_prev = {o["sha256"] for e in entries_prev for o in e[1].values()}
            hashes_cur = {o["sha256"] for e in entries_cur for o in e[1].values()}
            note = "image changed" if hashes_prev and hashes_cur and hashes_prev != hashes_cur else ""
            if entries_prev and entries_cur and passed_prev != passed_cur:
                rows.append(
                    {
                        "case": " ".join(key),
                        "measure": "status",
                        "prev": None,
                        "cur": None,
                        "n_prev": len(entries_prev),
                        "n_cur": len(entries_cur),
                        "delta": None,
                        "verdict": "now failing" if passed_prev else "now passing",
                        "note": note,
                    }
                )
                continue
            for label, metric, higher_better in measures:

                def values(entries: list[Any], metric: str | None = metric) -> list[float]:
                    out = []
                    for c, _, m in entries:
                        v = c["seconds"] if metric is None else m.get(metric)
                        if c["status"] == "pass" and v is not None:
                            out.append(v)
                    return out

                vp, vc = values(entries_prev), values(entries_cur)
                if not vp or not vc:
                    continue
                a, b = statistics.median(vp), statistics.median(vc)
                delta = 100 * (b - a) / a if a else 0.0
                worse = delta < 0 if higher_better else delta > 0
                outside = b < min(vp) or b > max(vp)
                if abs(delta) >= self.REGRESSION_PCT and outside:
                    verdict = "regression" if worse else "improvement"
                else:
                    verdict = "within noise"
                rows.append(
                    {
                        "case": " ".join(key),
                        "measure": label,
                        "prev": a,
                        "cur": b,
                        "n_prev": len(vp),
                        "n_cur": len(vc),
                        "delta": delta,
                        "verdict": verdict,
                        "note": note if metric is None else "",
                    }
                )
        return previous, current, len(runs_prev), len(runs_cur), rows

    # Backend -> CSS colour token. Colour follows the backend, never its position
    # among the backends a chart happens to show.
    BACKEND_ORDER: tuple[str, ...] = ("cuda", "vulkan", "cpu", "metal", "rocm", "sycl")

    @classmethod
    def _html_backends(
        cls,
        title: str,
        note: str,
        runs: dict[str, sqlite3.Row],
        rows: list[tuple[str, dict[str, tuple[float | None, str]]]],
    ) -> str:
        """Grouped horizontal bars: one row per case, one bar per backend. Each row
        is scaled to its own longest bar; the value label carries the magnitude."""
        esc = html.escape
        backends = sorted(runs, key=lambda b: cls.BACKEND_ORDER.index(b) if b in cls.BACKEND_ORDER else 99)
        parts = [f"<figure><figcaption>{esc(title)} <span>{esc(note)}</span></figcaption>", '<div class="legend">']
        for b in backends:
            r = runs[b]
            parts.append(
                f'<span><i style="background: var(--b-{esc(b)}, var(--text-3))"></i>{esc(b)} '
                f"(run {r['id']}, {esc(r['version'] or '?')})</span>"
            )
        parts.append("</div>")
        for case, values in rows:
            top = max((v for v, _ in values.values() if v is not None), default=0.0) or 1.0
            parts.append(f'<div class="cmp-row"><div class="cmp-case">{esc(case)}</div><div>')
            for b in backends:
                value, status = values.get(b, (None, "not run"))
                if value is None:
                    tip = f"{case}  {b}\n{status}"
                    parts.append(f'<div class="cmp-bar hit" data-tip="{esc(tip)}">{esc(status)}</div>')
                    continue
                tip = f"{case}  {b}\nrun {runs[b]['id']}  {runs[b]['version'] or '?'}\n{value:.2f}"
                parts.append(
                    f'<div class="cmp-bar hit" data-tip="{esc(tip)}"><span class="fill" '
                    f'style="width: calc((100% - 48px) * {value / top:.4f}); background: var(--b-{esc(b)}, var(--text-3))">'
                    f"</span>{value:.1f}</div>"
                )
            parts.append("</div></div>")
        parts.append("</figure>")
        return "".join(parts)

    def _html_versions(self, runs: list[sqlite3.Row]) -> list[str]:
        """The "version over version" section: per project, backend and target,
        the latest version against the one tested before it."""
        esc = html.escape
        groups: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
        for r in reversed(runs):  # oldest first
            if r["finished_at"] is not None:
                groups.setdefault((r["project"], r["backend"], r["target"]), []).append(r)
        compared, single = [], []
        for (project, backend, target), group in groups.items():
            result = self._version_rows(group)
            name = f"{project} / {backend} / {target}"
            if result is None:
                single.append(f"{name} ({group[-1]['version'] or '?'}, {len(group)} run{'s' * (len(group) != 1)})")
            else:
                compared.append((name, result))

        out = [
            "<h2>Version over version</h2>",
            f'<p class="meta">Same project, backend and target; the newest version against the next one below '
            f"it, by version number. Medians over passing runs. A regression or improvement moves the median by at least "
            f"{self.REGRESSION_PCT:g}% and leaves the range of the previous version's runs. Builds that share a "
            f"version string count as one version.</p>",
        ]
        if compared:
            flagged = [
                f"{name}: {sum(r['verdict'] in ('regression', 'now failing') for r in rows)} regression(s)"
                for name, (_, _, _, _, rows) in compared
                if any(r["verdict"] in ("regression", "now failing") for r in rows)
            ]
            out.append(
                '<p class="regression">' + esc("; ".join(flagged)) + "</p>"
                if flagged
                else '<p class="improvement">No regressions.</p>'
            )
        for name, (previous, current, n_prev, n_cur, rows) in compared:
            out.append(
                f"<h3>{esc(name)}: {esc(previous)} ({n_prev} run{'s' * (n_prev != 1)}) &rarr; "
                f"{esc(current)} ({n_cur} run{'s' * (n_cur != 1)})</h3>"
            )
            out.append(
                '<div class="scroll"><table><tr><th>case</th><th>measure</th>'
                f'<th class="num">{esc(previous)}</th><th class="num">{esc(current)}</th>'
                '<th class="num">change</th><th>verdict</th><th>note</th></tr>'
            )
            for r in rows:
                cls = r["verdict"].replace(" ", "-")
                flag = ' class="flag"' if r["verdict"] != "within noise" else ""
                prev = "-" if r["prev"] is None else f"{r['prev']:.2f} <small>(n={r['n_prev']})</small>"
                cur = "-" if r["cur"] is None else f"{r['cur']:.2f} <small>(n={r['n_cur']})</small>"
                delta = "" if r["delta"] is None else f"{r['delta']:+.1f}%"
                out.append(
                    f"<tr{flag}><td>{esc(r['case'])}</td><td>{esc(r['measure'])}</td>"
                    f'<td class="num">{prev}</td><td class="num">{cur}</td><td class="num">{delta}</td>'
                    f'<td class="{cls}">{esc(r["verdict"])}</td><td class="notes">{esc(r["note"])}</td></tr>'
                )
            out.append("</table></div>")
        if single:
            out.append(
                '<p class="meta">One version recorded so far, so nothing to compare: '
                + esc("; ".join(single))
                + ". Test the next version on the same backend and target to compare.</p>"
            )
        return out

    def write_report(self, out: Path, limit: int, backend: str | None = None, all_projects: bool = False) -> bool:
        """Write a self-contained HTML report: recent runs, then per project,
        backend and target the latest-vs-previous diff and a trend per case.
        Returns False, writing nothing, when there are no runs to report."""
        esc = html.escape
        runs = self._query(
            "SELECT * FROM runs WHERE (? OR project = ?) AND (? IS NULL OR backend = ?) ORDER BY id DESC",
            (all_projects, self.project, backend, backend),
        )
        if not runs:
            print(f"no runs recorded in {self.path}")
            return False

        def cls(status: str) -> str:
            return status if status in ("pass", "fail", "timeout", "skip") else "none"

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        body = [
            "<h1>Run history</h1>",
            f'<p class="meta">{esc(str(self.path))} &middot; generated {esc(self.now())} &middot; '
            f"{len(runs)} runs{'' if all_projects else ' of ' + esc(self.project)}</p>",
            *self._html_versions(runs),
            "<h3>Recent runs</h3>",
            '<div class="scroll"><table><tr><th class="num">id</th><th>started (UTC)</th><th>project</th>'
            '<th>backend</th><th>version</th><th>target</th><th>commit</th><th class="num">passed</th>'
            '<th class="num">secs</th><th>result</th></tr>',
        ]
        counts = {
            r["run_id"]: (r["ran"], r["passed"])
            for r in self._query(
                "SELECT run_id, COUNT(*) AS ran, SUM(status = 'pass') AS passed FROM cases GROUP BY run_id"
            )
        }
        for r in runs[:limit]:
            ran, passed = counts.get(r["id"], (0, 0))
            result = (
                "running or interrupted" if r["rc"] is None else ("pass" if r["rc"] == 0 else f"fail (rc={r['rc']})")
            )
            commit = (r["git_commit"] or "-")[:10] + ("+dirty" if r["git_dirty"] else "")
            body.append(
                f'<tr><td class="num">{r["id"]}</td><td>{esc(r["started_at"][:19])}</td><td>{esc(r["project"])}</td>'
                f"<td>{esc(r['backend'])}</td><td>{esc(r['version'] or '?')}</td><td>{esc(r['target'])}</td>"
                f'<td>{esc(commit)}</td><td class="num">{passed}/{ran}</td><td class="num">{secs(r["seconds"])}</td>'
                f'<td class="{cls("pass" if r["rc"] == 0 else "none" if r["rc"] is None else "fail")}">{esc(result)}</td></tr>'
            )
        body.append("</table></div>")

        # Latest finished run of each backend, per project and target.
        latest: dict[tuple[str, str], dict[str, sqlite3.Row]] = {}
        for r in runs:  # newest first, so the first run seen per backend is its latest
            if r["finished_at"] is not None:
                latest.setdefault((r["project"], r["target"]), {}).setdefault(r["backend"], r)
        for (project, target), by_backend in latest.items():
            if len(by_backend) < 2:
                continue
            cases = {b: self._cases(r["id"]) for b, r in by_backend.items()}
            keys = list(dict.fromkeys(k for c in cases.values() for k in c))
            body.append(f"<h2>{esc(project)} &middot; {esc(target)} &middot; backends side by side</h2>")
            body.append('<div class="grid">')
            measures: list[tuple[str, str, str | None]] = [
                ("seconds", "lower is faster", None),
                *((label, "higher is faster", metric) for metric, label in self.RATES),
            ]
            for title, direction, metric in measures:
                rows = []
                for key in keys:
                    values: dict[str, tuple[float | None, str]] = {}
                    for b, c in cases.items():
                        entry = c.get(key)
                        if entry is None:
                            values[b] = (None, "not run")
                        elif entry[0]["status"] != "pass":
                            values[b] = (None, entry[0]["status"])
                        else:
                            value = entry[0]["seconds"] if metric is None else entry[2].get(metric)
                            values[b] = (value, "pass" if value is not None else "no value")
                    if metric is not None and all(v is None for v, _ in values.values()):
                        continue  # no backend recorded this metric for the case
                    rows.append((" ".join(key), values))
                if rows:
                    body.append(
                        self._html_backends(
                            f"{title}", f"latest run per backend; {direction}; bars scaled per case", by_backend, rows
                        )
                    )
            body.append("</div>")

        groups: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
        for r in runs:
            if r["finished_at"] is not None:
                groups.setdefault((r["project"], r["backend"], r["target"]), []).append(r)
        for (project, run_backend, target), group in groups.items():
            group = list(reversed(group[:limit]))  # oldest first, for the trend
            latest = group[-1]
            body.append(f"<h2>{esc(project)} &middot; {esc(run_backend)} &middot; {esc(target)}</h2>")
            body.append(
                f'<p class="meta">{len(group)} finished runs shown &middot; latest {esc(latest["version"] or "?")} '
                f"on {esc(latest['host'])}, {esc(latest['started_at'][:19])} UTC</p>"
            )
            if len(group) < 2:
                body.append(
                    '<p class="meta">One finished run. The diff and trend charts appear after the next run '
                    "of this project, backend and target.</p>"
                )
            else:
                prev = group[-2]
                body.append(f"<h3>Run {latest['id']} vs run {prev['id']}</h3>")
                body.append(
                    '<div class="scroll"><table><tr><th>case</th>'
                    f'<th>run {prev["id"]}</th><th>run {latest["id"]}</th><th class="num">secs {prev["id"]}</th>'
                    f'<th class="num">secs {latest["id"]}</th><th class="num">delta</th><th>tok/s, outputs</th></tr>'
                )
                for row in self._diff_rows(prev["id"], latest["id"]):
                    changed = ' class="changed"' if row["status_a"] != row["status_b"] else ""
                    body.append(
                        f"<tr{changed}><td>{esc(row['case'])}</td>"
                        f'<td class="{cls(row["status_a"])}">{esc(row["status_a"])}</td>'
                        f'<td class="{cls(row["status_b"])}">{esc(row["status_b"])}</td>'
                        f'<td class="num">{secs(row["secs_a"])}</td><td class="num">{secs(row["secs_b"])}</td>'
                        f'<td class="num">{esc(row["delta"])}</td><td class="notes">{esc(", ".join(row["notes"]))}</td></tr>'
                    )
                body.append("</table></div>")

            per_run = [(r, self._cases(r["id"])) for r in group]
            keys = list(per_run[-1][1])
            figures = []
            for key in keys:
                series: list[tuple[str, str, Callable[[Any, dict[str, float]], float | None]]] = [
                    ("seconds", "s", lambda c, m: c["seconds"] if c["status"] == "pass" else None),
                    *(
                        (label, label, lambda c, m, metric=metric: m.get(metric) if c["status"] == "pass" else None)
                        for metric, label in self.RATES
                    ),
                ]
                for title, unit, get in series:
                    points: list[tuple[str, float | None, str, str]] = []
                    for r, cases in per_run:
                        entry = cases.get(key)
                        value = get(entry[0], entry[2]) if entry is not None else None
                        status = entry[0]["status"] if entry is not None else "not run"
                        hashes = ", ".join(f"{n} {o['sha256'][:8]}" for n, o in entry[1].items()) if entry else ""
                        tip = (
                            f"run {r['id']}  {r['version'] or '?'}\n{r['started_at'][:19]} UTC\n"
                            + (f"{value:.2f} {unit}" if value is not None else status)
                            + (f"\n{hashes}" if hashes else "")
                        )
                        points.append((str(r["id"]), value, tip, r["version"] or "?"))
                    if sum(p[1] is not None for p in points) < 2:
                        continue
                    figures.append(
                        f"<figure><figcaption>{esc(' '.join(key))} <span>{esc(title)}</span></figcaption>"
                        f"{self._svg_trend(points, unit)}</figure>"
                    )
            if figures:
                body.append(
                    "<h3>Trend per case (x: run id; dashed line: new version; failed and skipped runs leave a gap)</h3>"
                )
                body.append(f'<div class="grid">{"".join(figures)}</div>')

        page = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Run history</title><style>{self.REPORT_CSS}</style></head>"
            f'<body><main>{"".join(body)}</main><div id="tip" role="tooltip"></div>'
            f"<script>{self.REPORT_JS}</script></body></html>\n"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"wrote {out}")
        return True


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    """The directory layout every other object resolves against."""

    root: Path
    models_dir: Path
    data_dir: Path

    # The checkout keeps text under tests/media but audio under tests/samples,
    # so look in both rather than making the caller pick one with --data-dir.
    data_fallback_names: tuple[str, ...] = ("tests/media", "tests/samples")

    @staticmethod
    def find_root() -> Path:
        """Locate the project root: the cwd for subprocesses and the parent of
        ``models/`` and ``.venv/``.

        This file is checked in as ``<repo>/scripts/rwt.py`` but is also
        meant to be copied out standalone (as ``./rwt.py``) into a bare
        uv-managed wheel-test directory. Walking up to the nearest project
        marker handles both layouts; using ``__file__``'s own directory would
        resolve to ``<repo>/scripts`` in-repo and download models to
        ``scripts/models``.
        """
        here = Path(__file__).resolve().parent
        for candidate in (here, *here.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                return candidate
        return here

    @classmethod
    def from_environ(cls) -> Paths:
        root = cls.find_root()
        return cls(
            root=root,
            models_dir=Path(os.environ.get("CYLLAMA_MODELS_DIR", root / "models")),
            data_dir=Path(os.environ.get("CYLLAMA_DATA_DIR", root / "tests" / "media")),
        )

    @property
    def data_dirs(self) -> list[Path]:
        return [self.data_dir, *(self.root / name for name in self.data_fallback_names)]

    def find_data_asset(self, name: str) -> Path | None:
        """First existing copy of `name` in --data-dir or the checkout's data dirs."""
        for d in self.data_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        return None


# ---------------------------------------------------------------------------
# the environment under test
# ---------------------------------------------------------------------------


class Env:
    """The environment cyllama is tested in: its venv, backend and subprocesses.

    When ``venv`` is set, every subprocess runs that interpreter *directly*
    rather than through ``uv run``. This matters: ``uv run`` re-syncs whichever
    project owns the cwd, so run from this checkout it would build cyllama from
    source and test that instead of the installed wheel. ``venv=None`` restores
    the legacy ``uv run`` behaviour.
    """

    # Backend -> distribution on PyPI. Only the GPU backends get a renamed
    # distribution; `cpu` and `metal` are both the plain `cyllama` wheel, which
    # CI builds with GGML_METAL=1 on macOS and GGML_METAL=0 everywhere else.
    BACKENDS: dict[str, str] = {
        "cpu": "cyllama",
        "metal": "cyllama",
        "cuda": "cyllama-cuda12",
        "vulkan": "cyllama-vulkan",
        "rocm": "cyllama-rocm",
        "sycl": "cyllama-sycl",
    }

    # Distribution -> backend, for detection. Inverting BACKENDS would be
    # ambiguous for `cyllama`, so resolve that one by platform: the macOS wheel
    # is the Metal wheel, and there is no CPU-only macOS wheel to confuse it with.
    DISTRIBUTIONS: dict[str, str] = {
        **{dist: b for b, dist in BACKENDS.items() if dist != "cyllama"},
        "cyllama": "metal" if sys.platform == "darwin" else "cpu",
    }

    # Default env for a given backend. Existing values in os.environ take
    # precedence -- only unset keys are populated from these defaults, so
    # callers can always override by exporting the variable themselves.
    BACKEND_ENV_DEFAULTS: dict[str, dict[str, str]] = {
        # Every subprocess here goes through `uv run`, which re-syncs the project
        # environment first. Against an installed wheel that is a no-op, but in an
        # editable checkout it *rebuilds the extension* -- and the backend is chosen
        # from the environment at compile time, so without GGML_CUDA=1 the rebuild
        # links a CPU-only extension against CUDA static libs and every test dies
        # with `undefined symbol: ggml_backend_cuda_reg`. Set it so a dev checkout
        # rebuilds for the backend it is being asked to test.
        "cuda": {"GGML_CUDA": "1"},
        "rocm": {"GGML_HIP": "1"},
        "sycl": {"GGML_SYCL": "1"},
        # Same reasoning, plus: pin Vulkan to a specific device by default;
        # override with GGML_VK_VISIBLE_DEVICES=... in the caller's env if needed.
        "vulkan": {"GGML_VULKAN": "1", "GGML_VK_VISIBLE_DEVICES": "1"},
    }

    _DETECT_SRC = """
import importlib.metadata as md
for dist, backend in {distributions!r}.items():
    try:
        md.distribution(dist)
        print(backend)
        break
    except md.PackageNotFoundError:
        pass
"""

    def __init__(
        self,
        paths: Paths,
        venv: Path | None = None,
        venv_python_version: str | None = None,
        uv: str | None = None,
    ) -> None:
        self.paths = paths
        # The venv under test; None means fall back to `uv run`.
        self.venv = venv
        # Interpreter `uv venv` should build the target env from (--python).
        # Left unset, uv picks its own default, which is not necessarily the
        # version a given wheel was built for.
        self.venv_python_version = venv_python_version
        # While set, `run` copies each child's stderr here as well as to ours.
        self.capture: bytearray | None = None
        # Resolve `uv` once. Everything this script shells out to Python for is
        # routed through `uv run` so it executes inside the project's uv venv
        # regardless of how the script itself was launched.
        self.uv = uv or shutil.which("uv") or "uv"

    # -- venv plumbing ------------------------------------------------------

    @staticmethod
    def venv_python(venv: Path) -> Path:
        """Interpreter path inside `venv`, on either the Windows or POSIX layout."""
        win = venv / "Scripts" / "python.exe"
        if win.exists():
            return win
        posix = venv / "bin" / "python"
        if posix.exists():
            return posix
        return win if os.name == "nt" else posix

    def ensure_venv(self, venv: Path) -> Path:
        """Create `venv` if it does not exist yet; return its interpreter."""
        py = self.venv_python(venv)
        if not py.exists():
            print(f"creating venv at {venv}")
            cmd = [self.uv, "venv", str(venv)]
            if self.venv_python_version:
                cmd += ["--python", self.venv_python_version]
            subprocess.run(cmd, check=True)
            py = self.venv_python(venv)
        return py

    def python_cmd(self) -> list[str]:
        """argv prefix that runs Python in the environment under test."""
        if self.venv is not None:
            return [str(self.venv_python(self.venv))]
        return [self.uv, "run", "python"]

    def pip_install(
        self,
        spec: list[str],
        upgrade: bool = False,
        reinstall: bool = False,
        extra: list[str] | None = None,
    ) -> int:
        """Install `spec` (plus any --with packages) into the environment under test."""
        cmd = [self.uv, "pip", "install"]
        if self.venv is not None:
            cmd += ["--python", str(self.ensure_venv(self.venv))]
        if upgrade:
            cmd.append("--upgrade")
        if reinstall:
            cmd.append("--reinstall")
        return self.run(cmd + spec + list(extra or []))

    # -- subprocesses -------------------------------------------------------

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
        """Kill `proc` and every process it spawned.

        ``proc.kill()`` reaps only the direct child. A venv's python.exe re-execs
        the real interpreter, so a timed-out image run leaves that grandchild alive
        holding several GiB of VRAM -- and every later test in the matrix then OOMs
        or crawls, which silently invalidates the whole run's timings. Take the
        entire tree down instead.
        """
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"warning: could not fully reap pid {proc.pid}", file=sys.stderr)

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        check: bool = False,
        timeout: float | None = None,
    ) -> int:
        """Run a subprocess; return the exit code.

        Unlike previous revisions, `check=False` is the default so callers
        can accumulate failures across a smoke-test matrix. Pass
        ``check=True`` to restore the old fail-fast behaviour.
        """
        print(f"$ {' '.join(cmd)}", flush=True)
        full_env = os.environ.copy()
        # Redirected stdout on Windows defaults to the ANSI codepage, and the sd log
        # callback emits byte-level BPE markers (U+0120, U+010A) that cp1252 cannot
        # encode -- one UnicodeEncodeError traceback per log line once the output is
        # piped to a file. Force UTF-8 so a logged run matches a console one.
        full_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            full_env.update(env)
        capture = self.capture
        proc = subprocess.Popen(
            cmd,
            cwd=self.paths.root,
            env=full_env,
            start_new_session=os.name != "nt",
            stderr=subprocess.PIPE if capture is not None else None,
        )
        reader = None
        if capture is not None:
            reader = threading.Thread(target=self._tee, args=(proc.stderr, capture), daemon=True)
            reader.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"error: command timed out after {timeout}s", file=sys.stderr)
            self._kill_tree(proc)
            rc = 124  # conventional timeout exit code
        if reader is not None:
            reader.join(timeout=10)
        if check and rc != 0:
            sys.exit(rc)
        return rc

    @staticmethod
    def _tee(pipe: Any, sink: bytearray) -> None:
        """Copy `pipe` to our stderr as it arrives, appending it to `sink`."""
        out = sys.stderr.buffer
        for chunk in iter(lambda: pipe.read1(1 << 16), b""):
            out.write(chunk)
            out.flush()
            sink += chunk

    def cyllama(self, argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
        return self.run([*self.python_cmd(), "-m", "cyllama", *argv], env=env, timeout=timeout)

    def cyllama_module(
        self,
        module: str,
        argv: list[str],
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> int:
        return self.run([*self.python_cmd(), "-m", module, *argv], env=env, timeout=timeout)

    def has_module(self, name: str) -> bool:
        """Whether `name` is importable in the environment under test."""
        proc = subprocess.run(
            [*self.python_cmd(), "-c", f"import {name}"],
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    # -- backend detection --------------------------------------------------

    def _detect_backend_in_venv(self, venv: Path) -> str | None:
        py = self.venv_python(venv)
        if not py.exists():
            return None
        proc = subprocess.run(
            [str(py), "-c", self._DETECT_SRC.format(distributions=self.DISTRIBUTIONS)],
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or None

    def detect_backend(self) -> str | None:
        # With an explicit target venv, ask *it* what is installed. importlib.metadata
        # here would describe the interpreter running this script, which under
        # `uv run` from the checkout is the project env, not the wheel under test.
        if self.venv is not None:
            return self._detect_backend_in_venv(self.venv)
        for dist, backend in self.DISTRIBUTIONS.items():
            try:
                md.distribution(dist)
                return backend
            except md.PackageNotFoundError:
                continue
        return None

    _DIST_SRC = """
import hashlib, importlib.metadata as md
for name in {names!r}:
    try:
        d = md.distribution(name)
    except md.PackageNotFoundError:
        continue
    print(name, d.version, hashlib.sha256((d.read_text("RECORD") or "").encode()).hexdigest())
    break
"""

    def installed_dist(self) -> tuple[str, str, str] | None:
        """(distribution, version, sha256 of its RECORD) in the environment under test.

        RECORD holds a hash of every installed file, so its hash tells two builds
        of one version apart. An editable install's RECORD does not change on rebuild.
        """
        proc = subprocess.run(
            [*self.python_cmd(), "-c", self._DIST_SRC.format(names=list(self.DISTRIBUTIONS))],
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        parts = proc.stdout.split()
        return (parts[0], parts[1], parts[2]) if proc.returncode == 0 and len(parts) == 3 else None

    def env_for(self, backend: str) -> dict[str, str]:
        """Return default env overrides for a backend, skipping keys the
        caller has already set in the surrounding environment."""
        defaults = self.BACKEND_ENV_DEFAULTS.get(backend, {})
        return {k: v for k, v in defaults.items() if k not in os.environ}

    def require_backend(self, requested: str | None) -> str:
        detected = self.detect_backend()
        if requested and detected and requested != detected:
            print(
                f"warning: requested backend '{requested}' but '{detected}' is installed",
                file=sys.stderr,
            )
        backend = requested or detected
        if not backend:
            flags = ",".join("--" + b for b in self.BACKENDS)
            if self.venv is not None:
                print(
                    f"error: no cyllama backend installed in {self.venv}."
                    f"\n  Install from the index: {SCRIPT_NAME} install --venv {self.venv} {{{flags}}}"
                    f"\n  ...or a local wheel:    {SCRIPT_NAME} install --venv {self.venv} --wheel <path>",
                    file=sys.stderr,
                )
            else:
                print(
                    f"error: no cyllama backend installed. Run: {SCRIPT_NAME} install {{{flags}}}",
                    file=sys.stderr,
                )
            sys.exit(2)
        return backend

    # Prints the ggml device types after loading every backend. ggml logs to
    # stderr, so stdout carries only this line.
    _DEVICES_SRC = """
from cyllama.llama import llama_cpp as cy
cy.ggml_backend_load_all()
print(",".join(d["type"] for d in cy.ggml_backend_dev_info()))
"""

    # Command that shows whether the driver for a GPU backend is working.
    DRIVER_CHECKS: dict[str, str] = {
        "cuda": "nvidia-smi",
        "rocm": "rocm-smi",
        "vulkan": "vulkaninfo --summary",
        "sycl": "sycl-ls",
    }

    def preflight(self, backend: str) -> str | None:
        """Import cyllama once up front; return an error message, or None if fine.

        Every test shells out through `uv run`, which re-syncs the project first.
        Against an installed wheel that is a no-op. In an editable checkout it
        rebuilds the extension -- but only when the *sources* changed, never
        because the environment did, so an extension previously built for another
        backend is reused as-is. Linked against this backend's static libs it then
        fails to import, and without this check that arrives once per test as an
        `undefined symbol` traceback with no hint of the cause.

        For a GPU backend it also requires ggml to register a GPU device. When
        the driver is missing, ggml falls back to the CPU and every test still
        passes.
        """
        proc = subprocess.run(
            [*self.python_cmd(), "-c", self._DEVICES_SRC],
            cwd=self.paths.root,
            env={**os.environ, **self.env_for(backend)},
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            types = proc.stdout.strip().split(",")
            if backend == "cpu" or {"GPU", "iGPU"} & set(types):
                return None
            check = self.DRIVER_CHECKS.get(backend)
            hint = f"\n  Check the driver: {check}" if check else ""
            return f"backend '{backend}' registered no GPU device (devices: {','.join(types) or 'none'}){hint}"
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        hint = ""
        if self.venv is not None:
            hint = (
                f"\n  Environment under test: {self.venv_python(self.venv)}"
                f"\n  Install from the index:   {SCRIPT_NAME} install --venv {self.venv} --{backend}"
                f"\n  ...or a local wheel:      {SCRIPT_NAME} install --venv {self.venv} --wheel <path-to-wheel>"
            )
        elif "undefined symbol" in tail:
            env_key = next(iter(self.BACKEND_ENV_DEFAULTS.get(backend, {})), None)
            if env_key:
                hint = (
                    f"\n  The installed cyllama was not built for '{backend}'. In an editable"
                    f"\n  checkout, rebuild it:  {env_key}=1 uv pip install -e ."
                )
        return f"cannot import cyllama: {tail}{hint}"


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    """Where to fetch a model from.

    One of repo_id (HF Hub) or url (direct http) must be set.
    """

    filename: str
    repo_id: str | None = None
    hf_filename: str | None = None  # defaults to filename
    url: str | None = None
    notes: str = ""

    def hub_filename(self) -> str:
        return self.hf_filename or self.filename


class ModelRegistry:
    """Known models and data assets, and how to get them onto disk."""

    JFK_WAV_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

    # Which tests need which models.
    SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]
    RAG_REQUIREMENTS: list[str] = ["qwen3-4b", "bge-small-en"]

    # One text per line -- the format `cyllama embed -f` expects. Deliberately
    # includes a cluster about mortality so the `--similarity "death and dying"`
    # query in the embed case has something to rank above its 0.5 threshold.
    GENERATED_CORPUS: list[str] = [
        "The old man knew that he was dying, and he felt no fear of it.",
        "Death comes for everyone eventually, and grief is the price of having loved.",
        "Mourners gathered at the graveside in the cold morning air.",
        "He had spent his last years writing about mortality and the end of life.",
        "The hospice nurse spoke gently about what the final days would be like.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The compiler performs constant folding before emitting machine code.",
        "Mount Kilimanjaro is the highest free-standing mountain in the world.",
        "She sold the bakery and moved to a small town near the coast.",
        "Quicksort has an average time complexity of O(n log n).",
        "The bridge was rebuilt after the flood washed away its central span.",
        "A leopard was found frozen near the western summit of the mountain.",
    ]

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.sources = self.default_sources()
        self.apply_env_overrides()

    @staticmethod
    def default_sources() -> dict[str, ModelSource]:
        """Best-effort defaults -- overridable via CYLLAMA_MODEL_<KEY>=repo_id:file
        or by placing files in the models dir yourself. Use `list-models` to inspect.
        """
        return {
            "llama-3.2-1b": ModelSource(
                filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                url="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf",
            ),
            "qwen3-4b": ModelSource(
                filename="Qwen3-4B-Q8_0.gguf",
                repo_id="Qwen/Qwen3-4B-GGUF",
                url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
            ),
            "gemma-e4b": ModelSource(
                filename="gemma-4-E4B-it-Q5_K_M.gguf",
                repo_id="",  # override via env if/when available
                notes="set CYLLAMA_MODEL_GEMMA_E4B=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q5_K_M.gguf",
            ),
            "z-image-turbo": ModelSource(
                filename="z_image_turbo-Q6_K.gguf",
                repo_id="",
                notes="set CYLLAMA_MODEL_Z_IMAGE_TURBO=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q6_K.gguf",
            ),
            "ae": ModelSource(
                filename="ae.safetensors",
                repo_id="black-forest-labs/FLUX.1-schnell",
                hf_filename="ae.safetensors",
                url="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            ),
            "bge-small-en": ModelSource(
                filename="bge-small-en-v1.5-q8_0.gguf",
                repo_id="CompendiumLabs/bge-small-en-v1.5-gguf",
                url="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
            ),
            "whisper-base-en": ModelSource(
                filename="ggml-base.en.bin",
                repo_id="ggerganov/whisper.cpp",
                url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            ),
        }

    def apply_env_overrides(self) -> None:
        """Allow overriding repo ids via env vars (CYLLAMA_MODEL_<KEY>=repo:file)."""
        for key, src in self.sources.items():
            env_key = "CYLLAMA_MODEL_" + key.upper().replace("-", "_")
            val = os.environ.get(env_key)
            if not val:
                continue
            if ":" in val:
                repo, fname = val.split(":", 1)
                src.repo_id = repo
                src.hf_filename = fname
            else:
                src.repo_id = val

    # -- downloads ----------------------------------------------------------

    @staticmethod
    def download_urllib(url: str, dest: Path) -> None:
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_report = time.monotonic()
        bytes_read = 0
        chunk = 1024 * 1024  # 1 MiB
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total_hdr = r.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bytes_read += len(buf)
                now = time.monotonic()
                if now - last_report >= 2.0:
                    if total:
                        pct = 100.0 * bytes_read / total
                        print(
                            f"  {bytes_read / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"  {bytes_read / 1e6:.1f} MB", flush=True)
                    last_report = now
        tmp.rename(dest)

    @staticmethod
    def download_hf(repo_id: str, filename: str, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                "error: huggingface_hub not installed. Install with: pip install huggingface_hub",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"downloading {repo_id}:{filename} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Land the file directly in the models dir rather than copying from the
        # HF cache. Newer huggingface_hub uses `local_dir_use_symlinks=False`
        # and places the file at `<local_dir>/<filename>`; older releases
        # fall back to the cache path which we then copy.
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            # Older huggingface_hub without local_dir kwarg.
            out = hf_hub_download(repo_id=repo_id, filename=filename)
        out_path = Path(out)
        if out_path != dest:
            shutil.copyfile(out_path, dest)

    # -- lookups ------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.paths.models_dir / self.sources[key].filename

    def ensure_model(self, key: str) -> Path:
        src = self.sources[key]
        dest = self.path_for(key)
        if dest.exists():
            return dest
        if src.url:
            self.download_urllib(src.url, dest)
        elif src.repo_id:
            self.download_hf(src.repo_id, src.hub_filename(), dest)
        else:
            raise ModelSourceUnavailable(f"no source configured for model '{key}' ({src.filename}). {src.notes}")
        return dest

    def ensure_models(self, keys: list[str]) -> dict[str, Path]:
        return {k: self.ensure_model(k) for k in keys}

    # -- data assets --------------------------------------------------------
    #
    # These are inputs rather than models. In the checkout they already exist
    # under tests/media; standalone they do not, so each has a fallback --
    # jfk.wav is fetched from whisper.cpp, and the corpus is synthesised rather
    # than downloaded, since the one in the repo is a copyrighted short story.

    def ensure_corpus(self) -> Path:
        """Path to a line-per-text corpus, preferring the checkout's own."""
        repo_copy = self.paths.find_data_asset("corpus1.txt")
        if repo_copy is not None:
            return repo_copy
        generated = self.paths.models_dir / "corpus_generated.txt"
        if not generated.exists():
            print(f"writing generated corpus -> {generated}")
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("\n".join(self.GENERATED_CORPUS) + "\n", encoding="utf-8")
        return generated

    def ensure_audio(self) -> Path:
        """Path to the jfk.wav sample, downloading it if the checkout lacks one."""
        repo_copy = self.paths.find_data_asset("jfk.wav")
        if repo_copy is not None:
            return repo_copy
        dest = self.paths.models_dir / "jfk.wav"
        if not dest.exists():
            self.download_urllib(self.JFK_WAV_URL, dest)
        return dest


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts in ~/projects/demo/scripts)
# ---------------------------------------------------------------------------

TestFn = Callable[[str, "float | None"], int]


class TestSuite:
    """The smoke-test cases, grouped into families.

    Every case has the same signature -- ``(backend, timeout) -> exit code`` --
    and its docstring is the one-line description `list tests` and the generated
    Makefile print, so keep them short.
    """

    # Every test family, in the order `test-all` runs them: cheap and
    # fast-failing first, the multi-minute image cases last.
    FAMILY_ORDER: tuple[str, ...] = ("embed", "transcribe", "gen", "rag", "sd")

    # What `run --fast` runs in place of `test-all`. The sd cases dominate the
    # wall clock and mostly re-exercise the same three modules, so the third --
    # cpu-offload plus flash-attn, the most machinery of the three -- stands in
    # for all of them. gen-3 is left out rather than the family being named as a
    # whole: `gemma-e4b` is a 5.5 GB download, the largest gen model and the one
    # least likely to be on disk. If the download fails the case is a skip, and a
    # skip is rc=2 -- which would stop the sequence before `clean` over a missing
    # model rather than a bad wheel.
    FAST_TARGETS: tuple[str, ...] = ("test-gen-1", "test-gen-2", "test-sd-3")

    # Z-Image-Turbo is distilled for 8 steps without guidance (upstream
    # docs/z_image.md). The CLI defaults, 20 steps at cfg 7.0, cost 5x the passes.
    # A fixed seed makes a backend's images comparable from one release to the next.
    SD_TURBO_SAMPLING: tuple[str, ...] = ("--steps", "8", "--cfg-scale", "1.0", "--seed", "42")
    SD_WIDTH, SD_HEIGHT = 512, 1024
    # Below this in every channel an image is blank: black from a NaN render, or
    # one flat colour. A real render is in the tens.
    SD_MIN_STDDEV = 2.0

    # Rows of the `--stats` table the gen cases print -> metric names in the
    # run history. Times are seconds.
    STATS_ROWS: dict[str, str] = {
        "Prompt tokens": "prompt_tokens",
        "Generated tokens": "generated_tokens",
        "Prompt eval time": "prompt_seconds",
        "Generation time": "generation_seconds",
        "Total time": "total_seconds",
        "Tokens/second": "tokens_per_second",
    }

    @classmethod
    def parse_stats(cls, text: str) -> dict[str, float]:
        """Metrics from the last `--stats` table in `text`; empty if there is none."""
        metrics: dict[str, float] = {}
        for m in re.finditer(r"^\s+([A-Za-z/ ]+?)\s+\|\s+([0-9.]+)", text, re.MULTILINE):
            name = cls.STATS_ROWS.get(m.group(1))
            if name:
                metrics[name] = float(m.group(2))
        return metrics

    # Human-readable section headings for the generated Makefile's help text.
    FAMILY_TITLES: dict[str, str] = {
        "embed": "Embedding",
        "transcribe": "Transcription",
        "gen": "Generation",
        "rag": "RAG",
        "sd": "Stable Diffusion",
    }

    def __init__(self, env: Env, models: ModelRegistry) -> None:
        self.env = env
        self.models = models
        self.families: dict[str, dict[str, TestFn]] = {
            "embed": {"1": self.embed_1},
            "transcribe": {"1": self.transcribe_1},
            "gen": {"1": self.gen_1, "2": self.gen_2, "3": self.gen_3},
            "rag": {"1": self.rag_1, "2": self.rag_2},
            "sd": {"1": self.sd_1, "2": self.sd_2, "3": self.sd_3},
        }
        # Declared separately from FAMILY_ORDER so a family added to one and not
        # the other is caught here rather than silently skipped by `test-all`.
        assert tuple(self.families) == self.FAMILY_ORDER, "families must match FAMILY_ORDER"
        # Same reasoning: a renamed case would otherwise turn `run --fast` into an
        # argparse KeyError deep in the sequence, after the install step has run.
        unknown = [t for t in self.FAST_TARGETS if t not in self.targets()]
        assert not unknown, f"FAST_TARGETS names no such target: {unknown}"

    # -- stable diffusion ---------------------------------------------------

    def sd_output(self, n: str) -> str:
        """Filename the sd case `n` writes its image to.

        The cases run with the project root as cwd, so their output lands there.
        Named here rather than inline in each case so `clean` sweeps exactly the
        files the suite produces instead of globbing the root for `*.png`.
        """
        return f"z_turbo_{n}.png"

    def rag_db(self) -> Path:
        """Vector store the persistent rag case builds. Named here for the same
        reason as :meth:`sd_output`: `clean` sweeps it, and only what it writes."""
        return self.env.paths.root / "vector.db"

    def images(self) -> list[Path]:
        """The images the sd cases write, one per case."""
        return [self.env.paths.root / self.sd_output(n) for n in sorted(self.families["sd"])]

    def outputs(self) -> list[Path]:
        """Every file the suite leaves in the project root."""
        return [*self.images(), self.rag_db()]

    def run_sd(self, n: str, argv: list[str], backend: str, timeout: float | None) -> int:
        """Run sd case `n` with its own `argv`, then check the image it wrote."""
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        out = self.env.paths.root / self.sd_output(n)
        # A stale image from an earlier run would otherwise pass the check.
        out.unlink(missing_ok=True)
        rc = self.env.cyllama_module(
            "cyllama.sd",
            [
                "txt2img",
                "--diffusion-model",
                str(paths["z-image-turbo"]),
                "--vae",
                str(paths["ae"]),
                "--llm",
                str(paths["qwen3-4b"]),
                *self.SD_TURBO_SAMPLING,
                "-H",
                str(self.SD_HEIGHT),
                "-W",
                str(self.SD_WIDTH),
                "-o",
                self.sd_output(n),
                *argv,
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )
        return rc or self.check_image(out)

    def check_image(self, path: Path) -> int:
        """Fail an image of the wrong size, or one with no variation in any channel.

        Exit 0 from the CLI only means an image was written; a NaN render still
        writes one, all black.
        """
        try:
            width, height, channels, pixels = read_png(path)
        except (OSError, ValueError, zlib.error) as e:
            print(f"error: {path.name}: {e}", file=sys.stderr)
            return 1
        if (width, height) != (self.SD_WIDTH, self.SD_HEIGHT):
            print(
                f"error: {path.name} is {width}x{height}, expected {self.SD_WIDTH}x{self.SD_HEIGHT}",
                file=sys.stderr,
            )
            return 1
        spread = max(channel_stddevs(pixels, channels))
        print(f"-- {path.name}: {width}x{height}, max channel stddev {spread:.1f}")
        if spread < self.SD_MIN_STDDEV:
            print(
                f"error: {path.name} is blank (max channel stddev {spread:.2f} < {self.SD_MIN_STDDEV})", file=sys.stderr
            )
            return 1
        return 0

    def sd_1(self, backend: str, timeout: float | None) -> int:
        """z_turbo te-on-cpu."""
        # Unqualified, this case is a pure-GPU run needing ~9.4 GiB (3.9 text
        # encoder + 5.5 diffusion) and OOMs on anything smaller: upstream
        # master-731 dropped `free_params_immediately`, so the conditioner's
        # weights now stay resident for the life of the context instead of being
        # freed once the prompt is encoded. Parking the text encoder's weights in
        # RAM frees enough for the diffusion model while every module still
        # computes on the GPU -- unlike test 2, which moves *all* the weights.
        #
        # --vae-tiling is not optional here. Placement alone still dies in VAE
        # decode: at 512x1024 it wants a 3328 MiB compute buffer with the 5.5 GiB
        # of diffusion weights still resident, and no `--params-backend` spelling
        # helps because that is a compute buffer, not weights (`te=cpu,vae=cpu`
        # fails identically). Tiling is what shrinks it.
        #
        # Measured on an 8 GiB RTX 4060 at 20 steps, cfg 7.0: 3.17 s/it, 69 s end
        # to end. `--auto-fit` also fits but declines the GPU altogether on a
        # single-GPU box (~143 s/it), which no wheel-test timeout would survive.
        return self.run_sd("1", ["--params-backend", "te=cpu", "--vae-tiling", "-p", "a lovely cat"], backend, timeout)

    def sd_2(self, backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload."""
        return self.run_sd("2", ["--offload-to-cpu", "--vae-on-cpu", "-p", "a lovely cat"], backend, timeout)

    def sd_3(self, backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload + flash-attn."""
        return self.run_sd(
            "3",
            ["--offload-to-cpu", "--diffusion-fa", "-p", "a lovely plump blue-eyed cat"],
            backend,
            timeout,
        )

    # -- generation ---------------------------------------------------------

    def gen_1(self, backend: str, timeout: float | None) -> int:
        """Llama-3.2-1B short prompt."""
        model = self.models.ensure_model("llama-3.2-1b")
        return self.env.cyllama(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Explain quantum entanglement in one paragraph.",
                "-n",
                "256",
                "--stats",
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def gen_2(self, backend: str, timeout: float | None) -> int:
        """Qwen3-4B streamed."""
        model = self.models.ensure_model("qwen3-4b")
        return self.env.cyllama(
            ["gen", "-m", str(model), "-p", "Write a haiku about GPUs.", "-n", "256", "--stream", "--stats"],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def gen_3(self, backend: str, timeout: float | None) -> int:
        """Gemma-4-E4B streamed."""
        model = self.models.ensure_model("gemma-e4b")
        return self.env.cyllama(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "List three interesting facts about octopuses.",
                "-n",
                "512",
                "--temperature",
                "0.7",
                "--stream",
                "--stats",
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    # -- embedding ----------------------------------------------------------

    def embed_1(self, backend: str, timeout: float | None) -> int:
        """corpus similarity ranking."""
        model = self.models.ensure_model("bge-small-en")
        corpus = self.models.ensure_corpus()
        return self.env.cyllama(
            [
                "embed",
                "-m",
                str(model),
                "-f",
                str(corpus),
                "--similarity",
                "death and dying",
                "--threshold",
                "0.5",
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    # -- transcription ------------------------------------------------------

    def transcribe_1(self, backend: str, timeout: float | None) -> int:
        """jfk.wav speech-to-text."""
        # The invariant is that transcription works on a bare wheel install: the
        # wheels declare no dependencies, so nothing on this path may import a
        # third-party package. Do not gate on numbers being present -- gating on
        # numpy would fail a *correctly* built wheel, which is the whole point.
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        rc = self.env.cyllama(
            ["transcribe", "-f", str(audio), "-m", str(model)],
            env=self.env.env_for(backend),
            timeout=timeout,
        )
        if rc != 0 and not self.env.has_module("numpy"):
            # Wheels built before numpy was removed from whisper/cli.py import it at
            # module scope while declaring no dependency on it, so they die on the
            # import rather than on anything whisper did.
            print(
                "hint: this wheel may predate the numpy removal in whisper/cli.py."
                "\n  Re-running with --with numpy will confirm that diagnosis;"
                "\n  if it then passes, the wheel needs rebuilding, not a dependency.",
                file=sys.stderr,
            )
        return rc

    # -- rag ----------------------------------------------------------------

    def rag_1(self, backend: str, timeout: float | None) -> int:
        """in-memory index + query."""
        paths = self.models.ensure_models(ModelRegistry.RAG_REQUIREMENTS)
        corpus = self.models.ensure_corpus()
        return self.env.cyllama(
            [
                "rag",
                "-m",
                str(paths["qwen3-4b"]),
                "-e",
                str(paths["bge-small-en"]),
                "-f",
                str(corpus),
                # The case script omits -p and drops into an interactive chat loop,
                # which a smoke test cannot drive; a single query exercises the same
                # index -> retrieve -> generate path and then exits.
                "-p",
                "What does this text say about death?",
                "-n",
                "128",
                "--sources",
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def rag_2(self, backend: str, timeout: float | None) -> int:
        """persistent sqlite vector store (build + reopen)."""
        paths = self.models.ensure_models(ModelRegistry.RAG_REQUIREMENTS)
        corpus = self.models.ensure_corpus()
        db = self.rag_db()
        if db.exists():
            db.unlink()  # start from nothing so the create path is covered

        def query(prompt: str) -> int:
            return self.env.cyllama(
                [
                    "rag",
                    "-m",
                    str(paths["qwen3-4b"]),
                    "-e",
                    str(paths["bge-small-en"]),
                    "-f",
                    str(corpus),
                    "--db",
                    str(db),
                    "-p",
                    prompt,
                    "-n",
                    "128",
                ],
                env=self.env.env_for(backend),
                timeout=timeout,
            )

        rc = query("What does this text say about death?")
        if rc != 0:
            return rc
        if not db.exists():
            print(f"error: --db was given but no store was created at {db}", file=sys.stderr)
            return 1
        # Second pass reopens the existing store instead of re-embedding: the whole
        # point of --db, and the only part a single run would not cover.
        print(f"-- reopening existing store ({db.stat().st_size} bytes)")
        return query("What is the mountain in this text?")

    # -- target bookkeeping -------------------------------------------------

    def targets(self) -> dict[str, tuple[str, str]]:
        """Map each ``test-*`` target name to the (family, case) it runs.

        One token per test -- ``test-all``, ``test-gen-all``, ``test-sd-3`` -- so
        the CLI and the generated Makefile name the same things.
        """
        targets: dict[str, tuple[str, str]] = {"test-all": ("all", "all")}
        for fam, mapping in self.families.items():
            for n in sorted(mapping):
                targets[f"test-{fam}-{n}"] = (fam, n)
            targets[f"test-{fam}-all"] = (fam, "all")
        return targets

    def describe(self, kind: str, n: str) -> str:
        """One-line description of a target, from the case's docstring."""
        if kind == "all":
            return "every test in every family"
        if n == "all":
            return f"all {kind} tests"
        return (self.families[kind][n].__doc__ or "").strip()

    def collect_runs(self, kind: str, n: str) -> list[tuple[str, str]]:
        """Expand ('all'|<family>, 'all'|'1'|...) into concrete (kind, n) pairs."""
        kinds = list(self.families) if kind == "all" else [kind]
        runs: list[tuple[str, str]] = []
        for k in kinds:
            mapping = self.families[k]
            if n == "all":
                runs.extend((k, nk) for nk in sorted(mapping))
            elif n in mapping:
                runs.append((k, n))
            elif kind != "all":
                # An explicit `test embed 3` is a mistake worth reporting; the same
                # number under `test all 3` just means "the families that have a 3".
                print(
                    f"error: no test '{n}' in family '{k}' (have: {', '.join(sorted(mapping))})",
                    file=sys.stderr,
                )
                sys.exit(2)
        if not runs:
            print(f"error: no tests matched kind={kind} n={n}", file=sys.stderr)
            sys.exit(2)
        return runs

    def run_case(self, kind: str, n: str, backend: str, timeout: float | None) -> int:
        return self.families[kind][n](backend, timeout)


# ---------------------------------------------------------------------------
# generated Makefile
# ---------------------------------------------------------------------------


class MakefileRenderer:
    """Renders the Makefile whose rules mirror this script's own targets."""

    PY_VAR = "uv run ./rwt.py"

    def __init__(self, env: Env, suite: TestSuite) -> None:
        self.env = env
        self.suite = suite
        self.lines: list[str] = []

    def render(self) -> str:
        self.lines = []
        backends = list(self.env.BACKENDS)

        family_targets: dict[str, list[str]] = {
            fam: [f"test-{fam}-{n}" for n in sorted(mapping)] + [f"test-{fam}-all"]
            for fam, mapping in self.suite.families.items()
        }
        width = max(len(t) for ts in family_targets.values() for t in ts) + 2

        # Group .PHONY into readable lines
        groups = [
            ["help", "sync", "info", "clean", "reset"],
            backends,
            [f"run-{b}" for b in backends],
            [f"run-{b}-fast" for b in backends],
            ["list-models", "list-tests", "download", "runs", "runs-diff", "report"],
            *family_targets.values(),
            ["test-all"],
        ]
        phony_lines = " \\\n\t\t".join(" ".join(g) for g in groups if g)

        add = self.lines.append
        add("")
        add(f"PY := {self.PY_VAR}")
        add("")
        add(f".PHONY: {phony_lines}")
        add("")
        add("help:")
        add('\t@echo "Available targets (frontend for $(PY)):"')
        add('\t@echo ""')
        add('\t@echo "  Setup:"')
        add('\t@echo "    sync         - uv sync dependencies"')
        add('\t@echo "    info         - show cyllama backend info"')
        add('\t@echo "    clean        - remove .venv and any test images"')
        add('\t@echo "    reset        - clean + sync"')
        for b in backends:
            dist = self.env.BACKENDS[b]
            add(f'\t@echo "    {b:<12} - install {dist}"')
        add('\t@echo ""')
        add('\t@echo "  Models:"')
        add('\t@echo "    list-models  - list known models and whether they are on disk"')
        add('\t@echo "    download     - download all known models (use $(PY) download <key> for one)"')

        for fam, mapping in self.suite.families.items():
            title = self.suite.FAMILY_TITLES.get(fam, fam)
            add('\t@echo ""')
            add(f'\t@echo "  {title} tests (backend auto-detected):"')
            for n in sorted(mapping):
                doc = (mapping[n].__doc__ or "").strip().rstrip(".")
                label = f"test-{fam}-{n}"
                add(f'\t@echo "    {label:<{width}}- {doc}"')
            label = f"test-{fam}-all"
            add(f'\t@echo "    {label:<{width}}- run all {fam} tests"')

        add('\t@echo ""')
        add('\t@echo "  Full cycle (install + test-all + clean):"')
        for b in backends:
            add(f'\t@echo "    run-{b:<8} - install, test and clean the {b} backend"')
        fast = ", ".join(self.suite.FAST_TARGETS)
        add(f'\t@echo "    run-<backend>-fast - as above, but {fast} in place of test-all"')
        add('\t@echo ""')
        add('\t@echo "    list         - list test targets and models"')
        add('\t@echo "    test-all     - run every test in every family"')
        add('\t@echo "    runs         - list recorded runs"')
        add('\t@echo "    runs-diff    - compare the latest run with the one before it"')
        add('\t@echo "    report       - write an HTML report of the run history and open it"')

        self.rule("sync", "sync")
        self.rule("info", "info")
        self.rule("clean", "clean")
        self.rule("reset", "reset")
        for b in backends:
            self.rule(b, f"install --{b}")
        for b in backends:
            self.rule(f"run-{b}", f"run --{b}")
            self.rule(f"run-{b}-fast", f"run --{b} --fast")
        self.rule("list-models", "list models")
        self.rule("list-tests", "list tests")
        self.rule("download", "download all")
        self.rule("runs", "runs list")
        self.rule("runs-diff", "runs diff")
        self.rule("report", "report")
        for target in self.suite.targets():
            if target != "test-all":
                self.rule(target, f"test {target}")
        self.rule("test-all", "test test-all")
        add("")
        return "\n".join(self.lines)

    def rule(self, target: str, args: str) -> None:
        self.lines.append("")
        self.lines.append(f"{target}:")
        self.lines.append(f"\t@$(PY) {args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class Cli:
    """Argparse wiring and the subcommand implementations.

    The parser is built against the *defaults* (so --help can quote them), then
    :meth:`configure` rebuilds the collaborators from what was actually parsed.
    """

    def __init__(self) -> None:
        self.paths = Paths.from_environ()
        # Where the --cpu/--cuda/--vulkan/... shorthands look for their venv.
        # Relative names resolve against root so the shorthand means the same
        # thing from any cwd.
        self.venv_prefix = os.environ.get("CYLLAMA_VENV_PREFIX", ".venv-")
        self.env = Env(self.paths)
        self.models = ModelRegistry(self.paths)
        self.suite = TestSuite(self.env, self.models)
        self.runlog = RunLog(PROJECT)

    # -- configuration ------------------------------------------------------

    def configure(self, args: argparse.Namespace) -> None:
        """Apply parsed options; every collaborator is rebuilt from them."""
        if getattr(args, "models_dir", None):
            self.paths.models_dir = Path(args.models_dir).expanduser().resolve()
        if getattr(args, "data_dir", None):
            self.paths.data_dir = Path(args.data_dir).expanduser().resolve()

        venv: Path | None = None
        if getattr(args, "venv", None):
            venv = Path(args.venv).expanduser().resolve()
        elif getattr(args, "backend", None):
            # --cuda etc. only fills in what was not given explicitly, so
            # `--cuda --venv /tmp/x` still targets /tmp/x.
            venv = (self.paths.root / f"{self.venv_prefix}{args.backend}").resolve()
        self.env.venv = venv
        self.env.venv_python_version = getattr(args, "python", None) or None

    # -- simple commands ----------------------------------------------------

    def cmd_info(self, _args: argparse.Namespace) -> int:
        backend = self.env.detect_backend()
        venv = self.env.venv
        target = str(self.env.venv_python(venv)) if venv is not None else sys.executable
        print(f"{'python:':<9}{target}")
        print(f"{'backend:':<9}{backend or '(none)'}")
        print(f"{'models:':<9}{self.paths.models_dir}")
        if backend:
            self.env.cyllama(["info"])
        return 0

    def cmd_sync(self, _args: argparse.Namespace) -> int:
        return self.env.run([self.env.uv, "sync"])

    def cmd_clean(self, args: argparse.Namespace) -> int:
        venv = self.env.venv if self.env.venv is not None else self.paths.root / ".venv"
        if venv.exists():
            print(f"removing {venv}")
            shutil.rmtree(venv)
        # The sd cases write their images into the project root, so a run leaves
        # z_turbo_*.png behind for the next `git status` to report.
        keep = self.suite.images() if getattr(args, "keep_images", False) else []
        for out in self.suite.outputs():
            if not out.exists():
                continue
            if out in keep:
                print(f"keeping {out}")
            else:
                print(f"removing {out}")
                out.unlink()
        return 0

    def cmd_reset(self, args: argparse.Namespace) -> int:
        rc = self.cmd_clean(args)
        if rc != 0:
            return rc
        return self.cmd_sync(args)

    # -- install ------------------------------------------------------------

    @staticmethod
    def resolve_install_spec(args: argparse.Namespace) -> list[str] | None:
        """What ``--wheel`` asks to install, or None if it was not given.

        The value is either a local artifact or a requirement for the index, told
        apart by shape rather than by a second flag: a URL or anything carrying a
        path separator or a ``.whl`` suffix is a file, everything else is a spec
        handed to ``uv pip install`` as written (``cyllama-cuda12``,
        ``cyllama-vulkan==0.4.3``, ``cyllama-cuda12[extra]``).
        """
        value = args.wheel
        if not value:
            return None

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return [value]  # a URL; uv resolves it itself

        path = Path(value).expanduser()
        looks_local = path.suffix == ".whl" or path.exists() or "/" in value or "\\" in value
        if not looks_local:
            return [value]

        resolved = path.resolve()
        if not resolved.exists():
            print(f"error: wheel not found: {resolved}", file=sys.stderr)
            sys.exit(2)
        return [str(resolved)]

    def cmd_install(self, args: argparse.Namespace) -> int:
        if args.version and args.wheel:
            print("error: --version and --wheel both say what to install; give one", file=sys.stderr)
            return 2
        spec = self.resolve_install_spec(args)
        if spec is None:
            # No --wheel: the backend names the distribution to fetch from the index.
            backend = getattr(args, "backend", None)
            if not backend:
                flags = "/".join("--" + b for b in self.env.BACKENDS)
                print(
                    f"error: give a backend ({flags}), or --wheel <path-or-spec>",
                    file=sys.stderr,
                )
                return 2
            dist = self.env.BACKENDS[backend]
            spec = [f"{dist}=={args.version}" if args.version else dist]
        return self.env.pip_install(spec, upgrade=args.upgrade, reinstall=args.reinstall, extra=args.extra)

    # -- registries ---------------------------------------------------------

    def cmd_download(self, args: argparse.Namespace) -> int:
        keys = list(self.models.sources) if args.key == "all" else [args.key]
        failures = 0
        for k in keys:
            try:
                path = self.models.ensure_model(k)
                print(f"ok: {k} -> {path}")
            except ModelSourceUnavailable as e:
                print(f"skip: {k}: {e}", file=sys.stderr)
                failures += 1
        return 1 if failures else 0

    def cmd_list_models(self, _args: argparse.Namespace) -> int:
        for key, src in self.models.sources.items():
            source = f"hf:{src.repo_id}:{src.hub_filename()}" if src.repo_id else (src.url or "(no source configured)")
            on_disk = "YES" if (self.paths.models_dir / src.filename).exists() else "no"
            print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
            if src.notes and not src.repo_id and not src.url:
                print(f"{'':<16} note: {src.notes}")
        return 0

    def cmd_list_tests(self, _args: argparse.Namespace) -> int:
        targets = self.suite.targets()
        width = max(len(t) for t in targets)
        for target, (kind, n) in targets.items():
            print(f"{target:<{width}}  {self.suite.describe(kind, n)}")
        return 0

    def cmd_list(self, args: argparse.Namespace) -> int:
        """`list` with no argument shows both registries; `list tests|models` narrows."""
        what = getattr(args, "what", "all")
        rc = 0
        if what in ("tests", "all"):
            if what == "all":
                print("tests:")
            rc |= self.cmd_list_tests(args)
        if what in ("models", "all"):
            if what == "all":
                print("\nmodels:")
            rc |= self.cmd_list_models(args)
        return rc

    def cmd_gen_makefile(self, args: argparse.Namespace) -> int:
        content = MakefileRenderer(self.env, self.suite).render()
        if args.output:
            Path(args.output).write_text(content)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(content)
        return 0

    # -- test ---------------------------------------------------------------

    @staticmethod
    def _use_color(no_color: bool) -> bool:
        if no_color or os.environ.get("NO_COLOR"):
            return False
        return sys.stdout.isatty()

    def cmd_test(self, args: argparse.Namespace) -> int:
        kind, n = self.suite.targets()[args.target]

        # --dry-run promises to touch nothing, so it precedes every other step.
        if args.dry_run:
            backend = getattr(args, "backend", None) or self.env.detect_backend() or "?"
            for k, case in self.suite.collect_runs(kind, n):
                print(f"would run: {k} {case} (backend={backend})")
            return 0

        backend = self.env.require_backend(getattr(args, "backend", None))
        runs = self.suite.collect_runs(kind, n)

        problem = self.env.preflight(backend)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

        color = self._use_color(args.no_color)
        green = "\033[32m" if color else ""
        red = "\033[31m" if color else ""
        reset = "\033[0m" if color else ""

        if not args.no_record:
            dist = self.env.installed_dist()
            self.runlog.start(
                target=args.target,
                backend=backend,
                root=self.paths.root,
                version=dist[1] if dist else None,
                artifact=dist[0] if dist else None,
                artifact_sha256=dist[2] if dist else None,
                extra={"venv": str(self.env.venv) if self.env.venv else None, "env": self.env.env_for(backend)},
            )

        results: list[tuple[str, str, int, float]] = []
        for k, case in runs:
            print(f"\n=== {k} test {case} (backend={backend}) ===")
            started = time.monotonic()
            skipped: str | None = None
            # Only the gen cases print `--stats`; every other case keeps a
            # terminal stderr.
            self.env.capture = bytearray() if k == "gen" else None
            try:
                rc = self.suite.run_case(k, case, backend, args.timeout)
            except ModelSourceUnavailable as e:
                print(f"skip: {e}", file=sys.stderr)
                rc = 2
                skipped = str(e)
            finally:
                captured, self.env.capture = self.env.capture, None
            secs = time.monotonic() - started
            results.append((k, case, rc, secs))
            outputs = [self.paths.root / self.suite.sd_output(case)] if k == "sd" else []
            metrics = self.suite.parse_stats(captured.decode("utf-8", "replace")) if captured else {}
            self.runlog.case(k, case, rc, secs, skipped, outputs, metrics)
            if rc != 0 and args.fail_fast:
                break

        # Summary
        print("\n=== summary ===")
        worst = 0
        for k, case, rc, secs in results:
            status = f"{green}PASS{reset}" if rc == 0 else f"{red}FAIL (rc={rc}){reset}"
            print(f"  {k} {case}: {status}  ({secs:.1f}s)")
            worst = max(worst, rc)
        passed = sum(1 for r in results if r[2] == 0)
        total = sum(r[3] for r in results)
        print(f"{passed}/{len(results)} passed in {total:.1f}s")
        self.runlog.finish(worst)
        return worst

    def cmd_runs(self, args: argparse.Namespace) -> int:
        backend = getattr(args, "backend", None)
        if args.action == "list":
            if args.ids:
                print("error: `runs list` takes no ids", file=sys.stderr)
                return 2
            return self.runlog.print_list(args.limit, backend, args.all_projects)
        if len(args.ids) > 2:
            print("error: `runs diff` takes at most two ids", file=sys.stderr)
            return 2
        ids: list[int | None] = [None] * (2 - len(args.ids)) + list(args.ids)
        return self.runlog.print_diff(ids[0], ids[1], backend)

    def cmd_report(self, args: argparse.Namespace) -> int:
        out = Path(args.output).expanduser() if args.output else self.runlog.path.with_name("report.html")
        written = self.runlog.write_report(out, args.limit, getattr(args, "backend", None), args.all_projects)
        if written and not args.no_open:
            webbrowser.open(out.resolve().as_uri())
        return 0

    # -- run ----------------------------------------------------------------

    def run_targets(self, args: argparse.Namespace) -> list[str]:
        """The test targets one `run` invocation covers, in order."""
        if not args.fast:
            return [args.target or "test-all"]
        if args.target is not None:
            print(
                f"error: --fast already names its targets ({', '.join(self.suite.FAST_TARGETS)});"
                f" drop it to run '{args.target}' alone",
                file=sys.stderr,
            )
            sys.exit(2)
        return list(self.suite.FAST_TARGETS)

    def cmd_run(self, args: argparse.Namespace) -> int:
        """install -> test... -> clean, stopping at the first step that fails.

        A failure leaves the venv in place rather than cleaning up after it: the
        thing worth inspecting when a wheel fails is the environment it failed in,
        and `clean` is one command away once it has been looked at.
        """

        def test_step(target: str) -> Callable[[argparse.Namespace], int]:
            def step(a: argparse.Namespace) -> int:
                a.target = target
                return self.cmd_test(a)

            return step

        targets = self.run_targets(args)
        steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
            ("install", self.cmd_install),
            *((f"test {t}", test_step(t)) for t in targets),
            ("clean", self.cmd_clean),
        ]

        if args.dry_run:
            # `test --dry-run` promises to touch nothing, and `run` inherits that
            # promise for the whole sequence: print the steps, run none of them.
            where = f" --venv {self.env.venv}" if self.env.venv is not None else ""
            for name, _ in steps:
                verb, _, target = name.partition(" ")
                if verb == "clean" and args.keep_images:
                    target = "--keep-images"
                elif verb == "install" and args.version:
                    target = f"--version {args.version}"
                print(f"would run: {SCRIPT_NAME} {verb}{where}{' ' + target if target else ''}")
            print()
            for _, step in steps[1 : 1 + len(targets)]:
                step(args)
            return 0

        for i, (name, step) in enumerate(steps):
            print(f"\n=== {name} ===")
            rc = step(args)
            if rc != 0:
                skipped = ", ".join(n for n, _ in steps[i + 1 :])
                print(f"\nerror: {name} failed (rc={rc}); skipping {skipped}", file=sys.stderr)
                return rc
        return 0

    # -- argparse -----------------------------------------------------------

    def common_parser(self) -> argparse.ArgumentParser:
        """Options accepted both before and after the subcommand."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--venv",
            metavar="PATH",
            default=argparse.SUPPRESS,
            help="virtualenv to test against; `install` creates it if missing. Every "
            "subprocess runs this interpreter directly instead of `uv run`, so the "
            "installed wheel is what gets tested even from inside the source checkout.",
        )
        c.add_argument(
            "--models-dir",
            "--models_dir",
            metavar="PATH",
            dest="models_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding the GGUF/safetensors models (default: {self.paths.models_dir})",
        )
        shorthand = c.add_mutually_exclusive_group()
        for backend in self.env.BACKENDS:
            shorthand.add_argument(
                f"--{backend}",
                dest="backend",
                action="store_const",
                const=backend,
                default=argparse.SUPPRESS,
                help=f"test the {backend} backend, in {self.venv_prefix}{backend} unless --venv says otherwise",
            )
        c.add_argument(
            "--data-dir",
            "--data_dir",
            metavar="PATH",
            dest="data_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding corpus1.txt / jfk.wav (default: {self.paths.data_dir})",
        )
        return c

    @staticmethod
    def install_parser() -> argparse.ArgumentParser:
        """Options that only mean something while writing to the venv."""
        i = argparse.ArgumentParser(add_help=False)
        i.add_argument(
            "--wheel",
            metavar="WHEEL|SPEC",
            default=None,
            help="override what to install: a local wheel "
            "(dist/cyllama_cuda12-0.4.3-cp312-abi3-win_amd64.whl) or a pinned "
            "requirement (cyllama-vulkan==0.4.3). Usually unnecessary -- without "
            "it the latest release of the backend's distribution is fetched from "
            "the index (--cuda -> cyllama-cuda12).",
        )
        i.add_argument(
            "--version",
            metavar="X",
            default=None,
            help="release of the backend's distribution to install (--cuda --version 0.5.2 -> "
            "cyllama-cuda12==0.5.2), e.g. to record a baseline for a version comparison. "
            "Default: the latest release.",
        )
        i.add_argument(
            "--with",
            dest="extra",
            action="append",
            metavar="PKG",
            help="extra package to install alongside the wheel (repeatable), "
            "e.g. --with numpy when diagnosing a wheel that predates a fix.",
        )
        i.add_argument(
            "--python",
            metavar="VERSION",
            help="interpreter for a venv created here (e.g. 3.12); passed to `uv venv --python`",
        )
        i.add_argument(
            "--upgrade",
            action="store_true",
            help="pass --upgrade to uv pip install",
        )
        i.add_argument(
            "--reinstall",
            action="store_true",
            help="pass --reinstall to uv pip install",
        )
        return i

    @staticmethod
    def clean_parser() -> argparse.ArgumentParser:
        """Options for what `clean` removes; shared by `clean` and `run`."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--keep-images",
            action="store_true",
            help="leave the images the sd tests wrote (z_turbo_*.png) in the project root",
        )
        return c

    @staticmethod
    def test_parser() -> argparse.ArgumentParser:
        """Options that shape a test run; shared by `test` and `run`."""
        t = argparse.ArgumentParser(add_help=False)
        t.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="per-test timeout in seconds (default: no timeout)",
        )
        t.add_argument(
            "--fail-fast",
            action="store_true",
            help="stop at the first failing test instead of running the full matrix",
        )
        t.add_argument(
            "--dry-run",
            action="store_true",
            help="print the test matrix without downloading or invoking anything",
        )
        t.add_argument(
            "--no-color",
            action="store_true",
            help="disable colored PASS/FAIL output in the summary",
        )
        t.add_argument(
            "--no-record",
            action="store_true",
            help=f"do not record the run in the run history ({RunLog.default_path()})",
        )
        return t

    def build_parser(self) -> argparse.ArgumentParser:
        common = self.common_parser()
        p = argparse.ArgumentParser(
            description="cyllama wheel tester",
            parents=[common],
            epilog=("example: rwt.py install --cuda && rwt.py test --cuda test-all --models-dir models"),
        )
        _sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

        class sub:  # noqa: N801 - thin shim so add_parser always inherits `common`
            @staticmethod
            def add_parser(
                name: str,
                parents: Sequence[argparse.ArgumentParser] = (),
                **kw: Any,
            ) -> argparse.ArgumentParser:
                return _sub.add_parser(name, parents=[common, *parents], **kw)

        sub.add_parser("info", help="show python/backend/models info").set_defaults(func=self.cmd_info)
        sub.add_parser("sync", help="uv sync project dependencies").set_defaults(func=self.cmd_sync)
        sub.add_parser(
            "clean",
            parents=[self.clean_parser()],
            help="remove the venv and any images the sd tests left behind",
        ).set_defaults(func=self.cmd_clean)
        sub.add_parser("reset", help="clean + sync").set_defaults(func=self.cmd_reset)

        inst = sub.add_parser(
            "install",
            parents=[self.install_parser()],
            help="install a cyllama wheel into --venv, creating it if needed",
        )
        inst.set_defaults(func=self.cmd_install)

        dl = sub.add_parser("download", help="download a model (or 'all')")
        dl.add_argument("key", choices=[*self.models.sources.keys(), "all"])
        dl.set_defaults(func=self.cmd_download)

        lst = sub.add_parser("list", help="list test targets and models (or one of them)")
        lst.add_argument(
            "what",
            nargs="?",
            choices=["tests", "models", "all"],
            default="all",
            help="which registry to show (default: both)",
        )
        lst.set_defaults(func=self.cmd_list)

        # The flat names this script used before `list` existed. Kept working, but
        # out of --help so there is one obvious spelling.
        sub.add_parser("list-models").set_defaults(func=self.cmd_list_models)
        sub.add_parser("list-tests").set_defaults(func=self.cmd_list_tests)

        gm = sub.add_parser("gen-makefile", help="generate the Makefile from this script's registries")
        gm.add_argument("-o", "--output", help="write to file instead of stdout (e.g. -o Makefile)")
        gm.set_defaults(func=self.cmd_gen_makefile)

        # `test` takes one target name -- `test-sd-3` rather than `test sd 3`, so a
        # target is a single token and matches the Makefile rule of the same name.
        t = sub.add_parser("test", parents=[self.test_parser()], help="run a test target (see `list tests`)")
        t.add_argument(
            "target",
            choices=list(self.suite.targets()),
            metavar="TARGET",
            help="one of the targets `list tests` prints, e.g. test-all, test-gen-1",
        )
        t.set_defaults(func=self.cmd_test)

        # `run` is the whole cycle in one command, so a wheel can be checked out of
        # a clean machine without three invocations that must agree on the backend.
        r = sub.add_parser(
            "run",
            parents=[self.install_parser(), self.test_parser(), self.clean_parser()],
            help="install, test, then clean -- stopping at the first failure",
        )
        r.add_argument(
            "target",
            nargs="?",
            default=None,
            choices=list(self.suite.targets()),
            metavar="[TARGET]",
            help="the target to run (default: test-all)",
        )
        r.add_argument(
            "--fast",
            action="store_true",
            help="the short cycle: run "
            + ", ".join(self.suite.FAST_TARGETS)
            + " in place of test-all, skipping the image cases that dominate the wall clock",
        )
        r.set_defaults(func=self.cmd_run)

        rs = sub.add_parser(
            "runs",
            help=f"list recorded runs, or diff two of them (history in {RunLog.default_path()})",
        )
        rs.add_argument("action", nargs="?", choices=["list", "diff"], default="list")
        rs.add_argument(
            "ids",
            nargs="*",
            type=int,
            metavar="ID",
            help="diff: `B` compares B with the run before it; `A B` compares the two; "
            "none compares the latest run with the one before it",
        )
        rs.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="list: how many runs (default: 20)",
        )
        rs.add_argument(
            "--all-projects",
            action="store_true",
            help="list: include every project's runs, not only " + PROJECT + "'s",
        )
        rs.set_defaults(func=self.cmd_runs)

        rep = sub.add_parser("report", help="write an HTML report of the run history and open it in the browser")
        rep.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help=f"where to write it (default: {RunLog.default_path().with_name('report.html')})",
        )
        rep.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="recent runs listed, and runs per project/backend/target trend (default: 20)",
        )
        rep.add_argument(
            "--all-projects",
            action="store_true",
            help="include every project's runs, not only " + PROJECT + "'s",
        )
        rep.add_argument("--no-open", action="store_true", help="write the report without opening it")
        rep.set_defaults(func=self.cmd_report)

        return p

    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        self.configure(args)
        return int(args.func(args) or 0)


def main() -> None:
    sys.exit(Cli().main())


if __name__ == "__main__":
    main()
