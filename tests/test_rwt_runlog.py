"""Tests for the run history in scripts/rwt.py (RunLog and its cmd_test hook)."""

import hashlib
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

RWT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rwt.py"


@pytest.fixture(scope="module")
def rwt():
    spec = importlib.util.spec_from_file_location("rwt", RWT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rwt"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("rwt", None)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "runs" / "db.sqlite"


def record(rwt, db_path, tmp_path, *, version, image_bytes, sd_rc=0, backend="cuda", target="test-all"):
    """One complete run: a passing gen case, an sd case with an image, a skip."""
    log = rwt.RunLog("cyllama", db_path)
    log.start(target=target, backend=backend, root=tmp_path, version=version, artifact="cyllama-cuda12")
    image = tmp_path / "z_turbo_3.png"
    image.write_bytes(image_bytes)
    log.case("gen", "1", 0, 2.0)
    log.case("sd", "3", sd_rc, 40.0, outputs=[image])
    log.case("gen", "3", 2, 0.1, skipped="no source")
    log.finish(max(0, sd_rc))
    return log


def test_records_run_cases_and_outputs(rwt, db_path, tmp_path):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"png-a")
    db = sqlite3.connect(db_path)
    run = db.execute("SELECT project, backend, version, rc, finished_at FROM runs").fetchone()
    assert run[:4] == ("cyllama", "cuda", "0.4.3", 0)
    assert run[4] is not None
    cases = db.execute("SELECT family, n, status, rc, detail FROM cases ORDER BY id").fetchall()
    assert cases == [
        ("gen", "1", "pass", 0, None),
        ("sd", "3", "pass", 0, None),
        ("gen", "3", "skip", None, "no source"),
    ]
    out = db.execute("SELECT name, bytes, sha256 FROM outputs").fetchall()
    assert out == [("z_turbo_3.png", 5, hashlib.sha256(b"png-a").hexdigest())]
    assert db.execute("PRAGMA user_version").fetchone()[0] == rwt.RunLog.SCHEMA_VERSION


@pytest.mark.parametrize(
    ("rc", "skipped", "status"),
    [(0, None, "pass"), (1, None, "fail"), (124, None, "timeout"), (2, "why", "skip")],
)
def test_status(rwt, rc, skipped, status):
    assert rwt.RunLog.status(rc, skipped) == status


def test_missing_output_is_not_recorded(rwt, db_path, tmp_path):
    log = rwt.RunLog("cyllama", db_path)
    log.start(target="test-sd-3", backend="cuda", root=tmp_path)
    log.case("sd", "3", 1, 5.0, outputs=[tmp_path / "absent.png"])
    log.finish(1)
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 0


def test_unfinished_run_has_no_finished_at(rwt, db_path, tmp_path):
    log = rwt.RunLog("cyllama", db_path)
    log.start(target="test-all", backend="cpu", root=tmp_path)
    log.case("gen", "1", 0, 1.0)
    row = sqlite3.connect(db_path).execute("SELECT finished_at, rc FROM runs").fetchone()
    assert row == (None, None)


def test_newer_schema_disables_recording(rwt, db_path, tmp_path, capsys):
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    db.execute(f"PRAGMA user_version = {rwt.RunLog.SCHEMA_VERSION + 1}")
    db.close()
    log = rwt.RunLog("cyllama", db_path)
    log.start(target="test-all", backend="cpu", root=tmp_path)
    log.case("gen", "1", 0, 1.0)
    log.finish(0)
    assert not log.enabled
    assert log.run_id is None
    assert "run history disabled" in capsys.readouterr().err


def test_diff_defaults_to_previous_run_with_same_key(rwt, db_path, tmp_path, capsys):
    record(rwt, db_path, tmp_path, version="0.4.2", image_bytes=b"png-a")
    record(rwt, db_path, tmp_path, version="0.4.2", image_bytes=b"png-a", backend="vulkan")
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"png-bb", sd_rc=1)
    capsys.readouterr()
    log = rwt.RunLog("cyllama", db_path)
    assert log.print_diff() == 0
    out = capsys.readouterr().out
    # Run 3 (cuda) compares with run 1 (cuda), not run 2 (vulkan).
    assert out.split("  run      1")[1].split()[0] == "3"
    assert "* version  0.4.2" in out
    assert "  backend  cuda" in out
    assert "* sd 3          pass     fail" in out
    assert "z_turbo_3.png changed (5 -> 6 bytes)" in out
    assert "  gen 1         pass     pass" in out


def test_diff_identical_outputs(rwt, db_path, tmp_path, capsys):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"same")
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"same")
    capsys.readouterr()
    assert rwt.RunLog("cyllama", db_path).print_diff(1, 2) == 0
    assert "z_turbo_3.png identical" in capsys.readouterr().out


def test_diff_without_history(rwt, db_path, tmp_path, capsys):
    log = rwt.RunLog("cyllama", db_path)
    assert log.print_diff() == 2
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    assert log.print_diff() == 2
    assert "no earlier cyllama cuda test-all run" in capsys.readouterr().err


def test_list_filters_by_project(rwt, db_path, tmp_path, capsys):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    other = rwt.RunLog("chimera", db_path)
    other.start(target="test-all", backend="cuda", root=tmp_path, version="0.2.16")
    other.finish(0)
    capsys.readouterr()
    rwt.RunLog("cyllama", db_path).print_list(20)
    out = capsys.readouterr().out
    assert "cyllama" in out and "chimera" not in out
    assert " 2/3 " in out
    rwt.RunLog("cyllama", db_path).print_list(20, all_projects=True)
    assert "chimera" in capsys.readouterr().out


def test_list_does_not_create_db(rwt, db_path, capsys):
    rwt.RunLog("cyllama", db_path).print_list(20)
    assert not db_path.exists()
    assert "no runs recorded" in capsys.readouterr().out


def test_cmd_test_records(rwt, db_path, tmp_path, monkeypatch):
    """`test` writes one run with a row per case; --no-record and --dry-run write nothing."""
    cli = rwt.Cli()
    cli.runlog = rwt.RunLog("cyllama", db_path)
    cli.paths.root = tmp_path
    monkeypatch.setattr(cli.env, "require_backend", lambda requested: "cpu")
    monkeypatch.setattr(cli.env, "preflight", lambda backend: None)
    monkeypatch.setattr(cli.env, "installed_dist", lambda: ("cyllama", "0.4.3", "ab" * 32))

    def run_case(kind, n, backend, timeout):
        if kind == "sd":
            (tmp_path / cli.suite.sd_output(n)).write_bytes(b"img")
        return 0 if n != "2" else 1

    monkeypatch.setattr(cli.suite, "run_case", run_case)

    assert cli.main(["test", "--no-color", "--no-record", "test-sd-all"]) == 1
    assert cli.main(["test", "--dry-run", "test-sd-all"]) == 0
    assert not db_path.exists()

    assert cli.main(["test", "--no-color", "test-sd-all"]) == 1
    db = sqlite3.connect(db_path)
    run = db.execute("SELECT target, backend, version, artifact, artifact_sha256, rc FROM runs").fetchall()
    assert run == [("test-sd-all", "cpu", "0.4.3", "cyllama", "ab" * 32, 1)]
    statuses = db.execute("SELECT n, status FROM cases ORDER BY id").fetchall()
    assert statuses == [("1", "pass"), ("2", "fail"), ("3", "pass")]
    assert db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 3


def stats_table(capsys):
    """stderr of the CLI's own `--stats` table, so a format change breaks this test."""
    from types import SimpleNamespace

    from cyllama.__main__ import _print_stats_table

    capsys.readouterr()
    _print_stats_table(
        SimpleNamespace(
            prompt_tokens=12,
            generated_tokens=256,
            prompt_time=0.05,
            generation_time=2.5,
            total_time=2.55,
            tokens_per_second=102.4,
        )
    )
    return capsys.readouterr().err


def test_parse_stats_reads_cli_table(rwt, capsys):
    text = "log line | not a stat\n" + stats_table(capsys)
    assert rwt.TestSuite.parse_stats(text) == {
        "prompt_tokens": 12,
        "generated_tokens": 256,
        "prompt_seconds": 0.05,
        "generation_seconds": 2.5,
        "total_seconds": 2.55,
        "tokens_per_second": 102.4,
    }
    assert rwt.TestSuite.parse_stats("no table here") == {}


def test_env_run_tees_stderr_when_capturing(rwt, tmp_path, capfd):
    env = rwt.Env(rwt.Paths(root=tmp_path, models_dir=tmp_path, data_dir=tmp_path))
    script = "import sys; print('out'); sys.stderr.write('err-line\\n'); sys.exit(3)"
    env.capture = bytearray()
    assert env.run([sys.executable, "-c", script]) == 3
    # The child writes stderr in text mode, so Windows emits CRLF.
    assert bytes(env.capture) == b"err-line" + os.linesep.encode()
    seen = capfd.readouterr()
    assert "err-line" in seen.err and "out" in seen.out
    env.capture = None
    assert env.run([sys.executable, "-c", script]) == 3
    assert "err-line" in capfd.readouterr().err


def test_metrics_recorded_and_diffed(rwt, db_path, tmp_path, capsys):
    for tps in (100.0, 110.0):
        log = rwt.RunLog("cyllama", db_path)
        log.start(target="test-gen-1", backend="cuda", root=tmp_path)
        log.case("gen", "1", 0, 2.0, metrics={"tokens_per_second": tps, "generated_tokens": 256})
        log.finish(0)
    rows = sqlite3.connect(db_path).execute("SELECT name, value FROM metrics ORDER BY id").fetchall()
    assert rows[:2] == [("generated_tokens", 256.0), ("tokens_per_second", 100.0)]
    capsys.readouterr()
    rwt.RunLog("cyllama", db_path).print_diff()
    assert "tok/s 100.0 -> 110.0 (+10.0%)" in capsys.readouterr().out


def test_cmd_test_records_gen_stats(rwt, db_path, tmp_path, monkeypatch, capsys):
    """A gen case's `--stats` table on stderr ends up in the metrics table."""
    table = stats_table(capsys)
    cli = rwt.Cli()
    cli.runlog = rwt.RunLog("cyllama", db_path)
    cli.paths.root = tmp_path
    monkeypatch.setattr(cli.env, "require_backend", lambda requested: "cpu")
    monkeypatch.setattr(cli.env, "preflight", lambda backend: None)
    monkeypatch.setattr(cli.env, "installed_dist", lambda: None)
    script = f"import sys; sys.stderr.write({table!r})"
    monkeypatch.setattr(
        cli.suite, "run_case", lambda kind, n, backend, timeout: cli.env.run([sys.executable, "-c", script])
    )
    assert cli.main(["test", "--no-color", "test-gen-1"]) == 0
    assert cli.env.capture is None
    tps = sqlite3.connect(db_path).execute("SELECT value FROM metrics WHERE name = 'tokens_per_second'").fetchall()
    assert tps == [(102.4,)]


def test_report(rwt, db_path, tmp_path, capsys):
    """Recent-runs table, a diff per group, a trend per case and measure, a
    version marker, a gap for a failed case, and escaped text."""
    record(rwt, db_path, tmp_path, version="0.4.2", image_bytes=b"a")
    record(rwt, db_path, tmp_path, version="0.4.2", image_bytes=b"a", sd_rc=1)
    record(rwt, db_path, tmp_path, version="0.4.3<x>", image_bytes=b"b")
    out = tmp_path / "r" / "report.html"
    assert rwt.RunLog("cyllama", db_path).write_report(out, 20) is True
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert "cyllama &middot; cuda &middot; test-all" in page
    assert "Run 3 vs run 2" in page
    assert 'class="fail">fail<' in page  # run 2's sd case in the diff
    assert "0.4.3&lt;x&gt;" in page and "0.4.3<x>" not in page
    assert page.count("<figure>") == 2  # gen 1 seconds, sd 3 seconds; gen 3 never passes
    assert page.count('class="release"') == 2  # one version change per chart
    # sd 3 failed in run 2, so its line breaks: two one-point segments, no polyline.
    sd = page[page.index("sd 3 <span>") :]
    assert "<polyline" not in sd.split("</figure>")[0]


def test_report_empty(rwt, db_path, tmp_path, capsys):
    out = tmp_path / "report.html"
    assert rwt.RunLog("cyllama", db_path).write_report(out, 20) is False
    assert not out.exists()
    assert "no runs recorded" in capsys.readouterr().out


def test_svg_trend_ticks(rwt):
    svg = rwt.RunLog._svg_trend([("1", 3.7, "t", "a"), ("2", None, "t", "a"), ("3", 4.1, "t", "b")], "s")
    # Half the max is 2.05; the next nice step is 2.5, so the axis tops out at 5.
    assert [t for t in (">0<", ">2.5<", ">5<") if t in svg] == [">0<", ">2.5<", ">5<"]
    assert svg.count("<polyline") == 0  # the None splits the only two points


def test_cli_report_writes_and_opens(rwt, db_path, tmp_path, monkeypatch):
    """`report` writes beside the database and opens it; --no-open and -o behave."""
    opened = []
    monkeypatch.setattr(rwt.webbrowser, "open", opened.append)
    cli = rwt.Cli()
    cli.runlog = rwt.RunLog("cyllama", db_path)
    if hasattr(cli, "configure"):
        monkeypatch.setattr(cli, "configure", lambda args: None)

    assert cli.main(["report"]) == 0  # empty history: nothing written, nothing opened
    assert opened == []

    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    assert cli.main(["report"]) == 0
    default = db_path.parent / "report.html"
    assert default.exists()
    assert opened == [default.resolve().as_uri()]

    custom = tmp_path / "custom.html"
    assert cli.main(["report", "--no-open", "-o", str(custom), "-n", "5", "--all-projects"]) == 0
    assert custom.exists()
    assert len(opened) == 1


def test_default_path_is_not_hidden(rwt, monkeypatch):
    """Snap browsers cannot read hidden directories, so the report would not open."""
    monkeypatch.delenv("RUNS_DB", raising=False)
    path = rwt.RunLog.default_path()
    assert path == Path("~/config/runs/db.sqlite").expanduser()
    assert not any(part.startswith(".") for part in path.relative_to(Path.home()).parts)


def test_report_explains_single_run_group(rwt, db_path, tmp_path):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    assert rwt.RunLog("cyllama", db_path).write_report(out, 20) is True
    page = out.read_text()
    assert "One finished run" in page
    assert "<figure>" not in page


def test_report_compares_backends(rwt, db_path, tmp_path):
    """Latest run per backend side by side; colour follows the backend; a failed
    case shows its status; a case with no metric on any backend is left out."""
    record(rwt, db_path, tmp_path, version="0.4.2", image_bytes=b"a", backend="vulkan")  # superseded below
    log = rwt.RunLog("cyllama", db_path)
    log.start(target="test-all", backend="vulkan", root=tmp_path, version="0.4.3")
    log.case("gen", "1", 0, 2.5, metrics={"tokens_per_second": 80.0})
    log.case("sd", "3", 1, 30.0)
    log.finish(1)
    log.start(target="test-all", backend="cuda", root=tmp_path, version="0.4.3")
    log.case("gen", "1", 0, 1.5, metrics={"tokens_per_second": 120.0})
    log.case("sd", "3", 0, 20.0)
    log.finish(0)
    out = tmp_path / "report.html"
    assert rwt.RunLog("cyllama", db_path).write_report(out, 20) is True
    page = out.read_text()
    section = page[page.index("backends side by side") :]
    assert "cuda (run 3, 0.4.3)" in section and "vulkan (run 2, 0.4.3)" in section  # latest per backend
    assert section.index("cuda (run") < section.index("vulkan (run")  # fixed backend order
    assert "var(--b-cuda" in section and "var(--b-vulkan" in section
    assert ">fail</div>" in section  # vulkan sd 3
    tok = section[section.index("tok/s <span>") :]
    tok = tok[: tok.index("</figure>")]
    assert "gen 1" in tok and "sd 3" not in tok and "120.0" in tok and "80.0" in tok


def test_report_no_comparison_for_one_backend(rwt, db_path, tmp_path):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    rwt.RunLog("cyllama", db_path).write_report(out, 20)
    assert "backends side by side" not in out.read_text()


def version_history(rwt, db_path, tmp_path, runs):
    """Record `runs`: (version, {case: (rc, seconds, tok/s or None, image bytes or None)})."""
    log = rwt.RunLog("cyllama", db_path)
    image = tmp_path / "z_turbo_3.png"
    for version, cases in runs:
        log.start(target="test-all", backend="cuda", root=tmp_path, version=version)
        for (family, n), (rc, secs, tps, img) in cases.items():
            outputs = []
            if img is not None:
                image.write_bytes(img)
                outputs = [image]
            metrics = {"tokens_per_second": tps} if tps is not None else None
            log.case(family, n, rc, secs, outputs=outputs, metrics=metrics)
        log.finish(0)
    return log


def verdicts(rows):
    return {(r["case"], r["measure"]): r["verdict"] for r in rows}


def test_version_rows_verdicts(rwt, db_path, tmp_path):
    old = {
        ("gen", "1"): (0, 10.0, 100.0, None),
        ("gen", "2"): (0, 10.0, 50.0, None),
        ("rag", "1"): (0, 5.0, None, None),
        ("sd", "3"): (0, 40.0, None, b"a"),
        ("embed", "1"): (0, 2.0, None, None),
    }
    log = version_history(
        rwt,
        db_path,
        tmp_path,
        [
            ("1.0", old),
            ("1.0", {**old, ("rag", "1"): (0, 7.0, None, None)}),  # rag 1 ranges 5..7 in 1.0
            (
                "1.1",
                {
                    ("gen", "1"): (0, 12.0, 80.0, None),  # +20% seconds, -20% tok/s
                    ("gen", "2"): (0, 10.5, 51.0, None),  # +5%: under the threshold
                    ("rag", "1"): (0, 6.6, None, None),  # +10% on the median but inside 5..7
                    ("sd", "3"): (0, 30.0, None, b"b"),  # -25%, new image
                    ("embed", "1"): (1, 2.0, None, None),  # passed in 1.0, fails in 1.1
                },
            ),
        ],
    )
    group = log._query("SELECT * FROM runs ORDER BY id")
    previous, current, n_prev, n_cur, rows = log._version_rows(group)
    assert (previous, current, n_prev, n_cur) == ("1.0", "1.1", 2, 1)
    assert verdicts(rows) == {
        ("gen 1", "seconds"): "regression",
        ("gen 1", "tok/s"): "regression",
        ("gen 2", "seconds"): "within noise",
        ("gen 2", "tok/s"): "within noise",
        ("rag 1", "seconds"): "within noise",
        ("sd 3", "seconds"): "improvement",
        ("embed 1", "status"): "now failing",
    }
    sd = next(r for r in rows if r["case"] == "sd 3")
    assert sd["note"] == "image changed"
    assert next(r for r in rows if r["case"] == "rag 1")["prev"] == 6.0  # median of 5 and 7


def test_version_rows_compares_with_last_different_version(rwt, db_path, tmp_path):
    case = {("gen", "1"): (0, 10.0, None, None)}
    log = version_history(rwt, db_path, tmp_path, [("1.0", case), ("1.1", case), ("1.2", case)])
    group = log._query("SELECT * FROM runs ORDER BY id")
    assert log._version_rows(group)[:2] == ("1.1", "1.2")
    assert log._version_rows(group[:1]) is None


def test_version_rows_orders_by_number_not_test_order(rwt, db_path, tmp_path):
    """A baseline tested after the newer release still compares old -> new."""
    case = {("gen", "1"): (0, 10.0, None, None)}
    log = version_history(rwt, db_path, tmp_path, [("0.10.0", case), ("0.9.3", case), ("0.6.0", case)])
    group = log._query("SELECT * FROM runs ORDER BY id")
    assert log._version_rows(group)[:4] == ("0.9.3", "0.10.0", 1, 1)


def test_report_version_section(rwt, db_path, tmp_path):
    slow = {("gen", "1"): (0, 15.0, None, None)}
    version_history(rwt, db_path, tmp_path, [("1.0", {("gen", "1"): (0, 10.0, None, None)}), ("1.1", slow)])
    out = tmp_path / "report.html"
    rwt.RunLog("cyllama", db_path).write_report(out, 20)
    page = out.read_text()
    section = page[page.index("Version over version") : page.index("Recent runs")]
    assert "cyllama / cuda / test-all: 1 regression(s)" in section
    assert "1.0 (1 run) &rarr; 1.1 (1 run)" in section
    assert 'class="regression">regression<' in section


def test_report_version_section_single_version(rwt, db_path, tmp_path):
    record(rwt, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    rwt.RunLog("cyllama", db_path).write_report(out, 20)
    page = out.read_text()
    assert "nothing to compare: cyllama / cuda / test-all (0.4.3, 1 run)" in page
    assert "No regressions" not in page


@pytest.mark.parametrize(
    ("argv", "spec"),
    [
        (["install", "--cuda"], ["cyllama-cuda12"]),
        (["install", "--vulkan", "--version", "0.5.1"], ["cyllama-vulkan==0.5.1"]),
    ],
)
def test_install_version_pins_backend_distribution(rwt, monkeypatch, argv, spec):
    cli = rwt.Cli()
    seen = []
    monkeypatch.setattr(cli.env, "pip_install", lambda s, **kw: seen.append(s) or 0)
    assert cli.main(argv) == 0
    assert seen == [spec]


def test_install_version_and_wheel_conflict(rwt, monkeypatch, capsys):
    cli = rwt.Cli()
    monkeypatch.setattr(cli.env, "pip_install", lambda s, **kw: pytest.fail("must not install"))
    assert cli.main(["install", "--cuda", "--version", "0.5.1", "--wheel", "cyllama-cuda12"]) == 2
    assert "give one" in capsys.readouterr().err
