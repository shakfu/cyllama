"""scripts/ci_rename_package.py renames only `[project] name`."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("ci_rename_package", ROOT / "scripts" / "ci_rename_package.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_rename_package"] = module
    spec.loader.exec_module(module)
    return module


PYPROJECT = """\
[project]
name = "cyllama"

version = "0.4.6"
description = "café — Łódź"

[tool.other]
name = "cyllama"
"""


@pytest.fixture
def pyproject(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT, encoding="utf-8")
    return path


def test_renames_project_only(script, pyproject):
    assert script.main(["cyllama-cuda12", "--pyproject", str(pyproject)]) == 0
    expected = PYPROJECT.replace('name = "cyllama"\n\n', 'name = "cyllama-cuda12"\n\n', 1)
    assert pyproject.read_text(encoding="utf-8") == expected


def test_idempotent(script, pyproject):
    assert script.main(["cyllama-rocm", "--pyproject", str(pyproject)]) == 0
    once = pyproject.read_text(encoding="utf-8")
    assert script.main(["cyllama-rocm", "--pyproject", str(pyproject)]) == 0
    assert pyproject.read_text(encoding="utf-8") == once


def test_other_table_name_does_not_count(script, tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "other"\n\n[tool.x]\nname = "cyllama"\n', encoding="utf-8")
    assert script.main(["cyllama-sycl", "--pyproject", str(path)]) == 1
    assert 'name = "cyllama-sycl"' not in path.read_text(encoding="utf-8")


def test_rejects_unlisted_name(script, pyproject):
    assert script.main(["cyllama-cuda13", "--pyproject", str(pyproject)]) == 2
    assert pyproject.read_text(encoding="utf-8") == PYPROJECT


def test_real_pyproject_matches(script):
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start, end = script.project_table(text)
    assert script.name_pattern(script.CANONICAL).search(text[start:end])
