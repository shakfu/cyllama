#!/usr/bin/env python3
"""Rewrite the `[project] name` in pyproject.toml for a per-backend GPU wheel.

The GPU wheels publish under per-backend distribution names, so
CIBW_BEFORE_BUILD renames the project before each build. That edit
previously existed in three spellings across seven workflows -- GNU
`sed -i`, BSD `sed -i ''`, and a Python one-liner -- each with its own
verification step, and all three had to be kept in step by hand.

pyproject.toml is located relative to this file, not the working
directory: cibuildwheel invokes the Windows hook before its `cd {project}`,
so a cwd-relative path resolves differently per platform.

Usage:
    python scripts/ci_rename_package.py cyllama-cuda12
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CANONICAL = "cyllama"

# A typo'd name would otherwise build and upload a wheel under a
# distribution nobody owns.
ALLOWED = {
    CANONICAL,
    "cyllama-cuda12",
    "cyllama-rocm",
    "cyllama-sycl",
    "cyllama-vulkan",
}

DEFAULT_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def name_pattern(name: str) -> re.Pattern[str]:
    """Match a top-level `name = "<name>"` line, tolerant of spacing.

    Anchored to the line start so a `name` key nested in another table
    (authors, entry points) can never match.
    """
    return re.compile(rf'^name\s*=\s*"{re.escape(name)}"\s*$', re.MULTILINE)


def rename(path: Path, new_name: str) -> int:
    text = path.read_text()

    # Idempotent: cibuildwheel runs CIBW_BEFORE_BUILD once per wheel, so a
    # job building more than one invokes this repeatedly against a tree it
    # already renamed.
    if name_pattern(new_name).search(text):
        print(f"already renamed to {new_name} in {path}")
        return 0

    pattern = name_pattern(CANONICAL)
    if not pattern.search(text):
        sys.stderr.write(
            f'ERROR: no `name = "{CANONICAL}"` line in {path}; '
            "already renamed to something else, or the file was reformatted\n"
        )
        return 1

    path.write_text(pattern.sub(f'name = "{new_name}"', text, count=1))

    # Read back rather than trusting the write: this is the step the three
    # shell spellings each had, and the reason they were worth keeping.
    if not name_pattern(new_name).search(path.read_text()):
        sys.stderr.write(f'ERROR: `name = "{new_name}"` absent from {path} after rewrite\n')
        return 1

    print(f"renamed {CANONICAL} -> {new_name} in {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename the project in pyproject.toml for a GPU wheel build.")
    parser.add_argument("new_name", help=f"target distribution name, one of: {', '.join(sorted(ALLOWED))}")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="path to pyproject.toml (default: the one beside this script's parent)",
    )
    args = parser.parse_args(argv)

    if args.new_name not in ALLOWED:
        sys.stderr.write(f"ERROR: '{args.new_name}' is not an allowed name: {sorted(ALLOWED)}\n")
        return 2
    if not args.pyproject.is_file():
        sys.stderr.write(f"ERROR: pyproject not found: {args.pyproject}\n")
        return 2

    return rename(args.pyproject, args.new_name)


if __name__ == "__main__":
    sys.exit(main())
