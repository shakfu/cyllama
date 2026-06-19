#!/usr/bin/env python3
"""Format all code in src/xllamacpp directory.

This script formats:
- Python files (.py) using black
- C/C++ files (.cpp, .h) using clang-format 22, matching CI
- Cython files (.pyx, .pxd) are skipped (black support is limited)

The _version.py file is excluded from formatting as per pyproject.toml config.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src" / "xllamacpp"
CI_CLANG_FORMAT_MAJOR = "22"


def get_files_to_format():
    """Get lists of files by type."""
    python_files = []
    cpp_files = []

    for file_path in SRC_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix
        name = file_path.name

        # Skip _version.py (excluded in pyproject.toml)
        if name == "_version.py":
            continue
        if name == "xllamacpp.cpp":
            continue

        if suffix == ".py":
            python_files.append(str(file_path))
        elif suffix in (".cpp", ".h"):
            cpp_files.append(str(file_path))
        # Note: .pyx and .pxd (Cython) files are skipped
        # Black has limited support for Cython syntax

    return python_files, cpp_files


def to_repo_relative(files):
    """Convert absolute file paths to repository-relative POSIX paths."""
    return [Path(file_path).relative_to(ROOT_DIR).as_posix() for file_path in files]


def format_python(files):
    """Format Python files using black."""
    if not files:
        print("No Python files to format.")
        return 0

    print(f"Formatting {len(files)} Python file(s) with black...")
    cmd = ["black", "--config", str(ROOT_DIR / "pyproject.toml")] + files
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error formatting Python files: {result.stderr}", file=sys.stderr)
        return result.returncode

    print("Python files formatted successfully.")
    if result.stdout:
        print(result.stdout)
    return 0


def format_cpp(files):
    """Format C/C++ files using the same clang-format version as CI."""
    if not files:
        print("No C/C++ files to format.")
        return 0

    print(f"Formatting {len(files)} C/C++ file(s) with clang-format {CI_CLANG_FORMAT_MAJOR}...")

    relative_files = to_repo_relative(files)
    check = subprocess.run(["clang-format", "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        print(
            f"Error: clang-format not found. Install clang-format {CI_CLANG_FORMAT_MAJOR}.",
            file=sys.stderr,
        )
        return 1
    if f"version {CI_CLANG_FORMAT_MAJOR}." not in check.stdout:
        print(
            f"Warning: using {check.stdout.strip()}; CI uses clang-format {CI_CLANG_FORMAT_MAJOR}.",
            file=sys.stderr,
        )

    cmd = ["clang-format", "-i", "--style=file", *relative_files]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error formatting C/C++ files: {result.stderr}", file=sys.stderr)
        return result.returncode

    print("C/C++ files formatted successfully.")
    return 0


def main():
    python_files, cpp_files = get_files_to_format()

    print(f"Found {len(python_files)} Python file(s) and {len(cpp_files)} C/C++ file(s) to format.")
    print(f"Target directory: {SRC_DIR}")
    print()

    exit_code = 0

    exit_code |= format_python(python_files)
    print()
    exit_code |= format_cpp(cpp_files)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
