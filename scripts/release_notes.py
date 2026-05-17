#!/usr/bin/env python3
"""Extract a versioned section from CHANGELOG.md into a release-notes file.

Looks up the literal `## [<version>]` heading in CHANGELOG.md, captures
the block up to the next `## [` heading, strips leading/trailing blank
lines, prepends a `## Changes since the last Release` header
(configurable), and writes the result to the output path.

Falls back to `## [Unreleased]` if the version-named section is missing
or empty — useful when a tag was pushed before the heading was renamed.

Exit codes:
  0  section found + written to --output
  2  no matching section (neither version nor Unreleased); output not written

Used by .github/workflows/<release-workflow>.yml to populate the GitHub release body
from CHANGELOG.md. The dot-in-version-as-regex-metachar pitfall that the
shell/awk version had to work around does not exist here: this script
compares headings with `==` against a Python string.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def extract_section(changelog_text: str, version: str) -> Optional[str]:
    """Return the body under `## [<version>]`, or None if the heading is absent.

    The body is everything between the matched heading and the next
    `## [` heading (or end of file), inclusive of blank lines. The
    caller is responsible for trimming and decorating.
    """
    header = f"## [{version}]"
    out: list[str] = []
    in_section = False
    for line in changelog_text.splitlines():
        if line == header:
            in_section = True
            continue
        if in_section and line.startswith("## ["):
            break
        if in_section:
            out.append(line)
    return "\n".join(out) if in_section else None


def strip_blank_edges(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a CHANGELOG.md section into a release notes file."
    )
    parser.add_argument(
        "version",
        help='Release version, e.g. "0.1.2". The script first looks for '
        "`## [<version>]`; if missing or empty, falls back to "
        "`## [Unreleased]`.",
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="release-notes.md",
        help="Where to write the extracted notes (default: %(default)s).",
    )
    parser.add_argument(
        "--header",
        default="Changes since the last Release",
        help='Text after the leading "## " on line 1 of the output '
        '(default: "%(default)s").',
    )
    args = parser.parse_args(argv)

    text = Path(args.changelog).read_text()

    section = extract_section(text, args.version)
    if section is None or not section.strip():
        print(
            f"no '## [{args.version}]' section in {args.changelog}; "
            f"falling back to '## [Unreleased]'",
            file=sys.stderr,
        )
        section = extract_section(text, "Unreleased")

    if section is None or not section.strip():
        print(
            f"no CHANGELOG section found for '{args.version}' or 'Unreleased'",
            file=sys.stderr,
        )
        return 2

    body = strip_blank_edges(section)
    Path(args.output).write_text(f"## {args.header}\n\n{body}\n")
    print(f"wrote {args.output} ({len(body.splitlines())} body lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
