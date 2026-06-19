#!/usr/bin/env python3
"""Report changed C/C++ fields and enum members in staged llama.cpp headers."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
HEADER_ROOT = REPO_ROOT / "src" / "llama.cpp" / "include"
BINDING_FILES = [
    REPO_ROOT / "src" / "xllamacpp" / "xllamacpp.pxd",
    REPO_ROOT / "src" / "xllamacpp" / "xllamacpp.pyx",
]


@dataclass(frozen=True)
class Field:
    owner: str
    name: str
    ctype: str


@dataclass(frozen=True)
class EnumMember:
    owner: str
    name: str


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_show_head(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    result = run_git(["show", f"HEAD:{rel}"], check=False)
    if result.returncode != 0:
        return ""
    return result.stdout


def changed_headers() -> list[Path]:
    result = run_git(
        [
            "diff",
            "--name-only",
            "--",
            "src/llama.cpp/include",
            "src/llama.cpp/src",
        ]
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        path = REPO_ROOT / line
        if path.suffix in {".h", ".hpp", ".hh", ".hxx"} and path.exists():
            paths.append(path)
    return paths


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def normalize_type(value: str) -> str:
    return " ".join(value.replace("*", " * ").replace("&", " & ").split())


def split_field_decl(statement: str) -> list[tuple[str, str]]:
    statement = statement.strip().rstrip(";").strip()
    if not statement:
        return []
    if any(token in statement for token in ("(", ")", "{", "}")):
        return []
    if statement.startswith(("typedef ", "using ", "static_assert", "enum ")):
        return []
    if re.match(r"^(public|private|protected)\s*:", statement):
        return []

    fields: list[tuple[str, str]] = []
    parts = [part.strip() for part in statement.split(",")]
    base_type = ""
    for index, part in enumerate(parts):
        part = part.split("=", 1)[0].strip()
        match = re.match(r"(?P<type>.+?)\s*(?P<name>[A-Za-z_]\w*)(?:\s*\[[^\]]+\])?$", part)
        if not match:
            continue
        if index == 0:
            base_type = match.group("type").strip()
            ctype = base_type
        else:
            ctype = base_type
            if part.startswith(("*", "&")):
                ctype = f"{base_type} {part[0]}"
        name = match.group("name")
        if name in {"const", "volatile", "struct", "class"}:
            continue
        fields.append((name, normalize_type(ctype)))
    return fields


def extract_fields(text: str) -> dict[tuple[str, str], Field]:
    clean = strip_comments(text)
    lines = clean.splitlines()
    fields: dict[tuple[str, str], Field] = {}
    owner: str | None = None
    depth = 0
    pending_owner: str | None = None
    statement = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if owner is None:
            match = re.match(r"(?:typedef\s+)?(?:struct|class)\s+([A-Za-z_]\w*)\b", line)
            if match:
                pending_owner = None if ";" in line else match.group(1)
            if pending_owner and "{" in line:
                owner = pending_owner
                depth = line.count("{") - line.count("}")
                pending_owner = None
                after = line.split("{", 1)[1]
                if after.strip():
                    statement += " " + after
            elif not match:
                pending_owner = None
            continue

        depth += line.count("{") - line.count("}")
        body_line = line
        if "}" in body_line:
            body_line = body_line.split("}", 1)[0]
        statement += " " + body_line

        while ";" in statement:
            current, statement = statement.split(";", 1)
            for name, ctype in split_field_decl(current + ";"):
                fields[(owner, name)] = Field(owner=owner, name=name, ctype=ctype)

        if depth <= 0:
            owner = None
            statement = ""

    return fields


def extract_enum_members(text: str) -> dict[tuple[str, str], EnumMember]:
    clean = strip_comments(text)
    enums: dict[tuple[str, str], EnumMember] = {}
    pattern = re.compile(
        r"(?:typedef\s+)?enum(?:\s+(?:class\s+)?(?P<name>[A-Za-z_]\w*))?\s*\{(?P<body>.*?)\}\s*(?P<alias>[A-Za-z_]\w*)?",
        flags=re.S,
    )
    for match in pattern.finditer(clean):
        owner = match.group("alias") or match.group("name")
        if not owner:
            continue
        for raw_member in match.group("body").split(","):
            member = raw_member.split("=", 1)[0].strip()
            member = re.sub(r"\s+", " ", member)
            if not re.fullmatch(r"[A-Za-z_]\w*", member):
                continue
            enums[(owner, member)] = EnumMember(owner=owner, name=member)
    return enums


def load_bindings() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in BINDING_FILES:
        values[path.name] = path.read_text() if path.exists() else ""
    return values


def binding_status(field: Field, bindings: dict[str, str]) -> str:
    owner_pattern = re.compile(rf"\b(ctypedef\s+struct|cdef\s+cppclass)\s+{re.escape(field.owner)}\b")
    field_pattern = re.compile(rf"\b{re.escape(field.name)}\b")
    owner_files = [name for name, text in bindings.items() if owner_pattern.search(text)]
    field_files = [name for name, text in bindings.items() if field_pattern.search(text)]
    if owner_files and field_files:
        return f"owner+field referenced ({', '.join(sorted(set(owner_files + field_files)))})"
    if owner_files:
        return f"owner referenced, field missing ({', '.join(owner_files)})"
    if field_files:
        return f"field referenced, owner not declared ({', '.join(field_files)})"
    return "not referenced"


def enum_binding_status(member: EnumMember, bindings: dict[str, str]) -> str:
    owner_pattern = re.compile(
        rf"\b(cpdef\s+enum|cdef\s+enum)\s+{re.escape(member.owner)}\b"
    )
    member_pattern = re.compile(rf"\b{re.escape(member.name)}\b")
    owner_files = [name for name, text in bindings.items() if owner_pattern.search(text)]
    member_files = [name for name, text in bindings.items() if member_pattern.search(text)]
    if owner_files and member_files:
        return f"owner+member referenced ({', '.join(sorted(set(owner_files + member_files)))})"
    if owner_files:
        return f"owner referenced, member missing ({', '.join(owner_files)})"
    if member_files:
        return f"member referenced, owner not declared ({', '.join(member_files)})"
    return "not referenced"


def main() -> int:
    headers = changed_headers()
    if not headers:
        print("No changed llama.cpp headers found under src/llama.cpp/include or src/llama.cpp/src.")
        return 0

    bindings = load_bindings()
    any_changes = False

    for path in headers:
        old_fields = extract_fields(git_show_head(path))
        new_fields = extract_fields(path.read_text())
        keys = sorted(set(old_fields) | set(new_fields))
        changes: list[tuple[str, Field, Field | None]] = []
        for key in keys:
            old = old_fields.get(key)
            new = new_fields.get(key)
            if old and not new:
                changes.append(("removed", old, None))
            elif new and not old:
                changes.append(("added", new, None))
            elif old and new and old.ctype != new.ctype:
                changes.append(("type-changed", new, old))

        old_enums = extract_enum_members(git_show_head(path))
        new_enums = extract_enum_members(path.read_text())
        enum_keys = sorted(set(old_enums) | set(new_enums))
        enum_changes: list[tuple[str, EnumMember]] = []
        for key in enum_keys:
            old = old_enums.get(key)
            new = new_enums.get(key)
            if old and not new:
                enum_changes.append(("removed", old))
            elif new and not old:
                enum_changes.append(("added", new))

        if changes or enum_changes:
            any_changes = True
            print(path.relative_to(REPO_ROOT))
            for kind, field, old in changes:
                if old is None:
                    type_info = field.ctype
                else:
                    type_info = f"{old.ctype} -> {field.ctype}"
                print(f"  field {kind}: {field.owner}.{field.name}: {type_info}")
                print(f"    bindings: {binding_status(field, bindings)}")
            for kind, member in enum_changes:
                print(f"  enum {kind}: {member.owner}.{member.name}")
                print(f"    bindings: {enum_binding_status(member, bindings)}")

    if not any_changes:
        print("Changed headers found, but no struct/class field or enum member changes were detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
