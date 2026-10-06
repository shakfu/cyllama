"""Guards that the ``.pxd`` files declare every non-deprecated symbol of the C
headers cyllama binds: ``llama.h``, ``mtmd.h``, ``mtmd-helper.h`` and ``gguf.h``.

cyllama binds the public C API completely and nothing from ``libcommon``.
When llama.cpp is upgraded, new functions, enums, structs, typedefs and
struct fields fail this test until they are declared in the matching ``.pxd``
or listed in the header's ``skipped`` map with a reason (``"struct.field"``
for a field). ``ggml.h`` is not covered: cyllama binds the parts of ggml it
uses, not its graph-building API.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE = REPO_ROOT / "thirdparty" / "llama.cpp" / "include"
PXD_DIR = REPO_ROOT / "src" / "cyllama" / "llama"

CASES = {
    "llama.h": {"macro": "LLAMA_API", "prefix": "llama_", "pxd": "llama.pxd", "skipped": {}},
    "mtmd.h": {
        "macro": "MTMD_API",
        "prefix": "mtmd_",
        "pxd": "mtmd.pxd",
        "skipped": {
            "mtmd_get_memory_usage": "C++ only (std::map), marked unstable upstream",
            "mtmd_memory_usage": "C++ only (std::map), marked unstable upstream",
            **{
                f"mtmd_{n}_deleter": "C++ unique_ptr helper"
                for n in ("context", "bitmap", "batch", "input_chunk", "input_chunks")
            },
        },
    },
    "mtmd-helper.h": {
        "macro": "MTMD_API",
        "prefix": "mtmd_",
        "pxd": "mtmd.pxd",
        "skipped": {f"mtmd_helper_{n}_deleter": "C++ unique_ptr helper" for n in ("gen_audio", "video")},
    },
    "gguf.h": {"macro": "GGML_API", "prefix": "gguf_", "pxd": "gguf.pxd", "skipped": {}},
}


def _strip_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"^[ \t]*#[^\n]*", "", text, flags=re.MULTILINE)


def _header_symbols(text: str, macro: str, prefix: str) -> dict:
    text = _strip_c(text)
    functions = set()
    for stmt in text.split(";"):
        if macro not in stmt or "DEPRECATED" in stmt:
            continue
        m = re.search(rf"\b({prefix}\w+)\s*\(", stmt[stmt.index(macro) :])
        if m:
            functions.add(m.group(1))
    return {
        "function": functions,
        "enum": set(re.findall(rf"\benum\s+({prefix}\w+)\s*\{{", text)),
        "struct": set(re.findall(rf"\bstruct\s+({prefix}\w+)\s*[{{;]", text)),
        "typedef": set(re.findall(rf"typedef[^;]*?\(\s*\*\s*({prefix}\w+)\s*\)", text))
        | set(re.findall(rf"typedef[^;(]*?\b({prefix}\w+)\s*;", text)),
    }


def _member_name(decl: str):
    m = re.search(r"\(\s*\*\s*(\w+)\s*\)", decl)  # function pointer member
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*$", decl.strip())
    return m.group(1) if m else None


def _header_fields(text: str, prefix: str) -> dict:
    text = _strip_c(text)
    fields = {}
    for m in re.finditer(rf"\bstruct\s+({prefix}\w+)\s*\{{", text):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = re.sub(r"\bunion\s*\{|\}", "", text[m.end() : i - 1])  # pxd flattens unions
        fields[m.group(1)] = {n for d in body.split(";") if d.strip() and (n := _member_name(d))}
    return fields


def _pxd_fields(text: str, prefix: str) -> dict:
    fields, current = {}, None
    for line in text.splitlines():
        code = line.split("#", 1)[0].rstrip()
        m = re.match(rf"\s*(?:ctypedef|cdef)\s+struct\s+({prefix}\w+)\s*:", code)
        if m:
            current = fields.setdefault(m.group(1), set())
            indent = None
            continue
        if current is None or not code.strip():
            continue
        lead = len(code) - len(code.lstrip())
        if indent is None:
            indent = lead
        if lead < indent:
            current = None
            continue
        name = _member_name(code.rstrip(";"))
        if name and name != "pass":
            current.add(name)
    return fields


def _pxd_symbols(text: str, prefix: str) -> dict:
    text = re.sub(r"#[^\n]*", "", text)
    return {
        # a cname string binds the C function under a different Python-side name
        "function": set(re.findall(rf"\b({prefix}\w+)\s*\(", text)) | set(re.findall(rf'"({prefix}\w+)"', text)),
        "enum": set(re.findall(rf"\benum\s+({prefix}\w+)", text)),
        "struct": set(re.findall(rf"\bstruct\s+({prefix}\w+)", text)),
        "typedef": set(re.findall(rf"\(\s*\*\s*({prefix}\w+)\s*\)", text))
        | set(re.findall(rf"ctypedef\s+[^\n]*?\b({prefix}\w+)\s*$", text, flags=re.MULTILINE)),
    }


def _load(header: str) -> dict:
    case = CASES[header]
    h = (INCLUDE / header).read_text()
    p = (PXD_DIR / case["pxd"]).read_text()
    return {
        "symbols": _header_symbols(h, case["macro"], case["prefix"]),
        "pxd_symbols": _pxd_symbols(p, case["prefix"]),
        "fields": _header_fields(h, case["prefix"]),
        "pxd_fields": _pxd_fields(p, case["prefix"]),
        "skipped": case["skipped"],
        "pxd": case["pxd"],
    }


PARSED = {h: _load(h) for h in CASES}
KINDS = ("function", "enum", "struct", "typedef")


@pytest.mark.parametrize("header, kind", [(h, k) for h in CASES for k in KINDS])
def test_symbols_declared(header, kind):
    d = PARSED[header]
    have = d["pxd_symbols"][kind]
    if kind == "typedef":
        have = have | d["pxd_symbols"]["struct"]  # `typedef struct X X;` is declared as struct X
    missing = sorted(d["symbols"][kind] - have - set(d["skipped"]))
    assert not missing, f"{header} {kind}s not declared in {d['pxd']}: {missing}"


@pytest.mark.parametrize("header, struct", [(h, s) for h in CASES for s in sorted(PARSED[h]["fields"])])
def test_struct_fields_declared(header, struct):
    d = PARSED[header]
    if struct in d["skipped"] or struct not in d["pxd_fields"]:
        pytest.skip("struct not declared; reported by test_symbols_declared")
    missing = sorted(f for f in d["fields"][struct] - d["pxd_fields"][struct] if f"{struct}.{f}" not in d["skipped"])
    assert not missing, f"{struct} fields not declared in {d['pxd']}: {missing}"


def test_header_parse_is_not_empty():
    # Guards against the regexes silently matching nothing after a header reformat.
    assert len(PARSED["llama.h"]["symbols"]["function"]) > 200
    assert "llama_decode" in PARSED["llama.h"]["symbols"]["function"]
    assert "llama_load_model_from_file" not in PARSED["llama.h"]["symbols"]["function"]  # deprecated
    assert "mtmd_tokenize" in PARSED["mtmd.h"]["symbols"]["function"]
    assert "gguf_init_from_file" in PARSED["gguf.h"]["symbols"]["function"]


@pytest.mark.parametrize("header", list(CASES))
def test_skipped_entries_are_current(header):
    d = PARSED[header]
    every = set().union(*d["symbols"].values())
    every |= {f"{s}.{f}" for s, fs in d["fields"].items() for f in fs}
    stale = sorted(set(d["skipped"]) - every)
    assert not stale, f"skipped names no longer in {header}: {stale}"
