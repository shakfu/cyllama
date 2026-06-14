"""Guards against silent drift between hand-maintained Cython enum mirrors and
the upstream C headers they mirror.

The ``.pxd`` files declare ``cdef enum`` blocks by hand, copying the
enumerators from llama.cpp's headers. When llama.cpp is upgraded, new
enumerators (or reordered/removed ones) can silently desync the mirror: because
the entries are unvalued-and-sequential, a single missing entry shifts the
integer value of every entry after it, and a missing trailing entry leaves the
``*_COUNT`` sentinel wrong. Neither is caught by the C compiler unless that
exact enumerator is referenced at a call site.

This test recomputes the integer value of every *active* (non-commented)
enumerator on both sides using C enum semantics (sequential from 0, an explicit
``= N`` resets the running value) and asserts the two name->value maps are
identical. A mismatch points at exactly which enumerator drifted.

If an enumerator is intentionally removed upstream, comment it out in the
``.pxd`` (matching the header) rather than deleting the line, so the mirror
stays visually aligned with the upstream block.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_INCLUDE = REPO_ROOT / "thirdparty" / "llama.cpp" / "include"
PXD_DIR = REPO_ROOT / "src" / "cyllama" / "llama"

# Each case mirrors one C enum into one .pxd cdef enum.
#   header / pxd:       files to read
#   header_re / pxd_re: regex whose group(1) opens the enum body
#   prefix:             enumerator name prefix (also the stop sentinel anchor)
CASES = [
    {
        "id": "ggml_type",
        "header": LLAMA_INCLUDE / "ggml.h",
        "pxd": PXD_DIR / "ggml.pxd",
        "header_open": r"enum\s+ggml_type\s*\{",
        "pxd_open": r"cdef\s+enum\s+ggml_type\s*:",
        "prefix": "GGML_TYPE_",
    },
    {
        "id": "ggml_op",
        "header": LLAMA_INCLUDE / "ggml.h",
        "pxd": PXD_DIR / "ggml.pxd",
        "header_open": r"enum\s+ggml_op\s*\{",
        "pxd_open": r"cdef\s+enum\s+ggml_op\s*:",
        "prefix": "GGML_OP_",
    },
]


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _strip_pxd_comments(text: str) -> str:
    return re.sub(r"#[^\n]*", "", text)


def _parse_int(token: str) -> int:
    token = token.strip()
    return int(token, 16) if token.lower().startswith("0x") else int(token)


def _enum_body(text: str, open_re: str, prefix: str) -> str:
    """Return the text of the enum body from its opening token up to and
    including the terminating ``<prefix>COUNT`` enumerator."""
    m = re.search(open_re, text)
    assert m is not None, f"could not locate enum opening matching {open_re!r}"
    start = m.end()
    # Stop at the COUNT sentinel so unrelated trailing content is excluded.
    stop = re.search(rf"{prefix}COUNT\b", text[start:])
    assert stop is not None, f"could not find {prefix}COUNT after enum opening"
    return text[start : start + stop.end()]


def _enum_values(body: str, prefix: str) -> dict:
    """Map each active enumerator name to its resolved integer value using C
    enum semantics (start at 0; ``= N`` resets; otherwise previous + 1)."""
    values: dict[str, int] = {}
    next_implicit = 0
    entry_re = re.compile(rf"({re.escape(prefix)}\w+)\s*(?:=\s*(-?(?:0x[0-9a-fA-F]+|\d+)))?")
    for name, explicit in entry_re.findall(body):
        if explicit:
            value = _parse_int(explicit)
        else:
            value = next_implicit
        values[name] = value
        next_implicit = value + 1
    return values


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_enum_mirror_matches_header(case):
    header_text = _strip_c_comments(case["header"].read_text())
    pxd_text = _strip_pxd_comments(case["pxd"].read_text())

    header_body = _enum_body(header_text, case["header_open"], case["prefix"])
    pxd_body = _enum_body(pxd_text, case["pxd_open"], case["prefix"])

    header_values = _enum_values(header_body, case["prefix"])
    pxd_values = _enum_values(pxd_body, case["prefix"])

    # Sanity: we actually parsed something.
    assert len(header_values) > 1, f"parsed no enumerators from header for {case['id']}"
    assert len(pxd_values) > 1, f"parsed no enumerators from pxd for {case['id']}"

    missing = {k: header_values[k] for k in header_values.keys() - pxd_values.keys()}
    extra = {k: pxd_values[k] for k in pxd_values.keys() - header_values.keys()}
    mismatched = {
        k: (header_values[k], pxd_values[k])
        for k in header_values.keys() & pxd_values.keys()
        if header_values[k] != pxd_values[k]
    }

    assert not (missing or extra or mismatched), (
        f"{case['id']} mirror is out of sync with the upstream header.\n"
        f"  in header but missing from .pxd: {missing}\n"
        f"  in .pxd but not in header: {extra}\n"
        f"  value mismatches (header, pxd): {mismatched}\n"
        f"Update {case['pxd'].relative_to(REPO_ROOT)} to match "
        f"{case['header'].relative_to(REPO_ROOT)}."
    )
