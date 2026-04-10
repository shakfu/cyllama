"""Internal model file validation helpers.

Used across cyllama modules to surface clear, actionable errors for the
common failure modes (missing file, unreadable file, wrong format, empty
file, truncated header, absurd context size) before handing the path to
the underlying C++ libraries, which otherwise tend to fail with opaque
NULL-pointer returns, raw assertions, or — worst of all — segfaults that
the caller cannot catch.

The audit goal is "no segfaults from bad inputs", and the only reliable
way to enforce that from inside a single process is to refuse the bad
input *before* it crosses the FFI boundary.
"""

from __future__ import annotations

import os
import struct
from typing import Optional

# GGUF files start with the literal ASCII magic "GGUF".
GGUF_MAGIC = b"GGUF"

# llama.cpp/gguf.h: #define GGUF_VERSION 3. Versions 1 and 2 still appear
# in older models in the wild and llama.cpp continues to load them.
GGUF_KNOWN_VERSIONS = (1, 2, 3)

# Sanity caps for the GGUF header counts. Real models have at most a few
# thousand tensors and a few hundred KV metadata pairs; anything wildly
# larger almost certainly means the file is corrupt or not actually GGUF.
GGUF_MAX_TENSORS = 1_000_000
GGUF_MAX_KV_PAIRS = 1_000_000

# Legacy whisper.cpp ggml files store GGML_FILE_MAGIC = 0x67676d6c written
# as a little-endian uint32, which lands on disk as the four bytes
# 0x6c, 0x6d, 0x67, 0x67 -- i.e. "lmgg" when read as ASCII.
# Newer whisper.cpp models use GGUF instead. Accept either.
GGML_LEGACY_MAGIC_LE = b"\x6c\x6d\x67\x67"
WHISPER_VALID_MAGICS = (GGUF_MAGIC, GGML_LEGACY_MAGIC_LE)


def validate_model_file(
    path: str,
    *,
    kind: str = "model",
    expected_magic: Optional[bytes] = None,
) -> None:
    """Validate that *path* points to a readable, non-empty model file.

    Args:
        path: Filesystem path to validate.
        kind: Human-readable label used in error messages (e.g. "GGUF model",
            "whisper model", "VAE").
        expected_magic: If provided, the first ``len(expected_magic)`` bytes
            of the file must equal this byte string. Pass ``None`` to skip
            the magic-number check (e.g. for formats with multiple magics).

    Raises:
        TypeError: if *path* is not a string.
        ValueError: if *path* is empty, not a regular file, empty on disk,
            or has the wrong magic header.
        FileNotFoundError: if the path does not exist.
        IsADirectoryError: if the path resolves to a directory.
        PermissionError: if the file exists but is not readable.
    """
    if not isinstance(path, str):
        raise TypeError(f"{kind} path must be a string, got {type(path).__name__}")
    if not path:
        raise ValueError(f"{kind} path must be a non-empty string")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} file not found: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"{kind} path is a directory, expected a file: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"{kind} path is not a regular file (special file or broken link): {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{kind} file is not readable, check permissions: {path}")

    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise PermissionError(f"failed to stat {kind} file {path}: {e}") from e
    if size == 0:
        raise ValueError(f"{kind} file is empty: {path}")

    if expected_magic is not None:
        try:
            with open(path, "rb") as f:
                header = f.read(len(expected_magic))
        except OSError as e:
            raise PermissionError(f"failed to read {kind} file {path}: {e}") from e
        if header != expected_magic:
            raise ValueError(
                f"{path} does not look like a valid {kind} file "
                f"(expected magic {expected_magic!r}, got {header!r}). "
                "The file may be corrupt, truncated, or in a different format."
            )


def validate_gguf_file(path: str, *, kind: str = "GGUF model") -> None:
    """Validate a GGUF file beyond the bare magic.

    Reads and sanity-checks the 24-byte fixed header (magic, version,
    tensor_count, kv_count). This catches truncated or corrupt files
    that would otherwise reach llama.cpp / stable-diffusion.cpp and
    crash inside the GGUF parser instead of returning NULL.

    Raises:
        Same path-level exceptions as :func:`validate_model_file`, plus
        ``ValueError`` if the GGUF header itself is malformed.
    """
    # Path-level checks first (existence, perms, non-empty, magic).
    validate_model_file(path, kind=kind, expected_magic=GGUF_MAGIC)

    try:
        with open(path, "rb") as f:
            header = f.read(24)
    except OSError as e:
        raise PermissionError(f"failed to read {kind} header from {path}: {e}") from e

    if len(header) < 24:
        raise ValueError(
            f"{path} is too small to contain a valid GGUF header "
            f"(need at least 24 bytes, got {len(header)}). "
            "The file is truncated or not a GGUF file."
        )

    # Layout: char[4] magic | uint32 version | uint64 tensor_count | uint64 kv_count
    # All little-endian per the GGUF spec.
    _magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", header)

    if version not in GGUF_KNOWN_VERSIONS:
        raise ValueError(
            f"{path} has unsupported GGUF version {version} "
            f"(this build understands versions {GGUF_KNOWN_VERSIONS}). "
            "The file may be corrupt or produced by a much newer tool."
        )

    if tensor_count == 0 or tensor_count > GGUF_MAX_TENSORS:
        raise ValueError(
            f"{path} has implausible GGUF tensor_count={tensor_count} "
            f"(expected 1..{GGUF_MAX_TENSORS}). The file is corrupt or truncated."
        )

    if kv_count > GGUF_MAX_KV_PAIRS:
        raise ValueError(
            f"{path} has implausible GGUF kv_count={kv_count} "
            f"(expected 0..{GGUF_MAX_KV_PAIRS}). The file is corrupt or truncated."
        )


def validate_whisper_file(path: str, *, kind: str = "whisper model") -> None:
    """Validate a whisper.cpp model file.

    Whisper accepts both the legacy ggml format and newer GGUF, so we
    can't pin to a single magic. Instead we read the first 4 bytes and
    require them to match one of the known whisper magics. This catches
    arbitrary garbage files that would otherwise crash whisper.cpp's
    header parser.
    """
    # Path-level checks (no magic check yet — we'll do it ourselves).
    validate_model_file(path, kind=kind)

    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except OSError as e:
        raise PermissionError(f"failed to read {kind} header from {path}: {e}") from e

    if len(header) < 4:
        raise ValueError(f"{path} is too small to contain a magic header (need at least 4 bytes, got {len(header)}).")

    if header not in WHISPER_VALID_MAGICS:
        accepted = ", ".join(repr(m) for m in WHISPER_VALID_MAGICS)
        raise ValueError(
            f"{path} does not look like a valid {kind} file "
            f"(first 4 bytes={header!r}, expected one of: {accepted}). "
            "Whisper accepts either legacy ggml or newer GGUF formats."
        )
