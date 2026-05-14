"""``quarto_render`` -- wraps the local ``quarto`` CLI to render
`.qmd` / `.md` / `.ipynb` to a wide range of pandoc-backed output
formats.

The model gets two usage modes (matches the margo / infer-app
implementations one-to-one):

  1. RENDER EXISTING:   pass ``input`` only.
  2. CREATE-AND-RENDER: pass ``content`` (full quarto source). The tool
                        writes it to ``input`` -- or a slugged path under
                        the output dir if ``input`` is also omitted --
                        then invokes quarto.

Quarto is *not* a Python dependency; the tool just shells out. Only
register it when ``quarto_available()`` returns True; otherwise calls
raise a clear "install quarto" error rather than crashing mid-spawn.
"""

import os as _os
import re
import shutil as _shutil
import subprocess as _subprocess
from pathlib import Path as _Path
from typing import List, Optional

from .core import tool


# Allowlist of output formats. Bounding ``--to`` prevents the model from
# forwarding arbitrary strings to pandoc.
_QUARTO_FORMATS: frozenset[str] = frozenset(
    {
        "html",
        "pdf",
        "docx",
        "pptx",
        "odt",
        "rtf",
        "epub",
        "revealjs",
        "beamer",
        "latex",
        "markdown",
        "gfm",
        "asciidoc",
        "typst",
        "ipynb",
        "jats",
        "mediawiki",
        "commonmark",
    }
)

# Cap a single render at 10 minutes. Quarto runs range from sub-second
# (cached HTML) to many minutes (uncached PDF with computations); we err
# generous and rely on the agent's own cancellation as the real escape hatch.
_QUARTO_RENDER_TIMEOUT_SECONDS = 10 * 60.0

# Pulls the YAML ``title:`` value from quarto frontmatter. Used to derive a
# meaningful filename when the model supplies ``content`` without pinning
# ``input``.
_QUARTO_TITLE_RE = re.compile(r"(?m)^title:\s*['\"]?(.+?)['\"]?\s*$")
_QUARTO_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Matches quarto's ``Output created: <path>`` line. The path is relative
# to the input's directory; it's the only reliable signal of where the
# artifact landed without re-implementing quarto's output-naming rules.
_QUARTO_OUTPUT_CREATED_RE = re.compile(r"(?m)^Output created:\s*(.+?)\s*$")


def quarto_available() -> bool:
    """Return True if the ``quarto`` binary is on PATH.

    Agent surfaces should gate registration of :func:`quarto_render` on
    this -- cyllama does not bundle the binary, and the tool is useless
    without it.
    """
    return _shutil.which("quarto") is not None


def default_quarto_output_dir() -> _Path:
    """Return the directory used by the create-and-render path.

    Resolves to ``$CYLLAMA_QUARTO_OUTPUT_DIR`` if set, otherwise
    ``~/Documents/cyllama/output/``. Created on demand. Files generated
    by the create-and-render path land here -- both the ``.qmd`` source
    and the rendered artifact -- so the user can find them in
    Finder/Explorer without hunting through OS temp paths.
    """
    override = _os.environ.get("CYLLAMA_QUARTO_OUTPUT_DIR")
    if override:
        dir_ = _Path(override).expanduser()
    else:
        dir_ = _Path.home() / "Documents" / "cyllama" / "output"
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


def _quarto_slug_from_content(content: str) -> str:
    """Filesystem-friendly slug derived from the YAML ``title:`` line.

    Returns ``"document"`` if no title is present or the slug would be
    empty. Capped at 64 chars to keep paths reasonable on filesystems
    with shorter limits.
    """
    m = _QUARTO_TITLE_RE.search(content)
    if m is None:
        return "document"
    raw = m.group(1).strip().lower()
    slug = _QUARTO_SLUG_RE.sub("-", raw).strip("-")
    if not slug:
        return "document"
    if len(slug) > 64:
        slug = slug[:64].strip("-")
    return slug or "document"


def _quarto_unique_path(dir_: _Path, base: str, ext: str) -> _Path:
    """Return ``<dir>/<base><ext>`` if free, else ``<dir>/<base>-<n><ext>``.

    Avoids silently overwriting an existing artifact when the user renders
    two documents with the same YAML title into the same output dir.
    """
    candidate = dir_ / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    for n in range(2, 1000):
        c = dir_ / f"{base}-{n}{ext}"
        if not c.exists():
            return c
    # Pathological: hand back the original and let quarto overwrite -- better
    # than failing the call entirely.
    return candidate


# Hand-curated example. The auto-generated one would put the literal
# string "example" in every field, which is useless: `content` is a
# multi-line Quarto document, `input` is a filesystem path, and `to` is
# from an allowlist. Small models pattern-match on this example heavily
# so it needs to teach the create-and-render shape concretely.
_QUARTO_EXAMPLE_CONTENT = (
    "---\n"
    'title: "How to Boil an Egg"\n'
    "format: pptx\n"
    "---\n"
    "\n"
    "# Step 1: Bring water to a boil\n"
    "\n"
    "- Use a pot large enough to submerge the eggs\n"
    "- Add a pinch of salt\n"
    "\n"
    "# Step 2: Add the eggs\n"
    "\n"
    "- Lower in gently with a spoon\n"
    "- Time from when water returns to boil\n"
)


@tool
def quarto_render(
    input: str = "",
    content: str = "",
    to: str = "html",
    output_dir: str = "",
) -> str:
    """Creates and/or renders a Quarto markdown document.

    Two usage modes:

    1. RENDER EXISTING: pass ``input`` pointing to an existing ``.qmd`` /
       ``.md`` / ``.ipynb`` file (or quarto project directory). Leave
       ``content`` empty.
    2. CREATE-AND-RENDER: pass ``content`` with the full Quarto source
       (YAML frontmatter + markdown body). The tool writes it to
       ``input`` (or a slugged path under the output dir if ``input`` is
       omitted) and then renders. Use this mode whenever the user asks
       you to *make* or *generate* a document -- there is no separate
       file-write tool, so the model must supply the source via
       ``content``.

    Quarto documents start with a YAML frontmatter block delimited by
    ``---`` lines, declaring the title and (optionally) a ``format:``
    map. The ``format:`` map keys output formats to per-format option
    blocks. Multiple formats may coexist under the same ``format:`` key.
    If the document's YAML already pins a format, omit the ``to``
    argument so quarto uses what the author specified; pass ``to`` only
    when the user explicitly asks for a different target.

    Args:
        input: Path to an existing input file (``.qmd``, ``.md``,
            ``.ipynb``) or quarto project directory. When ``content`` is
            supplied this is the destination path the tool writes
            ``content`` to before rendering. May be empty if ``content``
            is supplied (a slug-derived path is used).
        content: Inline Quarto document source (YAML frontmatter +
            markdown body). When provided, the tool writes this to
            ``input`` before rendering. Required when ``input`` does not
            yet exist.
        to: Output format. One of: ``html``, ``pdf``, ``docx``, ``pptx``,
            ``odt``, ``rtf``, ``epub``, ``revealjs``, ``beamer``,
            ``latex``, ``markdown``, ``gfm``, ``asciidoc``, ``typst``,
            ``ipynb``, ``jats``, ``mediawiki``, ``commonmark``. Defaults
            to ``html``.
        output_dir: Optional directory for the rendered output, relative
            to the input's directory. When omitted quarto writes
            alongside the input.

    Returns:
        Quarto's combined stdout/stderr, followed by an ``Output file:
        <absolute path>`` line and a markdown link the model should
        paste verbatim when telling the user where the document is.
    """
    if not quarto_available():
        raise RuntimeError(
            "quarto CLI not found on PATH. Install via `brew install quarto` "
            "(macOS) or see https://quarto.org/docs/get-started/."
        )

    input_path = input.strip()
    body = content
    if not input_path and not body.strip():
        raise ValueError("either input (existing file) or content (inline source) is required")

    fmt = to.strip() or "html"
    if fmt not in _QUARTO_FORMATS:
        raise ValueError(f"unsupported format {fmt!r} (allowed: {', '.join(sorted(_QUARTO_FORMATS))})")

    # CREATE-AND-RENDER: materialize content to disk before invoking quarto.
    if body.strip():
        if not input_path:
            dest_dir = default_quarto_output_dir()
            target = _quarto_unique_path(dest_dir, _quarto_slug_from_content(body), ".qmd")
        else:
            target = _Path(input_path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        input_path = str(target)

    abs_input = _Path(input_path).expanduser().resolve()
    if not abs_input.exists():
        raise FileNotFoundError(f"input not found: {abs_input}")

    cmd: List[str] = ["quarto", "render", str(abs_input), "--to", fmt]
    cwd = str(abs_input.parent)
    od = output_dir.strip()
    if od:
        cmd.extend(["--output-dir", od])

    try:
        proc = _subprocess.run(
            cmd,
            cwd=cwd,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            timeout=_QUARTO_RENDER_TIMEOUT_SECONDS,
            text=True,
            check=False,
        )
    except _subprocess.TimeoutExpired as exc:
        partial = (exc.output or "").strip() if isinstance(exc.output, str) else ""
        raise RuntimeError(f"quarto render timed out after {_QUARTO_RENDER_TIMEOUT_SECONDS:.0f}s\n{partial}") from exc

    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"quarto render failed (exit {proc.returncode}):\n{output}")

    # Surface the absolute output path so the model can present it as a
    # usable file:// link rather than echoing quarto's relative line.
    out_path: Optional[_Path] = None
    m = _QUARTO_OUTPUT_CREATED_RE.search(output)
    if m is not None:
        p = _Path(m.group(1))
        if not p.is_absolute():
            p = _Path(cwd) / p
        out_path = p

    parts: List[str] = []
    if output:
        parts.append(output)
        parts.append("")  # blank line separator
    if out_path is not None:
        parts.append(f"Output file: {out_path}")
        parts.append(f"Markdown link to use verbatim in your reply: [{out_path.name}](file://{out_path})")
    else:
        parts.append(f"quarto render {abs_input} --to {fmt}: ok")
    return "\n".join(parts)


# Attached after decoration rather than via @tool(example_args=...) so mypy
# can keep its strict-decorator typing on the bare ``@tool`` form. The
# ``Tool`` dataclass is mutable, so this is safe.
quarto_render.example_args = {
    "content": _QUARTO_EXAMPLE_CONTENT,
    "to": "pptx",
}
