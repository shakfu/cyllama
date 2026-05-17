"""Tests for the ``quarto_render`` tool in ``cyllama.agents.tools``.

Mirrors the test surface in
``~/projects/personal/margo/pkg/margo/agent/tools_quarto_test.go``:

* Argument-validation tests run unconditionally.
* Slug-derivation tests run unconditionally (pure-Python).
* The real ``quarto render`` invocations are gated on
  :func:`quarto_available` so the suite stays portable on hosts that
  don't have the binary.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cyllama.agents.tools import (
    _quarto_slug_from_content,
    _quarto_unique_path,
    default_quarto_output_dir,
    quarto_available,
    quarto_render,
)


# Skip marker for tests that need the quarto CLI on PATH.
needs_quarto = pytest.mark.skipif(not quarto_available(), reason="quarto not on PATH")


class TestQuartoSlug:
    """Pure-Python slug derivation; no subprocess involved."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            ('---\ntitle: "How to Boil an Egg"\n---\n', "how-to-boil-an-egg"),
            ("---\ntitle: 'Mixed/Punct & Whitespace!'\n---\n", "mixed-punct-whitespace"),
            ("---\ntitle: bare value\n---\n", "bare-value"),
            ("no frontmatter at all", "document"),
            ('---\ntitle: "   "\n---', "document"),
        ],
    )
    def test_slug_cases(self, content: str, expected: str):
        assert _quarto_slug_from_content(content) == expected

    def test_long_title_is_capped(self):
        long_title = "a" * 200
        slug = _quarto_slug_from_content(f"---\ntitle: {long_title}\n---\n")
        assert len(slug) <= 64
        assert slug == "a" * 64


class TestQuartoUniquePath:
    def test_returns_base_when_free(self, tmp_path: Path):
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc.qmd"

    def test_increments_on_collision(self, tmp_path: Path):
        (tmp_path / "doc.qmd").write_text("x")
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc-2.qmd"

    def test_keeps_incrementing(self, tmp_path: Path):
        (tmp_path / "doc.qmd").write_text("x")
        (tmp_path / "doc-2.qmd").write_text("x")
        (tmp_path / "doc-3.qmd").write_text("x")
        p = _quarto_unique_path(tmp_path, "doc", ".qmd")
        assert p == tmp_path / "doc-4.qmd"


class TestQuartoRenderArgValidation:
    """Pre-flight validation runs before any subprocess call, so these
    tests work even when quarto is not installed -- *unless* the binary
    happens to be present, in which case the validation still wins because
    the bad args are rejected before exec."""

    @needs_quarto
    def test_empty_input_and_content_rejected(self):
        with pytest.raises(ValueError, match="input.*content"):
            quarto_render(input="", content="", to="html")

    @needs_quarto
    def test_unsupported_format_rejected(self):
        with pytest.raises(ValueError, match="unsupported format"):
            quarto_render(input="x.qmd", content="", to="rot13")

    def test_missing_quarto_surfaces_install_hint(self, monkeypatch: pytest.MonkeyPatch):
        # Force quarto_available -> False regardless of host install.
        monkeypatch.setattr("cyllama.agents.tools.quarto._shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="quarto CLI not found"):
            quarto_render(input="x.qmd", content="", to="html")


class TestQuartoDefaultOutputDir:
    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        target = tmp_path / "my-outputs"
        monkeypatch.setenv("CYLLAMA_QUARTO_OUTPUT_DIR", str(target))
        result = default_quarto_output_dir()
        assert result == target
        assert result.is_dir()

    def test_quarto_render_ships_curated_example(self):
        # The auto-generated example would put "example" in every string
        # field; verify quarto_render carries a hand-curated one that
        # demonstrates the create-and-render shape.
        assert quarto_render.example_args is not None
        assert "content" in quarto_render.example_args
        assert quarto_render.example_args["to"] == "pptx"
        # The example content must actually be a quarto document the slug
        # extractor will accept (i.e. has a YAML title line).
        from cyllama.agents.tools import _quarto_slug_from_content

        slug = _quarto_slug_from_content(quarto_render.example_args["content"])
        assert slug == "how-to-boil-an-egg"
        # And the rendered prompt must contain the curated example, not "example".
        rendered = quarto_render.to_prompt_string()
        assert "How to Boil an Egg" in rendered
        assert '"to": "example"' not in rendered

    def test_default_under_documents(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CYLLAMA_QUARTO_OUTPUT_DIR", raising=False)
        result = default_quarto_output_dir()
        # Don't assert is_dir() here — the helper creates it and we don't
        # want test runs to leave that directory behind. Just check the
        # resolved path shape.
        assert result.name == "output"
        assert result.parent.name == "cyllama"


@needs_quarto
class TestQuartoRenderLive:
    """Exercises the real ``quarto`` CLI. Skipped when not on PATH."""

    def test_render_existing_html(self, tmp_path: Path):
        src = tmp_path / "doc.qmd"
        src.write_text("---\ntitle: t\n---\n\nhello\n", encoding="utf-8")

        out = quarto_render(input=str(src), content="", to="html")
        rendered = tmp_path / "doc.html"
        assert rendered.exists(), f"expected {rendered} to exist; got:\n{out}"
        assert f"Output file: {rendered}" in out
        assert f"[doc.html](file://{rendered})" in out

    def test_create_and_render_writes_input(self, tmp_path: Path):
        dst = tmp_path / "presentation.qmd"
        src = '---\ntitle: "t"\nformat: html\n---\n\nbody\n'

        out = quarto_render(input=str(dst), content=src, to="html")
        assert dst.exists(), "qmd was not written before render"
        rendered = tmp_path / "presentation.html"
        assert rendered.exists(), f"expected {rendered} to exist; got:\n{out}"
        assert f"Output file: {rendered}" in out
        assert f"[presentation.html](file://{rendered})" in out

    def test_create_and_render_slug_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Point the default output dir at a tmp_path-scoped location so
        # the test doesn't pollute the user's real ~/Documents.
        monkeypatch.setenv("CYLLAMA_QUARTO_OUTPUT_DIR", str(tmp_path))
        src = '---\ntitle: "How to Boil an Egg"\nformat: html\n---\n\nbody\n'

        out = quarto_render(input="", content=src, to="html")
        rendered = tmp_path / "how-to-boil-an-egg.html"
        assert rendered.exists(), f"expected {rendered}; got:\n{out}"
        assert f"[how-to-boil-an-egg.html](file://{rendered})" in out

        # Second render of the same title must not silently overwrite.
        out2 = quarto_render(input="", content=src, to="html")
        assert "how-to-boil-an-egg-2" in out2

    def test_input_not_found(self, tmp_path: Path):
        ghost = tmp_path / "does-not-exist.qmd"
        with pytest.raises(FileNotFoundError):
            quarto_render(input=str(ghost), content="", to="html")
