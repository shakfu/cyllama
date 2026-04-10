"""Tests for the vendored-jinja2 chat-template path on LLM.

These cover three layers:

1. **The vendored jinja2 itself** -- a smoke test that the vendor copy
   imports cleanly, that compiled templates resolve runtime symbols
   from `cyllama._vendor.jinja2.runtime` (not from a system jinja2 or
   nowhere), that undefined variables render as empty strings (catches
   the `_MissingType` leak that motivated the compiler.py rewrite),
   and that the `set` inside `if not x is defined` pattern used by
   real Llama-3 templates actually escapes the if block.

2. **`LLM._apply_jinja_template`** -- end-to-end against the standard
   test model fixture (`Llama-3.2-1B-Instruct-Q8_0.gguf`). Verifies
   that the rendered prompt contains the expected Llama-3 chat tokens
   (`<|start_header_id|>`, `<|eot_id|>`, etc.), that
   `add_generation_prompt=True` appends the trailing assistant header,
   and that `add_generation_prompt=False` does not.

3. **`LLM._apply_template` fallback wiring** -- verifies that the
   wrapper tries the Jinja path first and falls back to the legacy
   substring-heuristic path on `TemplateError`. The pipeline-level
   tests in `test_rag_pipeline.py` cover the next layer up
   (`RAGPipeline._chat_with_fallback`'s three-tier fallback), so this
   file only verifies the binding -> wrapper boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GEMMA_MODEL = ROOT / "models" / "Gemma-4-E4B-it-Q5_K_M.gguf"


# ---------------------------------------------------------------------------
# Layer 1: vendored jinja2 itself
# ---------------------------------------------------------------------------


class TestVendoredJinja2:
    """Smoke tests against `cyllama._vendor.jinja2` directly.

    These would fail if the vendoring procedure (sed-rewriting markupsafe
    imports, sed-rewriting compiler.py's `from jinja2.runtime import`
    emit string) was incomplete or regressed during a re-vendor.
    """

    def test_vendor_imports_cleanly(self):
        from cyllama._vendor import jinja2  # noqa: F401
        from cyllama._vendor.jinja2 import Environment  # noqa: F401
        from cyllama._vendor.jinja2.exceptions import TemplateError  # noqa: F401
        from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: F401

    def test_undefined_variable_renders_as_empty_string(self):
        """An unset variable must render as `''`, not as the literal
        string `'missing'`. Catches the `_MissingType` sentinel leak
        that happens when compiled templates pull runtime symbols from
        a different `jinja2` module than the one the runtime check
        runs in -- the `if rv is missing:` identity check fails
        between the two `missing` sentinels and the sentinel leaks
        into rendered output.
        """
        from cyllama._vendor.jinja2 import Environment

        env = Environment()
        result = env.from_string("[{{ undefined_var }}]").render()
        assert result == "[]", (
            f"undefined variable rendered as {result!r}; expected '[]'. "
            f"This usually means compiler.py's `from jinja2.runtime import` "
            f"emit string was not rewritten to point at "
            f"cyllama._vendor.jinja2.runtime."
        )

    def test_compiled_template_uses_vendored_runtime(self):
        """Inspect the compiled source for a tiny template and assert
        the `from ... import` line points at the vendored runtime, not
        at top-level `jinja2.runtime`."""
        from cyllama._vendor.jinja2 import Environment

        compiled = Environment().compile("[{{ x }}]", raw=True)
        assert "from cyllama._vendor.jinja2.runtime import " in compiled, (
            f"compiled template imports from the wrong runtime module:\n"
            f"{compiled.splitlines()[0]}"
        )
        assert "from jinja2.runtime import " not in compiled

    def test_set_inside_is_defined_if_escapes_block(self):
        """Real Llama-3 chat templates use the pattern
        `{% if not x is defined %}{% set x = ... %}{% endif %}` to
        provide defaults for caller-supplied variables. Jinja2's
        documented behaviour is that `if` does not introduce a scope,
        so the set should escape the if block. Pinning this here
        because if Jinja2 ever changes that behaviour, every chat
        template that uses the pattern would silently break.
        """
        from cyllama._vendor.jinja2 import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tpl = env.from_string(
            "{%- if not x is defined %}"
            "{%- set x = 'default-value' %}"
            "{%- endif %}"
            "[{{ x }}]"
        )
        assert tpl.render() == "[default-value]"

    def test_sandbox_environment_supports_loopcontrols(self):
        """`ImmutableSandboxedEnvironment` + `loopcontrols` extension
        is the exact configuration `_apply_jinja_template` uses, and
        also what HuggingFace's `apply_chat_template` uses. Pin it so
        a future re-vendor can't accidentally drop the extension."""
        from cyllama._vendor.jinja2 import ext as _ext
        from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[_ext.loopcontrols],
        )
        # `break` only works under loopcontrols
        tpl = env.from_string(
            "{%- for i in range(10) %}{{ i }}{%- if i == 2 %}{%- break %}{%- endif %}{%- endfor %}"
        )
        assert tpl.render() == "012"

    def test_raise_exception_global_propagates_template_error(self):
        """Gemma's chat template (and many others) call
        `{{ raise_exception('...') }}` to abort rendering on bad input.
        We provide `raise_exception` as a global that throws
        `TemplateError`. The exception type matters because the
        wrapper's fallback is gated on `_JinjaTemplateError`."""
        from cyllama._vendor.jinja2.exceptions import TemplateError
        from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise TemplateError(msg)

        env = ImmutableSandboxedEnvironment()
        env.globals["raise_exception"] = raise_exception
        tpl = env.from_string("{{ raise_exception('boom') }}")
        with pytest.raises(TemplateError, match="boom"):
            tpl.render()


# ---------------------------------------------------------------------------
# Layer 2: LLM._apply_jinja_template against the real test model
# ---------------------------------------------------------------------------


class TestApplyJinjaTemplate:
    """End-to-end against the standard test model fixture.

    Uses the `model_path` fixture from conftest.py, which auto-skips
    when `models/Llama-3.2-1B-Instruct-Q8_0.gguf` is missing.
    """

    def test_renders_llama3_chat_format(self, model_path):
        """The rendered prompt must contain the canonical Llama-3
        instruct tokens. If the substring heuristic in llama.cpp's
        legacy path was being used, this would also work -- but the
        Jinja path also handles the `Today Date:` and
        `Cutting Knowledge Date:` substitutions inside the system
        header, so we check for those too."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        prompt = llm._apply_jinja_template(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi."},
            ]
        )
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "You are helpful." in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "Hi." in prompt
        # add_generation_prompt=True is the default
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        # Llama-3.2's template injects today's date via strftime_now,
        # which means our `strftime_now` global was actually called.
        assert "Today Date:" in prompt

    def test_add_generation_prompt_false_omits_assistant_header(self, model_path):
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        prompt = llm._apply_jinja_template(
            [{"role": "user", "content": "Hi."}],
            add_generation_prompt=False,
        )
        assert "Hi." in prompt
        # No trailing assistant header
        assert not prompt.rstrip().endswith("assistant<|end_header_id|>")

    def test_user_only_message(self, model_path):
        """User-only conversation (no system message) must work."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        prompt = llm._apply_jinja_template(
            [{"role": "user", "content": "Just a user message."}]
        )
        assert "Just a user message." in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt

    def test_multi_turn_conversation(self, model_path):
        """Three turns must all be present in the rendered prompt in
        the right order."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        prompt = llm._apply_jinja_template(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ]
        )
        # All three must appear in order
        i_q1 = prompt.find("Q1")
        i_a1 = prompt.find("A1")
        i_q2 = prompt.find("Q2")
        assert -1 < i_q1 < i_a1 < i_q2

    def test_invalid_message_raises_value_error(self, model_path):
        """Validation errors must come out as `ValueError`/`TypeError`
        consistently with the legacy path. Templates with mismatched
        roles, missing content, etc. should not silently produce
        garbage."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        with pytest.raises(ValueError, match="missing 'content'"):
            llm._apply_jinja_template([{"role": "user"}])
        with pytest.raises(ValueError, match="missing or invalid 'role'"):
            llm._apply_jinja_template([{"content": "no role"}])
        with pytest.raises(TypeError):
            llm._apply_jinja_template(["not a dict"])


# ---------------------------------------------------------------------------
# Layer 3: LLM._apply_template wrapper fallback wiring
# ---------------------------------------------------------------------------


class TestApplyTemplateFallback:
    """The wrapper must try the Jinja path first and fall back to the
    legacy substring-heuristic path on TemplateError. We can't easily
    construct a model whose embedded template will raise without
    using a real model that does this (Gemma is the canonical example
    but isn't a CI fixture), so most of this layer is covered by the
    pipeline-level fallback tests in test_rag_pipeline.py.
    """

    def test_wrapper_uses_jinja_path_when_template_is_none(self, model_path):
        """Calling `_apply_template` with `template=None` should
        produce the same output as calling `_apply_jinja_template`
        directly, because the wrapper's first branch is the Jinja path."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        msgs = [{"role": "user", "content": "Hello."}]

        via_wrapper = llm._apply_template(msgs)
        via_direct = llm._apply_jinja_template(msgs)
        assert via_wrapper == via_direct

    def test_wrapper_uses_legacy_path_when_template_is_named(self, model_path):
        """When the caller explicitly passes `template='llama3'` (or
        any named template), the wrapper must skip the Jinja path
        because named-template lookup is a feature of the legacy
        path."""
        from cyllama.api import LLM

        llm = LLM(model_path, verbose=False)
        msgs = [{"role": "user", "content": "Hello."}]

        prompt = llm._apply_template(msgs, template="llama3")
        # Llama-3 named template produces a Llama-3 shaped prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "Hello." in prompt


# ---------------------------------------------------------------------------
# Layer 4: Gemma 4 regression -- the model that motivated all of this
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not GEMMA_MODEL.exists(),
    reason=f"Gemma 4 model not found at {GEMMA_MODEL}",
)
class TestGemma4Regression:
    """Pin the Gemma 4 fix end-to-end. Before the vendored-jinja2 path,
    `LLM(gemma).chat(...)` raised `RuntimeError: Failed to apply chat
    template` immediately on the first call because llama.cpp's
    substring heuristic doesn't recognise Gemma 4's `<|turn>` markers
    (it looks for `<start_of_turn>`, which Gemma 2/3 use).

    After the vendored jinja2 path, the embedded Jinja template is
    evaluated directly and produces the correct prompt. This test
    pins that behaviour so a future regression in the binding or the
    fallback wiring is caught immediately.
    """

    def test_jinja_path_renders_gemma4_template(self):
        from cyllama.api import LLM

        llm = LLM(str(GEMMA_MODEL), verbose=False)
        try:
            prompt = llm._apply_jinja_template(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi."},
                ]
            )
            # Gemma 4 specifically uses `<|turn>` markers, NOT
            # `<start_of_turn>` (which is what Gemma 2/3 use, and what
            # llama.cpp's substring heuristic looks for). The presence
            # of `<|turn>` confirms we're rendering Gemma 4's actual
            # embedded template, not a fallback.
            assert "<|turn>" in prompt
            assert "You are helpful." in prompt
            assert "Hi." in prompt
        finally:
            # Explicit cleanup so the GPU resources are released before
            # the next test loads the same model. Without this, two
            # Gemma 4 contexts can briefly overlap on Metal because
            # Python GC is non-deterministic, and the second load
            # occasionally fails with `llama_decode failed`.
            llm.close()

    def test_chat_method_works_end_to_end(self):
        """The full LLM.chat() flow must work for Gemma 4. Before the
        fix, this raised RuntimeError on the first call."""
        from cyllama.api import LLM, GenerationConfig

        llm = LLM(
            str(GEMMA_MODEL),
            config=GenerationConfig(max_tokens=50, temperature=0.0, n_gpu_layers=99),
            verbose=False,
        )
        try:
            response = llm.chat(
                [
                    {"role": "system", "content": "You are helpful. Answer in one sentence."},
                    {"role": "user", "content": "What is the capital of France?"},
                ]
            )
            text = str(response).lower()
            assert "paris" in text, f"expected 'paris' in response, got: {text!r}"
        finally:
            llm.close()
