"""Parity test for the `llama.llama_cpp` public surface.

This test pins the names the rest of the package — most importantly the
agent module — depends on. It exists primarily to catch drift between the
sibling projects: cyllama (Cython) and inferna (nanobind) both expose
`<pkg>.llama.llama_cpp` as the public binding facade, and shared Python
code does `from ..llama.llama_cpp import LlamaSampler` etc. without
caring which binding strategy backs it.

If a new method is added to `LlamaSampler` in one project but the other
project's facade isn't updated to re-export it, agent code breaks in
that project alone. The discoverable failure mode is "agent works in
cyllama but AttributeErrors in inferna" — exactly the kind of bug that
shared-source porting can hide. Running the same test in both repos
detects the gap mechanically.

The list below is the *minimum* surface the agent + api + batching
modules need. Add to it whenever a new dependency is introduced.
"""

from __future__ import annotations

import pytest

from cyllama.llama import llama_cpp as L


# Names required by the agent layer (constrained.py) plus core api/batching.
REQUIRED_NAMES: tuple[str, ...] = (
    # Sampler chain — required by ConstrainedAgent.
    "LlamaSampler",
    "LlamaSamplerChainParams",
    # Batch helpers — required by ConstrainedAgent's grammar generation.
    "llama_batch_get_one",
    # Core model / context — required by api.LLM.
    "LlamaModel",
    "LlamaContext",
    "LlamaBatch",
    "LlamaVocab",
    # Parameter wrappers — required by api.LLM construction.
    "LlamaModelParams",
    "LlamaContextParams",
)


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_llama_cpp_exports_name(name: str) -> None:
    """Every required name must resolve and be callable or a type."""
    attr = getattr(L, name, None)
    assert attr is not None, (
        f"llama_cpp facade missing required name: {name!r}. "
        "If this fires only in inferna, the nanobind re-export shim is "
        "out of sync with the Cython surface. If it fires in both, the "
        "name was removed and the agent layer needs updating."
    )
    assert callable(attr) or isinstance(attr, type), (
        f"llama_cpp.{name} resolved but is neither callable nor a type (got {type(attr).__name__})"
    )


def test_llama_cpp_sampler_grammar_method() -> None:
    """LlamaSampler must expose add_grammar — ConstrainedAgent calls it."""
    sampler_cls = L.LlamaSampler
    assert hasattr(sampler_cls, "add_grammar"), (
        "LlamaSampler.add_grammar is required by ConstrainedAgent's "
        "GrammarConstrainedLLM._ensure_sampler_with_grammar. If this "
        "fires only in inferna, the nanobind binding does not expose "
        "the grammar sampler method that cyllama's Cython binding does."
    )
