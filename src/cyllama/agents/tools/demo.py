"""``DEMO_TOOLS`` -- the minimal reference set of pre-built tools that
ships with cyllama.

These tools are auto-registered when ``cyllama chat`` invokes any
``/agent*`` command (see ``cyllama.llama.chat._run_agent_command``), and
are exported from ``cyllama.agents`` so library users can pass them to
their own agents (or omit them entirely).

The set was chosen so each tool illustrates one distinct pattern:

* :func:`current_time`    -- zero-arg lookup with optional validated parameter.
* :func:`calculator`      -- safe expression evaluation via an AST allowlist.
* :func:`word_count`      -- multi-statistic string analyzer.
* :func:`search_wikipedia`-- bounded network IO.
* :func:`quarto_render`   -- subprocess + filesystem write with two usage modes.

Anti-goals (deliberately *not* included):

* No filesystem tools. A safe filesystem tool needs a per-instance sandbox
  root and resolved-path checks; that doesn't fit a global module-level
  helper.
* No arbitrary ``http_get``. A general HTTP fetcher needs an allowlist
  the agent author chooses; shipping one with an open allowlist would
  be unsafe, and shipping one with a closed allowlist would be useless.
* No shell or eval. These are the canonical footguns; if a user wants
  them they should opt in explicitly.

Tuples are used so consumers can't mutate the shared list and
accidentally desync different agents in the same process.

``quarto_render`` is included unconditionally; runtime checks raise a
clear "install quarto" error if the binary is missing, so adding it to
the collection doesn't make non-quarto installs surface weird failures
at import time. Agent surfaces that want to hide unavailable tools
entirely should gate on :func:`cyllama.agents.tools.quarto_available`
themselves.
"""

from typing import Tuple

from .calculator import calculator
from .core import Tool
from .current_time import current_time
from .quarto import quarto_render
from .search_wikipedia import search_wikipedia
from .word_count import word_count


DEMO_TOOLS: Tuple[Tool, ...] = (
    current_time,
    calculator,
    word_count,
    search_wikipedia,
    quarto_render,
)
