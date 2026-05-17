#!/usr/bin/env python3
"""
Chat implementation equivalent to build/llama.cpp/examples/simple-chat/simple-chat.cpp

This module provides a Python implementation of the chat example using the cyllama wrapper.
"""

import glob as _glob
import os
import sys
import time
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from ..defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    LLAMA_DEFAULT_SEED,
)
from ..utils.color import white, magenta, grey, cyan, red, bold, END, esc, FG_END

from .llama_cpp import (
    LlamaModel,
    LlamaContext,
    LlamaSampler,
    LlamaChatMessage,
    LlamaModelParams,
    LlamaContextParams,
    LlamaSamplerChainParams,
    ggml_backend_load_all,
    llama_batch_get_one,
    disable_logging,
)

# Vendored jinja2 for chat-template rendering (same path as api.py).
try:
    from cyllama._vendor.jinja2.exceptions import TemplateError as _JinjaTemplateError
except ImportError:  # pragma: no cover
    _JinjaTemplateError = type("_JinjaTemplateError", (Exception,), {})


SLASH_COMMANDS: Tuple[Tuple[str, str], ...] = (
    ("/help", "show this help message"),
    ("/exit", "exit the chat"),
    ("/clear", "clear conversation history"),
    ("/regen", "regenerate the last assistant reply"),
    ("/read", "/read <file> - queue a file as context for the next message"),
    ("/glob", "/glob <pattern> - queue files matching a glob as context"),
    ("/tools", "list the tools available to /agent"),
    ("/agent", "/agent <task> - run a ReAct agent"),
    ("/agent-strict", "/agent-strict <task> - run a constrained agent"),
    ("/agent-contract", "/agent-contract <task> - run a contract agent"),
    ("/agent-plan", "/agent-plan <task> - run a plan-then-act agent"),
    ("/agent-reflect", "/agent-reflect <task> - run a reflect-and-revise agent"),
)

SLASH_COMMAND_NAMES: Tuple[str, ...] = tuple(name for name, _ in SLASH_COMMANDS)

FILE_ARG_COMMANDS: Tuple[str, ...] = ("/read", "/glob")


def print_usage() -> None:
    """Print usage information"""
    print("\nexample usage:")
    print(f"\n    {sys.argv[0]} -m model.gguf [-c context_size] [-ngl n_gpu_layers]")
    print()


class Chat:
    """Chat interface using cyllama"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        ngl: int = DEFAULT_N_GPU_LAYERS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        min_p: float = DEFAULT_MIN_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        seed: int = LLAMA_DEFAULT_SEED,
    ):
        """Initialize the chat with model and parameters"""
        # Set up error-only logging (skip for now to avoid issues)
        # set_log_callback(lambda level, text: sys.stderr.write(text) if level >= 3 else None)

        # Load dynamic backends
        ggml_backend_load_all()

        # Initialize model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = ngl
        self.model = LlamaModel(model_path, model_params)

        # Get vocab
        self.vocab = self.model.get_vocab()

        # Initialize context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_ctx
        self.context = LlamaContext(self.model, ctx_params)

        # Initialize sampler with caller-provided parameters
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        if repeat_penalty != 1.0:
            self.sampler.add_penalties(64, repeat_penalty, 0.0, 0.0)
        self.sampler.add_top_k(top_k)
        self.sampler.add_top_p(top_p, 1)
        self.sampler.add_min_p(min_p, 1)
        self.sampler.add_temp(temperature)
        self.sampler.add_dist(seed)

        # Initialize chat state (simplified - use list of dicts instead of LlamaChatMessage objects)
        self.messages: List[Dict[str, str]] = []
        self.prev_len = 0
        self.n_past = 0  # Track position in context

        # Store parameters for creating fresh contexts
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.ngl = ngl
        self.max_tokens = max_tokens

        # Session-level statistics
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_prompt_time = 0.0
        self.total_generation_time = 0.0

        # Lazily-constructed high-level LLM for /agent* commands.
        # Built on first /agent use so users who never invoke agents
        # don't pay for a second model handle.
        self._agent_llm: Any = None

    def _apply_template(
        self,
        messages: List[Any],
        add_assistant_msg: bool = True,
    ) -> str:
        """Apply chat template using vendored jinja2 with C API fallback.

        Mirrors the two-tier approach in ``cyllama.api.LLM._apply_template``:
        try the vendored jinja2 interpreter first (handles Gemma 4, Qwen3,
        and any template the C heuristic doesn't recognise), then fall back
        to the C ``llama_chat_apply_template`` on failure.
        """
        # --- Tier 1: vendored jinja2 ---
        try:
            return self._apply_jinja_template(messages, add_assistant_msg)
        except _JinjaTemplateError:
            pass
        except Exception:
            pass

        # --- Tier 2: C API (substring heuristic) ---
        tmpl = self.model.get_default_chat_template()
        if tmpl:
            chat_msgs = []
            for msg in messages:
                if isinstance(msg, LlamaChatMessage):
                    chat_msgs.append(msg)
                else:
                    chat_msgs.append(LlamaChatMessage(role=msg["role"], content=msg["content"]))
            return cast(str, self.model.chat_apply_template(tmpl, chat_msgs, add_assistant_msg))

        # --- Tier 3: simple User/Assistant formatting ---
        conversation = ""
        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            content = msg["content"] if isinstance(msg, dict) else msg.content
            if role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
            elif role == "system":
                conversation += f"System: {content}\n"
        if add_assistant_msg:
            conversation += "Assistant:"
        return conversation

    def _apply_jinja_template(
        self,
        messages: List[Any],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the model's embedded chat template via vendored jinja2."""
        import json
        from datetime import datetime

        from cyllama._vendor.jinja2 import ext as _jinja2_ext
        from cyllama._vendor.jinja2.exceptions import TemplateError
        from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

        template_str = self.model.get_default_chat_template()
        if not template_str:
            raise TemplateError("Model has no embedded chat template")

        vocab = self.model.get_vocab()
        bos_id = vocab.token_bos()
        eos_id = vocab.token_eos()
        bos_token = vocab.token_to_piece(bos_id, special=True) if bos_id >= 0 else ""
        eos_token = vocab.token_to_piece(eos_id, special=True) if eos_id >= 0 else ""

        def raise_exception(message: str) -> str:
            raise TemplateError(message)

        def tojson_filter(
            value: Any,
            ensure_ascii: bool = False,
            indent: Optional[int] = None,
            separators: Optional[Tuple[str, str]] = None,
            sort_keys: bool = False,
        ) -> str:
            return json.dumps(
                value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
            )

        def strftime_now(fmt: str) -> str:
            return datetime.now().strftime(fmt)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[_jinja2_ext.loopcontrols],
        )
        env.filters["tojson"] = tojson_filter
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now

        # Normalise messages to dicts for the template
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_dicts.append(msg)
            else:
                msg_dicts.append({"role": msg.role, "content": msg.content})

        compiled = env.from_string(template_str)
        return cast(
            str,
            compiled.render(
                messages=msg_dicts,
                bos_token=bos_token,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt,
            ),
        )

    def generate(self, prompt: str, on_token: Optional[Callable[[str], None]] = None) -> str:
        """Generate response for the given prompt using a fresh context.

        Args:
            prompt: The full templated prompt to generate from.
            on_token: Optional callback invoked with each token piece as
                it is generated.  Used for streaming output.
        """
        # Create a fresh context for this generation to avoid state issues
        fresh_context = LlamaContext(self.model, self.context.params, verbose=False)

        response = ""
        n_past = 0
        n_generated = 0

        # Tokenize the prompt
        prompt_tokens = self.vocab.tokenize(prompt, True, True)

        if not prompt_tokens:
            raise ValueError("Failed to tokenize the prompt")

        # Create batch for the prompt starting from position 0
        batch = llama_batch_get_one(prompt_tokens, 0)
        n_past = len(prompt_tokens)

        # Decode the initial batch (prompt eval)
        t_prompt_start = time.perf_counter()
        try:
            ret = fresh_context.decode(batch)
            if ret != 0:
                print(f"Warning: decode returned {ret}")
        except Exception as e:
            print(f"Decode failed with error: {e}")
            return response
        t_prompt_end = time.perf_counter()

        # Generation loop
        max_tokens = self.max_tokens
        t_gen_start = time.perf_counter()

        for i in range(max_tokens):
            # Check context size
            n_ctx = fresh_context.n_ctx
            if n_past >= n_ctx - 1:  # Leave room for at least one more token
                print(END)
                print("context size exceeded", file=sys.stderr)
                break

            # Sample next token
            new_token_id = self.sampler.sample(fresh_context, -1)

            # Check if it's end of generation
            if self.vocab.is_eog(new_token_id):
                break

            # Convert token to piece and add to response
            try:
                piece = self.vocab.token_to_piece(new_token_id, 0, True)
                response += piece
                n_generated += 1
                if on_token is not None:
                    on_token(piece)

            except Exception as e:
                print(f"Failed to convert token to piece: {e}")
                break

            # Create batch with the new token at the correct position
            batch = llama_batch_get_one([new_token_id], n_past)
            n_past += 1

            # Decode for next iteration
            try:
                ret = fresh_context.decode(batch)
                if ret != 0:
                    print(f"Warning: decode returned {ret}")
            except Exception as e:
                print(f"Decode failed at token {i + 1}: {e}")
                break

        t_gen_end = time.perf_counter()

        # Accumulate session stats
        self.total_prompt_tokens += len(prompt_tokens)
        self.total_generated_tokens += n_generated
        self.total_prompt_time += t_prompt_end - t_prompt_start
        self.total_generation_time += t_gen_end - t_gen_start

        return response.strip()

    def print_session_stats(self) -> None:
        """Print a formatted table of accumulated session statistics."""
        total_time = self.total_prompt_time + self.total_generation_time
        tps = self.total_generated_tokens / self.total_generation_time if self.total_generation_time > 0 else 0.0
        rows = [
            ("Prompt tokens", str(self.total_prompt_tokens)),
            ("Generated tokens", str(self.total_generated_tokens)),
            ("Prompt eval time", f"{self.total_prompt_time:.2f} s"),
            ("Generation time", f"{self.total_generation_time:.2f} s"),
            ("Total time", f"{total_time:.2f} s"),
            ("Tokens/second", f"{tps:.2f}"),
        ]
        key_width = max(len(r[0]) for r in rows)
        val_width = max(len(r[1]) for r in rows)
        width = key_width + val_width + 5
        line = "-" * width
        print(line, file=sys.stderr)
        for key, val in rows:
            print(f"  {key:<{key_width}} | {val:>{val_width}}", file=sys.stderr)
        print(line, file=sys.stderr)

    def print_banner(self) -> None:
        """Print startup banner with model info and available commands."""
        from .. import __version__

        model_name = os.path.basename(self.model_path)

        #     print(f"build      : inferna v{__version__}")
        #     print(f"model      : {model_name}")
        #     print("modalities : text")

        left = f"{bold(cyan('cyllama'))} v{__version__} chat"
        right = model_name
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 80
        print(f"{left}{right:>{cols - (len(left) - 19)}}")

        print()
        print(f"type {cyan('/help')} to list available commands, or {cyan('/exit')} to quit")
        print()

    def _get_agent_llm(self) -> Any:
        """Lazily build a high-level :class:`cyllama.api.LLM` for agent calls.

        Reused across /agent invocations within the same chat session.
        Constructed with the same model + ctx + ngl the Chat was given.
        """
        if self._agent_llm is None:
            from ..api import GenerationConfig, LLM

            self._agent_llm = LLM(
                self.model_path,
                config=GenerationConfig(
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.ngl,
                    max_tokens=self.max_tokens,
                ),
                verbose=False,
            )
        return self._agent_llm

    @staticmethod
    def _render_agent_event(ev: Any) -> None:
        """Print a single :class:`AgentEvent` to stdout with type-appropriate color.

        ANSWER events print without color (matching normal chat output).
        Trace events (THOUGHT/ACTION/OBSERVATION/...) print in dim color
        prefixed with the event type and optional ``source`` tag.
        """
        from ..agents.types import EventType

        etype = ev.type
        content = ev.content or ""
        source = (ev.metadata or {}).get("source")
        tag = f"[{etype.value}{' ' + source if source else ''}]"

        if etype == EventType.ANSWER:
            # Final answer prints uncolored. Composed-agent answers
            # tag themselves with source="final"; intermediate
            # planner/worker answers also flow here but are equally
            # legible without extra styling.
            if source and source != "final":
                print(grey(tag), content)
            else:
                print(content)
            return

        if etype == EventType.ERROR:
            print(red(f"{tag} {content}"), file=sys.stderr)
            return

        if etype == EventType.THOUGHT:
            print(cyan(tag), grey(content))
            return
        if etype == EventType.ACTION:
            print(magenta(tag), content)
            return
        if etype == EventType.OBSERVATION:
            print(grey(f"{tag} {content}"))
            return

        # CONTRACT_CHECK / CONTRACT_VIOLATION / future types: render
        # as a generic trace line so we never silently drop events.
        print(grey(f"{tag} {content}"))

    def _run_agent_command(self, kind: str, task: str) -> None:
        """Execute one /agent* slash command.

        Streams events from :func:`cyllama.agents.stream_agent`,
        rendering each via :meth:`_render_agent_event`. The final
        answer is appended to the chat history so subsequent turns
        can reference it.
        """
        from ..agents import stream_agent
        from ..agents.tools import DEMO_TOOLS
        from ..agents.types import EventType

        if not task.strip():
            print(f"usage: /agent{('-' + kind) if kind != 'react' else ''} <task>")
            return

        llm = self._get_agent_llm()
        # Echo the task as a user message in the transcript so chat
        # history reflects what the agent was asked to do.
        self.messages.append({"role": "user", "content": task})

        final_answer = ""
        try:
            for ev in stream_agent(kind, llm, task, tools=list(DEMO_TOOLS)):
                self._render_agent_event(ev)
                if ev.type == EventType.ANSWER:
                    src = (ev.metadata or {}).get("source")
                    # For composed kinds (plan/reflect) only the
                    # "final" answer is the canonical reply. For
                    # simple kinds there's no source tag, so any
                    # ANSWER is the final.
                    if src in (None, "final"):
                        final_answer = ev.content or ""
        except KeyboardInterrupt:
            print(FG_END)
            return
        except Exception as e:  # noqa: BLE001
            print(red(f"agent error: {e}"), file=sys.stderr)
            # Roll back the user echo so a bad command doesn't poison history.
            self.messages.pop()
            return

        if final_answer:
            self.messages.append({"role": "assistant", "content": final_answer})

    def _run_turn(self, stream: bool) -> Optional[Tuple[str, float, float]]:
        """Generate a response for the current self.messages.

        Returns (response, prompt_tps, gen_tps) or None on error.
        """
        prompt = self._apply_template(self.messages, add_assistant_msg=True)

        prompt_tokens_before = self.total_prompt_tokens
        gen_tokens_before = self.total_generated_tokens
        prompt_time_before = self.total_prompt_time
        gen_time_before = self.total_generation_time

        if stream:
            response = self.generate(
                prompt,
                on_token=lambda piece: print(piece, end="", flush=True),
            )
            print()
        else:
            response = self.generate(prompt)
            print(response)

        prompt_tokens = self.total_prompt_tokens - prompt_tokens_before
        gen_tokens = self.total_generated_tokens - gen_tokens_before
        prompt_time = self.total_prompt_time - prompt_time_before
        gen_time = self.total_generation_time - gen_time_before

        prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0.0
        gen_tps = gen_tokens / gen_time if gen_time > 0 else 0.0
        return response, prompt_tps, gen_tps

    def _load_files(self, paths: List[str]) -> str:
        """Read files; return concatenated content with markers. Prints errors to stderr."""
        chunks: List[str] = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    chunks.append(f"--- BEGIN FILE: {p} ---\n{f.read()}\n--- END FILE: {p} ---")
            except OSError as e:
                print(f"error reading {p}: {e}", file=sys.stderr)
        return "\n\n".join(chunks)

    def chat_loop(self, stream: bool = True, stats: bool = False) -> None:
        """Main chat loop.

        Args:
            stream: If True (default), print tokens as they are
                generated.  If False, buffer the full response before
                printing.
            stats: If True, print session statistics on exit.
        """
        # Enable readline-style line editing and persistent history
        # (up/down arrows cycle through prior turns, Ctrl-R reverse
        # search, etc.). Gracefully no-ops on platforms without
        # readline. Uses a separate history file from `cyllama rag` so
        # the two REPLs don't pollute each other.
        from .._internal.readline import setup_history, history_path_for, setup_completer

        setup_history(history_path_for("chat"))
        restore_completer = setup_completer(SLASH_COMMAND_NAMES, FILE_ARG_COMMANDS)

        # Clear screen + scrollback and home the cursor before the
        # banner so the chat starts on a fresh viewport. ANSI escapes
        # work on macOS/Linux terminals and on Windows 10+ consoles
        # (which have VT processing enabled by default).
        print("\033[2J\033[3J\033[H", end="", flush=True)

        self.print_banner()

        pending_context: str = ""

        try:
            while True:
                # print(bold(green("> ")) + esc(1, 32), end="", flush=True)
                # print(bold(yellow("> ")) + esc(1, 33), end="", flush=True)
                print(bold(white("> ")) + esc(1, 37), end="", flush=True)
                try:
                    raw = input()
                except (EOFError, KeyboardInterrupt):
                    print(esc(22, 39))
                    break
                print(esc(22, 39), end="", flush=True)

                user_input = raw.strip()
                if not user_input:
                    continue

                # Slash commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0]
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd == "/help":
                        width = max(len(name) for name, _ in SLASH_COMMANDS)
                        print("available commands:")
                        for name, desc in SLASH_COMMANDS:
                            print(f"  {cyan(name.ljust(width))}  {desc}")
                        continue
                    if cmd == "/exit":
                        break
                    if cmd == "/clear":
                        self.messages.clear()
                        pending_context = ""
                        print("(history cleared)")
                        continue
                    if cmd == "/regen":
                        # Find the last user message; drop everything after it.
                        last_user_idx = next(
                            (i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i]["role"] == "user"),
                            None,
                        )
                        if last_user_idx is None:
                            print("(nothing to regenerate)")
                            continue
                        del self.messages[last_user_idx + 1 :]
                        try:
                            result = self._run_turn(stream)
                            if result is not None:
                                response, p_tps, g_tps = result
                                self.messages.append({"role": "assistant", "content": response})
                                print(magenta(f"[ Prompt: {p_tps:.1f} t/s | Generation: {g_tps:.1f} t/s ]"))
                                print()
                        except KeyboardInterrupt:
                            print(FG_END)
                            break
                        continue
                    if cmd == "/read":
                        if not arg:
                            print("usage: /read <file>")
                            continue
                        content = self._load_files([arg])
                        if content:
                            pending_context = pending_context + ("\n\n" if pending_context else "") + content
                            print(f"(queued {arg}; will be prepended to your next message)")
                        continue
                    if cmd == "/glob":
                        if not arg:
                            print("usage: /glob <pattern>")
                            continue
                        matches = sorted(_glob.glob(arg, recursive=True))
                        if not matches:
                            print(f"(no files matched: {arg})")
                            continue
                        content = self._load_files(matches)
                        if content:
                            pending_context = pending_context + ("\n\n" if pending_context else "") + content
                            print(f"(queued {len(matches)} file(s); will be prepended to your next message)")
                        continue

                    if cmd == "/tools":
                        # Lazy-import so users who never list/use tools don't
                        # pay the import cost on chat start.
                        from ..agents.tools import DEMO_TOOLS

                        if not DEMO_TOOLS:
                            print("(no tools registered)")
                            continue
                        width = max(len(t.name) for t in DEMO_TOOLS)
                        print("available tools (for /agent*):")
                        for t in DEMO_TOOLS:
                            # First non-empty line of the tool's description
                            # is conventionally the one-line summary. Avoid
                            # dumping multi-paragraph docstrings here.
                            summary = next(
                                (line.strip() for line in t.description.splitlines() if line.strip()),
                                "",
                            )
                            print(f"  {cyan(t.name.ljust(width))}  {summary}")
                        continue

                    # /agent and /agent-* family
                    if cmd == "/agent" or cmd.startswith("/agent-"):
                        agent_kind_map = {
                            "/agent": "react",
                            "/agent-strict": "constrained",
                            "/agent-contract": "contract",
                            "/agent-plan": "plan",
                            "/agent-reflect": "reflect",
                        }
                        agent_kind = agent_kind_map.get(cmd)
                        if agent_kind is None:
                            print(f"unknown agent command: {cmd}")
                            continue
                        self._run_agent_command(agent_kind, arg)
                        continue

                    print(f"unknown command: {cmd}")
                    continue

                # Prepend any queued file context to this user message
                if pending_context:
                    content = pending_context + "\n\n" + user_input
                    pending_context = ""
                else:
                    content = user_input

                self.messages.append({"role": "user", "content": content})

                try:
                    result = self._run_turn(stream)
                    if result is not None:
                        response, p_tps, g_tps = result
                        self.messages.append({"role": "assistant", "content": response})
                        print(magenta(f"[ Prompt: {p_tps:.1f} t/s | Generation: {g_tps:.1f} t/s ]"))
                        print()

                except KeyboardInterrupt:
                    print(FG_END)
                    break
                except Exception as e:
                    print(FG_END, end="")
                    print(f"\nError: {e}", file=sys.stderr)
                    # Remove the user message that caused the error
                    self.messages.pop()
        finally:
            # Always reset terminal colors on exit
            print(END, end="", flush=True)
            if restore_completer is not None:
                restore_completer()
            if stats and self.total_generated_tokens > 0:
                print(file=sys.stderr)
                self.print_session_stats()


def main() -> None:
    """Main entry point"""
    disable_logging()
    parser = argparse.ArgumentParser(description="Simple chat using cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("-c", "--context", type=int, default=2048, help="Context size")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS, help="Number of GPU layers")
    parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (default: %(default)s)"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k sampling (default: %(default)s)")
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p (nucleus) sampling (default: %(default)s)"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Min-p sampling threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=DEFAULT_REPEAT_PENALTY,
        help="Repetition penalty, 1.0 = disabled (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=LLAMA_DEFAULT_SEED, help="Random seed (default: %(default)s)")
    parser.add_argument(
        "--no-stream", action="store_true", help="Buffer full response before printing (default: stream)"
    )
    parser.add_argument("--stats", action="store_true", help="Show session statistics on exit")

    args = parser.parse_args()

    try:
        chat = Chat(
            args.model,
            n_ctx=args.context,
            ngl=args.n_gpu_layers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
            seed=args.seed,
        )
        chat.chat_loop(stream=not args.no_stream, stats=args.stats)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
