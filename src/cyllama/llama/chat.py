#!/usr/bin/env python3
"""
Chat implementation equivalent to build/llama.cpp/examples/simple-chat/simple-chat.cpp

This module provides a Python implementation of the chat example using the cyllama wrapper.
"""

import sys
import argparse
from typing import List, Dict

from .._defaults import DEFAULT_MAX_TOKENS, DEFAULT_N_GPU_LAYERS
from ..utils.color import green, yellow, END

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


def print_usage():
    """Print usage information"""
    print("\nexample usage:")
    print(f"\n    {sys.argv[0]} -m model.gguf [-c context_size] [-ngl n_gpu_layers]")
    print()


class Chat:
    """Chat interface using cyllama"""

    def __init__(self, model_path: str, n_ctx: int = 2048, ngl: int = DEFAULT_N_GPU_LAYERS, max_tokens: int = DEFAULT_MAX_TOKENS):
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

        # Initialize sampler
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        self.sampler.add_min_p(0.05, 1)
        self.sampler.add_temp(0.8)
        self.sampler.add_dist(1337)  # Use fixed seed like LLAMA_DEFAULT_SEED

        # Initialize chat state (simplified - use list of dicts instead of LlamaChatMessage objects)
        self.messages: List[Dict[str, str]] = []
        self.prev_len = 0
        self.n_past = 0  # Track position in context

        # Store parameters for creating fresh contexts
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.ngl = ngl
        self.max_tokens = max_tokens

    def _apply_template(
        self,
        messages: List,
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
                    chat_msgs.append(
                        LlamaChatMessage(role=msg["role"], content=msg["content"])
                    )
            return self.model.chat_apply_template(tmpl, chat_msgs, add_assistant_msg)

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
        messages: List,
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

        def tojson_filter(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

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
        return compiled.render(
            messages=msg_dicts,
            bos_token=bos_token,
            eos_token=eos_token,
            add_generation_prompt=add_generation_prompt,
        )

    def generate(self, prompt: str) -> str:
        """Generate response for the given prompt using a fresh context"""
        # Create a fresh context for this generation to avoid state issues
        fresh_context = LlamaContext(self.model, self.context.params, verbose=False)

        response = ""
        n_past = 0

        # Tokenize the prompt
        prompt_tokens = self.vocab.tokenize(prompt, True, True)

        if not prompt_tokens:
            raise ValueError("Failed to tokenize the prompt")

        # Create batch for the prompt starting from position 0
        batch = llama_batch_get_one(prompt_tokens, 0)
        n_past = len(prompt_tokens)

        # Decode the initial batch
        try:
            ret = fresh_context.decode(batch)
            if ret != 0:
                print(f"Warning: decode returned {ret}")
        except Exception as e:
            print(f"Decode failed with error: {e}")
            return response

        # Generation loop
        max_tokens = self.max_tokens

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

        return response.strip()

    def chat_loop(self):
        """Main chat loop"""
        # Enable readline-style line editing and persistent history
        # (up/down arrows cycle through prior turns, Ctrl-R reverse
        # search, etc.). Gracefully no-ops on platforms without
        # readline. Uses a separate history file from `cyllama rag` so
        # the two REPLs don't pollute each other.
        from .._readline import setup_history, history_path_for

        setup_history(history_path_for("chat"))

        while True:
            # Get user input
            print(green("> "), end="")
            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                break

            # Always use dict format for messages (works with both jinja
            # and C API paths via _apply_template)
            self.messages.append({"role": "user", "content": user_input})
            prompt = self._apply_template(self.messages, add_assistant_msg=True)

            # Generate response
            try:
                response = self.generate(prompt)
                print(yellow(response))


                self.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
                # Remove the user message that caused the error
                self.messages.pop()


def main():
    """Main entry point"""
    disable_logging()
    parser = argparse.ArgumentParser(description="Simple chat using cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("-c", "--context", type=int, default=2048, help="Context size")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS, help="Number of GPU layers")
    parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response")

    args = parser.parse_args()

    try:
        chat = Chat(args.model, args.context, args.n_gpu_layers, args.max_tokens)
        chat.chat_loop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
