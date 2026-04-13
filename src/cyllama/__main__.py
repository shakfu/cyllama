"""cyllama CLI: cyllama [command] (or python -m cyllama [command])"""

import argparse
import platform
import sys

from ._defaults import (
    LLAMA_DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_GPU_LAYERS,
)


def _parse_system_info(info_str: str) -> dict[str, str]:
    """Parse 'KEY = VALUE | KEY2 = VALUE2 |' format into a dict."""
    result = {}
    for part in info_str.split("|"):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result


def _cpu_features_from_info(info: dict[str, str]) -> list[str]:
    """Extract enabled CPU features from system info dict."""
    cpu_keys = [
        "NEON",
        "AVX",
        "AVX2",
        "AVX512",
        "FMA",
        "ARM_FMA",
        "F16C",
        "FP16_VA",
        "DOTPROD",
        "SSE3",
        "WASM_SIMD",
        "VSX",
    ]
    features = []
    for key in cpu_keys:
        for info_key, val in info.items():
            if info_key.strip().endswith(key) and val == "1":
                features.append(key)
                break
    return features


def _get_built_backends() -> list[str]:
    """Return GPU backend names enabled at build time (from _backend.py)."""
    try:
        from . import _backend
    except ImportError:
        return []
    _names = {
        "cuda": "CUDA",
        "vulkan": "Vulkan",
        "metal": "Metal",
        "hip": "HIP",
        "sycl": "SYCL",
        "opencl": "OpenCL",
        "blas": "BLAS",
    }
    return [name for attr, name in _names.items() if getattr(_backend, attr, False)]


def _get_loaded_backends() -> list[str]:
    """Return GPU backend names currently registered in the ggml registry."""
    try:
        from .llama import llama_cpp as cy

        return [r for r in cy.ggml_backend_reg_names() if r not in ("CPU",)]
    except Exception:
        return []


def _get_build_info() -> dict:
    """Load build info if available."""
    try:
        from . import _build_info

        return {k: v for k, v in vars(_build_info).items() if not k.startswith("_")}
    except ImportError:
        return {}


def cmd_info():
    """Print build and backend information."""
    from . import __version__

    print(f"cyllama {__version__}")
    print(f"Python {platform.python_version()} ({platform.platform()})")
    build_info = _get_build_info()
    print()

    # llama.cpp
    print("llama.cpp:")
    try:
        from .llama import llama_cpp as cy

        cy.llama_backend_init()
        cy.ggml_backend_load_all()
        llama_ver = build_info.get("llama_cpp_version", "unknown")
        print(f"  version:       {llama_ver}")
        print(f"  ggml version:  {cy.ggml_version()}")
        print(f"  ggml commit:   {cy.ggml_commit()}")
        built = _get_built_backends()
        print(f"  built:         {', '.join(built) if built else 'CPU only'}")
        print(f"  registries:    {', '.join(cy.ggml_backend_reg_names())}")
        devices = cy.ggml_backend_dev_info()
        if devices:
            print("  devices:")
            for dev in devices:
                print(f"    {dev['name']:20s} [{dev['type']:5s}]  {dev['description']}")
        print(f"  GPU offload:   {cy.llama_supports_gpu_offload()}")
        print(f"  MMAP support:  {cy.llama_supports_mmap()}")
        print(f"  MLOCK support: {cy.llama_supports_mlock()}")
        print(f"  RPC support:   {cy.llama_supports_rpc()}")
    except Exception as e:
        print(f"  not available ({e})")

    print()

    # whisper.cpp
    print("whisper.cpp:")
    try:
        from .whisper import whisper_cpp

        # Load backends so whisper sees GPU registries (mirrors what
        # every upstream whisper.cpp example does in main())
        whisper_cpp.ggml_backend_load_all()

        info_str = whisper_cpp.print_system_info()
        whisper_ver = build_info.get("whisper_cpp_version", "unknown")
        print(f"  version:       {whisper_ver}")
        print(f"  ggml version:  {build_info.get('whisper_cpp_ggml_version', whisper_cpp.version())}")
        info = _parse_system_info(info_str)
        features = _cpu_features_from_info(info)
        built = _get_built_backends()
        loaded = _get_loaded_backends()
        print(f"  built:         {', '.join(built) if built else 'CPU only'}")
        print(f"  backends:      {', '.join(loaded) if loaded else 'CPU'}")
        if features:
            print(f"  CPU features:  {', '.join(features)}")
    except Exception as e:
        print(f"  not available ({e})")

    print()

    # stable-diffusion.cpp
    print("stable-diffusion.cpp:")
    try:
        from .sd import get_system_info, ggml_backend_load_all as sd_load_backends

        # Load backends so sd sees GPU registries
        sd_load_backends()

        sd_ver = build_info.get("stable_diffusion_cpp_version", "unknown")
        print(f"  version:       {sd_ver}")
        print(f"  ggml version:  {build_info.get('stable_diffusion_cpp_ggml_version', 'unknown')}")
        info_str = get_system_info()
        info = _parse_system_info(info_str)
        features = _cpu_features_from_info(info)
        built = _get_built_backends()
        loaded = _get_loaded_backends()
        print(f"  built:         {', '.join(built) if built else 'CPU only'}")
        print(f"  backends:      {', '.join(loaded) if loaded else 'CPU'}")
        if features:
            print(f"  CPU features:  {', '.join(features)}")
    except Exception as e:
        print(f"  not available ({e})")


def cmd_version():
    """Print version."""
    from . import __version__

    print(__version__)


def cmd_generate(args):
    """Generate text from a prompt."""
    from .api import GenerationConfig, complete

    # Read prompt from arg, file, or stdin
    if args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file) as f:
            prompt = f.read()
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        print("Error: provide a prompt via -p, -f, or stdin", file=sys.stderr)
        return 1

    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        seed=args.seed,
    )

    try:
        if args.stream:
            for chunk in complete(prompt, args.model, config, stream=True, verbose=args.verbose):
                print(chunk, end="", flush=True)
            print()
        else:
            response = complete(prompt, args.model, config, verbose=args.verbose)
            if args.json:
                print(response.to_json())
            else:
                print(response)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    return 0


def cmd_chat(args):
    """Chat with a model."""
    from . import __version__
    import os

    model_name = os.path.basename(args.model)
    left = f"cyllama v{__version__} chat"
    right = model_name
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    print(f"{left}{right:>{cols - len(left)}}")

    if args.prompt:
        # Single-turn mode via high-level API
        from .api import GenerationConfig, chat

        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})

        config = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx_size,
            seed=args.seed,
        )

        if args.stream:
            for chunk in chat(messages, args.model, config, stream=True, verbose=args.verbose, template=args.template):
                print(chunk, end="", flush=True)
            print()
        else:
            response = chat(messages, args.model, config, verbose=args.verbose, template=args.template)
            if args.json:
                print(response.to_json())
            else:
                print(response)
    else:
        # Interactive mode - delegate to llama.chat
        sys.argv = [
            "cyllama chat",
            "-m",
            args.model,
            "-c",
            str(args.ctx_size),
            "-ngl",
            str(args.n_gpu_layers),
            "-n",
            str(args.max_tokens),
        ]
        from .llama.chat import main as chat_main

        chat_main()

    return 0


def cmd_embed(args):
    """Compute text embeddings."""
    from .rag.embedder import Embedder

    embedder = Embedder(
        args.model,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    # --dim: print dimensions and exit
    if args.dim:
        print(embedder.dimension)
        return 0

    # Collect texts
    texts = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        with open(args.file) as f:
            texts.extend(line.strip() for line in f if line.strip())
    if not sys.stdin.isatty():
        texts.extend(line.strip() for line in sys.stdin if line.strip())

    # --similarity: rank texts by cosine similarity to query
    if args.similarity:
        if not texts:
            print("Error: provide texts to rank via -t, -f, or stdin", file=sys.stderr)
            return 1
        query_emb = embedder.embed(args.similarity)
        text_embs = embedder.embed_batch(texts)
        # Cosine similarity (embeddings are already normalized by default)
        scores = []
        for text, emb in zip(texts, text_embs):
            dot = sum(a * b for a, b in zip(query_emb, emb))
            scores.append((dot, text))
        scores.sort(reverse=True)
        for score, text in scores:
            if score < args.threshold:
                break
            print(f"{score:6.4f}  {text}")
        return 0

    # Default: output raw embeddings as JSON
    if not texts:
        print("Error: provide text via -t, -f, or stdin", file=sys.stderr)
        return 1

    import json

    embeddings = embedder.embed_batch(texts)
    print(json.dumps([e.tolist() for e in embeddings]))
    return 0


def cmd_rag(args):
    """RAG: query documents with a language model."""
    from .rag import RAG, RAGConfig

    # The CLI defaults to the chat-template path because nearly every
    # GGUF used with `cyllama rag` in practice is a chat-tuned model
    # (Llama-3-Instruct, Qwen3-Chat, Mistral-Instruct, etc.) and the
    # raw-completion `Question:/Answer:` template causes those models to
    # leak their instruction-tuning artifacts (Qwen3 dumping its <think>
    # block, models re-roleplaying as the user, paragraph paraphrase
    # loops, etc.). `--no-chat-template` reverts to the legacy raw
    # completion path for base/completion models that need it.
    use_chat_template = not args.no_chat_template

    # --system is honoured by both paths, but goes to a different field:
    #   chat-template path -> system_prompt (native system message)
    #   raw-completion path -> baked into prompt_template
    prompt_template: str | None = None
    system_prompt: str | None = None
    if args.system:
        if use_chat_template:
            system_prompt = args.system
        else:
            prompt_template = f"""{args.system}

Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{{context}}

Question: {{question}}

Answer:"""

    config = RAGConfig(
        top_k=args.top_k,
        similarity_threshold=args.threshold or None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        # Streaming-level n-gram repetition detection stays on as a
        # belt-and-suspenders safety net: even with the chat-template
        # path doing the heavy lifting, the detector catches any residual
        # lexical loops on the small fraction of model/prompt
        # combinations the chat path doesn't fully tame.
        repetition_threshold=args.repetition_threshold,
        repetition_ngram=args.repetition_ngram,
        repetition_window=args.repetition_window,
        use_chat_template=use_chat_template,
        # Strip <think>...</think> reasoning blocks by default. Reasoning-
        # tuned models (Qwen3, DeepSeek-R1) emit a chain-of-thought block
        # before their actual answer when invoked via their native chat
        # template, and the block routinely consumes the entire 200-token
        # budget, leaving zero tokens for the answer the user actually
        # wanted. The system prompt also asks for `/no_think` mode and
        # forbids reasoning, but the stripper is the safety net for
        # models that ignore the directive.
        strip_think_blocks=not args.show_think,
        **({"system_prompt": system_prompt} if system_prompt else {}),
        **({"prompt_template": prompt_template} if prompt_template else {}),
    )

    # ------------------------------------------------------------------
    # Decide where the index lives, and whether we're (re)building or
    # reusing it. The decision matrix:
    #
    #   --db not given                       in-memory, must index files
    #   --db PATH, no DB, no files           error
    #   --db PATH, no DB, files              create DB, index files
    #   --db PATH, DB exists, no files       reuse DB, no indexing
    #   --db PATH, DB exists, files          reopen DB, append files
    #   --db PATH, DB exists, files+rebuild  delete DB, recreate, index
    #   --db PATH, DB exists, no files+rebuild  error (rebuild needs sources)
    #
    # The metadata-compatibility check inside VectorStore.__init__ runs
    # automatically on every reopen, so a misconfigured combination
    # (different embedder, different chunk size) raises a friendly
    # VectorStoreError before we get to indexing or querying.
    # ------------------------------------------------------------------
    import os
    from . import __version__

    has_sources = bool(args.file or args.dir)
    db_exists = args.db is not None and os.path.exists(args.db)

    if args.db is not None and args.rebuild:
        if not has_sources:
            print(
                "Error: --rebuild requires -f/-d (rebuilding from no sources would empty the index)",
                file=sys.stderr,
            )
            return 1
        if db_exists:
            try:
                os.remove(args.db)
            except OSError as e:
                print(f"Error: could not remove existing --db {args.db}: {e}", file=sys.stderr)
                return 1
            db_exists = False

    if args.db is not None and not db_exists and not has_sources:
        print(
            f"Error: --db {args.db} does not exist and no -f/-d provided to populate it",
            file=sys.stderr,
        )
        return 1

    if args.db is None and not has_sources:
        print(
            "Error: no documents loaded. Provide files via -f or directories via -d, "
            "or pass --db PATH to reuse an existing index",
            file=sys.stderr,
        )
        return 1

    # Construct RAG with the appropriate db_path. The default is
    # ":memory:" (in-memory, ephemeral); --db PATH switches to a
    # file-backed store. Either way the same RAG class handles it.
    rag_kwargs = {
        "embedding_model": args.embedding_model,
        "generation_model": args.model,
        "n_gpu_layers": args.n_gpu_layers,
        "config": config,
    }
    if args.db is not None:
        rag_kwargs["db_path"] = args.db

    try:
        rag = RAG(**rag_kwargs)
    except Exception as e:
        # VectorStore.__init__ raises VectorStoreError on metadata
        # mismatch (different embedder, different metric, different
        # chunk config). The message is already user-friendly so we
        # just print and exit cleanly rather than dumping a traceback.
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load documents (if any) into the store. If --db is given and
    # the DB already exists with no -f/-d, we skip indexing and just
    # query what's on disk. RAG.add_documents/add_texts now hash each
    # source and skip ones already in the store, so re-running with
    # the same -f corpus.txt against an existing DB is a no-op
    # (everything is reported as "skipped" and the user goes straight
    # to query mode).
    n_added = 0
    n_skipped = 0
    if has_sources:
        if args.file:
            try:
                result = rag.add_documents(args.file)
            except ValueError as e:
                # Raised when a file basename matches an existing
                # source but the content differs. The message
                # already names the file and tells the user how to
                # proceed; we just print and exit cleanly.
                print(f"Error: {e}", file=sys.stderr)
                rag.close()
                return 1
            n_added += len(result)
            n_skipped += len(result.skipped_labels)
        if args.dir:
            from .rag import load_directory

            for path in args.dir:
                docs = load_directory(path, glob=args.glob)
                try:
                    result = rag.add_texts([d.text for d in docs])
                except ValueError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    rag.close()
                    return 1
                n_added += len(result)
                n_skipped += len(result.skipped_labels)

    model_name = os.path.basename(args.model)
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    left = f"cyllama v{__version__} rag"
    print(f"{left}{model_name:>{cols - len(left)}}")

    # Status line: tell the user whether we indexed fresh, appended,
    # reused, or skipped on dedup. Useful for confirming --db worked
    # and for spotting "I expected new files to be picked up but
    # nothing happened" cases.
    total_chunks = len(rag.store)
    skipped_suffix = f" ({n_skipped} unchanged)" if n_skipped > 0 else ""
    if args.db is None:
        # In-memory mode -- dedup still applies if the user passes
        # the same file twice in one invocation, but the typical
        # case is "fresh build, all sources are new".
        print(f"{n_added} chunks indexed{skipped_suffix}")
    elif n_added == 0 and n_skipped == 0:
        print(f"reusing {total_chunks} chunks from {args.db}")
    elif n_added == 0:
        # All sources were already indexed -- pure reuse with
        # explicit confirmation that the user's -f/-d arguments
        # were considered and skipped on dedup.
        print(f"reusing {total_chunks} chunks from {args.db}{skipped_suffix}")
    elif total_chunks == n_added:
        # Fresh DB or first index of these sources.
        print(f"{n_added} chunks indexed -> {args.db}{skipped_suffix}")
    else:
        # Append: some sources were new, some were already there
        # (or some were dedup-skipped on this run).
        previous = total_chunks - n_added
        print(f"{n_added} new chunks appended to {args.db} ({previous} existing, {total_chunks} total){skipped_suffix}")

    if args.prompt:
        # Single query mode
        try:
            if args.stream:
                for chunk in rag.stream(args.prompt, config):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = rag.query(args.prompt, config)
                print(response.text)
                if args.sources:
                    print("\n--- Sources ---")
                    for src in response.sources:
                        print(f"  [{src.score:.4f}] {src.text[:120]}...")
        except KeyboardInterrupt:
            print("\nInterrupted", file=sys.stderr)
            rag.close()
            return 130
    else:
        # Interactive mode. Enable readline-style line editing and
        # persistent history (up/down arrows cycle through prior
        # questions, left/right edit, Ctrl-R reverse-search, etc.).
        # Gracefully no-ops on platforms without readline.
        from ._readline import setup_history, history_path_for

        setup_history(history_path_for("rag"))

        try:
            while True:
                print("\033[32m> \033[0m", end="")
                try:
                    question = input().strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    continue
                print("\033[33m", end="")
                for chunk in rag.stream(question, config):
                    print(chunk, end="", flush=True)
                print("\033[0m")
                if args.sources:
                    results = rag.search(question, k=config.top_k, threshold=config.similarity_threshold)
                    if results:
                        print("--- Sources ---")
                        for src in results:
                            print(f"  [{src.score:.4f}] {src.text[:120]}...")
                # Blank line between turns for visual separation between
                # the answer and the next prompt.
                print()
        except KeyboardInterrupt:
            pass

    rag.close()
    return 0


def _delegate(module_path, import_name="main"):
    """Delegate to a sub-module's main(), stripping the subcommand from sys.argv."""
    sys.argv = ["cyllama " + sys.argv[1]] + sys.argv[2:]
    import importlib

    mod = importlib.import_module(module_path, package="cyllama")
    return getattr(mod, import_name)()


def main():
    parser = argparse.ArgumentParser(
        prog="cyllama",
        description="cyllama CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- info / version ---------------------------------------------------
    subparsers.add_parser("info", help="Show build and backend information")
    subparsers.add_parser("version", help="Show version")

    # -- generate (alias: gen) --------------------------------------------
    gen_parser = subparsers.add_parser("generate", aliases=["gen"], help="Generate text from a prompt")
    gen_parser.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    gen_parser.add_argument("-p", "--prompt", help="Text prompt")
    gen_parser.add_argument("-f", "--file", help="Read prompt from file")
    gen_parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    gen_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    gen_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    gen_parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    gen_parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    gen_parser.add_argument("--repeat-penalty", type=float, default=DEFAULT_REPEAT_PENALTY)
    gen_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    gen_parser.add_argument("-c", "--ctx-size", type=int, default=None)
    gen_parser.add_argument("--seed", type=int, default=LLAMA_DEFAULT_SEED)
    gen_parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    gen_parser.add_argument("--json", action="store_true", help="Output as JSON")
    gen_parser.add_argument("--verbose", action="store_true")

    # -- chat -------------------------------------------------------------
    chat_parser = subparsers.add_parser("chat", help="Chat with a model")
    chat_parser.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    chat_parser.add_argument("-p", "--prompt", help="Single-turn message (omit for interactive)")
    chat_parser.add_argument("-s", "--system", help="System prompt")
    chat_parser.add_argument("--template", help="Chat template (e.g. chatml, llama3)")
    chat_parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    chat_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    chat_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    chat_parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    chat_parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P)
    chat_parser.add_argument("--repeat-penalty", type=float, default=DEFAULT_REPEAT_PENALTY)
    chat_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    chat_parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    chat_parser.add_argument("--seed", type=int, default=LLAMA_DEFAULT_SEED)
    chat_parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    chat_parser.add_argument("--json", action="store_true", help="Output as JSON")
    chat_parser.add_argument("--verbose", action="store_true")

    # -- embed ------------------------------------------------------------
    embed_parser = subparsers.add_parser("embed", help="Compute text embeddings")
    embed_parser.add_argument("-m", "--model", required=True, help="Path to GGUF embedding model")
    embed_parser.add_argument("-t", "--text", action="append", help="Text to embed (repeatable)")
    embed_parser.add_argument("-f", "--file", help="Read texts from file (one per line)")
    embed_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    embed_parser.add_argument("-c", "--ctx-size", type=int, default=512)
    embed_parser.add_argument(
        "--pooling", default="mean", choices=["mean", "cls", "last"], help="Pooling strategy (default: mean)"
    )
    embed_parser.add_argument("--no-normalize", action="store_true", help="Skip L2 normalization of embeddings")
    embed_parser.add_argument("--dim", action="store_true", help="Print embedding dimensions and exit")
    embed_parser.add_argument("--similarity", metavar="QUERY", help="Rank texts by similarity to QUERY")
    embed_parser.add_argument(
        "--threshold", type=float, default=0.0, help="Minimum similarity score to display (default: 0.0)"
    )

    # -- rag --------------------------------------------------------------
    rag_parser = subparsers.add_parser("rag", help="Query documents with RAG")
    rag_parser.add_argument("-m", "--model", required=True, help="Path to GGUF generation model")
    rag_parser.add_argument("-e", "--embedding-model", required=True, help="Path to GGUF embedding model")
    rag_parser.add_argument("-f", "--file", action="append", help="File to index (repeatable)")
    rag_parser.add_argument("-d", "--dir", action="append", help="Directory to index (repeatable)")
    rag_parser.add_argument("--glob", default="**/*", help="Glob pattern for directory loading (default: **/*)")
    rag_parser.add_argument("-p", "--prompt", help="Single query (omit for interactive)")
    rag_parser.add_argument("-s", "--system", help="System instruction (e.g. 'Answer in one paragraph')")
    rag_parser.add_argument("-n", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    rag_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    rag_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve")
    rag_parser.add_argument("--threshold", type=float, default=None, help="Minimum similarity threshold")
    rag_parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    rag_parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    rag_parser.add_argument("--sources", action="store_true", help="Show source chunks")
    rag_parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=(
            "Path to a SQLite vector store file. With --db, the index is "
            "persisted to disk and reopened on subsequent runs, so the "
            "corpus is embedded only once. If PATH does not exist it will "
            "be created. If PATH exists, the embedding model and chunking "
            "config must match the original (mismatch raises a clear "
            "error). Without --db, the index is held in-memory and "
            "rebuilt on every run (current default behavior). When --db "
            "is set and no -f/-d is given, the existing index is queried "
            "as-is without re-indexing."
        ),
    )
    rag_parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete the database at --db and recreate it from the "
            "current -f/-d sources. Use after switching embedding models "
            "or changing chunking config. Has no effect without --db."
        ),
    )
    rag_parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help=(
            "Use raw-completion prompting instead of the model's native "
            "chat template. The chat template is on by default because "
            "nearly all GGUF models used with `cyllama rag` are chat-tuned, "
            "and the raw-completion path causes them to leak instruction-"
            "tuning artifacts (Qwen3 <think> blocks, paragraph paraphrase "
            "loops, model re-roleplaying as the user). Pass this flag for "
            "base/completion models that need the legacy path."
        ),
    )
    rag_parser.add_argument(
        "--show-think",
        action="store_true",
        help=(
            "Show <think>...</think> reasoning blocks emitted by Qwen3, "
            "DeepSeek-R1, and other reasoning-tuned models. By default "
            "these blocks are stripped from the streamed output because "
            "they typically consume the entire --max-tokens budget on "
            "small budgets, leaving no room for the actual answer. Pass "
            "this flag for debugging or when you want the reasoning "
            "visible in the transcript."
        ),
    )
    rag_parser.add_argument(
        "--repetition-threshold",
        type=int,
        default=2,
        # Default of 2 means the detector fires on the FIRST repeat of a
        # 5-gram, not the second. The whole point of a loop guard is to
        # cut the loop as early as possible -- waiting for a third repeat
        # wastes tokens on content the user doesn't want. ngram=5 makes
        # exact 5-word phrase repeats themselves rare in non-loopy text,
        # so the false-positive risk is low.
        help="Stop generation after the same n-gram repeats this many times in the rolling window. 0 disables (default: 2)",
    )
    rag_parser.add_argument(
        "--repetition-ngram",
        type=int,
        default=5,
        help="Word-level n-gram length for repetition detection (default: 5)",
    )
    rag_parser.add_argument(
        "--repetition-window",
        type=int,
        default=300,
        help="Number of recent words kept by the repetition detector (default: 300, tuned for paragraph-length loops)",
    )

    # -- delegation commands ----------------------------------------------
    subparsers.add_parser("server", help="Start OpenAI-compatible API server")
    subparsers.add_parser("transcribe", help="Speech-to-text transcription")
    subparsers.add_parser("tts", help="Text-to-speech synthesis")
    subparsers.add_parser("sd", help="Stable Diffusion image generation")
    subparsers.add_parser("agent", help="Run agents")
    subparsers.add_parser("memory", help="Estimate GPU memory requirements")

    # Parse only the known args so delegation commands can pass through
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        cmd_info()
    elif args.command == "version":
        cmd_version()
    elif args.command in ("generate", "gen"):
        return cmd_generate(args)
    elif args.command == "chat":
        return cmd_chat(args)
    elif args.command == "embed":
        return cmd_embed(args)
    elif args.command == "rag":
        return cmd_rag(args)
    elif args.command == "server":
        return _delegate(".llama.server.__main__")
    elif args.command == "transcribe":
        return _delegate(".whisper.cli")
    elif args.command == "tts":
        return _delegate(".llama.tts")
    elif args.command == "sd":
        return _delegate(".sd.__main__")
    elif args.command == "agent":
        return _delegate(".agents.cli")
    elif args.command == "memory":
        return _delegate(".memory")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
