"""RAG Pipeline for combining retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from .repetition import NGramRepetitionDetector, ThinkBlockStripper
from .types import SearchResult

if TYPE_CHECKING:
    from ..api import LLM
    from .embedder import Embedder
    from .store import VectorStore


# Default prompt template for RAG queries (raw-completion path).
DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""


# Default system prompt for the chat-template path. This replaces the
# Question:/Answer: framing that some chat-tuned models (notably Qwen3)
# misinterpret as a continuation pattern and loop on.
#
# The trailing "/no_think" directive is a Qwen3-specific control token
# that disables its chain-of-thought mode for the turn (see Qwen3 model
# card). It's a literal text directive, so other models that don't
# recognise it just see it as part of the system prompt and ignore it --
# harmless for Llama, Mistral, Gemma, etc. The natural-language
# instruction immediately before it asks for the same behaviour in
# model-agnostic terms; together they save the small ``max_tokens``
# budget from being eaten by reasoning blocks the user never sees.
DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the "
    "provided context. If the context does not contain the information "
    "needed, say so plainly. Give your answer once and do not repeat or "
    "paraphrase it. Answer directly without showing any chain-of-thought "
    "or reasoning steps. /no_think"
)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline.

    Attributes:
        top_k: Number of documents to retrieve (default: 5)
        similarity_threshold: Minimum similarity score for retrieval (default: None)
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Generation temperature (default: 0.7)
        prompt_template: Template for formatting the RAG prompt
        context_separator: String to join retrieved documents (default: "\\n\\n")
        include_metadata: Whether to include metadata in context (default: False)
        repetition_window: Word-level rolling-window size for the streaming
            n-gram repetition detector. Default: 80.
        repetition_ngram: N-gram length used by the repetition detector.
            Default: 5.
        repetition_threshold: Repeat count that trips the detector and
            stops generation early. ``0`` disables the detector entirely
            (default). Set to e.g. ``3`` to enable.
        use_chat_template: When True, the pipeline calls
            ``generator.chat()`` with a system + user message instead of
            sending a raw completion prompt. Default: False. Use this for
            chat-tuned models that loop on the raw "Question:/Answer:"
            framing.
        system_prompt: System message used when ``use_chat_template`` is
            True. Defaults to a prompt that explicitly tells the model
            not to repeat or paraphrase its answer.
        strip_think_blocks: When True, ``<think>...</think>`` reasoning
            blocks emitted by Qwen3, DeepSeek-R1, and other reasoning-
            tuned models are removed from the streamed output before it
            reaches the caller. Default: True. Set to False if you want
            the reasoning visible (e.g. for debugging or transcript
            capture). The strip is implemented by
            :class:`cyllama.rag.repetition.ThinkBlockStripper`.
    """

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float | None = None

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7

    # Prompt template (raw-completion path)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    # Context formatting
    context_separator: str = "\n\n"
    include_metadata: bool = False

    # Repetition detection (streaming-level loop guard).
    # Defaults to off so the bare RAGConfig() preserves the historical
    # behaviour; opt in by setting repetition_threshold > 0. The CLI
    # turns this on by default because that's where the bug was hit.
    # Window size is tuned for paragraph-length loops (Qwen3-4B greedy
    # decoding), not just short phrase loops.
    repetition_window: int = 300
    repetition_ngram: int = 5
    repetition_threshold: int = 0

    # Chat-template prompting (alternative to raw completion). Off by
    # default for the same backwards-compat reason.
    use_chat_template: bool = False
    system_prompt: str | None = None

    # Reasoning-block stripping. Defaults OFF for the same backwards-
    # compat reason as the other streaming-path features: enabling it
    # forces the pipeline through the streaming code path (since the
    # strip is a per-chunk filter), which loses the rich
    # GenerationStats that the legacy non-streaming path returns. The
    # CLI flips this to True because reasoning-tuned models (Qwen3,
    # DeepSeek-R1) routinely consume the entire `max_tokens` budget on
    # their `<think>` block, leaving no room for the actual answer --
    # so the CLI surfaces, where users hit this bug, opt in by default.
    strip_think_blocks: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.similarity_threshold is not None and not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.repetition_threshold < 0:
            raise ValueError(f"repetition_threshold must be >= 0 (0 = disabled), got {self.repetition_threshold}")
        if self.repetition_threshold > 0:
            # Only validate the other repetition fields when the detector
            # is actually enabled, so a config that leaves them at zero
            # while disabling the feature is still legal.
            if self.repetition_ngram < 2:
                raise ValueError(
                    f"repetition_ngram must be >= 2 when repetition is enabled, got {self.repetition_ngram}"
                )
            if self.repetition_window < self.repetition_ngram:
                raise ValueError(
                    f"repetition_window ({self.repetition_window}) must be "
                    f">= repetition_ngram ({self.repetition_ngram})"
                )


@dataclass
class RAGResponse:
    """Response from a RAG query.

    Attributes:
        text: Generated response text
        sources: Retrieved documents used as context
        stats: Optional generation statistics
        query: Original query string
    """

    text: str
    sources: list[SearchResult]
    stats: Any | None = None  # GenerationStats when available
    query: str = ""

    def __str__(self) -> str:
        """Return the response text."""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "text": self.text,
            "query": self.query,
            "sources": [
                {
                    "id": s.id,
                    "text": s.text,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in self.sources
            ],
        }
        if self.stats is not None:
            result["stats"] = {  # type: ignore[assignment]
                "prompt_tokens": self.stats.prompt_tokens,
                "generated_tokens": self.stats.generated_tokens,
                "total_time": self.stats.total_time,
                "tokens_per_second": self.stats.tokens_per_second,
            }
        return result


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation.

    The RAGPipeline orchestrates the retrieval-augmented generation process:
    1. Embed the user's question
    2. Retrieve relevant documents from the vector store
    3. Format a prompt with the retrieved context
    4. Generate a response using the LLM

    Example:
        >>> from cyllama import LLM
        >>> from cyllama.rag import Embedder, VectorStore, RAGPipeline
        >>>
        >>> embedder = Embedder("models/bge-small.gguf")
        >>> store = VectorStore(dimension=embedder.dimension)
        >>> llm = LLM("models/llama.gguf")
        >>>
        >>> # Add some documents
        >>> docs = ["Python is a programming language.", "The sky is blue."]
        >>> embeddings = embedder.embed_batch(docs)
        >>> store.add(embeddings, docs)
        >>>
        >>> # Query
        >>> pipeline = RAGPipeline(embedder, store, llm)
        >>> response = pipeline.query("What is Python?")
        >>> print(response.text)
    """

    def __init__(
        self,
        embedder: "Embedder",
        store: "VectorStore",
        generator: "LLM",
        config: RAGConfig | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            embedder: Embedder for converting queries to vectors
            store: VectorStore for similarity search
            generator: LLM for generating responses
            config: RAG configuration (uses defaults if None)
        """
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.config = config or RAGConfig()
        # Tri-state cache for the chat template's system-role support.
        # `None` means we haven't probed yet; `True` means the canonical
        # `[system, user]` message shape works; `False` means it raised
        # a template-application error and we have to merge the system
        # content into the first user message before sending. Several
        # models distributed as GGUF (Gemma 2/3, some Mistral variants,
        # older instruct templates) reject a `system` role outright,
        # while Llama-3, Qwen, newer Mistral, etc. accept it. We probe
        # lazily on the first chat-template call rather than at __init__
        # time to keep this RAGPipeline class testable without a real
        # LLM, and we cache so subsequent queries don't pay the
        # try/except cost.
        self._system_role_supported: bool | None = None
        # Set True when both the canonical [system, user] AND the merged
        # [user] chat-template attempts have failed -- i.e. the model's
        # embedded chat template can't be applied at all by llama.cpp's
        # basic C API (cyllama doesn't currently link `common_chat.cpp`
        # which has full Jinja support, so any GGUF whose embedded Jinja
        # template doesn't match one of the hardcoded substring
        # heuristics in `llm_chat_detect_template` returns -1). When set,
        # `_generate_chunks` permanently routes through the raw-completion
        # path for this pipeline instance, regardless of
        # `cfg.use_chat_template`.
        self._chat_template_unusable: bool = False

    def query(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> RAGResponse:
        """Answer a question using RAG.

        Steps:
        1. Embed the question
        2. Retrieve relevant documents
        3. Format prompt (or chat messages) with context
        4. Generate response, optionally with streaming-level repetition
           detection or via the model's chat template

        Args:
            question: The question to answer
            config: Optional config override for this query

        Returns:
            RAGResponse with generated text and sources
        """
        cfg = config or self.config

        # 1. Embed the question
        query_embedding = self.embedder.embed(question)

        # 2. Retrieve relevant documents
        sources = self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

        gen_config = self._build_gen_config(cfg)

        # When none of the streaming-path features are enabled we keep
        # the legacy fast path that calls the generator non-streaming
        # and preserves the rich GenerationStats that come back on the
        # Response object. The streaming path used by the new features
        # cannot recover those stats from a chunk iterator.
        if cfg.repetition_threshold > 0 or cfg.use_chat_template or cfg.strip_think_blocks:
            chunks = list(self._generate_chunks(question, sources, cfg, gen_config))
            text = "".join(chunks)
            stats = None
        else:
            prompt = self._format_prompt(question, sources, cfg)
            response = self.generator(prompt, config=gen_config)
            text = str(response)
            stats = getattr(response, "stats", None)

        return RAGResponse(
            text=text,
            sources=sources,
            stats=stats,
            query=question,
        )

    def stream(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> Iterator[str]:
        """Stream response tokens for a question.

        Yields tokens as they are generated, useful for real-time display.
        Honours ``RAGConfig.repetition_threshold`` and
        ``RAGConfig.use_chat_template``.

        Args:
            question: The question to answer
            config: Optional config override for this query

        Yields:
            Response tokens as strings
        """
        cfg = config or self.config

        # 1. Embed the question
        query_embedding = self.embedder.embed(question)

        # 2. Retrieve relevant documents
        sources = self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

        gen_config = self._build_gen_config(cfg)
        yield from self._generate_chunks(question, sources, cfg, gen_config)

    def _build_gen_config(self, cfg: RAGConfig, *, force_completion: bool = False) -> Any:
        """Construct the underlying GenerationConfig from a RAGConfig.

        ``force_completion`` overrides ``cfg.use_chat_template`` so the
        chat-template -> raw-completion fallback path can rebuild the
        config with the right stop sequences without mutating the
        caller's RAGConfig.
        """
        from ..api import GenerationConfig

        # The Question:/Context:/Answer: stop sequences only make sense
        # for the raw-completion prompt template; they would otherwise
        # match a user question that happens to mention "Question:".
        completion_path = force_completion or not cfg.use_chat_template
        stop_sequences = ["Question:", "\nContext:", "\nAnswer:"] if completion_path else []
        return GenerationConfig(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop_sequences=stop_sequences,
        )

    def _build_chat_messages(
        self,
        question: str,
        sources: list[SearchResult],
        cfg: RAGConfig,
    ) -> list[dict[str, str]]:
        """Build chat messages for the chat-template generation path."""
        context = self._format_context(sources, cfg)
        system = cfg.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _merge_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return a copy of ``messages`` with any system content prepended
        to the first user message.

        Several chat templates (Gemma 2/3, some older Mistral instruct
        templates, a handful of community fine-tunes) reject a ``system``
        role outright -- they only know about ``user`` and ``assistant``.
        For those models we have to fold the system instructions into
        the first user message so the same downstream code path works.

        The merge keeps the original list intact (we copy) so callers
        that hold a reference to the canonical message list don't see
        it mutated under them.
        """
        system_chunks: list[str] = []
        rest: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system_chunks.append(m.get("content", ""))
            else:
                rest.append(dict(m))
        if not system_chunks or not rest:
            return rest or [dict(m) for m in messages]
        # Find the first user message and prepend the system content.
        for m in rest:
            if m.get("role") == "user":
                m["content"] = "\n\n".join(system_chunks + [m.get("content", "")])
                break
        return rest

    def _chat_with_fallback(
        self,
        messages: list[dict[str, str]],
        gen_config: Any,
    ) -> Iterator[str]:
        """Call ``generator.chat()`` with feature detection for the
        ``system`` role.

        On the first call we try the canonical ``[system, user]`` shape.
        If the chat template rejects it (some models -- notably Gemma --
        only support ``user``/``assistant``), we cache the failure on
        ``self._system_role_supported`` and fall back to merging the
        system content into the first user message. Subsequent calls
        skip straight to the merged shape so we don't pay the failed-
        attempt cost on every query.

        Any RuntimeError that does *not* look like a chat-template
        rejection is re-raised unchanged so genuine errors aren't
        masked.
        """
        # Already known to fail with system role -> merge up front.
        if self._system_role_supported is False:
            return self.generator.chat(
                self._merge_system_into_user(messages),
                config=gen_config,
                stream=True,
            )

        try:
            token_iter = self.generator.chat(messages, config=gen_config, stream=True)
        except RuntimeError as e:
            # llama.cpp's chat_apply_template raises with this exact
            # message; tolerate variations on "template" so wording
            # changes upstream don't silently break the fallback.
            if "template" not in str(e).lower():
                raise
            self._system_role_supported = False
            return self.generator.chat(
                self._merge_system_into_user(messages),
                config=gen_config,
                stream=True,
            )

        # First successful canonical call -> remember so future calls
        # don't have to retry the try/except.
        if self._system_role_supported is None:
            self._system_role_supported = True
        return token_iter

    def _generate_chunks(
        self,
        question: str,
        sources: list[SearchResult],
        cfg: RAGConfig,
        gen_config: Any,
    ) -> Iterator[str]:
        """Yield generated chunks from the chosen path, with optional
        streaming-level repetition detection and ``<think>`` block stripping.

        Both ``query()`` and ``stream()`` go through this helper so the
        chat-template branch, the think-block stripper, and the loop
        guard live in exactly one place.

        Filter order: model -> think-block strip -> repetition detect ->
        caller. The strip runs *before* the detector so the rolling
        window only sees user-visible text -- otherwise the detector
        would waste capacity on reasoning the user never reads, and
        could false-positive on phrases the model legitimately repeats
        between its think block and its answer.
        """
        # Three-tier path selection:
        #   1. Caller asked for chat template AND we haven't given up on
        #      it yet -> try the chat path (which itself has a system
        #      vs merged-user fallback inside `_chat_with_fallback`).
        #   2. Caller asked for chat template but we already discovered
        #      it's unusable for this model -> raw-completion fallback.
        #   3. Caller asked for raw completion outright -> raw-completion.
        # The chat path can also degrade INTO the raw-completion path
        # mid-call if both chat shapes raise a template error -- see
        # the except branch below.
        use_chat = cfg.use_chat_template and not self._chat_template_unusable
        token_iter: Iterator[str]
        if use_chat:
            messages = self._build_chat_messages(question, sources, cfg)
            try:
                token_iter = self._chat_with_fallback(messages, gen_config)
            except RuntimeError as e:
                # Both [system, user] and [merged user] chat shapes
                # failed with a template error. The model's embedded
                # chat template can't be applied by llama.cpp's basic
                # C API. Permanently downgrade this pipeline to the
                # raw-completion path so subsequent queries skip the
                # chat attempt entirely, and warn the user once so the
                # silent quality degradation isn't invisible.
                if "template" not in str(e).lower():
                    raise
                self._chat_template_unusable = True
                import warnings

                warnings.warn(
                    "Chat template could not be applied to this model "
                    "(both canonical [system, user] and merged-user "
                    "shapes failed). Falling back to raw-completion "
                    "prompting for the rest of this RAGPipeline "
                    "instance. Answer quality may be slightly worse "
                    "than the model's native chat format would give. "
                    "Pass --no-chat-template (or "
                    "RAGConfig(use_chat_template=False)) to suppress "
                    "this warning and use the completion path "
                    "directly.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Rebuild gen_config with completion-path stop
                # sequences -- the original was built without them
                # because cfg.use_chat_template was True.
                gen_config = self._build_gen_config(cfg, force_completion=True)
                prompt = self._format_prompt(question, sources, cfg)
                token_iter = self.generator(prompt, config=gen_config, stream=True)
        elif cfg.use_chat_template and self._chat_template_unusable:
            # Cached downgrade: rebuild gen_config for the completion
            # path on every call until the user explicitly opts out.
            gen_config = self._build_gen_config(cfg, force_completion=True)
            prompt = self._format_prompt(question, sources, cfg)
            token_iter = self.generator(prompt, config=gen_config, stream=True)
        else:
            prompt = self._format_prompt(question, sources, cfg)
            token_iter = self.generator(prompt, config=gen_config, stream=True)

        # Outermost filter: strip <think>...</think> blocks if enabled.
        if cfg.strip_think_blocks:
            token_iter = self._strip_think_blocks(token_iter)

        if cfg.repetition_threshold > 0:
            detector = NGramRepetitionDetector(
                window=cfg.repetition_window,
                ngram=cfg.repetition_ngram,
                threshold=cfg.repetition_threshold,
            )
            # Yield-then-feed by design: the chunk that completes the
            # repeating n-gram is allowed through to the caller before
            # we exit. This leaves a tiny dangling fragment (typically a
            # single word -- the start of the loop's next iteration) at
            # the end of the output, which acts as a visible "the guard
            # fired" tell so users can distinguish a loop-cut from a
            # max_tokens-cut. We tried feed-then-yield to suppress the
            # fragment but the abrupt cutoff made coherent answers look
            # truncated mid-sentence -- the dangling fragment is the
            # better UX trade.
            for chunk in token_iter:
                yield chunk
                if detector.feed(chunk):
                    return
        else:
            yield from token_iter

    @staticmethod
    def _strip_think_blocks(token_iter: Iterator[str]) -> Iterator[str]:
        """Wrap a token iterator with a ``ThinkBlockStripper`` so the
        downstream consumer never sees reasoning content."""
        stripper = ThinkBlockStripper()
        for chunk in token_iter:
            for cleaned in stripper.feed(chunk):
                if cleaned:
                    yield cleaned
        for cleaned in stripper.flush():
            if cleaned:
                yield cleaned

    def retrieve(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents without generation.

        Useful for debugging or when you only need retrieval.

        Args:
            question: The question to retrieve documents for
            config: Optional config override

        Returns:
            List of relevant SearchResults
        """
        cfg = config or self.config
        query_embedding = self.embedder.embed(question)
        return self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

    def _format_context(
        self,
        sources: list[SearchResult],
        config: RAGConfig,
    ) -> str:
        """Join retrieved sources into a single context string.

        Used by both the raw-completion prompt template and the
        chat-template message builder.
        """
        context_parts = []
        for source in sources:
            if config.include_metadata and source.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in source.metadata.items())
                context_parts.append(f"[{meta_str}]\n{source.text}")
            else:
                context_parts.append(source.text)
        return config.context_separator.join(context_parts)

    def _format_prompt(
        self,
        question: str,
        sources: list[SearchResult],
        config: RAGConfig,
    ) -> str:
        """Format the RAG prompt with retrieved context.

        Args:
            question: User's question
            sources: Retrieved documents
            config: Configuration for formatting

        Returns:
            Formatted prompt string
        """
        context = self._format_context(sources, config)
        return config.prompt_template.format(
            context=context,
            question=question,
        )

    def __repr__(self) -> str:
        return f"RAGPipeline(embedder={self.embedder!r}, store={self.store!r}, config={self.config!r})"
