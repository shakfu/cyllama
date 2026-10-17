"""Decision models: typed answers in one forward pass, no generation.

Port of llama.cpp's ``/v1/systemone`` (tools/server/server-decision.cpp,
TypeSafe System One API) to the public C API. A request has a ``state`` and
``questions``; each question is a ``choice``, ``score`` or ``noul`` (yes/no
probability). The model type comes from the ``<arch>.decision.type`` metadata.

Supported, each checked against upstream llama-server:

- ``laya`` (also Julia-1): one embedding row per option, at its marker token.
- ``lev``: logits of one label token per option at the last prompt token; a
  choice is evaluated with its options in both orders.
- ``kev``: scaled dot product of the hidden states of the last token and of
  the end token of each option.
- ``pplx-decider``, ``vllm-d3``: lev's label path, one option order.
- ``lfm2-d1``: logits of the label tokens of each option, the maximum per option.
- ``lfm2-d1-omni``: laya's embedding rows, prompt pieces cut to token budgets.

Text only: a request with images or audio is refused. openjev and nimble use
lev's label path but are untested; clef and vllm-decision2 need
``llama_batch_ext_set_decision_order``, which is not in ``llama.h``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from typing import Any, Optional, Tuple, Union

from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

from .llama_cpp import LlamaBatchExt, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams

DECISION_TYPES = (
    "openjev",
    "lev",
    "kev",
    "nimble",
    "laya",
    "clef",
    "pplx-decider",
    "vllm-decision2",
    "vllm-d3",
    "lfm2-d1",
    "lfm2-d1-omni",
)
SUPPORTED_TYPES = ("laya", "lev", "kev", "pplx-decider", "vllm-d3", "lfm2-d1", "lfm2-d1-omni")
_LABEL_TYPES = ("lev", "pplx-decider", "vllm-d3")  # one label token per option, A..Z then AA..ZZ
_EMBD_TYPES = ("laya", "kev", "lfm2-d1-omni")  # read the embeddings output
_NULL_STATE_TYPES = ("lfm2-d1", "lfm2-d1-omni", "vllm-d3")  # e.g. to ask about images only

_QUESTION_TYPES = ("choice", "score", "noul")  # index is the laya output column
_LAYA_MAX_OPTION_TOKENS = 48
_MAX_OPTIONS = 255
_LEV_N_RATINGS = 9  # lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# lfm2-d1-omni: text between the pieces of the prompt, and at the start of the pieces cut to a budget
_D1OMNI_MARKER = "<<d1omni:"
_D1OMNI_SEP = "<<d1omni:sep>>"
_D1OMNI_MARK_STATE = "<<d1omni:state>>"
_D1OMNI_MARK_QUESTION = "<<d1omni:question>>"
_D1OMNI_MARK_OPTION = "<<d1omni:option>>"
_D1OMNI_MAX_TOKENS = 16384


def get_decision_type(model: LlamaModel) -> Optional[str]:
    """Return the decision type of ``model``, ``None`` if it is not a decision model.

    An unrecognised type is returned as ``"unknown"``.
    """
    try:
        arch = model.meta_val_str("general.architecture")
        name = model.meta_val_str(f"{arch}.decision.type")
    except ValueError:
        return None
    return name if name in DECISION_TYPES else "unknown"


def _replace_text(val: Any, search: str, replace: str) -> Any:
    if isinstance(val, str):
        return val.replace(search, replace)
    if isinstance(val, list):
        return [_replace_text(v, search, replace) for v in val]
    if isinstance(val, dict):
        return {k: _replace_text(v, search, replace) for k, v in val.items()}
    return val


def _sort_keys(val: Any) -> Any:
    if isinstance(val, list):
        return [_sort_keys(v) for v in val]
    if isinstance(val, dict):
        return {k: _sort_keys(val[k]) for k in sorted(val)}
    return val


def _kev_render(val: Any, indent: int = 0) -> str:
    # flattens JSON into text, object keys kept as labels (kev/api.py: render)
    pad = "  " * indent
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, list):
        lines = []
        for item in val:
            lines.append(pad + "- " + _kev_render(item, indent + 1).lstrip(" \t\n\r"))
        return "\n".join(lines)
    if isinstance(val, dict):
        lines = []
        for key, item in val.items():
            nested = isinstance(item, (dict, list))
            lines.append(pad + key + (":\n" if nested else ": ") + _kev_render(item, indent + 1 if nested else 0))
        return "\n".join(lines)
    return json.dumps(val)


def _escape_special(text: str) -> str:
    # write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
    return _SPECIAL.sub("<¦\\1¦>", text)


def _kev_text(val: Any) -> str:
    return _escape_special(_kev_render(val))


def _d1omni_escape(val: Any) -> Any:
    # keys too (d1-omni prompt.py: escape)
    if isinstance(val, str):
        return _escape_special(val)
    if isinstance(val, list):
        return [_d1omni_escape(v) for v in val]
    if isinstance(val, dict):
        return {_escape_special(k): _d1omni_escape(v) for k, v in val.items()}
    return val


def _check_media(body: dict[str, Any]) -> None:
    """Refuse image, audio and video input, which upstream reads with mtmd."""
    videos = body.get("videos")
    if videos is not None and not (isinstance(videos, (list, dict)) and not videos):
        raise ValueError('"videos" is not supported')
    for key in ("files", "images"):
        files = body.get(key)
        if files is not None and not isinstance(files, list):
            raise ValueError(f'"{key}" must be an array')
        if files:
            raise NotImplementedError("image and audio input are not supported for decisions")
    state = body.get("state")
    messages = state.get("messages") if isinstance(state, dict) else state
    if not isinstance(messages, list):
        return
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_audio") and part["type"] in part:
                raise NotImplementedError("image and audio input are not supported for decisions")


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    sort_keys: bool = False,
) -> str:
    # same defaults as llama.cpp's jinja engine: no key sorting, no ASCII escaping
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _confidence_choice(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    uniform = 1.0 / len(probs)
    return max(0.0, (max(probs) - uniform) / (1.0 - uniform))


def _confidence_entropy(probs: list[float]) -> float:
    # vllm-d3: normalized entropy, as in the reference implementation
    if len(probs) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probs))))


def _confidence_score(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    n = len(probs)
    mode = probs.index(max(probs))
    dist = sum(p * abs(i - mode) for i, p in enumerate(probs))
    dist_uniform = sum(abs(i - (n - 1) / 2.0) / n for i in range(n))
    return max(0.0, 1.0 - dist / dist_uniform)


class DecisionModel:
    """Answers System One requests with a decision model.

    Example:
        >>> dm = DecisionModel("models/Laya-Q8_0.gguf")
        >>> dm.answer({"state": "I was charged twice.", "questions": {
        ...     "route": {"type": "choice", "instructions": "Which team?",
        ...               "criteria": {"billing": None, "shipping": None}}}})
    """

    def __init__(
        self,
        model: Union[str, LlamaModel],
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        cache_prompt: bool = True,
    ):
        """
        Args:
            model: GGUF path or a loaded model.
            n_ctx: Context size. The prompt of a question is evaluated in one
                batch, so it must fit.
            n_gpu_layers: Layers to offload when ``model`` is a path.
            n_threads: CPU threads; llama.cpp's default when ``None``.
            cache_prompt: As llama-server, a causal model keeps the KV cache of
                the prefix shared with the previous prompt, so answers depend on
                earlier questions by float rounding. ``False`` evaluates every
                prompt from the start: slower, and repeatable.

        Raises:
            ValueError: ``model`` is not a decision model, or lacks required metadata.
            NotImplementedError: the decision type is not supported yet.
        """
        if isinstance(model, str):
            mparams = LlamaModelParams()
            mparams.n_gpu_layers = n_gpu_layers
            model = LlamaModel(model, mparams, verbose=False)
        self.model = model
        self.type = get_decision_type(model)
        if self.type is None:
            raise ValueError("not a decision model (no <arch>.decision.type metadata)")
        if self.type not in SUPPORTED_TYPES:
            raise NotImplementedError(f"decision type {self.type!r} is not supported; supported: {SUPPORTED_TYPES}")

        self.vocab = model.get_vocab()
        arch = model.meta_val_str("general.architecture")
        prefix = f"{arch}.decision."

        try:
            source = model.get_default_chat_template_by_name("systemone")
        except Exception:
            source = None
        if not source:
            raise ValueError('decision model has no "systemone" template')
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        self._template = env.from_string(source)

        self.temperatures: dict[str, float] = {}
        self.score_bias: dict[int, list[float]] = {}  # number of levels -> offset of each level
        for i in range(model.meta_count()):
            key = model.meta_key_by_index(i)
            if key.startswith(prefix + "temperature."):
                temp = float(model.meta_val_str_by_index(i))
                if temp <= 0.0:
                    raise ValueError(f"invalid decision temperature: {key} = {temp}")
                self.temperatures[key[len(prefix + "temperature.") :]] = temp
            elif key.startswith(prefix + "score_bias."):
                parts = key[len(prefix + "score_bias.") :].split(".")
                n, level = (
                    (int(p) for p in parts)
                    if len(parts) == 2 and all(p.isascii() and p.isdigit() for p in parts)
                    else (0, 0)
                )
                if not 2 <= n <= 255 or level >= n:
                    raise ValueError(f"invalid decision score bias: {key}")
                self.score_bias.setdefault(n, [0.0] * n)[level] = float(model.meta_val_str_by_index(i))

        self.labels: list[int] = []  # lev: one label token per output
        self.label_texts: list[str] = []
        self.text_marker = ""
        self.n_options_max = _MAX_OPTIONS
        if self.type in _LABEL_TYPES:
            # label codes are A..Z then AA..ZZ, only the ones that are a single token are used
            codes = [chr(a) for a in range(65, 91)] + [chr(a) + chr(b) for a in range(65, 91) for b in range(65, 91)]
            for code in codes:
                toks = self.vocab.tokenize(code, add_special=False, parse_special=False)
                if len(toks) == 1 and len(self.labels) < _MAX_OPTIONS:
                    self.labels.append(toks[0])
                    self.label_texts.append(code)
            self.n_options_max = len(self.labels)
        elif self.type == "kev":
            # the hidden state of an option is read at the token that ends it
            toks = self.vocab.tokenize("<|box_end|>", add_special=False, parse_special=True)
            if len(toks) != 1:
                raise ValueError("decision model has no <|box_end|> token")
            self.token_marker = toks[0]
        elif self.type == "laya":
            self.token_marker = self.vocab.token_mask()
            self.token_sep = self.vocab.token_sep()
            if self.token_marker < 0 or self.token_sep < 0:
                raise ValueError("decision model has no mask or sep token")
            self.text_marker = self.vocab.token_to_piece(self.token_marker, 0, True)
            try:
                self.max_head_tokens = int(model.meta_val_str(prefix + "max_head_tokens"))
            except ValueError:
                self.max_head_tokens = 0
            if self.max_head_tokens <= 0:
                raise ValueError("decision model has no valid max_head_tokens")
        elif self.type == "lfm2-d1-omni":
            self.token_marker = self.vocab.token_mask()
            if self.token_marker < 0:
                raise ValueError("decision model has no mask token")

        cparams = LlamaContextParams()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_ctx
        if self.type in _EMBD_TYPES:
            # read from the embeddings output: every prompt token is an output, in one ubatch
            cparams.n_ubatch = n_ctx
            cparams.embeddings = True
            cparams.pooling_type = 0  # LLAMA_POOLING_TYPE_NONE
            cparams.n_outputs_max = n_ctx
            cparams.n_outputs_max_per_seq = 1
        if n_threads is not None:
            cparams.n_threads = n_threads
            cparams.n_threads_batch = n_threads
        cparams.swa_full = False  # llama-server's default, it decides which prefixes the memory keeps
        self.ctx = LlamaContext(model, cparams, verbose=False)
        self.n_batch = n_ctx
        self.cache_prompt = cache_prompt
        self._cached: list[int] = []  # the prompt in the KV cache
        self._lock = threading.Lock()  # one context, so one request at a time

    # -- request parsing -----------------------------------------------------

    def _parse_questions(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        if "state" not in body or (body["state"] is None and self.type not in _NULL_STATE_TYPES):
            raise ValueError('"state" must be provided')
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError('"questions" must be a non-empty object')

        parsed = []
        for qid, q in questions.items():

            def err(msg: str, qid: str = qid) -> ValueError:
                return ValueError(f"questions.{qid}: {msg}")

            if not isinstance(q, dict):
                raise err("must be an object")
            if q.get("instructions") is None and self.type != "vllm-d3":  # its prompt has a default question
                raise err('"instructions" must be provided')
            qtype = q.get("type")
            criteria = q.get("criteria")
            if qtype == "choice":
                if not isinstance(criteria, dict) or not criteria:
                    raise err('"criteria" must be a non-empty object')
                options = list(criteria.items())
            elif qtype == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise err('"criteria" must be an array of 2 to 10 levels')
                options = [(str(i), d) for i, d in enumerate(criteria)]
            elif qtype == "noul":
                if criteria is not None and not isinstance(criteria, dict):
                    raise err('"criteria" must be an object')
                criteria = criteria or {}
                options = []
                for k in ("false", "true"):
                    desc = criteria.get(k)
                    if k not in criteria and self.type == "lfm2-d1-omni":
                        desc = criteria.get("yes" if k == "true" else "no")
                    options.append((k, desc))
                if self.type == "lfm2-d1":
                    options.reverse()  # true first
            else:
                raise err('"type" must be one of: choice, score, noul')
            if len(options) > self.n_options_max:
                raise err(f"too many options ({len(options)}), this model supports at most {self.n_options_max}")
            parsed.append({"id": qid, "type": qtype, "instructions": q.get("instructions"), "options": options})
        return parsed

    # -- prompt --------------------------------------------------------------

    def _n_variants(self, question: dict[str, Any]) -> int:
        # lev shows the options of a choice in 2 orders, to cancel the preference for the first label
        if self.type == "lev" and question["type"] == "choice" and len(question["options"]) > 1:
            return 2
        return 1

    def _n_outputs(self, question: dict[str, Any]) -> int:
        if self.type == "lev" and question["type"] == "noul":
            return _LEV_N_RATINGS
        return len(question["options"])

    def _tokenize_label(self, text: str) -> list[int]:
        return list(self.vocab.tokenize(text, add_special=False, parse_special=False))

    def _d1_labels(self, question: dict[str, Any]) -> Tuple[list[str], list[list[int]]]:
        """lfm2-d1: label text and label tokens of each option, codes as in prompt.py of the model repo."""

        def single_tokens(forms: list[str]) -> list[int]:
            out: list[int] = []
            for form in forms:
                toks = self._tokenize_label(form)
                if len(toks) == 1 and toks[0] not in out:
                    out.append(toks[0])
            return out

        texts: list[str] = []
        groups: list[list[int]] = []
        options = question["options"]
        if question["type"] != "choice":
            for key, _ in options:
                if question["type"] == "score":
                    group = single_tokens([key])
                else:
                    group = single_tokens(["yes", "Yes", "YES"] if key == "true" else ["no", "No", "NO"])
                if not group:
                    raise RuntimeError(f"decision label is not a single token: {key}")
                texts.append(key)
                groups.append(group)
            return texts, groups

        n = len(options)
        if all(len(k.encode()) == 1 and k.isalpha() for k, _ in options):
            codes = [k for k, _ in options]
        elif n <= 26:
            codes = [chr(65 + i) for i in range(n)]
        else:
            codes = [f"{i:02d}" for i in range(n)]
        letters = [chr(c) for c in range(65, 91)]
        pool = (
            letters
            + [f"{i:02d}" for i in range(100)]
            + [chr(c) for c in range(97, 123)]
            + [f"#{i}" for i in range(200)]
            + [a + b for a in letters for b in letters]
        )

        used: list[int] = []

        def take(code: str) -> bool:
            toks = self._tokenize_label(code)
            if len(toks) != 1 or toks[0] in used:
                return False
            used.append(toks[0])
            texts.append(code)
            groups.append(toks + [t for t in single_tokens([" " + code]) if t != toks[0]])
            return True

        for code in codes:
            if not take(code) and not any(take(c) for c in pool):
                raise ValueError(f"no single-token label left for {n} options")
        return texts, groups

    def _render_options(self, question: dict[str, Any], variant: int) -> list[dict[str, Any]]:
        opts = question["options"] if variant == 0 else question["options"][::-1]
        label_texts = self._d1_labels(question)[0] if self.type == "lfm2-d1" else self.label_texts
        out = []
        for i, (key, desc) in enumerate(opts):
            option = {"key": key, "description": desc}
            if self.type == "kev":
                option["key"] = _kev_text(key)
                if desc is not None:
                    option["description"] = _kev_text(desc)
            if label_texts:
                option["label"] = label_texts[i]
            out.append(option)
        return out

    def _render(self, state: Any, question: dict[str, Any], variant: int) -> str:
        inp = {
            "id": question["id"],
            "type": question["type"],
            "instructions": question["instructions"],
            "state": state,
            "options": self._render_options(question, variant),
        }
        if self.type == "lev":
            inp = _sort_keys(inp)  # lev was trained with sorted keys
        if self.type == "kev":
            inp["state"] = _kev_text(state)
            inp["instructions"] = _kev_text(question["instructions"])
        if self.text_marker:
            # the input must not contain the marker of the options
            inp = _replace_text(inp, self.text_marker, " ")
        if self.type == "lfm2-d1-omni":
            inp = _replace_text(_d1omni_escape(inp), _D1OMNI_MARKER, "<<d1omni ")
            inp.update(
                audio=False,
                sep=_D1OMNI_SEP,
                mark_state=_D1OMNI_MARK_STATE,
                mark_question=_D1OMNI_MARK_QUESTION,
                mark_option=_D1OMNI_MARK_OPTION,
            )
        inp["images"] = []
        return str(self._template.render(**inp))

    def _laya_tokens(self, tokens: list[int], n_options: int) -> Tuple[list[int], list[int]]:
        """Cut question and options to max_head_tokens as in training; return (tokens, marker positions).

        The prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
        """
        marker, sep = self.token_marker, self.token_sep
        markers = [i for i, t in enumerate(tokens) if t == marker]
        invalid = RuntimeError("unexpected layout of the decision prompt")
        if len(markers) != n_options or markers[0] < 2 or tokens[markers[0] - 1] != sep or tokens[-1] != sep:
            raise invalid
        head_end = markers[0] - 1
        try:
            opts_end = tokens.index(sep, markers[-1])
        except ValueError:
            raise invalid from None
        if opts_end + 1 >= len(tokens):
            raise invalid

        options = [tokens[markers[i] : (markers[i + 1] if i + 1 < n_options else opts_end)] for i in range(n_options)]

        def set_max(n_max: int) -> int:
            for i, opt in enumerate(options):
                options[i] = opt[:n_max]
            return sum(len(o) for o in options)

        n_options_tokens = set_max(_LAYA_MAX_OPTION_TOKENS + 1)
        if n_options_tokens + 16 > self.max_head_tokens:
            # too many or too long options, shrink them evenly
            n_options_tokens = set_max(max(4, (self.max_head_tokens - min(self.max_head_tokens, 16)) // n_options))
        n_question_max = max(8, self.max_head_tokens - min(self.max_head_tokens, n_options_tokens))

        out = [tokens[0]] + tokens[1 : min(head_end, 1 + n_question_max)] + [sep]
        positions = []
        for opt in options:
            positions.append(len(out))
            out += opt
        out += tokens[opts_end:]
        return out, positions

    def _d1omni_tokens(self, prompt: str, n_options: int) -> Tuple[list[int], list[int]]:
        """Cut each piece of the prompt to its budget as in training (d1-omni prompt.py: encode); return (tokens, marker positions).

        The state gets the room that is left.
        """
        invalid = RuntimeError("unexpected layout of the decision prompt")
        pieces = prompt.split(_D1OMNI_SEP)
        if pieces[0] != "":  # the media markers, none here
            raise invalid
        n_max = _D1OMNI_MAX_TOKENS
        # the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
        n_budget = max(96, min(24 * n_options + 32, n_max // 2))
        n_option_max = max(2, (n_budget - 3 * n_options) // n_options)
        n_question_max = max(16, n_budget)

        head: list[int] = []
        body: Optional[list[int]] = None
        tail: list[int] = []
        for piece in pieces[1:]:
            n_piece_max = None
            is_state = piece.startswith(_D1OMNI_MARK_STATE)
            if is_state:
                piece = piece[len(_D1OMNI_MARK_STATE) :]
            elif piece.startswith(_D1OMNI_MARK_QUESTION):
                piece = piece[len(_D1OMNI_MARK_QUESTION) :]
                n_piece_max = n_question_max
            elif piece.startswith(_D1OMNI_MARK_OPTION):
                piece = piece[len(_D1OMNI_MARK_OPTION) :]
                n_piece_max = n_option_max
            toks = self.vocab.tokenize(piece, add_special=False, parse_special=True)[:n_piece_max]
            if is_state:
                if body is not None:
                    raise invalid
                body = toks
            else:
                (head if body is None else tail).extend(toks)
        if body is None:
            raise invalid

        n_room = n_max - min(n_max, len(head) + len(tail))
        tokens = (head + body[:n_room] + tail)[:n_max]
        markers = [i for i, t in enumerate(tokens) if t == self.token_marker]
        if len(markers) != n_options:
            raise ValueError("the options do not fit in the context")
        return tokens, markers

    # -- evaluation ----------------------------------------------------------

    def _decode(self, tokens: list[int], all_outputs: bool, pos_first: int = -1) -> int:
        """Evaluate ``tokens``; return n_past, the number of tokens reused from the KV cache.

        Batch index i is prompt position n_past + i. ``pos_first`` is the first position
        that is read, -1 if only the last one.
        """
        if len(tokens) > self.n_batch:
            raise ValueError(f"prompt has {len(tokens)} tokens, the batch size is {self.n_batch}; increase n_ctx")
        n_past = 0
        # llama-server: a non-causal context reads the whole prompt in one batch, it reuses nothing
        if self.cache_prompt and self.ctx.get_causal_attn():
            n_past = next(
                (i for i, (a, b) in enumerate(zip(self._cached, tokens)) if a != b), min(len(self._cached), len(tokens))
            )
            if pos_first >= 0:
                n_past = min(n_past, pos_first)
            if n_past > 0:
                # the memory must still hold what the first new token attends to (SWA, recurrent state);
                # a decision has no checkpoint to restore, so the prompt is evaluated again
                n_swa = self.model.n_swa
                pos_eval = n_past if n_past < len(tokens) else n_past - 1
                if self.ctx.memory_seq_pos_min(0) >= max(0, pos_eval - n_swa + 2 if n_swa > 0 else pos_eval):
                    n_past = 0
            n_past = min(n_past, len(tokens) - 1)  # evaluate at least one token
        self._cached = []
        if n_past == 0:
            self.ctx.kv_cache_clear()
        else:
            self.ctx.memory_seq_rm(0, n_past, -1)
        batch = LlamaBatchExt(self.ctx)
        for i in range(n_past, len(tokens)):
            batch.add_token(tokens[i], i, output=all_outputs or i == len(tokens) - 1)
        self.ctx.process(batch)
        self._cached = list(tokens)
        return n_past

    def _evaluate(self, state: Any, question: dict[str, Any], variant: int) -> Tuple[list[float], int]:
        """Return the raw scores of one variant of ``question`` and its prompt length."""
        prompt = self._render(state, question, variant)
        column = _QUESTION_TYPES.index(question["type"])
        if self.type == "lfm2-d1-omni":
            tokens, markers = self._d1omni_tokens(prompt, len(question["options"]))
            n_past = self._decode(tokens, all_outputs=True, pos_first=markers[0])
            return [self.ctx.get_embeddings_ith(m - n_past)[column] for m in markers], len(tokens)
        tokens = self.vocab.tokenize(prompt, add_special=False, parse_special=True)
        if self.type in _LABEL_TYPES:
            self._decode(tokens, all_outputs=False)
            logits = self.ctx.get_logits_ith(-1)
            return [logits[t] for t in self.labels[: self._n_outputs(question)]], len(tokens)
        if self.type == "lfm2-d1":
            self._decode(tokens, all_outputs=False)
            logits = self.ctx.get_logits_ith(-1)
            return [max(logits[t] for t in group) for group in self._d1_labels(question)[1]], len(tokens)
        if self.type == "laya":
            tokens, markers = self._laya_tokens(tokens, len(question["options"]))
            n_past = self._decode(tokens, all_outputs=True, pos_first=markers[0])
            return [self.ctx.get_embeddings_ith(m - n_past)[column] for m in markers], len(tokens)
        # kev: an option is read at its end token, the question at the last token
        markers = [i for i, t in enumerate(tokens) if t == self.token_marker]
        if len(markers) != len(question["options"]):
            raise RuntimeError("unexpected layout of the decision prompt")
        n_past = self._decode(tokens, all_outputs=True, pos_first=markers[0])
        q = self.ctx.get_embeddings_ith(len(tokens) - 1 - n_past)
        n = len(q) // 2
        scale = math.sqrt(n)
        scores = []
        for m in markers:
            k = self.ctx.get_embeddings_ith(m - n_past)
            scores.append(sum(q[i] * k[n + i] for i in range(n)) / scale)
        return scores, len(tokens)

    def _temperature(self, question: dict[str, Any]) -> float:
        n = len(question["options"])
        if self.type == "lev":
            bucket = "small" if n <= 8 else "mid" if n <= 26 else "large"
        else:
            bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
        for name in (f"{question['type']}.{bucket}", question["type"]):
            if name in self.temperatures:
                return self.temperatures[name]
        return 1.0

    def _format(self, question: dict[str, Any], variants: list[list[float]]) -> dict[str, Any]:
        # softmax over the outputs of each variant, then the average of the variants
        n = self._n_outputs(question)
        temp = self._temperature(question)
        bias = self.score_bias.get(n) if question["type"] == "score" else None
        probs = [0.0] * n
        for v, scores in enumerate(variants):
            if any(math.isnan(s) for s in scores):
                raise RuntimeError("the model could not evaluate the decision")
            if bias:
                scores = [s + b for s, b in zip(scores, bias)]
            top = max(scores)
            p = [math.exp((s - top) / temp) for s in scores]
            total = sum(p)
            for i in range(n):
                # the second variant is in the reverse order
                probs[i if v == 0 else n - 1 - i] += p[i] / total / len(variants)

        keys = [k for k, _ in question["options"]]
        answer = {"type": question["type"]}
        if question["type"] == "noul":
            if self.type == "lev":
                answer["noul"] = sum(p * i / (n - 1) for i, p in enumerate(probs))
            else:
                answer["noul"] = probs[keys.index("true")]
            return answer
        probabilities = dict(zip(keys, probs))
        if question["type"] == "choice":
            answer["choice"] = keys[probs.index(max(probs))]
            answer["probabilities"] = probabilities
            answer["confidence"] = _confidence_entropy(probs) if self.type == "vllm-d3" else _confidence_choice(probs)
        else:
            answer["score"] = sum(i * x for i, x in enumerate(probs))
            if self.type == "vllm-d3":
                # non-text descriptions as canonical JSON text
                answer["legend"] = {
                    k: d if isinstance(d, str) else json.dumps(_sort_keys(d), ensure_ascii=False, separators=(",", ":"))
                    for k, d in question["options"]
                }
            else:
                answer["legend"] = {k: d for k, d in question["options"]}
            answer["probabilities"] = probabilities
            answer["confidence"] = _confidence_entropy(probs) if self.type == "vllm-d3" else _confidence_score(probs)
        return answer

    def answer(self, request: Any) -> dict[str, Any]:
        """Answer a ``/v1/systemone`` request body.

        Returns:
            ``{"answers": {question_id: answer}, "usage": {...}}``, as the endpoint.

        Raises:
            ValueError: invalid request, or a prompt that does not fit in one batch.
            NotImplementedError: the request has images or audio.
        """
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        questions = self._parse_questions(request)
        _check_media(request)
        state = request["state"]
        answers = {}
        n_input = 0
        with self._lock:
            for q in questions:
                variants = []
                for v in range(self._n_variants(q)):
                    scores, n_tokens = self._evaluate(state, q, v)
                    variants.append(scores)
                    n_input += n_tokens
                answers[q["id"]] = self._format(q, variants)
        return {"answers": answers, "usage": {"input_tokens": n_input, "output_tokens": 0}}


def load_decision_model(
    model: LlamaModel, n_ctx: int, logger: Optional[logging.Logger] = None
) -> Optional[DecisionModel]:
    """Return a :class:`DecisionModel` for ``model``, or ``None`` if it is not a supported decision model."""
    dtype = get_decision_type(model)
    if dtype is None:
        return None
    try:
        return DecisionModel(model, n_ctx=n_ctx)
    except (NotImplementedError, ValueError) as e:
        (logger or logging.getLogger(__name__)).warning(f"/v1/systemone disabled: {e}")
        return None


def systemone_response(decision: Optional[DecisionModel], data: Any, model_name: str) -> Tuple[int, dict[str, Any]]:
    """Status and body for a ``/v1/systemone`` request, with llama-server's status codes."""
    if decision is None:
        return 501, {"error": {"type": "not_supported_error", "message": "the model is not a supported decision model"}}
    try:
        result = decision.answer(data)
    except ValueError as e:
        return 400, {"error": {"type": "invalid_request_error", "message": str(e)}}
    except NotImplementedError as e:
        return 501, {"error": {"type": "not_supported_error", "message": str(e)}}
    return 200, {"model": model_name, **result}
