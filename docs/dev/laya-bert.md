# Laya typed decisions on llama.cpp ModernBERT

Status: proposal. Nothing implemented.

## Goal

Run the three Laya checkpoints (`english`, `multilingual`, `typed-decisions`,
HF repo `convaiinnovations/laya`) from cyllama, with no new native code and no
second ggml. Output must match `laya.cpp` (https://github.com/lkarlslund/laya.cpp):
exact categories, numeric absolute error <= 1e-4.

Non-goals:

- laya.cpp's custom CUDA/Vulkan kernels, BF16 paths, Core ML.
- The JEV HTTP server. A `/v1/systemone` route can be added to cyllama's server later.
- Training, or checkpoints outside the two encoder profiles below.

## Model, as laya.cpp implements it

Sources: `docs/architecture.md`, `src/runtime.cpp`, `src/protocol.cpp`, laya.cpp
commit of 2026-09-26.

| | english / typed-decisions | multilingual |
|-|-|-|
| Encoder | ModernBERT, w=1024, 28 layers, 16 heads | ModernBERT, w=768, 22 layers, 12 heads |
| FFN intermediate | 2624 | 1152 |
| Vocab | 50368, byte-level BPE | 256000, metaspace BPE + byte fallback |
| RoPE base global / local | 160000 / 10000 | 160000 / 160000 |
| `max_len` / `head_max_len` | 512 / 192 | 1024 / 256 |

Common to both:

- Global attention on layers where `i % 3 == 0`. Elsewhere, local attention with
  `|q - k| <= 64`.
- Head dim 64. No biases in the encoder. GeGLU with exact-erf GELU.

`model.safetensors` holds the encoder (`encoder.*`) and a decision head:

1. `h = final_norm(encoder(ids)) + type_emb[qtype]`. `qtype` is choice=0, score=1, noul=2.
2. Two pre-norm `nn.TransformerEncoderLayer` blocks (`head.layers.{0,1}`):
   - LayerNorm with bias.
   - Global attention over the unpadded sequence. No positional encoding.
   - FFN of width 4w with ReLU.
3. Scorer on option-marker rows: `LayerNorm -> Linear(w,w) -> GELU -> Linear(w,1)`.
   This gives one logit per option.
4. Action head:
   - Input: `h[CLS]` plus 4 statistics from the uncalibrated option softmax.
     The statistics are top-1, top-1 minus top-2, normalised entropy, and `count/255`.
   - Layers: `Linear(w+4,256) -> GELU -> Linear(256, n_actions)`.
5. Calibration:
   - Divide option logits by a temperature: `temperature[qtype]`, overridden by
     `temperature_by_options["<type>:<bucket>"]`.
   - Softmax, then confidence and score. Round to 4 decimals.

Request formatting (`agent::prepare`):

- Layout: `[CLS] "<type> question: <instructions>" [SEP] ([MASK] " <option>")* [SEP] <state> [SEP]`.
- Option budget: 48 tokens per option. Combined token budgets apply to the heading and option lists.
- Requests over budget are rejected unless `allow_truncation` is set.
- Literal mask text in user input is replaced with a space.
- Text is NFC-normalised before BPE.

## Design

```
request JSON
   |  laya/protocol.py   (port of agent::prepare; tokenizes via LlamaModel vocab)
   v
token ids per question, marker positions, qtype
   |  LlamaContext.decode, embeddings mode, pooling NONE, one seq_id per question
   v
per-token encoder output [L, w]          <- llama.cpp ModernBERT graph, any backend
   |  laya/head.py       (numpy: type_emb, 2 head layers, scorer, action head)
   v
option logits, action logits
   |  laya/protocol.py   (port of agent::predict post-processing)
   v
JEV answer envelope
```

Split rationale:

- Encoder in llama.cpp. It accounts for 93% of FLOPs (english) or 87% (multilingual); see Cost.
  It runs on every backend cyllama already builds.
- Head in Python. llama.cpp has no architecture with this block type: pre-norm,
  biased LayerNorm, ReLU FFN, no RoPE. Adding one means patching llama.cpp.
- A GGUF cannot carry the head tensors either. The llama.cpp loader rejects
  tensors the architecture does not create. The head ships in a sidecar file.

### Files

| Path | Content |
|-|-|
| `scripts/laya_convert.py` | Checkpoint dir -> `<name>.gguf` + `<name>.head.npz` |
| `src/cyllama/laya/__init__.py` | `LayaClassifier` |
| `src/cyllama/laya/protocol.py` | `prepare()`, `format_answers()` |
| `src/cyllama/laya/head.py` | `DecisionHead` (numpy) |
| `src/cyllama/llama/llama_cpp.pyx` | Bulk per-token embedding accessor (new) |
| `tests/test_laya_protocol.py`, `test_laya_head.py`, `test_laya_e2e.py` | see Validation |
| `scripts/laya_parity.py` | Dev-only diff against `laya-cli` |

`cyllama.laya` lives in its own subpackage, not `integrations/`. It is a model
runtime, not an adapter to a third-party framework.

### API

```python
from cyllama.laya import LayaClassifier

clf = LayaClassifier("laya-english.gguf", n_gpu_layers=-1)  # finds laya-english.head.npz
clf.predict([{"state": "...", "questions": {"refund": {"type": "noul", "instructions": "..."}}}])
# -> [{"model": "laya-rl-agent", "answers": {...}, "usage": {...}}]   same schema as laya-cli
```

Other behaviour:

- `predict(..., raw=True)` returns the ids, markers, logits and action logits, as
  `laya-cli --raw` does. Used by the parity tests.
- One instance holds one llama context. It uses the Embedder's non-blocking busy-lock
  (`docs/dev/runtime-guard.md`).
- numpy is imported inside `cyllama.laya` only. This keeps the core zero-dependency
  and follows `src/cyllama/llama/mtmd` (conditional numpy import). It is exposed as
  the extra `cyllama[laya]`.

### Conversion (`scripts/laya_convert.py`)

1. Split `model.safetensors`:
   - `encoder.*` -> strip the prefix -> HF `ModernBertModel` names
     (`embeddings.tok_embeddings.weight`, `layers.N.attn.Wqkv.weight`, `final_norm.weight`).
   - Write them to a temp HF dir with `encoder/config.json` as `config.json`
     (`architectures: ["ModernBertModel"]`) and `tokenizer/*`.
2. Run llama.cpp `convert_hf_to_gguf.py --outtype f32` on the temp dir.
   - `ModernBertModel.set_gguf_parameters` (`conversion/bert.py:609`) writes the
     sliding window (128) and the pattern (3).
   - `base.py:1466` writes the local RoPE base from `rope_parameters.sliding_attention`.
     laya's `encoder/config.json` uses that key (`runtime.cpp:121`).
3. Write `<name>.head.npz`:
   - The `type_emb`, `head.*`, `scorer.*` and `act_head.*` arrays.
   - A `config` JSON string: `temperature`, `temperature_by_options`, `act_costs`,
     `max_len`, `head_max_len`, and the CLS/SEP/MASK/PAD token strings from
     `tokenizer/tokenizer_config.json`.
4. Record the source HF revision in both outputs.

The script is a dev tool. `convert_hf_to_gguf.py` needs torch, and cyllama does not
ship it. We publish converted files only if the model license permits it.

### Encoder call

- Context: `embeddings=True`, `pooling_type=NONE`, `n_ubatch = n_batch >= sum of tokens
  per decode`. A non-causal model needs each sequence inside one ubatch.
  Set `n_seq_max >=` questions per decode.
- Each question gets its own `seq_id`, positions `0..L-1`, and every token marked for output.
  llama.cpp masks attention across `seq_id`s, so laya's padding masks are not needed.
- Symmetric SWA matches laya's local mask. `llama-hparams.h:493` masks
  `|p1 - p0| > n_swa/2`, which is `> 64` for `n_swa = 128`.
- Sequences are built from ids. Each piece is tokenized with `add_special=False,
  parse_special=False`, and the special-token ids are inserted directly. The GGUF's
  `add_bos`/`add_eos`/`add_sep` flags never apply.
- Python applies `unicodedata.normalize("NFC", ...)` and mask neutralisation before
  tokenizing. llama.cpp's BPE path does not NFC-normalise (to verify in P0).

New accessor: `LlamaContext.get_embeddings_ith` builds a Python list one float at a
time. One question at L=256, w=1024 is 262k floats. Add
`LlamaContext.get_embeddings_rows(n_rows) -> memoryview` (float32, shape
`[n_rows, n_embd]`) over `llama_get_embeddings`. The caller passes `n_rows` because
llama.cpp does not expose `n_outputs`.

### Cost

These are estimates from parameter counts, not measurements.

Encoder MACs per token:

- english: 28 x (4w^2 attention + 7.7w^2 GeGLU) = 327w^2.
- multilingual: 22 x 8.5w^2 = 187w^2.

The head is 2 x 12w^2 = 24w^2, i.e. 7% (english) or 13% (multilingual) of encoder
FLOPs. Only `CLS` and marker rows are used from the last head layer. Computing that
layer's Q, output projection and FFN for those rows only, with K/V still over all
tokens, cuts the head to about 14w^2.

Absolute numbers: english at L=256 is about 13 GFLOP of head per question.
- numpy sgemm on CPU at 100-300 GFLOP/s gives 40-130 ms per question.
- laya.cpp on CUDA does 270-490 questions/s end to end (`docs/models.md`), i.e. 2-4 ms.

Consequences:

- **CPU deployments.** The head adds about 7-13% to encoder time. Acceptable.
- **GPU deployments.** The numpy head dominates: throughput is about 10-50x below
  laya.cpp. The fix is P4 below, not more numpy tuning.

### Precision

- Parity work uses an F32 GGUF.
- The checkpoint stores FP16 projection weights (`docs/architecture.md`), so an F16
  GGUF stores them losslessly. ggml's F16 matmul may still round activations to F16
  (CPU `vec_dot_type`; to verify), which can exceed the 1e-4 gate.
- Accept F16 and Q8_0 GGUFs only after the same acceptance run. Report their max
  error rather than assume it.

## Validation

The reference is `laya-cli --cpu --fp32`. It exposes `--prepare` (token ids and
markers), `--raw` (logits), and `LAYA_TRACE_DIR` (per-layer `.f32` tensors,
including `final-norm`). The corpus is laya.cpp's
`benchmarks/cases/acceptance-250.json` plus `tests/model-requests.json`
(36 long-context and multi-question cases).

| Level | Check | Gate |
|-|-|-|
| Tokens | `prepare()` ids/markers vs `laya-cli --prepare` | exact, all cases |
| Budgets | over-budget requests raise the same error class/message as laya | exact |
| Encoder | per-token output vs `LAYA_TRACE_DIR/final-norm.f32` | report max abs err; no gate, used for diagnosis |
| Logits | option and action logits vs `--raw` | report |
| Answers | `predict()` vs laya-cli output | exact category, abs err <= 1e-4 |

Repo tests, run under `make test`:

- `test_laya_protocol.py`: budgets, truncation, option rendering for list/object
  criteria, `noul` defaults, mask neutralisation. Uses a stub tokenizer. No model needed.
- `test_laya_head.py`: `DecisionHead` on a small random head (w=64), compared with
  a hand-written step-by-step numpy reference. Also calibration and bucket selection.
- `test_laya_e2e.py`: golden answers for about 20 corpus cases, generated once from
  laya-cli. Skipped when the GGUF is absent, like the other model-dependent tests.

`scripts/laya_parity.py` runs the full table above against a local laya.cpp build.
It is not part of `make test`.

## Phases

Each phase ends at its gate. Failing a gate stops the plan or changes it.

- **P0 spike: encoder parity, english.**
  - Steps: convert to an F32 GGUF; decode `smoke.json` cases; diff against `final-norm.f32`.
  - Exit: answers derived from our encoder output, run through laya's head, pass on `smoke.json`.
  - Stop if a mismatch traces to llama.cpp's ModernBERT graph and GGUF metadata
    cannot fix it. Candidates: RoPE type, embedding norm, layer-0 identity norm.
    Then the choice is an upstream llama.cpp PR or dropping the plan.
- **P1: protocol + numpy head, english and typed-decisions, CPU.**
  - Exit: all 250 + 36 cases pass the Answers gate; repo tests pass.
- **P2: GPU backends via `n_gpu_layers`.**
  - Exit: Answers gate on CUDA, Vulkan and Metal with the F32 GGUF.
  - Also produce an F16/Q8_0 error report.
- **P3: multilingual.**
  - `ModernBertModel.set_vocab` always calls `_set_vocab_gpt2` (`conversion/bert.py:603`).
    That expects byte-level BPE with a known pre-tokenizer hash, so it cannot
    represent a metaspace + byte-fallback vocab.
  - `laya_convert.py` subclasses the converter and writes the vocab through the
    SentencePiece/llama-HF path.
  - Exit: the Tokens gate on multilingual cases, then the Answers gate.
- **P4 (conditional): head on the ggml backend.**
  - Start only if a GPU user needs more than the numpy head delivers.
  - Build the head graph in Cython against `src/cyllama/llama/ggml.pxd`, on the
    backend the encoder uses.
  - It needs about 25M parameters (english), roughly 100 MB of F32 weights.

## Open questions

- Model license on `convaiinnovations/laya`. It decides whether converted files can
  be published or must be converted locally.
- The meaning of `act_costs` and the action classes. laya.cpp exposes only
  `act_probability = softmax(actions)[0]`. Port that and nothing more until the
  publisher documents the rest.
- Whether the multilingual encoder is mmBERT. Width 768, 22 layers and vocab 256000
  match mmBERT-base. That is inference from shapes. It affects which converter vocab
  path is correct.
- Whether typed-decisions shares the english tokenizer. laya.cpp runs the same 253
  byte-level fixtures on both (`docs/models.md`), which suggests yes. Confirm by
  hashing `tokenizer.json`.

## Alternatives considered

- **Bind laya.cpp.** Rejected: it string-patches `ggml-cuda.cu` at build time
  (`CMakeLists.txt:57-82`) and pins its own ggml. cyllama runs one ggml across
  llama, whisper and sd.
- **Call `laya-cli --server` over HTTP.** It gives full speed and exact parity with
  no port. Use it instead of this plan if the user already runs a GPU service and
  does not need in-process inference.
- **Constrained JSON from a small instruct LLM via cyllama grammars.** No new code
  at all, but it is a different model with different accuracy. It does not replace
  these checkpoints.
