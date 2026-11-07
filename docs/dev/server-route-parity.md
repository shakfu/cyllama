# Server route parity with llama-server

Status: plan, 2026-11-06, against llama.cpp `b11960`. Nothing below is implemented.

Goal: `PythonServer` and `EmbeddedServer` serve the routes of upstream `llama-server`.
Wheel users need this because wheels do not ship `llama-server` (see `LlamaServer`).

## Inventory

Upstream routes: `build/llama.cpp/tools/server/server.cpp:140-215`.
cyllama routes: `src/cyllama/llama/server/python.py:545-576`, `embedded.pyx:271-298`.
Both cyllama servers serve the same 5 routes.

| Group | Upstream routes | cyllama | Prerequisites |
|-|-|-|-|
| health | `/health`, `/v1/health` | `/health` | none |
| models | `/models`, `/v1/models` | `/v1/models` | none |
| chat | `/chat/completions`, `/v1/chat/completions` | v1 only, text only | P1 for media parts |
| chat extras | `/v1/chat/completions/control`, `/{v1/,}chat/completions/input_tokens` | no | none |
| completions | `/completion`, `/completions`, `/v1/completions` | no | none |
| responses | `/{v1/,}responses`, `/{v1/,}responses/input_tokens` | no | none |
| anthropic | `/v1/messages`, `/v1/messages/count_tokens` | no | none |
| infill | `/infill` | no | none |
| embeddings | `/embedding`, `/embeddings`, `/v1/embeddings` | v1 only | none |
| rerank | `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | no | rerank pooling |
| systemone | `/v1/systemone` | yes | none |
| tokens | `/tokenize`, `/detokenize`, `/apply-template` | no | none |
| admin | `/metrics`, `GET/POST /props`, `/slots`, `/slots/:id`, `GET/POST /lora-adapters` | no | none |
| audio in | `/{v1/,}audio/transcriptions` | no | P1, P2 |
| audio out | `/tts` | no | P3, P4 |
| realtime | `/v1/realtime` (WebSocket) | no | P1, P3, P4, P5 |
| streams | `GET/DELETE /v1/stream`, `/v1/streams/lookup` | no | router mode |

Out of scope: the `streams` group. It resumes conversations in upstream's multi-model
router mode (`server-models.cpp:2375`), which cyllama does not have.

## Prerequisites

- **P0. One handler layer.** Each server has its own dispatch and `_handle_*` methods.
  `embedded.pyx` imports only the slot and request types from `python.py`. 45 routes
  written twice is the main cost. Move handlers into transport-independent Python that
  returns a response object or a chunk iterator. Each server then only does I/O.
- **P1. Multimodal chat input.** Neither server loads an mmproj. Upstream transcription
  sends audio through an mtmd chat model (`server-context.cpp:6082`), not a separate ASR
  model. Needs `image_url` / `input_audio` content parts mapped to `MtmdBitmap`.
- **P2. Multipart form parsing.** `/v1/audio/transcriptions` takes `multipart/form-data`.
  cpp-httplib parses it (`req.files`); `http_shim` would need to pass the files through.
  Python 3.13 removed `cgi`, so `PythonServer` needs `email.parser` or a new dependency.
- **P3. Binary chunked responses.** Streaming `/tts` sends raw WAV with an unknown-length
  header. `http_shim.cpp:103` already uses `set_chunked_content_provider` with a content
  type. `PythonServer` needs `Transfer-Encoding: chunked` written by hand.
- **P4. Server mode from model capabilities.** A TTS model needs embeddings mode plus an
  mmproj, and has no chat path. Choose routes at load from `MtmdContext.capabilities`.
  This matches upstream: one model per process.
- **P5. WebSocket.** Vendored cpp-httplib 0.62.0 has a WebSocket server
  (`thirdparty/llama.cpp/include/cpp-httplib/httplib.h:2260`). The stdlib has none, so
  `PythonServer` needs RFC 6455 written by hand or a dependency.

## Open decisions

1. **Transcription backend.** Upstream path (mtmd audio chat model, P1), whisper
   (`cyllama.whisper`, already wrapped), or both chosen by the loaded model. Whisper needs
   no P1 and is the better ASR model for most users. The upstream path is the parity one.
2. **Which servers.** Full parity in both, or `/v1/realtime` in `EmbeddedServer` only.
   P5 is cheap there and costly in `PythonServer`.
3. **Response formats.** Upstream `/tts` serves `wav` and `pcm` only. OpenAI
   `/v1/audio/speech` also lists mp3/opus/aac/flac, which need an encoder dependency.
   Serve upstream's body on both paths, wav/pcm only?
4. **Aliases.** Upstream registers 2-4 paths per route. Serve all of them, or `/v1/*` only?
5. **Verification.** `make llama-server` builds the reference binary. Extend the
   `test_decision.py -k parity` pattern: the same request to both servers, the same
   response shape.

## Proposed order

1. P0. No behaviour change; current tests cover it.
2. Routes with no prerequisites: aliases, `/completions`, tokens, `/infill`, admin,
   `input_tokens`, `/v1/messages`, `/v1/responses`.
3. P4 + P3, then `/tts` (calls `MtmdTTSGenerator.stream()`).
4. P1, then media chat parts, then P2 and `/v1/audio/transcriptions`.
5. P5 and `/v1/realtime`.

Steps 2-5 are independent of each other once P0 is done.
