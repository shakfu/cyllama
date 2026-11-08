# Chatterbox voice-cloning latency

## Summary

Voice cloning with Chatterbox Turbo costs about 0.26 s of encoding per second of reference audio. The cost recurs on every `set_input` call. Token generation and vocoding run at the same speed as with the default voice.

Measured on an M1 MacBook Air (Metal), llama.cpp b12110 (`cd959ba`), `chatterbox-turbo-Q8_0.gguf` with `mmproj-chatterbox-turbo-F16.gguf`. The reference is `dtrump.wav` (38.3 s, 44.1 kHz stereo) or its first N seconds. The utterance is one sentence, about 2.8 s of audio.

| Reference | `set_input` | preprocess | encode | gen + vocode |
|-|-|-|-|-|
| default voice | 0.05 s | - | - | 2.9 s |
| 5 s | 1.27 s | 0.09 s | 1.14 s | 2.5 s |
| 10 s | 2.42 s | 0.18 s | 2.21 s | 2.8 s |
| 15 s | - | 0.21 s | 3.74 s | - |
| 20 s | 5.29 s | 0.23 s | 4.91 s | 3.7 s |
| 38 s | 9.95 s | 0.29 s | 9.52 s | 2.9 s |

`preprocess` is `mtmd_tokenize` (CPU feature extraction). `encode` is `mtmd_encode_chunk`. A repeated `set_input` on the same `MtmdBitmap` costs the same as the first.

## Causes

1. **The reference is re-encoded per utterance.** `chatterbox_gen_audio_pipeline::set_input` calls `encode_speaker` on `inp->speaker_ref` every time (`tools/mtmd/mtmd-helper-gen.cpp:1750`). `MtmdTTSGenerator.generate` calls `set_input` once per call. A passage split into 9 sentences with the 38 s reference spends about 90 s on encoding.

2. **Two of five reference features read the whole clip.** `mtmd_audio_preprocessor_chatterbox_ref::preprocess` (`tools/mtmd/mtmd-audio.cpp:1298`) builds five entries:

   | Entry | Feature | Cap |
   |-|-|-|
   | 0 | kaldi fbank for the CAMPPlus x-vector | none |
   | 1 | s3 tokenizer mel, flow side | 10 s |
   | 2 | s3 tokenizer mel, T3 side | 15 s (6 s multilingual) |
   | 3 | s3gen prompt mel at 24 kHz | 10 s |
   | 4 | voice-encoder mel, silence trimmed | none |

   The token count reaches 376 at 15 s and stays there. Encode time keeps rising: 3.74 s at 15 s, 4.91 s at 20 s, 9.52 s at 38 s.

3. **The voice encoder runs as scalar CPU code.** `cbx_ref_cond` (`tools/mtmd/mtmd.cpp:2440`) evaluates the 3-layer, 256-unit LSTM in nested C++ loops with `double` accumulators on one thread. It runs once per 160-frame window, stepping 77 frames, so consecutive windows overlap by about 50%. Each window costs about 216M multiply-adds. A 38 s clip has about 48 windows.

A `sample` profile of `mtmd_encode_chunk` on the 38 s clip gives 3164 main-thread samples:

- About 1600 are self time in one function called by `mtmd_encode_chunk`: `cbx_ref_encode`, with `cbx_ref_cond` inlined.
- 1127 wait in `-[_MTLCommandBuffer waitUntilCompleted]`, reached from that function's first graph call.

The extension has no symbols. Functions were matched by address offset against one shared start address. The 1127 Metal samples are attributed to the x-vector (entry 0) because `cbx_ref_xvec` is the first graph call in `cbx_ref_encode`. That attribution is an inference.

## Why cyllama cannot cache the encoding

`set_input` always resets the pipeline and encodes `speaker_ref`. A voice bitmap (`mtmd_bitmap_is_voice`) skips the encoder. The public API creates one only from a predefined voice name (`mtmd_gen_audio_get_voice`). Chatterbox voices also carry gen-side tensors (`voice.<name>.ref_tokens`, `ref_feat`, `ref_spk`) that are read from the GGUF by name. No call builds a voice bitmap from an encoded clip.

## Mitigations

For users, today:

- Trim the reference to 10-15 s. Encode drops to 2.2-3.7 s per call. Past 15 s, only the averaged x-vector and voice-encoder embeddings use the extra audio. Quality impact not measured.
- Synthesize paragraphs instead of sentences, to reduce `set_input` calls.

Upstream candidates:

- An API that returns a voice bitmap, with its gen-side data, from an encoded clip, so it can be encoded once.
- Run the voice-encoder LSTM as a ggml graph, or batch the windows with `float` accumulators.
- Cap entry 0 at 10 s. From memory, the reference Chatterbox computes the s3gen x-vector from the 10 s decoder clip. Not checked against the source.

## Reproducer

```python
import sys, time
import cyllama.llama.llama_cpp as cy

cy.ggml_backend_load_all()
mp = cy.LlamaModelParams()
mp.n_gpu_layers = 99
model = cy.LlamaModel("models/chatterbox-turbo-Q8_0.gguf", mp)
mctx = cy.MtmdContext("models/mmproj-chatterbox-turbo-F16.gguf", model)
for ref in sys.argv[1:]:
    bmp = cy.MtmdBitmap.from_file(mctx, ref)
    t0 = time.perf_counter()
    chunks = mctx.tokenize(mctx.marker, [bmp], True, True)
    t1 = time.perf_counter()
    audio = next(chunks[i] for i in range(len(chunks)) if chunks[i].type == cy.MtmdInputChunkType.AUDIO)
    mctx.encode_chunk(audio)
    t2 = time.perf_counter()
    print(f"{ref}: preprocess {t1 - t0:.2f}s encode {t2 - t1:.2f}s n_tokens {audio.n_tokens}")
```

Cut test references with `ffmpeg -i ref.wav -t 10 ref10.wav`.
