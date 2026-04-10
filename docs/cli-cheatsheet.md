# CLI Cheatsheet

Complete reference for all cyllama command-line interfaces.

## Two Ways to Run

| Form | Description |
|------|-------------|
| `cyllama <command>` | Unified CLI (recommended) |
| `python -m cyllama.<module>` | Direct sub-module invocation |

The unified CLI delegates to sub-module CLIs for `server`, `transcribe`, `tts`, `sd`, `agent`, and `memory`. The high-level commands (`generate`, `chat`, `embed`, `rag`) are implemented directly in the unified CLI using the Python API.

---

## cyllama generate

**Alias**: `gen`

Generate text from a prompt.

```bash
cyllama gen -m models/llama.gguf -p "What is Python?" --stream
cyllama gen -m models/llama.gguf -f prompt.txt --json
echo "Hello" | cyllama gen -m models/llama.gguf
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `-p, --prompt` | string | | Text prompt |
| `-f, --file` | string | | Read prompt from file (or stdin if `-p` and `-f` omitted) |
| `-n, --max-tokens` | int | 512 | Maximum tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top-k` | int | 40 | Top-k sampling |
| `--top-p` | float | 0.95 | Nucleus sampling |
| `--min-p` | float | 0.05 | Minimum probability threshold |
| `--repeat-penalty` | float | 1.1 | Repetition penalty |
| `-ngl, --n-gpu-layers` | int | 99 | GPU layers to offload |
| `-c, --ctx-size` | int | (auto) | Context window size |
| `--seed` | int | -1 | Random seed (-1 = random) |
| `--stream` | flag | | Stream tokens to stdout |
| `--json` | flag | | Output as JSON with stats |
| `--verbose` | flag | | Verbose output |

---

## cyllama chat

Interactive or single-turn chat with a model.

```bash
cyllama chat -m models/llama.gguf                          # interactive
cyllama chat -m models/llama.gguf -p "Explain gravity"     # single-turn
cyllama chat -m models/llama.gguf -s "You are a physicist" # with system prompt
cyllama chat -m models/llama.gguf -n 1024 --template chatml
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `-p, --prompt` | string | | Single-turn message (omit for interactive) |
| `-s, --system` | string | | System prompt |
| `--template` | string | | Chat template (e.g. chatml, llama3) |
| `-n, --max-tokens` | int | 512 | Maximum tokens per response |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top-k` | int | 40 | Top-k sampling |
| `--top-p` | float | 0.95 | Nucleus sampling |
| `--min-p` | float | 0.05 | Minimum probability threshold |
| `--repeat-penalty` | float | 1.1 | Repetition penalty |
| `-ngl, --n-gpu-layers` | int | 99 | GPU layers to offload |
| `-c, --ctx-size` | int | 2048 | Context window size |
| `--seed` | int | -1 | Random seed (-1 = random) |
| `--stream` | flag | | Stream tokens to stdout |
| `--json` | flag | | Output as JSON with stats |
| `--verbose` | flag | | Verbose output |

---

## cyllama embed

Generate embeddings and compute similarity.

```bash
cyllama embed -m models/bge-small.gguf -t "hello world" -t "another text"
cyllama embed -m models/bge-small.gguf -f texts.txt
cyllama embed -m models/bge-small.gguf --dim
cyllama embed -m models/bge-small.gguf --similarity "machine learning" -f corpus.txt --threshold 0.5
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF embedding model |
| `-t, --text` | string | | Text to embed (repeatable) |
| `-f, --file` | string | | Read texts from file (one per line) |
| `-ngl, --n-gpu-layers` | int | 99 | GPU layers to offload |
| `-c, --ctx-size` | int | 512 | Context window size |
| `--pooling` | choice | mean | Pooling strategy: `mean`, `cls`, `last` |
| `--no-normalize` | flag | | Skip L2 normalization |
| `--dim` | flag | | Print embedding dimensions and exit |
| `--similarity` | string | | Rank texts by similarity to this query |
| `--threshold` | float | 0.0 | Minimum similarity score to display |

---

## cyllama rag

Retrieval-augmented generation over local documents.

```bash
# Single query, ephemeral in-memory index (default)
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -p "How do I configure X?" --stream

# Interactive mode (omit -p)
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -f guide.md -f faq.md --sources

# With system instruction and retrieval tuning
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -d docs/ -s "Answer in one paragraph" -k 3 --threshold 0.4
```

### Persistent index (`--db`)

By default the vector index is held in memory and rebuilt on every
run. For corpora large enough that re-embedding is expensive, pass
`--db PATH` to persist the index to a SQLite file and reuse it on
subsequent runs:

```bash
# First run: index the corpus into a file
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -f docs/corpus.txt --db ./rag.db

# Subsequent runs: reuse the index, no -f needed
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    --db ./rag.db

# Append more files to the existing index
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -f docs/new.txt --db ./rag.db

# Rebuild from scratch (e.g. after switching embedding model)
cyllama rag -m models/llama.gguf -e models/bge-small.gguf \
    -f docs/corpus.txt --db ./rag.db --rebuild
```

The DB records the embedding model fingerprint (basename + file size),
chunk size, and chunk overlap when it's first created. Reopening with a
different embedding model, vector metric, or chunking config raises a
clear error rather than silently producing wrong rankings — pass
`--rebuild` to recreate the index against the new config.

### Corpus deduplication (automatic)

Each indexed file is hashed (md5 of its raw bytes) and the hash is
recorded in the DB's `embeddings_sources` table. Re-running with the
same `-f` files is a no-op on the indexing side — the files are
silently skipped and the user goes straight to query mode. The status
line surfaces the skip count:

```bash
$ cyllama rag -m ... -e ... -f corpus.txt --db ./rag.db
128 chunks indexed -> ./rag.db                          # first run

$ cyllama rag -m ... -e ... -f corpus.txt --db ./rag.db
reusing 128 chunks from ./rag.db (1 unchanged)          # second run, dedup fired

$ cyllama rag -m ... -e ... -f corpus.txt -f new.txt --db ./rag.db
3 new chunks appended to ./rag.db (128 existing, 131 total) (1 unchanged)
```

Editing a file in place (same basename, different content) is detected
as a hash mismatch and refused with a clear error message — rename the
file (treat it as a new source) or use `--rebuild` to recreate the
whole index from the new content. This prevents the index from silently
ending up with two versions of the same logical source.

`add_texts` (the directory-loading path used by `-d`) deduplicates the
same way, using a `text:<hash-prefix>` synthetic label since text
strings don't have a meaningful name.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF generation model |
| `-e, --embedding-model` | string | (required) | Path to GGUF embedding model |
| `-f, --file` | string | | File to index (repeatable) |
| `-d, --dir` | string | | Directory to index (repeatable) |
| `--glob` | string | `**/*` | Glob pattern for directory loading |
| `-p, --prompt` | string | | Single query (omit for interactive) |
| `-s, --system` | string | | System instruction (system prompt in chat mode) |
| `-n, --max-tokens` | int | 512 | Maximum tokens to generate |
| `--temperature` | float | 0.7 | Generation temperature |
| `-k, --top-k` | int | 5 | Number of chunks to retrieve |
| `--threshold` | float | (none) | Minimum similarity threshold |
| `-ngl, --n-gpu-layers` | int | 99 | GPU layers to offload |
| `--stream` | flag | | Stream output tokens |
| `--sources` | flag | | Show source chunks with similarity scores |
| `--db` | string | (none) | Path to persistent SQLite vector store |
| `--rebuild` | flag | | Delete `--db` and recreate from `-f`/`-d` |
| `--no-chat-template` | flag | | Use raw-completion path instead of chat template |
| `--show-think` | flag | | Show `<think>` reasoning blocks (default: stripped) |
| `--repetition-threshold` | int | 2 | Stop generation after n-gram repeats this many times (0 disables) |
| `--repetition-ngram` | int | 5 | Word-level n-gram length for repetition detection |
| `--repetition-window` | int | 300 | Number of recent words tracked by the repetition detector |

At least one document source (`-f`/`-d`) **or** an existing `--db` is required.

---

## cyllama server

Start an OpenAI-compatible HTTP server.

**Also**: `python -m cyllama.llama.server`

```bash
cyllama server -m models/llama.gguf
cyllama server -m models/llama.gguf --port 9090 --server-type python
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `--host` | string | 127.0.0.1 | Host to bind to |
| `--port` | int | 8080 | Port to listen on |
| `--ctx-size` | int | 2048 | Context window size |
| `--gpu-layers` | int | -1 | GPU layers to offload |
| `--n-parallel` | int | 1 | Number of parallel processing slots |
| `--server-type` | choice | embedded | Server implementation: `python` or `embedded` |

---

## cyllama transcribe

Transcribe audio files using whisper.cpp.

**Also**: `python -m cyllama.whisper.cli`

```bash
cyllama transcribe -m models/ggml-base.en.bin -f audio.wav
cyllama transcribe -m models/ggml-base.en.bin -f audio.wav -l auto -tr
cyllama transcribe -m models/ggml-base.en.bin -f audio.wav -osrt -o output
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f, --file` | string | | Input audio file (repeatable) |
| `-o, --output` | string | | Output file path (repeatable) |
| `-m, --model` | string | | Path to whisper model |
| `-t, --threads` | int | (auto) | Number of threads |
| `-p, --processors` | int | (auto) | Number of processors |
| `-l, --language` | string | en | Language code (or `auto`) |
| `-tr, --translate` | flag | | Translate to English |
| `-dl, --detect-language` | flag | | Detect language |

**Timing**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-ot, --offset-t` | int | 0 | Time offset in milliseconds |
| `-on, --offset-n` | int | 0 | Segment offset |
| `-d, --duration` | int | 0 | Duration in milliseconds |
| `-mc, --max-context` | int | -1 | Maximum context |
| `-ml, --max-len` | int | 0 | Maximum segment length |

**Sampling**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-bo, --best-of` | int | 5 | Best of N samples |
| `-bs, --beam-size` | int | 5 | Beam search size |
| `-wt, --word-thold` | float | 0.01 | Word probability threshold |
| `-et, --entropy-thold` | float | 2.40 | Entropy threshold |
| `-lpt, --logprob-thold` | float | -1.00 | Log probability threshold |
| `-tp, --temperature` | float | 0.0 | Temperature |
| `-tpi, --temperature-inc` | float | 0.2 | Temperature increment |

**Output formats** (flags, all off by default):

| Flag | Format |
|------|--------|
| `-otxt, --output-txt` | Plain text |
| `-ovtt, --output-vtt` | WebVTT |
| `-osrt, --output-srt` | SRT subtitles |
| `-owts, --output-wts` | Word timestamps |
| `-ocsv, --output-csv` | CSV |
| `-oj, --output-json` | JSON |
| `-ojf, --output-json-full` | Full JSON |
| `-olrc, --output-lrc` | LRC lyrics |

**Display**:

| Flag | Description |
|------|-------------|
| `-np, --no-prints` | Suppress output |
| `-ps, --print-special` | Print special tokens |
| `-pc, --print-colors` | Colorized output |
| `-pp, --print-progress` | Show progress |
| `-nt, --no-timestamps` | Omit timestamps |
| `-ng, --no-gpu` | Disable GPU |

---

## cyllama tts

Text-to-speech synthesis.

**Also**: `python -m cyllama.llama.tts`

```bash
cyllama tts -m models/tts.gguf -mv models/vocoder.gguf -p "Hello world"
cyllama tts -m models/tts.gguf -mv models/vocoder.gguf -p "Hello" -o speech.wav
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to text-to-codes model |
| `-mv, --vocoder-model` | string | (required) | Path to codes-to-speech model |
| `-p, --prompt` | string | (required) | Text to synthesize |
| `-o, --output` | string | output.wav | Output WAV file |
| `-c, --context` | int | 8192 | Context size |
| `-b, --batch` | int | 8192 | Batch size |
| `-ngl, --n-gpu-layers` | int | 99 | GPU layers to offload |
| `-n, --n-predict` | int | 4096 | Max tokens to predict |
| `--speaker-file` | string | | Speaker profile JSON file |
| `--use-guide-tokens` | flag | (on) | Use guide tokens (prevents hallucinations) |
| `--no-guide-tokens` | flag | | Disable guide tokens |

---

## cyllama sd

Stable Diffusion image and video generation.

**Also**: `python -m cyllama.sd`

### Subcommands

- `txt2img` (alias: `generate`) -- Text to image
- `img2img` -- Image to image
- `inpaint` -- Inpainting with mask
- `controlnet` -- ControlNet guided generation
- `video` -- Video generation (Wan, CogVideoX)
- `upscale` -- ESRGAN upscaling
- `convert` -- Model format conversion
- `info` -- System info

### txt2img / generate

```bash
cyllama sd txt2img -m models/sd.gguf -p "a sunset" -o sunset.png
cyllama sd txt2img --diffusion-model models/z_image.gguf --llm models/qwen.gguf \
    --vae models/ae.safetensors -p "a cat" -H 1024 -W 512 --diffusion-fa
```

### img2img

```bash
cyllama sd img2img -m models/sd.gguf -i input.png -p "oil painting style" --strength 0.7
```

### inpaint

```bash
cyllama sd inpaint -m models/sd.gguf -i input.png --mask mask.png -p "fill with flowers"
```

### controlnet

```bash
cyllama sd controlnet -m models/sd.gguf --control-net models/cn.gguf \
    --control-image edges.png -p "a house" --control-strength 0.9
```

### video

```bash
cyllama sd video -m models/wan.gguf -p "a cat walking" --video-frames 16 --fps 24
```

### Common Model Options

All generation subcommands (`txt2img`, `img2img`, `inpaint`, `controlnet`) share:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | | Path to model (or use `--diffusion-model`) |
| `--diffusion-model` | string | | Path to diffusion model |
| `--high-noise-diffusion-model` | string | | Path to high-noise diffusion model |
| `--vae` | string | | Path to VAE model |
| `--taesd` | string | | Path to TAESD model (fast preview) |
| `--clip-l` | string | | Path to CLIP-L model |
| `--clip-g` | string | | Path to CLIP-G model |
| `--clip-vision` | string | | Path to CLIP vision model |
| `--t5xxl` | string | | Path to T5-XXL model |
| `--llm` | string | | Path to LLM text encoder |
| `--llm-vision` | string | | Path to LLM vision encoder |
| `--tensor-type-rules` | string | | Tensor type rules |

### Common Generation Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-p, --prompt` | string | (required) | Text prompt |
| `-n, --negative` | string | | Negative prompt |
| `-o, --output` | string | output.png | Output path |
| `-W, --width` | int | 512 | Image width |
| `-H, --height` | int | 512 | Image height |
| `--steps` | int | 20 | Sampling steps |
| `--cfg-scale` | float | 7.0 | Classifier-free guidance scale |
| `-s, --seed` | int | -1 | Random seed (-1 = random) |
| `-b, --batch` | int | 1 | Batch count |
| `--clip-skip` | int | -1 | CLIP skip layers |

### Subcommand-Specific Options

**img2img / inpaint**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i, --init-img` | string | (required) | Path to init image |
| `--strength` | float | 0.75 (img2img), 1.0 (inpaint) | Denoising strength (0.0-1.0) |
| `--mask` | string | (inpaint only, required) | Path to mask image (white=inpaint) |

**controlnet**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--control-net` | string | (required) | Path to ControlNet model |
| `--control-image` | string | (required) | Path to control image |
| `--control-strength` | float | 0.9 | Control strength (0.0-1.0+) |
| `--canny` | flag | | Apply Canny edge detection to control image |

**video**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--video-frames` | int | 16 | Number of video frames |
| `--fps` | int | 24 | Frames per second for output |
| `-i, --init-img` | string | | Path to init image |
| `--end-img` | string | | Path to end image (for flf2v) |
| `--moe-boundary` | float | 0.875 | MoE boundary for Wan2.2 |

### Sampler Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sampler` | string | | Method: `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `tcd` |
| `--scheduler` | string | | Schedule: `discrete`, `karras`, `exponential`, `ays`, `gits` |
| `--eta` | float | 0.0 | Eta for DDIM/TCD samplers |
| `--rng` | choice | | RNG type: `std_default`, `cuda`, `cpu` |
| `--sampler-rng` | choice | | Sampler RNG type |
| `--prediction` | choice | | Prediction type: `eps`, `v`, `edm_v`, `sd3_flow`, `flux_flow`, `flux2_flow` |

### Guidance Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--slg-scale` | float | 0.0 | Skip layer guidance scale (0=disabled, 2.5 good for SD3.5) |
| `--skip-layer-start` | float | 0.01 | SLG enabling point |
| `--skip-layer-end` | float | 0.2 | SLG disabling point |
| `--guidance` | float | | Distilled guidance scale (for FLUX) |
| `--img-cfg-scale` | float | | Image CFG scale (inpaint / instruct-pix2pix) |

### Memory Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t, --threads` | int | -1 (auto) | Number of threads |
| `--offload-to-cpu` | flag | | Offload weights to CPU (low VRAM) |
| `--clip-on-cpu` | flag | | Keep CLIP on CPU |
| `--vae-on-cpu` | flag | | Keep VAE on CPU |
| `--control-net-cpu` | flag | | Keep ControlNet on CPU |
| `--diffusion-fa` | flag | | Flash attention in diffusion model |
| `--diffusion-conv-direct` | flag | | Direct convolution in diffusion |
| `--vae-conv-direct` | flag | | Direct convolution in VAE |

### VAE Tiling Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--vae-tiling` | flag | | Enable VAE tiling for large images |
| `--vae-tile-size` | string | 512x512 | VAE tile size |
| `--vae-tile-overlap` | float | 0.5 | VAE tile overlap fraction |

### Preview Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--preview` | choice | none | Preview mode: `none`, `proj`, `tae`, `vae` |
| `--preview-path` | string | ./preview.png | Preview output path |
| `--preview-interval` | int | 1 | Preview interval (steps) |
| `--preview-noisy` | flag | | Preview noisy instead of denoised |
| `--taesd-preview-only` | flag | | Use TAESD only for preview, not final decode |

### Misc Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lora-apply-mode` | choice | | LoRA mode: `auto`, `immediately`, `at_runtime` |
| `--flow-shift` | float | | Flow shift for SD3.x/Wan models |
| `--chroma-disable-dit-mask` | flag | | Disable DiT mask for Chroma |
| `--chroma-enable-t5-mask` | flag | | Enable T5 mask for Chroma |
| `--chroma-t5-mask-pad` | int | | T5 mask pad for Chroma |
| `-v, --verbose` | flag | | Verbose output |
| `--progress` | flag | | Show progress bar |

### upscale

```bash
cyllama sd upscale -m models/esrgan.gguf -i input.png -o output.png
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to ESRGAN model |
| `-i, --input` | string | (required) | Input image path |
| `-o, --output` | string | (required) | Output image path |
| `-f, --factor` | int | (model default) | Upscale factor |
| `-r, --repeats` | int | 1 | Upscale repeats |
| `-t, --threads` | int | -1 (auto) | Number of threads |
| `--offload-to-cpu` | flag | | Offload to CPU |
| `-v, --verbose` | flag | | Verbose output |

### convert

```bash
cyllama sd convert -i models/sd.safetensors -o models/sd.gguf -t q8_0
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i, --input` | string | (required) | Input model path |
| `-o, --output` | string | (required) | Output model path |
| `-t, --type` | string | f16 | Output type: `f32`, `f16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, etc. |
| `--vae` | string | | Path to VAE model |
| `--tensor-type-rules` | string | | Tensor type rules |
| `-v, --verbose` | flag | | Verbose output |

### info

```bash
cyllama sd info
```

No arguments. Prints stable-diffusion.cpp system info and available backends.

---

## cyllama agent

Agent framework CLI.

**Also**: `python -m cyllama.agents.cli`

### Subcommands

#### run

Run a ReAct agent with optional tools.

```bash
cyllama agent run -m models/llama.gguf -p "What is 25 * 4?"
cyllama agent run -m models/llama.gguf -f task.txt --enable-shell
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `-p, --prompt` | string | | Prompt to run |
| `-f, --prompt-file` | string | | File containing the prompt |
| `--system-prompt` | string | | Custom system prompt |
| `--max-iterations` | int | 10 | Maximum agent iterations |
| `--enable-shell` | flag | | Enable shell command tool |
| `-v, --verbose` | flag | | Verbose output |

#### acp

Run an agent with MCP (Model Context Protocol) servers.

```bash
cyllama agent acp -m models/llama.gguf --mcp-stdio "calc:python:calc_server.py"
cyllama agent acp -m models/llama.gguf --mcp-http "api:http://localhost:3000"
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `--mcp-stdio` | string | | MCP server via stdio `name:command:arg1:...` (repeatable) |
| `--mcp-http` | string | | MCP server via HTTP `name:url` (repeatable) |
| `--session-storage` | choice | memory | Session storage: `memory`, `file`, `sqlite` |
| `--session-path` | string | | Path for file/sqlite session storage |
| `--system-prompt` | string | | Custom system prompt |
| `--max-iterations` | int | 10 | Maximum agent iterations |
| `-v, --verbose` | flag | | Verbose output |

#### mcp-test

Test MCP server connectivity and tool listing.

```bash
cyllama agent mcp-test --stdio "calc:python:calc_server.py"
cyllama agent mcp-test --http "api:http://localhost:3000" --call-tool "add:{\"a\":1,\"b\":2}"
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stdio` | string | | MCP server via stdio `name:command:arg1:...` |
| `--http` | string | | MCP server via HTTP `name:url` |
| `--call-tool` | string | | Call a tool `tool_name:json_args` |
| `-v, --verbose` | flag | | Verbose output |

---

## cyllama memory

Estimate GPU memory requirements for a model.

**Also**: `python -m cyllama.memory`

```bash
cyllama memory models/llama.gguf
cyllama memory models/llama.gguf --gpu-memory 8192
cyllama memory models/llama.gguf --gpu-memory "4096,4096" --ctx-size 4096
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `model_path` | positional | (required) | Path to GGUF model file |
| `--gpu-memory` | string | | Available GPU memory in MB (multi-GPU: `"4096,4096"`) |
| `--ctx-size` | int | 2048 | Context size |
| `--batch-size` | int | 1 | Batch size |
| `--n-parallel` | int | 1 | Number of parallel sequences |
| `--kv-cache-type` | choice | f16 | KV cache precision: `f16`, `f32` |
| `--overview-only` | flag | | Show only memory overview |
| `--verbose` | flag | | Verbose output |

---

## cyllama info

Show build configuration and available backends.

```bash
cyllama info
```

No arguments.

---

## cyllama version

Print version number.

```bash
cyllama version
```

No arguments.

---

## Advanced: python -m cyllama.llama.cli

Low-level llama.cpp CLI with full parameter control. Not exposed through the unified `cyllama` command.

```bash
python -m cyllama.llama.cli -m models/llama.gguf -p "Hello" -n 128
python -m cyllama.llama.cli -m models/llama.gguf -cnv   # conversation mode
python -m cyllama.llama.cli -m models/llama.gguf -i      # interactive mode
```

### Model Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m, --model` | string | (required) | Path to GGUF model |
| `--lora` | string | | LoRA adapter path (implies `--no-mmap`) |
| `--lora-scaled` | PATH SCALE | | LoRA adapter with custom scaling |
| `--lora-base` | string | | Base model for LoRA layers |

### Context Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-c, --ctx-size` | int | 4096 | Context size |
| `-b, --batch-size` | int | 2048 | Batch size for prompt processing |
| `--ubatch` | int | 512 | Physical batch size |
| `--keep` | int | 0 | Tokens to keep from initial prompt |
| `--chunks` | int | -1 | Max chunks to process (-1 = unlimited) |
| `--grp-attn-n` | int | 1 | Group-attention factor |
| `--grp-attn-w` | int | 512 | Group-attention width |

### GPU Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-ngl, --n-gpu-layers` | int | -1 | GPU layers (-1 = default) |
| `--main-gpu` | int | 0 | GPU for scratch and small tensors |
| `--tensor-split` | string | | Tensor split ratios across GPUs |
| `--split-mode` | choice | layer | Split mode: `none`, `layer`, `row` |

### CPU Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t, --threads` | int | 4 | Compute threads |
| `-tb, --threads-batch` | int | 4 | Batch processing threads |
| `--no-mmap` | flag | | Do not memory-map model |
| `--mlock` | flag | | Lock model in RAM |
| `--numa` | flag | | NUMA optimizations |

### Generation Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-n, --n-predict` | int | -1 | Tokens to predict (-1 = inf, -2 = fill context) |
| `--top-k` | int | 40 | Top-k sampling |
| `--top-p` | float | 0.95 | Top-p sampling |
| `--min-p` | float | 0.05 | Min-p sampling |
| `--tfs` | float | 1.0 | Tail free sampling |
| `--typical` | float | 1.0 | Locally typical sampling |
| `--repeat-last-n` | int | 64 | Tokens considered for repeat penalty |
| `--repeat-penalty` | float | 1.1 | Repeat penalty |
| `--frequency-penalty` | float | 0.0 | Frequency penalty |
| `--presence-penalty` | float | 0.0 | Presence penalty |
| `--mirostat` | int | 0 | Mirostat mode (0=off, 1, 2) |
| `--mirostat-lr` | float | 0.1 | Mirostat learning rate |
| `--mirostat-ent` | float | 5.0 | Mirostat target entropy |
| `-l, --logit-bias` | string | | Logit bias (`TOKEN+BIAS` or `TOKEN-BIAS`) |
| `--temp` | float | 0.8 | Temperature |
| `--seed` | int | -1 | Random seed |

### RoPE Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rope-freq-base` | float | 0.0 | RoPE base frequency |
| `--rope-freq-scale` | float | 0.0 | RoPE frequency scale |
| `--yarn-ext-factor` | float | -1.0 | YaRN extrapolation mix |
| `--yarn-attn-factor` | float | 1.0 | YaRN magnitude scale |
| `--yarn-beta-fast` | float | 32.0 | YaRN low correction dim |
| `--yarn-beta-slow` | float | 1.0 | YaRN high correction dim |
| `--yarn-orig-ctx` | int | 0 | YaRN original context length |

### Prompt Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-p, --prompt` | string | | Prompt text |
| `-f, --file` | string | | Prompt file |
| `-e, --escape` | flag | | Process escape sequences |
| `--prompt-cache` | string | | Prompt cache file path |
| `--prompt-cache-all` | flag | | Save/load full prompt cache |
| `--prompt-cache-ro` | flag | | Read-only prompt cache |
| `--verbose-prompt` | flag | | Print prompt before generation |

### Interactive / Chat Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i, --interactive` | flag | | Interactive mode |
| `--interactive-first` | flag | | Interactive mode, wait for input immediately |
| `-ins, --instruct` | flag | | Instruction mode (Alpaca-style) |
| `-cnv, --conversation` | flag | | Conversation mode |
| `--no-cnv` | flag | | Disable conversation mode |
| `--single-turn` | flag | | Single-turn conversation |
| `--chat-template` | string | | Chat template name |
| `--sys, --system-prompt` | string | | System prompt |
| `--use-jinja` | flag | | Use Jinja2 for chat templates |
| `-r, --reverse-prompt` | string | | Stop at this string, return control |
| `--in-prefix` | string | | Prefix for user inputs |
| `--in-suffix` | string | | Suffix for user inputs |
| `--in-prefix-bos` | flag | | BOS before user inputs |
| `--multiline-input` | flag | | Allow multiline input |
| `--simple-io` | flag | | Simplified I/O for subprocesses |
| `--color` | flag | | Colorized output |

### Other Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--embedding` | flag | | Embedding mode |
| `--display-prompt` | flag | | Print prompt |
| `--no-display-prompt` | flag | | Don't print prompt |
| `--ctx-shift` | flag | | Enable context shifting |
| `--no-cache` | flag | | Disable KV cache |
| `--no-kv-offload` | flag | | Disable KV offload |
| `--no-flash-attn` | flag | | Disable flash attention |
| `--no-perf` | flag | | Disable performance metrics |
| `--timing` | flag | | Print timing info |
| `--log-disable` | flag | | Disable all logs |
| `--log-enable` | flag | | Enable logs |
| `--log-file` | string | | Log filename |
| `--log-new` | flag | | Don't resume previous log |
| `--log-append` | flag | | Append to existing log |
