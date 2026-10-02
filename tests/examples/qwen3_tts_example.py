#!/usr/bin/env python3
"""
Qwen3-TTS text-to-speech example using cyllama.

Generates speech with a Qwen3-TTS backbone and its mmproj (code predictor
plus vocoder) and writes a 24 kHz mono 16-bit WAV file.

Requirements:
    - models/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
    - models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
      (from https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF)

Usage:
    python tests/examples/qwen3_tts_example.py [--prompt TEXT] [--output PATH]

Examples:
    # Default prompt, writes build/qwen3_tts.wav
    python tests/examples/qwen3_tts_example.py

    # Custom prompt and language
    python tests/examples/qwen3_tts_example.py -p "Bonjour tout le monde" --lang fr

    # Clone a voice from reference audio
    python tests/examples/qwen3_tts_example.py --speaker-file speaker.wav
"""

import argparse
import os
import sys
import time

from cyllama.llama.tts import MtmdTTSGenerator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate speech with Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", "--model", default="models/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", help="Backbone model")
    parser.add_argument(
        "-mm", "--mmproj", default="models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", help="mmproj file"
    )
    parser.add_argument("-p", "--prompt", default="I say, how about some eggs?", help="Text to speak")
    parser.add_argument("-o", "--output", default="build/qwen3_tts.wav", help="Output WAV file")
    parser.add_argument("--lang", default="en", help="Language code: zh en de it pt es ja ko fr ru")
    parser.add_argument("--speaker-file", help="Reference audio (wav, mp3) for voice cloning")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed")
    args = parser.parse_args()

    for path in (args.model, args.mmproj):
        if not os.path.exists(path):
            print(f"Missing model file: {path}", file=sys.stderr)
            return 1
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with MtmdTTSGenerator(
        args.model,
        args.mmproj,
        lang=args.lang,
        speaker_file=args.speaker_file,
        seed=args.seed,
    ) as tts:
        start = time.perf_counter()
        wav = tts.generate(args.prompt, args.output)
        elapsed = time.perf_counter() - start
        # 44-byte WAV header, 2 bytes per sample
        seconds = (len(wav) - 44) / 2 / tts.sample_rate

    print(f"Wrote {args.output}: {seconds:.2f}s of audio in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
