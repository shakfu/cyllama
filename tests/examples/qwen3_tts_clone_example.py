#!/usr/bin/env python3
"""
Qwen3-TTS voice cloning example using cyllama.

Speaks the same text twice, once in the model's default voice and once in
the voice of a reference recording, so the two WAV files can be compared.
The reference passes through the mmproj speaker encoder; any length of
clean speech works, a few seconds or more is best.

The default reference, tests/samples/jfk.wav, is 11 seconds of
John F. Kennedy's 1961 inaugural address (public domain).

Requirements:
    - models/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
    - models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
      (from https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF)

Usage:
    python tests/examples/qwen3_tts_clone_example.py [--speaker-file PATH] [--prompt TEXT]

Writes:
    build/qwen3_tts_default.wav   default voice
    build/qwen3_tts_cloned.wav    reference voice
"""

import argparse
import gc
import os
import sys

from cyllama.llama.tts import MtmdTTSGenerator


def speak(args: argparse.Namespace, speaker_file: str | None, output: str) -> None:
    with MtmdTTSGenerator(args.model, args.mmproj, lang=args.lang, speaker_file=speaker_file, seed=args.seed) as tts:
        wav = tts.generate(args.prompt, output)
        # 44-byte WAV header, 2 bytes per sample
        seconds = (len(wav) - 44) / 2 / tts.sample_rate
    gc.collect()
    print(f"Wrote {output}: {seconds:.2f}s ({'voice of ' + speaker_file if speaker_file else 'default voice'})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone a voice with Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", "--model", default="models/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", help="Backbone model")
    parser.add_argument(
        "-mm", "--mmproj", default="models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", help="mmproj file"
    )
    parser.add_argument("--speaker-file", default="tests/samples/jfk.wav", help="Reference speech (wav, mp3) to clone")
    parser.add_argument(
        "-p", "--prompt", default="We choose to make speech from text, not because it is easy.", help="Text to speak"
    )
    parser.add_argument("--lang", default="en", help="Language code: zh en de it pt es ja ko fr ru")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed")
    parser.add_argument("--outdir", default="build", help="Output directory")
    args = parser.parse_args()

    for path in (args.model, args.mmproj, args.speaker_file):
        if not os.path.exists(path):
            print(f"Missing file: {path}", file=sys.stderr)
            return 1
    os.makedirs(args.outdir, exist_ok=True)

    speak(args, None, os.path.join(args.outdir, "qwen3_tts_default.wav"))
    speak(args, args.speaker_file, os.path.join(args.outdir, "qwen3_tts_cloned.wav"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
