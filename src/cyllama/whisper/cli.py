#!/usr/bin/env python3
"""
Whisper CLI - Python wrapper for whisper.cpp CLI functionality.

This module provides a command-line interface equivalent to the whisper.cpp CLI,
using the cyllama Whisper wrapper for transcription functionality.
"""

import argparse
import sys
import os
import json
import wave
import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading

# Import the whisper module
try:
    from . import whisper_cpp as wh
except ImportError:
    # When running as script, use absolute import
    import whisper_cpp as wh


class WhisperParams:
    """Parameters for Whisper CLI, equivalent to whisper_params struct."""

    def __init__(self):
        # Basic parameters
        self.n_threads = min(4, threading.active_count())
        self.n_processors = 1
        self.offset_t_ms = 0
        self.offset_n = 0
        self.duration_ms = 0
        self.progress_step = 5
        self.max_context = -1
        self.max_len = 0
        self.best_of = 5  # Default from whisper
        self.beam_size = 5  # Default from whisper
        self.audio_ctx = 0

        # Thresholds
        self.word_thold = 0.01
        self.entropy_thold = 2.40
        self.logprob_thold = -1.00
        self.no_speech_thold = 0.6
        self.grammar_penalty = 100.0
        self.temperature = 0.0
        self.temperature_inc = 0.2

        # Boolean flags
        self.debug_mode = False
        self.translate = False
        self.detect_language = False
        self.diarize = False
        self.tinydiarize = False
        self.split_on_word = False
        self.no_fallback = False
        self.output_txt = False
        self.output_vtt = False
        self.output_srt = False
        self.output_wts = False
        self.output_csv = False
        self.output_jsn = False
        self.output_jsn_full = False
        self.output_lrc = False
        self.no_prints = False
        self.print_special = False
        self.print_colors = False
        self.print_confidence = False
        self.print_progress = False
        self.no_timestamps = False
        self.log_score = False
        self.use_gpu = True
        self.flash_attn = False
        self.suppress_nst = False

        # String parameters
        self.language = "en"
        self.prompt = ""
        self.model = "models/ggml-base.en.bin"
        self.grammar = ""
        self.grammar_rule = ""
        self.tdrz_speaker_turn = " [SPEAKER_TURN]"
        self.suppress_regex = ""
        self.openvino_encode_device = "CPU"
        self.dtw = ""

        # File lists
        self.fname_inp: List[str] = []
        self.fname_out: List[str] = []

        # VAD parameters
        self.vad = False
        self.vad_model = ""
        self.vad_threshold = 0.5
        self.vad_min_speech_duration_ms = 250
        self.vad_min_silence_duration_ms = 100
        self.vad_max_speech_duration_s = float('inf')
        self.vad_speech_pad_ms = 30
        self.vad_samples_overlap = 0.1


def load_wav_file(filepath: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return samples as float32 array and sample rate."""
    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.readframes(-1)
        sound_info = wav_file.getparams()

        # Convert to float32 normalized to [-1, 1]
        if sound_info.sampwidth == 1:
            fmt = f'{len(frames)}B'
            samples = struct.unpack(fmt, frames)
            samples = [(s - 128) / 128.0 for s in samples]
        elif sound_info.sampwidth == 2:
            fmt = f'{len(frames) // 2}h'
            samples = struct.unpack(fmt, frames)
            samples = [s / 32768.0 for s in samples]
        elif sound_info.sampwidth == 3:
            # 24-bit samples
            samples = []
            for i in range(0, len(frames), 3):
                if i + 2 < len(frames):
                    sample = int.from_bytes(frames[i:i+3], byteorder='little', signed=True)
                    samples.append(sample / 8388608.0)  # 2^23
        elif sound_info.sampwidth == 4:
            fmt = f'{len(frames) // 4}i'
            samples = struct.unpack(fmt, frames)
            samples = [s / 2147483648.0 for s in samples]  # 2^31
        else:
            raise ValueError(f"Unsupported sample width: {sound_info.sampwidth}")

        return np.array(samples, dtype=np.float32), sound_info.framerate


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Simple resampling using linear interpolation."""
    if orig_sr == target_sr:
        return samples

    # Calculate the ratio
    ratio = orig_sr / target_sr
    new_length = int(len(samples) / ratio)

    # Create new indices
    old_indices = np.arange(len(samples))
    new_indices = np.linspace(0, len(samples) - 1, new_length)

    # Interpolate
    resampled = np.interp(new_indices, old_indices, samples)
    return resampled.astype(np.float32)


def escape_string_json(text: str) -> str:
    """Escape string for JSON output."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def escape_string_csv(text: str) -> str:
    """Escape string for CSV output."""
    return text.replace('"', '""')


def format_timestamp(t: int, always_include_hours: bool = False, decimal_marker: str = ',') -> str:
    """Format timestamp in HH:MM:SS,mmm format."""
    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec -= hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec -= min * (1000 * 60)
    sec = msec // 1000
    msec -= sec * 1000

    if always_include_hours or hr > 0:
        return f"{hr:02d}:{min:02d}:{sec:02d}{decimal_marker}{msec:03d}"
    else:
        return f"{min:02d}:{sec:02d}{decimal_marker}{msec:03d}"


def output_txt(ctx: wh.WhisperContext, params: WhisperParams, output_file) -> None:
    """Output transcription as plain text."""
    n_segments = ctx.full_n_segments()

    for i in range(n_segments):
        text = ctx.full_get_segment_text(i)
        if params.no_timestamps:
            print(text.strip(), file=output_file)
        else:
            t0 = ctx.full_get_segment_t0(i)
            t1 = ctx.full_get_segment_t1(i)
            print(f"[{format_timestamp(t0)} --> {format_timestamp(t1)}] {text.strip()}", file=output_file)


def output_vtt(ctx: wh.WhisperContext, params: WhisperParams, output_file) -> None:
    """Output transcription in WebVTT format."""
    print("WEBVTT", file=output_file)
    print("", file=output_file)

    n_segments = ctx.full_n_segments()

    for i in range(n_segments):
        text = ctx.full_get_segment_text(i).strip()
        t0 = ctx.full_get_segment_t0(i)
        t1 = ctx.full_get_segment_t1(i)

        print(f"{format_timestamp(t0, decimal_marker='.')} --> {format_timestamp(t1, decimal_marker='.')}", file=output_file)
        print(text, file=output_file)
        print("", file=output_file)


def output_srt(ctx: wh.WhisperContext, params: WhisperParams, output_file) -> None:
    """Output transcription in SRT format."""
    n_segments = ctx.full_n_segments()

    for i in range(n_segments):
        text = ctx.full_get_segment_text(i).strip()
        t0 = ctx.full_get_segment_t0(i)
        t1 = ctx.full_get_segment_t1(i)

        print(f"{i + 1}", file=output_file)
        print(f"{format_timestamp(t0)} --> {format_timestamp(t1)}", file=output_file)
        print(text, file=output_file)
        print("", file=output_file)


def output_csv(ctx: wh.WhisperContext, params: WhisperParams, output_file) -> None:
    """Output transcription in CSV format."""
    print("start,end,text", file=output_file)

    n_segments = ctx.full_n_segments()

    for i in range(n_segments):
        text = ctx.full_get_segment_text(i).strip()
        text_escaped = escape_string_csv(text)
        t0 = ctx.full_get_segment_t0(i)
        t1 = ctx.full_get_segment_t1(i)

        # Convert to milliseconds
        t0_ms = t0 * 10
        t1_ms = t1 * 10

        print(f"{t0_ms},{t1_ms},\"{text_escaped}\"", file=output_file)


def output_json(ctx: wh.WhisperContext, params: WhisperParams, output_file, full: bool = False) -> None:
    """Output transcription in JSON format."""
    result = {
        "text": "",
        "segments": []
    }

    n_segments = ctx.full_n_segments()

    # Build full text
    full_text = ""
    for i in range(n_segments):
        text = ctx.full_get_segment_text(i)
        full_text += text

    result["text"] = full_text.strip()

    # Build segments
    for i in range(n_segments):
        text = ctx.full_get_segment_text(i).strip()
        t0 = ctx.full_get_segment_t0(i)
        t1 = ctx.full_get_segment_t1(i)

        # Convert to seconds
        t0_s = t0 * 0.01
        t1_s = t1 * 0.01

        segment = {
            "id": i,
            "seek": 0,  # Not implemented in basic version
            "start": t0_s,
            "end": t1_s,
            "text": text,
            "tokens": [],
            "temperature": params.temperature,
            "avg_logprob": 0.0,  # Not implemented in basic version
            "compression_ratio": 0.0,  # Not implemented in basic version
            "no_speech_prob": ctx.full_get_segment_no_speech_prob(i) if hasattr(ctx, 'full_get_segment_no_speech_prob') else 0.0
        }

        if full:
            # Add token-level information
            n_tokens = ctx.full_n_tokens(i)
            tokens = []
            for j in range(n_tokens):
                token_text = ctx.full_get_token_text(i, j)
                token_id = ctx.full_get_token_id(i, j)
                token_p = ctx.full_get_token_p(i, j)

                tokens.append({
                    "text": token_text,
                    "id": token_id,
                    "probability": token_p
                })

            segment["tokens"] = tokens

        result["segments"].append(segment)

    json.dump(result, output_file, indent=2, ensure_ascii=False)


def output_lrc(ctx: wh.WhisperContext, params: WhisperParams, output_file) -> None:
    """Output transcription in LRC format."""
    n_segments = ctx.full_n_segments()

    for i in range(n_segments):
        text = ctx.full_get_segment_text(i).strip()
        t0 = ctx.full_get_segment_t0(i)

        # Convert to MM:SS.ss format
        t0_ms = t0 * 10
        minutes = t0_ms // (1000 * 60)
        t0_ms -= minutes * (1000 * 60)
        seconds = t0_ms / 1000.0

        print(f"[{minutes:02d}:{seconds:05.2f}]{text}", file=output_file)


def create_whisper_params_from_args(params: WhisperParams) -> wh.WhisperFullParams:
    """Convert CLI parameters to WhisperFullParams."""
    whisper_params = wh.WhisperFullParams()

    # Set basic parameters
    whisper_params.n_threads = params.n_threads
    whisper_params.offset_ms = params.offset_t_ms
    whisper_params.duration_ms = params.duration_ms
    whisper_params.translate = params.translate
    whisper_params.no_context = False  # Default
    whisper_params.no_timestamps = params.no_timestamps
    whisper_params.single_segment = False  # Default
    whisper_params.print_special = params.print_special
    whisper_params.print_progress = params.print_progress
    whisper_params.print_realtime = False  # Default
    whisper_params.print_timestamps = not params.no_timestamps
    whisper_params.token_timestamps = not params.no_timestamps
    whisper_params.temperature = params.temperature

    # Set language if specified
    if params.language:
        whisper_params.language = params.language

    return whisper_params


def process_file(input_file: str, params: WhisperParams) -> None:
    """Process a single audio file."""
    if not params.no_prints:
        print(f"Processing: {input_file}")

    # Load audio file
    try:
        samples, sample_rate = load_wav_file(input_file)
    except Exception as e:
        print(f"Error loading audio file {input_file}: {e}", file=sys.stderr)
        return

    # Resample to 16kHz if needed
    if sample_rate != wh.WHISPER.SAMPLE_RATE:
        if not params.no_prints:
            print(f"Resampling from {sample_rate}Hz to {wh.WHISPER.SAMPLE_RATE}Hz")
        samples = resample_audio(samples, sample_rate, wh.WHISPER.SAMPLE_RATE)

    # Initialize Whisper context
    try:
        ctx = wh.WhisperContext(params.model)
    except Exception as e:
        print(f"Error loading model {params.model}: {e}", file=sys.stderr)
        return

    if not params.no_prints:
        print(f"Model loaded: {params.model}")

    # Create whisper parameters
    whisper_params = create_whisper_params_from_args(params)

    # Process audio - we need to enable the full method first
    try:
        # This will need to be uncommented in whisper_cpp.pyx
        result = ctx.full(samples, whisper_params)
        if result != 0:
            print(f"Whisper processing failed with error {result}", file=sys.stderr)
            return
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return

    if not params.no_prints:
        print("Transcription completed")

    # Generate output files
    base_name = Path(input_file).stem

    if params.output_txt:
        output_path = f"{base_name}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_txt(ctx, params, f)
        if not params.no_prints:
            print(f"Text output saved to: {output_path}")

    if params.output_vtt:
        output_path = f"{base_name}.vtt"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_vtt(ctx, params, f)
        if not params.no_prints:
            print(f"VTT output saved to: {output_path}")

    if params.output_srt:
        output_path = f"{base_name}.srt"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_srt(ctx, params, f)
        if not params.no_prints:
            print(f"SRT output saved to: {output_path}")

    if params.output_csv:
        output_path = f"{base_name}.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_csv(ctx, params, f)
        if not params.no_prints:
            print(f"CSV output saved to: {output_path}")

    if params.output_jsn:
        output_path = f"{base_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_json(ctx, params, f, full=False)
        if not params.no_prints:
            print(f"JSON output saved to: {output_path}")

    if params.output_jsn_full:
        output_path = f"{base_name}_full.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_json(ctx, params, f, full=True)
        if not params.no_prints:
            print(f"Full JSON output saved to: {output_path}")

    if params.output_lrc:
        output_path = f"{base_name}.lrc"
        with open(output_path, 'w', encoding='utf-8') as f:
            output_lrc(ctx, params, f)
        if not params.no_prints:
            print(f"LRC output saved to: {output_path}")

    # Default output to console if no specific output format requested
    if not any([params.output_txt, params.output_vtt, params.output_srt,
                params.output_csv, params.output_jsn, params.output_jsn_full, params.output_lrc]):
        output_txt(ctx, params, sys.stdout)


def parse_arguments() -> WhisperParams:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Whisper CLI - Speech-to-text transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cyllama.whisper.cli -f audio.wav
  python -m cyllama.whisper.cli -f audio.wav --output-srt --output-vtt
  python -m cyllama.whisper.cli -m models/ggml-medium.en.bin -f audio.wav --translate
        """
    )

    params = WhisperParams()

    # Input/output files
    parser.add_argument('-f', '--file', dest='fname_inp', action='append',
                        help='Input audio file path')
    parser.add_argument('-o', '--output', dest='fname_out', action='append',
                        help='Output file path')

    # Model parameters
    parser.add_argument('-m', '--model', default=params.model,
                        help='Model path (default: %(default)s)')

    # Processing parameters
    parser.add_argument('-t', '--threads', type=int, default=params.n_threads,
                        help='Number of threads (default: %(default)s)')
    parser.add_argument('-p', '--processors', type=int, default=params.n_processors,
                        help='Number of processors (default: %(default)s)')
    parser.add_argument('-ot', '--offset-t', type=int, default=params.offset_t_ms,
                        help='Time offset in milliseconds (default: %(default)s)')
    parser.add_argument('-on', '--offset-n', type=int, default=params.offset_n,
                        help='Segment offset (default: %(default)s)')
    parser.add_argument('-d', '--duration', type=int, default=params.duration_ms,
                        help='Duration in milliseconds (default: %(default)s)')
    parser.add_argument('-mc', '--max-context', type=int, default=params.max_context,
                        help='Maximum context (default: %(default)s)')
    parser.add_argument('-ml', '--max-len', type=int, default=params.max_len,
                        help='Maximum length (default: %(default)s)')

    # Decoding parameters
    parser.add_argument('-bo', '--best-of', type=int, default=params.best_of,
                        help='Best of N samples (default: %(default)s)')
    parser.add_argument('-bs', '--beam-size', type=int, default=params.beam_size,
                        help='Beam search size (default: %(default)s)')
    parser.add_argument('-wt', '--word-thold', type=float, default=params.word_thold,
                        help='Word probability threshold (default: %(default)s)')
    parser.add_argument('-et', '--entropy-thold', type=float, default=params.entropy_thold,
                        help='Entropy threshold (default: %(default)s)')
    parser.add_argument('-lpt', '--logprob-thold', type=float, default=params.logprob_thold,
                        help='Log probability threshold (default: %(default)s)')
    parser.add_argument('-tp', '--temperature', type=float, default=params.temperature,
                        help='Temperature (default: %(default)s)')
    parser.add_argument('-tpi', '--temperature-inc', type=float, default=params.temperature_inc,
                        help='Temperature increment (default: %(default)s)')

    # Language and translation
    parser.add_argument('-l', '--language', default=params.language,
                        help='Language code (default: %(default)s)')
    parser.add_argument('-tr', '--translate', action='store_true',
                        help='Translate to English')
    parser.add_argument('-dl', '--detect-language', action='store_true',
                        help='Detect language')

    # Output formats
    parser.add_argument('-otxt', '--output-txt', action='store_true',
                        help='Output plain text')
    parser.add_argument('-ovtt', '--output-vtt', action='store_true',
                        help='Output WebVTT')
    parser.add_argument('-osrt', '--output-srt', action='store_true',
                        help='Output SRT')
    parser.add_argument('-owts', '--output-wts', action='store_true',
                        help='Output word timestamps')
    parser.add_argument('-ocsv', '--output-csv', action='store_true',
                        help='Output CSV')
    parser.add_argument('-oj', '--output-json', dest='output_jsn', action='store_true',
                        help='Output JSON')
    parser.add_argument('-ojf', '--output-json-full', dest='output_jsn_full', action='store_true',
                        help='Output full JSON')
    parser.add_argument('-olrc', '--output-lrc', action='store_true',
                        help='Output LRC')

    # Display options
    parser.add_argument('-np', '--no-prints', action='store_true',
                        help='No prints')
    parser.add_argument('-ps', '--print-special', action='store_true',
                        help='Print special tokens')
    parser.add_argument('-pc', '--print-colors', action='store_true',
                        help='Print with colors')
    parser.add_argument('-pp', '--print-progress', action='store_true',
                        help='Print progress')
    parser.add_argument('-nt', '--no-timestamps', action='store_true',
                        help='Do not print timestamps')
    parser.add_argument('-ng', '--no-gpu', action='store_true',
                        help='Disable GPU')

    args = parser.parse_args()

    # Map arguments to parameters
    if args.fname_inp:
        params.fname_inp = args.fname_inp
    if args.fname_out:
        params.fname_out = args.fname_out

    params.model = args.model
    params.n_threads = args.threads
    params.n_processors = args.processors
    params.offset_t_ms = args.offset_t
    params.offset_n = args.offset_n
    params.duration_ms = args.duration
    params.max_context = args.max_context
    params.max_len = args.max_len

    params.best_of = args.best_of
    params.beam_size = args.beam_size
    params.word_thold = args.word_thold
    params.entropy_thold = args.entropy_thold
    params.logprob_thold = args.logprob_thold
    params.temperature = args.temperature
    params.temperature_inc = args.temperature_inc

    params.language = args.language
    params.translate = args.translate
    params.detect_language = args.detect_language

    params.output_txt = args.output_txt
    params.output_vtt = args.output_vtt
    params.output_srt = args.output_srt
    params.output_wts = args.output_wts
    params.output_csv = args.output_csv
    params.output_jsn = args.output_jsn
    params.output_jsn_full = args.output_jsn_full
    params.output_lrc = args.output_lrc

    params.no_prints = args.no_prints
    params.print_special = args.print_special
    params.print_colors = args.print_colors
    params.print_progress = args.print_progress
    params.no_timestamps = args.no_timestamps
    params.use_gpu = not args.no_gpu

    return params


def main():
    """Main entry point."""
    params = parse_arguments()

    # Validate input files
    if not params.fname_inp:
        print("Error: No input files specified. Use -f or --file to specify input files.", file=sys.stderr)
        sys.exit(1)

    # Check model exists
    if not os.path.exists(params.model):
        print(f"Error: Model file not found: {params.model}", file=sys.stderr)
        sys.exit(1)

    # Process each input file
    for input_file in params.fname_inp:
        if input_file != "-" and not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            continue

        try:
            process_file(input_file, params)
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing {input_file}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()