#!/usr/bin/env python3
"""
TTS (Text-to-Speech) implementation equivalent to build/llama.cpp/tools/tts/tts.cpp

This module provides a Python implementation of the TTS example using the cyllama wrapper.
Based on OuteTTS model for high-quality text-to-speech synthesis.
"""

import sys
import os
import argparse
import math
import re
import threading
import struct
from typing import List, Optional, Dict, Any, Tuple
import wave

from . import (
    LlamaModel, LlamaContext, LlamaSampler, LlamaBatch, LlamaChatMessage,
    LlamaModelParams, LlamaContextParams, LlamaSamplerChainParams,
    set_log_callback, ggml_backend_load_all, llama_batch_get_one,
    disable_logging
)


def print_usage():
    """Print usage information"""
    print("\nexample usage:")
    print(f"\n    {sys.argv[0]} -m model.gguf -p \"Hello!\"")
    print()


class WavHeader:
    """WAV file header structure"""
    def __init__(self, sample_rate: int, data_size: int):
        self.riff = b'RIFF'
        self.wave = b'WAVE'
        self.fmt = b'fmt '
        self.fmt_chunk_size = 16
        self.audio_format = 1  # PCM
        self.num_channels = 1  # Mono
        self.sample_rate = sample_rate
        self.bits_per_sample = 16
        self.byte_rate = sample_rate * self.num_channels * (self.bits_per_sample // 8)
        self.block_align = self.num_channels * (self.bits_per_sample // 8)
        self.data = b'data'
        self.data_size = data_size
        self.chunk_size = 36 + data_size

    def to_bytes(self) -> bytes:
        """Convert header to bytes"""
        return struct.pack('<4sI4s4sIHHIIHH4sI',
                          self.riff, self.chunk_size, self.wave, self.fmt,
                          self.fmt_chunk_size, self.audio_format, self.num_channels,
                          self.sample_rate, self.byte_rate, self.block_align,
                          self.bits_per_sample, self.data, self.data_size)


def save_wav16(filename: str, data: List[float], sample_rate: int) -> bool:
    """Save audio data as 16-bit WAV file"""
    try:
        header = WavHeader(sample_rate, len(data) * 2)  # 2 bytes per sample

        with open(filename, 'wb') as f:
            f.write(header.to_bytes())

            # Convert float samples to 16-bit PCM
            for sample in data:
                pcm_sample = int(max(-32768, min(32767, sample * 32767.0)))
                f.write(struct.pack('<h', pcm_sample))

        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}", file=sys.stderr)
        return False


def fill_hann_window(length: int, periodic: bool = True) -> List[float]:
    """Generate Hann window"""
    offset = 0 if periodic else -1
    window = []
    for i in range(length):
        value = 0.5 * (1.0 - math.cos((2.0 * math.pi * i) / (length + offset)))
        window.append(value)
    return window


def twiddle(k: int, N: int) -> Tuple[float, float]:
    """Compute twiddle factors for FFT"""
    angle = 2 * math.pi * k / N
    return math.cos(angle), math.sin(angle)


def irfft(n: int, inp_cplx: List[float]) -> List[float]:
    """Inverse real FFT (very basic implementation)"""
    N = n // 2 + 1

    # Extract real and imaginary parts
    real_input = [inp_cplx[2*i] for i in range(N)]
    imag_input = [inp_cplx[2*i + 1] for i in range(N)]

    real_output = [0.0] * n

    for k in range(n):
        for m in range(N):
            twiddle_real, twiddle_imag = twiddle(k * m, n)
            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag

    # Normalize
    return [x / N for x in real_output]


def fold(data: List[float], n_out: int, n_win: int, n_hop: int, n_pad: int) -> List[float]:
    """Fold operation equivalent to torch.nn.functional.fold"""
    output = [0.0] * n_out

    col_idx = 0
    for w_col in range((len(data) // n_win)):
        start = w_col * n_hop - n_pad
        end = start + n_win

        for w_im in range(start, end):
            if 0 <= w_im < n_out and col_idx < len(data):
                output[w_im] += data[col_idx]
            col_idx += 1

    # Return trimmed output
    return output[n_pad:n_out-n_pad]


def embd_to_audio(embd: List[float], n_codes: int, n_embd: int, n_threads: int = 4) -> List[float]:
    """Convert embeddings to audio using inverse spectrogram operations"""
    n_fft = 1280
    n_hop = 320
    n_win = 1280
    n_pad = (n_win - n_hop) // 2
    n_out = (n_codes - 1) * n_hop + n_win

    # Generate Hann window
    hann = fill_hann_window(n_fft, True)

    n_spec = n_embd * n_codes

    # Transpose embeddings: E[k*n_codes + l] = embd[l*n_embd + k]
    E = [0.0] * n_spec
    for l in range(n_codes):
        for k in range(n_embd):
            E[k * n_codes + l] = embd[l * n_embd + k]

    # Convert to complex spectrogram
    S = [0.0] * n_spec
    for k in range(n_embd // 2):
        for l in range(n_codes):
            mag = math.exp(E[k * n_codes + l])
            phi = E[(k + n_embd // 2) * n_codes + l]

            # Clamp magnitude
            if mag > 1e2:
                mag = 1e2

            S[2 * (k * n_codes + l)] = mag * math.cos(phi)
            S[2 * (k * n_codes + l) + 1] = mag * math.sin(phi)

    # Transpose spectrogram for IRFFT
    ST = [0.0] * n_spec
    for l in range(n_codes):
        for k in range(n_embd // 2):
            ST[l * n_embd + 2 * k] = S[2 * (k * n_codes + l)]
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1]

    # Apply IRFFT and windowing
    res = []
    hann2 = []

    for l in range(n_codes):
        # Apply IRFFT to get time-domain signal
        frame_spec = ST[l * n_embd:(l + 1) * n_embd]
        frame_audio = irfft(n_fft, frame_spec)

        # Apply window
        windowed_frame = [frame_audio[i] * hann[i] for i in range(n_fft)]
        squared_hann = [hann[i] * hann[i] for i in range(n_fft)]

        res.extend(windowed_frame)
        hann2.extend(squared_hann)

    # Overlap-add using fold operation
    audio = fold(res, n_out, n_win, n_hop, n_pad)
    env = fold(hann2, n_out, n_win, n_hop, n_pad)

    # Normalize by envelope
    for i in range(len(audio)):
        if env[i] > 0:
            audio[i] /= env[i]

    return audio


# Number to words conversion
ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
}

TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"
}


def convert_less_than_thousand(num: int) -> str:
    """Convert number less than 1000 to words"""
    result = ""

    if num >= 100:
        result += ONES[num // 100] + " hundred "
        num %= 100

    if num >= 20:
        result += TENS[num // 10]
        if num % 10 > 0:
            result += "-" + ONES[num % 10]
    elif num > 0:
        result += ONES[num]

    return result


def number_to_words(number_str: str) -> str:
    """Convert number string to words"""
    try:
        decimal_pos = number_str.find('.')
        integer_part = number_str[:decimal_pos] if decimal_pos != -1 else number_str

        int_number = int(integer_part)
        result = ""

        if int_number == 0:
            result = "zero"
        else:
            if int_number >= 1000000000:
                billions = int_number // 1000000000
                result += convert_less_than_thousand(billions) + " billion "
                int_number %= 1000000000

            if int_number >= 1000000:
                millions = int_number // 1000000
                result += convert_less_than_thousand(millions) + " million "
                int_number %= 1000000

            if int_number >= 1000:
                thousands = int_number // 1000
                result += convert_less_than_thousand(thousands) + " thousand "
                int_number %= 1000

            if int_number > 0:
                result += convert_less_than_thousand(int_number)

        # Handle decimal part
        if decimal_pos != -1:
            result += " point"
            decimal_part = number_str[decimal_pos + 1:]
            for digit in decimal_part:
                result += " " + ONES[int(digit)]

        return result.strip()
    except:
        return " "


def replace_numbers_with_words(input_text: str) -> str:
    """Replace numbers in text with their word equivalents"""
    pattern = r'\d+(\.\d+)?'

    def replace_func(match):
        return number_to_words(match.group())

    return re.sub(pattern, replace_func, input_text)


def process_text(text: str, tts_version: str = "0.2") -> str:
    """Process text for TTS input (based on OuteTTS prompt processor)"""
    # Convert numbers to words
    processed_text = replace_numbers_with_words(text)

    # Convert to lowercase
    processed_text = processed_text.lower()

    # Replace special characters with spaces
    processed_text = re.sub(r'[-_/,\.\\]', ' ', processed_text)

    # Remove non-alphabetic characters (except spaces)
    processed_text = re.sub(r'[^a-z\s]', '', processed_text)

    # Collapse multiple spaces
    processed_text = re.sub(r'\s+', ' ', processed_text)

    # Trim whitespace
    processed_text = processed_text.strip()

    # Replace spaces with separator tokens
    separator = "<|space|>" if tts_version == "0.3" else "<|text_sep|>"
    processed_text = re.sub(r'\s', separator, processed_text)

    return processed_text


def prepare_guide_tokens(vocab, text: str, tts_version: str = "0.2") -> List[int]:
    """Prepare guide tokens to prevent hallucinations"""
    delimiter = "<|space|>" if tts_version == "0.3" else "<|text_sep|>"

    result = []

    # First token is always a newline
    newline_tokens = vocab.tokenize("\n", False, True)
    if newline_tokens:
        result.append(newline_tokens[0])

    # Split by delimiter and tokenize each word
    words = text.split(delimiter)

    for word in words:
        if word:
            word_tokens = vocab.tokenize(word, False, True)
            if word_tokens:
                result.append(word_tokens[0])

    return result


class TTSGenerator:
    """Text-to-Speech generator using OuteTTS models"""

    def __init__(self,
                 ttc_model_path: str,  # text-to-codes model
                 cts_model_path: str,  # codes-to-speech model
                 n_ctx: int = 8192,
                 n_batch: int = 8192,
                 ngl: int = 99,
                 n_predict: int = 4096,
                 speaker_file: Optional[str] = None,
                 use_guide_tokens: bool = True):
        """Initialize TTS with models and parameters"""

        # Load dynamic backends
        ggml_backend_load_all()

        # Initialize text-to-codes model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = ngl
        self.model_ttc = LlamaModel(ttc_model_path, model_params)
        self.vocab = self.model_ttc.get_vocab()

        # Initialize text-to-codes context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        self.context_ttc = LlamaContext(self.model_ttc, ctx_params)

        # Initialize codes-to-speech model
        model_params_cts = LlamaModelParams()
        model_params_cts.n_gpu_layers = ngl
        self.model_cts = LlamaModel(cts_model_path, model_params_cts)

        # Initialize codes-to-speech context (embedding mode)
        ctx_params_cts = LlamaContextParams()
        ctx_params_cts.n_ctx = n_ctx
        ctx_params_cts.n_batch = n_batch
        ctx_params_cts.n_ubatch = n_batch
        self.context_cts = LlamaContext(self.model_cts, ctx_params_cts)

        # Set embedding mode for codes-to-speech
        self.context_cts.set_embeddings_mode(True)

        # Initialize sampler for text-to-codes generation
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        self.sampler.add_top_k(4)
        self.sampler.add_temp(0.8)
        self.sampler.add_dist(1337)  # Use fixed seed

        # Store parameters
        self.n_predict = n_predict
        self.use_guide_tokens = use_guide_tokens
        self.speaker_file = speaker_file

        # Determine TTS version
        chat_template = self.model_ttc.get_default_chat_template()
        if chat_template and "outetts-0.3" in chat_template:
            self.tts_version = "0.3"
        else:
            self.tts_version = "0.2"

        # Load speaker data if provided
        if speaker_file:
            self.load_speaker(speaker_file)
        else:
            self.setup_default_speaker()

    def load_speaker(self, speaker_file: str):
        """Load speaker profile from JSON file"""
        import json
        try:
            with open(speaker_file, 'r') as f:
                speaker_data = json.load(f)

            # Extract version if available
            if 'version' in speaker_data:
                self.tts_version = speaker_data['version']

            # Build audio text and data from speaker
            self.audio_text = self.audio_text_from_speaker(speaker_data)
            self.audio_data = self.audio_data_from_speaker(speaker_data)

        except Exception as e:
            print(f"Error loading speaker file: {e}", file=sys.stderr)
            self.setup_default_speaker()

    def setup_default_speaker(self):
        """Setup default speaker profile"""
        # Default speaker profile based on OuteTTS en_male_1
        separator = "<|space|>" if self.tts_version == "0.3" else "<|text_sep|>"

        self.audio_text = f"<|text_start|>the{separator}overall{separator}package{separator}from{separator}just{separator}two{separator}people{separator}is{separator}pretty{separator}remarkable{separator}sure{separator}i{separator}have{separator}some{separator}critiques{separator}about{separator}some{separator}of{separator}the{separator}gameplay{separator}aspects{separator}but{separator}its{separator}still{separator}really{separator}enjoyable{separator}and{separator}it{separator}looks{separator}lovely{separator}"

        # This is a simplified version - in practice you'd want the full audio data
        self.audio_data = "<|audio_start|>\n"
        if self.tts_version == "0.3":
            # Convert format for v0.3
            self.audio_data = self.audio_data.replace("<|code_start|>", "").replace("<|code_end|>", "<|space|>")

    def audio_text_from_speaker(self, speaker_data: Dict) -> str:
        """Extract audio text from speaker data"""
        audio_text = "<|text_start|>"
        separator = "<|space|>" if self.tts_version == "0.3" else "<|text_sep|>"

        if 'words' in speaker_data:
            for word in speaker_data['words']:
                audio_text += word['word'] + separator

        return audio_text

    def audio_data_from_speaker(self, speaker_data: Dict) -> str:
        """Extract audio data from speaker data"""
        audio_data = "<|audio_start|>\n"

        if 'words' in speaker_data:
            code_start = "" if self.tts_version == "0.3" else "<|code_start|>"
            code_end = "<|space|>" if self.tts_version == "0.3" else "<|code_end|>"

            for word in speaker_data['words']:
                word_text = word['word']
                duration = word['duration']
                codes = word['codes']

                entry = f"{word_text}<|t_{duration:.2f}|>{code_start}"
                for code in codes:
                    entry += f"<|{code}|>"
                entry += code_end + "\n"
                audio_data += entry

        return audio_data

    def generate_codes(self, text: str) -> List[int]:
        """Generate audio codes from text"""
        print(f"Processing text: '{text}'")

        # Process input text
        processed_text = process_text(text, self.tts_version)
        print(f"Processed text: '{processed_text}'")

        # Prepare guide tokens if enabled
        guide_tokens = []
        if self.use_guide_tokens:
            guide_tokens = prepare_guide_tokens(self.vocab, processed_text, self.tts_version)

        # Build prompt
        prompt_tokens = []

        # Start with system prompt
        start_tokens = self.vocab.tokenize("<|im_start|>\n", True, True)
        prompt_tokens.extend(start_tokens)

        # Add audio text
        audio_text_tokens = self.vocab.tokenize(self.audio_text, False, True)
        prompt_tokens.extend(audio_text_tokens)

        # Add processed user text
        text_tokens = self.vocab.tokenize(processed_text, False, True)
        prompt_tokens.extend(text_tokens)

        # Add text end marker
        end_tokens = self.vocab.tokenize("<|text_end|>\n", False, True)
        prompt_tokens.extend(end_tokens)

        # Add audio data (speaker profile) - this provides the voice template
        if self.speaker_file:
            audio_data_tokens = self.vocab.tokenize(self.audio_data, False, True)
            prompt_tokens.extend(audio_data_tokens)
        else:
            # For default speaker, we need to include the template but limit generation
            # Add just the first few entries of the audio_data as examples
            lines = self.audio_data.split('\n')
            limited_audio_data = '\n'.join(lines[:10])  # Just first few entries
            audio_data_tokens = self.vocab.tokenize(limited_audio_data, False, True)
            prompt_tokens.extend(audio_data_tokens)

        print(f"Prompt size: {len(prompt_tokens)} tokens")

        # Create batch for initial prompt
        batch = LlamaBatch(n_tokens=max(len(prompt_tokens), 1), embd=0, n_seq_max=1)

        # Add prompt tokens to batch
        batch.add_sequence(prompt_tokens, 0, False)

        # Mark last token for logits
        batch.set_last_logits_to_true()

        # Decode initial prompt
        ret = self.context_ttc.decode(batch)
        if ret != 0:
            print(f"Warning: initial decode returned {ret}")

        # Generation loop
        codes = []
        n_past = len(prompt_tokens)
        guide_token_idx = 0
        next_token_uses_guide_token = True

        # Limit generation to avoid excessive output
        max_new_tokens = min(self.n_predict, len(guide_tokens) * 3 if guide_tokens else 100)

        for i in range(max_new_tokens):
            if n_past >= self.context_ttc.n_ctx - 1:
                print("Context size exceeded")
                break

            # Sample next token
            try:
                new_token_id = self.sampler.sample(self.context_ttc, -1)
            except Exception as e:
                print(f"Sampling failed: {e}")
                break

            # Use guide token if applicable - this forces the model to generate the target words
            if (guide_tokens and next_token_uses_guide_token and
                guide_token_idx < len(guide_tokens) and
                not self.vocab.is_control(new_token_id) and
                not self.vocab.is_eog(new_token_id)):
                new_token_id = guide_tokens[guide_token_idx]
                guide_token_idx += 1

            # Check if next token should use guide token (after newline)
            next_token_uses_guide_token = (new_token_id == 198)  # newline token

            codes.append(new_token_id)

            # Check for end of generation
            if self.vocab.is_eog(new_token_id):
                print(f"Stopped at EOG token after {i+1} tokens")
                break

            # If we've used all guide tokens and generated some audio, we can stop
            if guide_tokens and guide_token_idx >= len(guide_tokens):
                # Generate a few more tokens for the audio codes, then stop
                if i > len(guide_tokens) + 50:  # Allow some audio generation
                    print(f"Completed guided generation after {i+1} tokens")
                    break

            # Create batch for next token
            batch.set_batch([new_token_id], n_past, True)
            n_past += 1

            # Decode next token
            ret = self.context_ttc.decode(batch)
            if ret != 0:
                print(f"Warning: decode returned {ret}")
                break

        # Batch will be automatically freed when it goes out of scope

        print(f"Generated {len(codes)} tokens")

        # Filter to audio codes only (token range 151672-155772)
        audio_codes = [t for t in codes if 151672 <= t <= 155772]
        print(f"Filtered to {len(audio_codes)} audio codes")

        # Adjust token values for vocoder input
        adjusted_codes = [t - 151672 for t in audio_codes]

        return adjusted_codes

    def codes_to_audio(self, codes: List[int]) -> List[float]:
        """Convert audio codes to audio samples"""
        n_codes = len(codes)

        # Create batch for codes
        batch = LlamaBatch(n_tokens=n_codes, embd=0, n_seq_max=1)

        batch.add_sequence(codes, 0, True)

        # Encode codes to get embeddings
        try:
            self.context_cts.encode(batch)
        except Exception as e:
            print(f"Error encoding batch: {e}")
            return []

        # Get embeddings for all tokens at once
        n_embd = self.model_cts.n_embd

        try:
            # Try to get all embeddings at once (should be n_codes * n_embd floats)
            embeddings = self.context_cts.get_embeddings()
            print(f"Got embeddings: {len(embeddings)} floats")

            # If we got fewer embeddings than expected, collect them individually (slower fallback)
            if len(embeddings) < n_codes * n_embd:
                print(f"Insufficient embeddings, collecting individually...")
                embeddings = []
                for i in range(n_codes):
                    if i % 100 == 0:
                        print(f"Processing token {i}/{n_codes}")
                    token_embeddings = self.context_cts.get_embeddings_ith(i)
                    embeddings.extend(token_embeddings)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

        if not embeddings:
            print("Error: no embeddings returned")
            return []

        print(f"Debug: collected embeddings length: {len(embeddings)}, n_codes: {n_codes}, n_embd: {n_embd}")
        print(f"Debug: expected embeddings length: {n_codes * n_embd}")

        # Convert embeddings to audio
        try:
            audio = embd_to_audio(embeddings, n_codes, n_embd, 4)
        except Exception as e:
            print(f"Error in embd_to_audio: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Batch will be automatically freed when it goes out of scope

        # Zero out first 0.25 seconds to remove artifacts
        sample_rate = 24000
        zero_samples = sample_rate // 4
        for i in range(min(zero_samples, len(audio))):
            audio[i] = 0.0

        return audio

    def generate(self, text: str, output_file: str = "output.wav") -> bool:
        """Generate speech from text and save to file"""
        try:
            print(f"Generating speech for: '{text}'")

            # Generate audio codes from text
            codes = self.generate_codes(text)
            if not codes:
                print("Error: no codes generated")
                return False

            # Convert codes to audio
            audio = self.codes_to_audio(codes)
            if not audio:
                print("Error: no audio generated")
                return False

            # Save to WAV file
            sample_rate = 24000
            success = save_wav16(output_file, audio, sample_rate)

            if success:
                print(f"Audio written to file '{output_file}'")
                print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

            return success

        except Exception as e:
            print(f"Error generating speech: {e}", file=sys.stderr)
            return False


def main():
    """Main entry point"""
    disable_logging()

    parser = argparse.ArgumentParser(description="Text-to-Speech using cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to text-to-codes model file")
    parser.add_argument("-mv", "--vocoder-model", required=True, help="Path to codes-to-speech model file")
    parser.add_argument("-p", "--prompt", required=True, help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file")
    parser.add_argument("-c", "--context", type=int, default=8192, help="Context size")
    parser.add_argument("-b", "--batch", type=int, default=8192, help="Batch size")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="Number of GPU layers")
    parser.add_argument("-n", "--n-predict", type=int, default=4096, help="Number of tokens to predict")
    parser.add_argument("--speaker-file", help="Speaker profile JSON file")
    parser.add_argument("--use-guide-tokens", action="store_true", help="Use guide tokens to prevent hallucinations")

    args = parser.parse_args()

    try:
        tts = TTSGenerator(
            ttc_model_path=args.model,
            cts_model_path=args.vocoder_model,
            n_ctx=args.context,
            n_batch=args.batch,
            ngl=args.n_gpu_layers,
            n_predict=args.n_predict,
            speaker_file=args.speaker_file,
            use_guide_tokens=args.use_guide_tokens
        )

        success = tts.generate(args.prompt, args.output)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()