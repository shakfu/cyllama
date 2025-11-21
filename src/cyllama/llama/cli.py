#!/usr/bin/env python3
"""
Python CLI implementation of llama.cpp main tool using cyllama wrapper.

This replicates the functionality of build/llama.cpp/tools/main/main.cpp
in Python using the cyllama wrapper.
"""

import argparse
import sys
import os
import signal
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from . import llama_cpp as cy



class LlamaCLI:
    """Main CLI class that replicates the functionality of llama.cpp main tool."""
    
    def __init__(self):
        self.model: Optional[cy.LlamaModel] = None
        self.ctx: Optional[cy.LlamaContext] = None
        self.sampler: Optional[cy.LlamaSampler] = None
        self.vocab: Optional[cy.LlamaVocab] = None
        self.is_interacting = False
        self.need_insert_eot = False
        
        # Performance tracking
        self.t_main_start = 0
        self.n_decode = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._sigint_handler)
    
    def _sigint_handler(self, signo, frame):
        """Handle SIGINT (Ctrl+C) signals."""
        if signo == signal.SIGINT:
            if not self.is_interacting and hasattr(self, 'interactive') and self.interactive:
                self.is_interacting = True
                self.need_insert_eot = True
            else:
                print("\n")
                self._print_performance()
                print("Interrupted by user")
                sys.exit(130)
    
    def _print_performance(self):
        """Print performance statistics."""
        if self.t_main_start > 0:
            t_main_end = cy.ggml_time_us()
            elapsed = (t_main_end - self.t_main_start) / 1000000.0
            if elapsed > 0:
                speed = self.n_decode / elapsed
                print(f"decoded {self.n_decode} tokens in {elapsed:.2f} s, speed: {speed:.2f} t/s")
        
        if self.sampler:
            self.sampler.print_perf_data()
        if self.ctx:
            self.ctx.print_perf_data()
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(path)
    
    def _file_is_empty(self, path: str) -> bool:
        """Check if file is empty."""
        try:
            return os.path.getsize(path) == 0
        except OSError:
            return True
    
    def _print_usage(self, prog_name: str):
        """Print usage information."""
        print(f"\nexample usage:\n")
        print(f"\n  text generation:     {prog_name} -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 --no-cnv\n")
        print(f"\n  chat (conversation): {prog_name} -m your_model.gguf --sys \"You are a helpful assistant\"\n")
        print()
    
    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Python implementation of llama.cpp main tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Model parameters
        parser.add_argument("-m", "--model", type=str, required=True,
                          help="model path")
        parser.add_argument("--lora", type=str, default="",
                          help="apply LoRA adapter (implies --no-mmap)")
        parser.add_argument("--lora-scaled", type=str, nargs=2, metavar=("PATH", "SCALE"),
                          help="apply LoRA adapter with user defined scaling S (implies --no-mmap)")
        parser.add_argument("--lora-base", type=str, default="",
                          help="optional model to use as a base for the layers modified by the LoRA adapter")
        
        # Context parameters
        parser.add_argument("-c", "--ctx-size", type=int, default=4096,
                          help="size of the prompt context (default: 4096)")
        parser.add_argument("-b", "--batch-size", type=int, default=2048,
                          help="batch size for prompt processing (default: 2048)")
        parser.add_argument("--ubatch", type=int, default=512,
                          help="physical batch size for prompt processing (default: 512)")
        parser.add_argument("--keep", type=int, default=0,
                          help="number of tokens to keep from the initial prompt (default: 0, -1 = all)")
        parser.add_argument("--chunks", type=int, default=-1,
                          help="max number of chunks to process (-1 = unlimited)")
        parser.add_argument("--grp-attn-n", type=int, default=1,
                          help="group-attention factor (default: 1)")
        parser.add_argument("--grp-attn-w", type=int, default=512,
                          help="group-attention width (default: 512)")
        parser.add_argument("--print-freq", type=int, default=-1,
                          help="print token count every N tokens (-1 = disabled)")
        
        # RoPE parameters
        parser.add_argument("--rope-freq-base", type=float, default=0.0,
                          help="RoPE base frequency (default: loaded from model)")
        parser.add_argument("--rope-freq-scale", type=float, default=0.0,
                          help="RoPE frequency scaling factor (default: loaded from model)")
        parser.add_argument("--yarn-ext-factor", type=float, default=-1.0,
                          help="YaRN extrapolation mix factor (default: -1.0)")
        parser.add_argument("--yarn-attn-factor", type=float, default=1.0,
                          help="YaRN magnitude scaling factor (default: 1.0)")
        parser.add_argument("--yarn-beta-fast", type=float, default=32.0,
                          help="YaRN low correction dim (default: 32.0)")
        parser.add_argument("--yarn-beta-slow", type=float, default=1.0,
                          help="YaRN high correction dim (default: 1.0)")
        parser.add_argument("--yarn-orig-ctx", type=int, default=0,
                          help="YaRN original context length (default: 0)")
        
        # GPU parameters
        parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=-1,
                          help="number of layers to store in VRAM (-1 = use default)")
        parser.add_argument("--main-gpu", type=int, default=0,
                          help="the GPU that is used for scratch and small tensors")
        parser.add_argument("--tensor-split", type=str, default="",
                          help="how split tensors should be distributed across GPUs")
        parser.add_argument("--split-mode", type=str, default="layer",
                          choices=["none", "layer", "row"],
                          help="how to split the model across GPUs")
        
        # CPU parameters
        parser.add_argument("-t", "--threads", type=int, default=4,
                          help="number of threads to use during computation (default: 4)")
        parser.add_argument("-tb", "--threads-batch", type=int, default=4,
                          help="number of threads to use during batch and prompt processing (default: 4)")
        parser.add_argument("--threads-batch-infer", type=int, default=-1,
                          help="number of threads to use during batch inference (default: same as --threads-batch)")
        parser.add_argument("--no-mmap", action="store_true",
                          help="do not memory-map model (slower load but may reduce pageouts if not using mlock)")
        parser.add_argument("--mlock", action="store_true",
                          help="force system to keep model in RAM rather than swapping or compressing")
        parser.add_argument("--numa", action="store_true",
                          help="attempt optimizations that help on some NUMA systems")
        
        # Generation parameters
        parser.add_argument("-n", "--n-predict", type=int, default=-1,
                          help="number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)")
        parser.add_argument("--top-k", type=int, default=40,
                          help="top-k sampling (default: 40, 0 = disabled)")
        parser.add_argument("--top-p", type=float, default=0.95,
                          help="top-p sampling (default: 0.95, 1.0 = disabled)")
        parser.add_argument("--min-p", type=float, default=0.05,
                          help="min-p sampling (default: 0.05, 0.0 = disabled)")
        parser.add_argument("--tfs", type=float, default=1.0,
                          help="tail free sampling, parameter z (default: 1.0, 1.0 = disabled)")
        parser.add_argument("--typical", type=float, default=1.0,
                          help="locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)")
        parser.add_argument("--repeat-last-n", type=int, default=64,
                          help="last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)")
        parser.add_argument("--repeat-penalty", type=float, default=1.1,
                          help="penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)")
        parser.add_argument("--frequency-penalty", type=float, default=0.0,
                          help="repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)")
        parser.add_argument("--presence-penalty", type=float, default=0.0,
                          help="repeat alpha presence penalty (default: 0.0, 0.0 = disabled)")
        parser.add_argument("--mirostat", type=int, default=0,
                          help="use Mirostat sampling (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)")
        parser.add_argument("--mirostat-lr", type=float, default=0.1,
                          help="Mirostat learning rate, parameter eta (default: 0.1)")
        parser.add_argument("--mirostat-ent", type=float, default=5.0,
                          help="Mirostat target entropy, parameter tau (default: 5.0)")
        parser.add_argument("-l", "--logit-bias", type=str, default="",
                          help="modifies the likelihood of token appearing in the completion")
        parser.add_argument("--temp", type=float, default=0.8,
                          help="temperature (default: 0.8)")
        parser.add_argument("--seed", type=int, default=-1,
                          help="random seed (default: -1, use random seed for < 0)")
        
        # Prompt parameters
        parser.add_argument("-p", "--prompt", type=str, default="",
                          help="prompt")
        parser.add_argument("-f", "--file", type=str, default="",
                          help="prompt file to read from")
        parser.add_argument("-e", "--escape", action="store_true",
                          help="process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)")
        parser.add_argument("--prompt-cache", type=str, default="",
                          help="path to file for caching evaluated prompt")
        parser.add_argument("--prompt-cache-all", action="store_true",
                          help="if specified, a user is able to save/load prompt cache to/from a file")
        parser.add_argument("--prompt-cache-ro", action="store_true",
                          help="open the prompt cache read-only and do not update it")
        parser.add_argument("--verbose-prompt", action="store_true",
                          help="print prompt before generation")
        
        # Interactive parameters
        parser.add_argument("-i", "--interactive", action="store_true",
                          help="run in interactive mode")
        parser.add_argument("--interactive-first", action="store_true",
                          help="run in interactive mode and wait for input right away")
        parser.add_argument("-ins", "--instruct", action="store_true",
                          help="run in instruction mode (use with Alpaca models)")
        parser.add_argument("--multiline-input", action="store_true",
                          help="allows you to write or paste multiple lines without ending each in '\\'")
        parser.add_argument("--simple-io", action="store_true",
                          help="improves compatibility with subprocesses and limited consoles")
        parser.add_argument("--color", action="store_true",
                          help="colorise output to distinguish prompt and user input from generations")
        parser.add_argument("-r", "--reverse-prompt", type=str, default="",
                          help="stop generation at PROMPT, return control in interactive mode")
        parser.add_argument("--in-prefix", type=str, default="",
                          help="string to prefix user inputs with (default: empty)")
        parser.add_argument("--in-suffix", type=str, default="",
                          help="string to suffix after user inputs with (default: empty)")
        parser.add_argument("--in-prefix-bos", action="store_true",
                          help="prefix BOS to user inputs")
        parser.add_argument("--display-prompt", action="store_true",
                          help="print the prompt before generation")
        
        # Chat parameters
        parser.add_argument("--chat-template", type=str, default="",
                          help="chat template to use")
        parser.add_argument("--sys", "--system-prompt", type=str, default="",
                          help="system prompt")
        parser.add_argument("-cnv", "--conversation", action="store_true",
                          help="enable conversation mode (default: auto)")
        parser.add_argument("--no-cnv", action="store_true",
                          help="disable conversation mode")
        parser.add_argument("--single-turn", action="store_true",
                          help="disable conversation mode for a single turn")
        parser.add_argument("--use-jinja", action="store_true",
                          help="use Jinja2 template engine for chat templates")
        
        # Other parameters
        parser.add_argument("--embedding", action="store_true",
                          help="run in embedding mode")
        parser.add_argument("--no-display-prompt", action="store_true",
                          help="don't print prompt")
        parser.add_argument("--ctx-shift", action="store_true",
                          help="enable context shifting")
        parser.add_argument("--no-cache", action="store_true",
                          help="disable KV cache")
        parser.add_argument("--no-kv-offload", action="store_true",
                          help="disable KV offload")
        parser.add_argument("--no-flash-attn", action="store_true",
                          help="disable flash attention")
        parser.add_argument("--no-perf", action="store_true",
                          help="disable performance metrics")
        parser.add_argument("--timing", action="store_true",
                          help="print timing information")
        parser.add_argument("--log-disable", action="store_true",
                          help="disable all logs")
        parser.add_argument("--log-enable", action="store_true",
                          help="explicitly enable logs")
        parser.add_argument("--log-file", type=str, default="",
                          help="specify a log filename (without extension)")
        parser.add_argument("--log-new", action="store_true",
                          help="don't resume previous logging")
        parser.add_argument("--log-append", action="store_true",
                          help="don't truncate the old log file")
        
        return parser.parse_args()
    
    def _load_model(self, args: argparse.Namespace):
        """Load the model and initialize context."""
        print(f"llama backend init")
        
        # Initialize backend
        cy.llama_backend_init()
        if args.numa:
            cy.llama_numa_init(cy.GGML_NUMA_STRATEGY_DISTRIBUTE)
        
        # Load dynamic backends
        cy.ggml_backend_load_all()
        
        # Model parameters
        model_params = cy.LlamaModelParams()
        model_params.n_gpu_layers = args.n_gpu_layers
        model_params.use_mmap = not args.no_mmap
        model_params.use_mlock = args.mlock
        
        # Load model
        print(f"load the model and apply lora adapter, if any")
        self.model = cy.LlamaModel(args.model, model_params)
        if self.model is None:
            print(f"error: unable to load model")
            sys.exit(1)
        
        # Get vocabulary
        self.vocab = self.model.get_vocab()
        
        # Context parameters
        ctx_params = cy.LlamaContextParams()
        ctx_params.n_ctx = args.ctx_size
        ctx_params.n_batch = args.batch_size
        ctx_params.n_ubatch = args.ubatch
        ctx_params.n_threads = args.threads
        ctx_params.n_threads_batch = args.threads_batch
        ctx_params.rope_freq_base = args.rope_freq_base
        ctx_params.rope_freq_scale = args.rope_freq_scale
        ctx_params.yarn_ext_factor = args.yarn_ext_factor
        ctx_params.yarn_attn_factor = args.yarn_attn_factor
        ctx_params.yarn_beta_fast = args.yarn_beta_fast
        ctx_params.yarn_beta_slow = args.yarn_beta_slow
        ctx_params.yarn_orig_ctx = args.yarn_orig_ctx
        ctx_params.no_perf = args.no_perf
        
        # Create context
        self.ctx = cy.LlamaContext(self.model, ctx_params)
        
        # Initialize sampler
        sampler_params = cy.LlamaSamplerChainParams()
        sampler_params.no_perf = args.no_perf
        
        self.sampler = cy.LlamaSampler(sampler_params)
        
        # Add sampling methods - start with greedy for simplicity
        self.sampler.add_greedy()
        
        print(f"sampler seed: {self.sampler.get_seed()}")
        print(f"generate: n_ctx = {self.ctx.n_ctx}, n_batch = {args.batch_size}, n_predict = {args.n_predict}, n_keep = {args.keep}")
    
    def _load_prompt(self, args: argparse.Namespace) -> str:
        """Load prompt from file or use provided prompt."""
        prompt = args.prompt
        
        if args.file:
            if not self._file_exists(args.file):
                print(f"error: file '{args.file}' does not exist")
                sys.exit(1)
            
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read()
        
        if args.escape:
            # Process escape sequences
            prompt = prompt.replace('\\n', '\n')
            prompt = prompt.replace('\\r', '\r')
            prompt = prompt.replace('\\t', '\t')
            prompt = prompt.replace("\\'", "'")
            prompt = prompt.replace('\\"', '"')
            prompt = prompt.replace('\\\\', '\\')
        
        return prompt
    
    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize the prompt."""
        if not prompt:
            return []
        
        return self.vocab.tokenize(prompt, add_special=True, parse_special=True)
    
    def _generate_text(self, args: argparse.Namespace, prompt_tokens: List[int]):
        """Generate text based on the prompt."""
        if not prompt_tokens:
            if self.vocab.get_add_bos():
                prompt_tokens = [self.vocab.token_bos()]
            else:
                print("input is empty")
                sys.exit(-1)
        
        # Check if prompt is too long
        if len(prompt_tokens) > self.ctx.n_ctx - 4:
            print(f"prompt is too long ({len(prompt_tokens)} tokens, max {self.ctx.n_ctx - 4})")
            sys.exit(1)
        
        # Prepare batch for prompt
        batch = cy.llama_batch_get_one(prompt_tokens)
        
        # Start timing
        self.t_main_start = cy.ggml_time_us()
        
        # Process prompt
        n_past = 0
        for i in range(0, len(prompt_tokens), args.batch_size):
            n_eval = min(args.batch_size, len(prompt_tokens) - i)
            batch_tokens = prompt_tokens[i:i + n_eval]
            batch = cy.llama_batch_get_one(batch_tokens, n_past)
            
            try:
                self.ctx.decode(batch)
                n_past += n_eval
            except ValueError as e:
                print(f"error: {e}")
                sys.exit(1)
        
        # Generate tokens
        n_remain = args.n_predict if args.n_predict > 0 else 1000  # Default limit
        n_pos = len(prompt_tokens)
        
        if args.verbose_prompt:
            print(f"prompt: '{self.vocab.detokenize(prompt_tokens)}'")
            print(f"number of tokens in prompt = {len(prompt_tokens)}")
            for i, token in enumerate(prompt_tokens):
                piece = self.vocab.token_to_piece(token, special=False)
                print(f"{token:6d} -> '{piece}'")
            print()
        
        # Print prompt if requested
        if args.display_prompt and not args.no_display_prompt:
            print(self.vocab.detokenize(prompt_tokens), end="", flush=True)
        
        # Generate response
        response = ""
        for i in range(n_remain):
            # Sample next token
            new_token_id = self.sampler.sample(self.ctx, -1)
            
            # Check for end of generation
            if self.vocab.is_eog(new_token_id):
                break
            
            # Convert token to text
            piece = self.vocab.token_to_piece(new_token_id, special=True)
            response += piece
            print(piece, end="", flush=True)
            
            # Prepare next batch
            batch = cy.llama_batch_get_one([new_token_id], n_pos)
            n_pos += 1
            
            try:
                self.ctx.decode(batch)
                self.n_decode += 1
            except ValueError as e:
                print(f"error: {e}")
                break
        
        print()  # New line after generation
        
        # Print performance
        self._print_performance()
        
        return response
    
    def _interactive_mode(self, args: argparse.Namespace):
        """Run in interactive mode."""
        print("== Running in interactive mode. ==")
        print(" - Press Ctrl+C to interject at any time.")
        print(" - Press Return to return control to the AI.")
        print(" - To return control without starting a new line, end your input with '/'.")
        print(" - If you want to submit another line, end your input with '\\'.")
        print()
        
        self.interactive = True
        self.is_interacting = args.interactive_first
        
        while True:
            if self.is_interacting:
                print("\n> ", end="", flush=True)
                
                # Read user input
                try:
                    user_input = input()
                except EOFError:
                    print("EOF by user")
                    break
                
                if not user_input:
                    continue
                
                # Process input
                if user_input.endswith('\\'):
                    # Multi-line input
                    user_input = user_input[:-1]
                    while True:
                        try:
                            line = input()
                            if line.endswith('\\'):
                                user_input += line[:-1]
                            else:
                                user_input += line
                                break
                        except EOFError:
                            break
                
                # Add input prefix/suffix
                if args.in_prefix:
                    user_input = args.in_prefix + user_input
                if args.in_suffix:
                    user_input = user_input + args.in_suffix
                
                # Tokenize and generate
                input_tokens = self.vocab.tokenize(user_input, add_special=False, parse_special=True)
                if input_tokens:
                    self._generate_text(args, input_tokens)
                
                self.is_interacting = False
    
    def run(self):
        """Main entry point."""
        args = self._parse_args()
        
        # Handle embedding mode
        if args.embedding:
            print("************")
            print("please use the 'embedding' tool for embedding calculations")
            print("************")
            return 0
        
        # Validate context size
        if args.ctx_size != 0 and args.ctx_size < 8:
            print("warning: minimum context size is 8, using minimum size.")
            args.ctx_size = 8
        
        # Load model
        self._load_model(args)
        
        # Load prompt
        prompt = self._load_prompt(args)
        
        # Tokenize prompt
        prompt_tokens = self._tokenize_prompt(prompt)
        
        # Run in interactive mode or generate text
        if args.interactive or args.interactive_first:
            self._interactive_mode(args)
        else:
            self._generate_text(args, prompt_tokens)
        
        return 0


def main():
    """Main entry point for the CLI."""
    cy.disable_logging()
    cli = LlamaCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
