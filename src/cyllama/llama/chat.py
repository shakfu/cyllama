#!/usr/bin/env python3
"""
Chat implementation equivalent to build/llama.cpp/examples/simple-chat/simple-chat.cpp

This module provides a Python implementation of the chat example using the cyllama wrapper.
"""

import sys
import argparse
from typing import List, Optional, Dict, Any

from . import (
    LlamaModel, LlamaContext, LlamaSampler, LlamaBatch, LlamaChatMessage,
    LlamaModelParams, LlamaContextParams, LlamaSamplerChainParams,
    set_log_callback, ggml_backend_load_all, llama_batch_get_one,
    disable_logging
)


def print_usage():
    """Print usage information"""
    print("\nexample usage:")
    print(f"\n    {sys.argv[0]} -m model.gguf [-c context_size] [-ngl n_gpu_layers]")
    print()


class Chat:
    """Chat interface using cyllama"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, ngl: int = 99):
        """Initialize the chat with model and parameters"""
        # Set up error-only logging (skip for now to avoid issues)
        # set_log_callback(lambda level, text: sys.stderr.write(text) if level >= 3 else None)
        
        # Load dynamic backends
        ggml_backend_load_all()
        
        # Initialize model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = ngl
        self.model = LlamaModel(model_path, model_params)
        
        # Get vocab
        self.vocab = self.model.get_vocab()
        
        # Initialize context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_ctx
        self.context = LlamaContext(self.model, ctx_params)
        
        # Initialize sampler
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        self.sampler.add_min_p(0.05, 1)
        self.sampler.add_temp(0.8)
        self.sampler.add_dist(1337)  # Use fixed seed like LLAMA_DEFAULT_SEED
        
        # Initialize chat state (simplified - use list of dicts instead of LlamaChatMessage objects)
        self.messages: List[Dict[str, str]] = []
        self.prev_len = 0
        self.n_past = 0  # Track position in context
        
        # Store parameters for creating fresh contexts
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.ngl = ngl
    
    def generate(self, prompt: str) -> str:
        """Generate response for the given prompt using a fresh context"""
        # Create a fresh context for this generation to avoid state issues
        fresh_context = LlamaContext(self.model, self.context.params, verbose=False)
        
        response = ""
        n_past = 0
        
        # Tokenize the prompt
        prompt_tokens = self.vocab.tokenize(prompt, True, True)
        
        if not prompt_tokens:
            raise ValueError("Failed to tokenize the prompt")
        
        # Create batch for the prompt starting from position 0
        batch = llama_batch_get_one(prompt_tokens, 0)
        n_past = len(prompt_tokens)
        
        # Decode the initial batch
        try:
            ret = fresh_context.decode(batch)
            if ret != 0:
                print(f"Warning: decode returned {ret}")
        except Exception as e:
            print(f"Decode failed with error: {e}")
            return response
        
        # Generation loop
        max_tokens = 50  # Reasonable limit for chat responses
        consecutive_spaces = 0  # Track consecutive spaces/newlines to avoid infinite generation
        
        for i in range(max_tokens):
            # Check context size
            n_ctx = fresh_context.n_ctx
            if n_past >= n_ctx - 1:  # Leave room for at least one more token
                print("\033[0m")
                print("context size exceeded", file=sys.stderr)
                break
            
            # Sample next token
            new_token_id = self.sampler.sample(fresh_context, -1)
            
            # Check if it's end of generation
            if self.vocab.is_eog(new_token_id):
                break
            
            # Convert token to piece and add to response
            try:
                piece = self.vocab.token_to_piece(new_token_id, 0, True)
                
                # Stop generating if we get too many consecutive spaces/newlines
                if piece.strip() == "":
                    consecutive_spaces += 1
                    if consecutive_spaces >= 3:  # Stop after 3 consecutive whitespace tokens
                        break
                else:
                    consecutive_spaces = 0
                
                print(piece, end='', flush=True)
                response += piece
                
            except Exception as e:
                print(f"Failed to convert token to piece: {e}")
                break
            
            # Create batch with the new token at the correct position
            batch = llama_batch_get_one([new_token_id], n_past)
            n_past += 1
            
            # Decode for next iteration
            try:
                ret = fresh_context.decode(batch)
                if ret != 0:
                    print(f"Warning: decode returned {ret}")
            except Exception as e:
                print(f"Decode failed at token {i+1}: {e}")
                break
        
        return response.strip()
    
    def chat_loop(self):
        """Main chat loop"""
        while True:
            # Get user input
            print("\033[32m> \033[0m", end='')
            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                break
            
            tmpl = self.model.get_default_chat_template()

            if tmpl:
                # Use LlamaChatMessage objects for template-based chat
                self.messages.append(LlamaChatMessage(role="user", content=user_input))
                prompt = self.model.chat_apply_template(tmpl, self.messages, add_assistant_msg=True)
            else:
                # Add user message to conversation history using dict format
                self.messages.append({"role": "user", "content": user_input})

                # Format conversation with User/Assistant format (this works well with this model)
                conversation = ""
                for msg in self.messages:
                    if msg["role"] == "user":
                        conversation += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        conversation += f"Assistant: {msg['content']}\n"
                
                # Add prompt for the assistant to continue
                conversation += "Assistant:"
                
                # Extract the prompt (new part since last message)
                prompt = conversation[self.prev_len:]
            
            # Generate response
            print("\033[33m", end='')
            try:
                response = self.generate(prompt)
                print("\n\033[0m")
                
                # Add assistant message
                if tmpl:
                    # Use LlamaChatMessage for template-based chat
                    self.messages.append(LlamaChatMessage(role="assistant", content=response))
                else:
                    # Use dict format for non-template chat
                    self.messages.append({"role": "assistant", "content": response})
                    
                    # Update previous length for next iteration
                    full_conversation = ""
                    for msg in self.messages:
                        if msg["role"] == "user":
                            full_conversation += f"User: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            full_conversation += f"Assistant: {msg['content']}\n"
                    self.prev_len = len(full_conversation)
                
            except Exception as e:
                print(f"\nError generating response: {e}", file=sys.stderr)
                # Remove the user message that caused the error
                self.messages.pop()


def main():
    """Main entry point"""
    disable_logging()
    parser = argparse.ArgumentParser(description="Simple chat using cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    parser.add_argument("-c", "--context", type=int, default=2048, help="Context size")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="Number of GPU layers")
    
    args = parser.parse_args()
    
    try:
        chat = Chat(args.model, args.context, args.n_gpu_layers)
        chat.chat_loop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()