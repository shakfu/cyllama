#!/usr/bin/env python3
"""
Quick start script for interactive cyllama usage.

Usage:
    python start.py -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
"""
import sys
import argparse

sys.path.insert(0, 'src')

import cyllama.cyllama as cy
from cyllama import Llama

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick start for cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()

    llm = Llama(model_path=args.model)