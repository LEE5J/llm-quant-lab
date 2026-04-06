#!/usr/bin/env python3
"""Backward-compatible wrapper for llmcompressor GPTQ path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization.gptq import main

if __name__ == "__main__":
    main()
