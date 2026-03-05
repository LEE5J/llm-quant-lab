#!/usr/bin/env python3
"""Backward-compatible wrapper.
Use scripts/quantization/awq.py directly for new workflows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.quantization.awq import main


if __name__ == "__main__":
    main()
