#!/usr/bin/env python3
"""Backward-compatible wrapper.
Use scripts/benchmark/smoke_infer.py directly for new workflows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark.smoke_infer import main


if __name__ == "__main__":
    main()
