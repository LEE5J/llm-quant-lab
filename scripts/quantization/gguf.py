#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="GGUF export placeholder")
    p.add_argument("--model-src", required=True, help="HF model dir or repo")
    p.add_argument("--output-file", required=True, help="output .gguf file path")
    p.add_argument("--qtype", default="Q4_K_M", help="e.g. Q4_K_M, Q5_K_M, Q8_0")
    args = p.parse_args()

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    # TODO: hook llama.cpp convert_hf_to_gguf.py here for real conversion.
    print("[DRY] GGUF export placeholder")
    print(f"MODEL_SRC={args.model_src}")
    print(f"OUT_FILE={out}")
    print(f"QTYPE={args.qtype}")


if __name__ == "__main__":
    main()
