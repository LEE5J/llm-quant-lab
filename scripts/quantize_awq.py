#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="AWQ quantization stub")
    p.add_argument("--model-id", required=True)
    p.add_argument("--w-bit", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # TODO: 실제 AWQ 파이프라인 연결
    # from awq import AutoAWQForCausalLM
    # from transformers import AutoTokenizer
    # ... calibrate -> quantize -> save ...

    print(f"[DRY] AWQ quantization placeholder")
    print(f"model_id={args.model_id}, w_bit={args.w_bit}, group_size={args.group_size}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
