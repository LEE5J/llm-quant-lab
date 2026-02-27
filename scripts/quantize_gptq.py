#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="GPTQ quantization stub")
    p.add_argument("--model-id", required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # TODO: AutoGPTQ/Optimum GPTQ 실제 양자화 연결

    print(f"[DRY] GPTQ quantization placeholder")
    print(f"model_id={args.model_id}, bits={args.bits}, group_size={args.group_size}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
