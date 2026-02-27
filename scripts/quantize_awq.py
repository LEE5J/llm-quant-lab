#!/usr/bin/env python3
"""
LLM Compressor quantization entry (replaces AutoAWQ path).

Usage example:
python scripts/quantize_awq.py \
  --model-id NCSOFT/Llama-VARCO-8B-Instruct \
  --output-dir results/varco8b-llmc-awq4 \
  --max-seq-length 2048
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Quantize model with llm-compressor (AWQ-like flow)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except Exception as e:
        raise SystemExit(
            "llmcompressor import failed. Install deps first:\n"
            "  uv pip install -r requirements.txt\n"
            f"detail: {e}"
        )

    print("[1/4] loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        device_map="auto",
    )

    print("[2/4] building quantization recipe...")
    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        )
    ]

    print("[3/4] running llm-compressor oneshot...")
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=str(out),
        max_seq_length=args.max_seq_length,
    )

    print("[4/4] saving tokenizer...")
    tokenizer.save_pretrained(str(out))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
