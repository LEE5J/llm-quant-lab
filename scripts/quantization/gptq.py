#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def main():
    p = argparse.ArgumentParser(description="GPTQ quantization via GPTQModel (ModelCloud)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--max-calib-samples", type=int, default=64)
    p.add_argument("--max-calib-seq-len", type=int, default=512)
    p.add_argument("--desc-act", action="store_true")
    p.add_argument("--damp-percent", type=float, default=0.1)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import gptqmodel  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "gptqmodel import failed. Install GPTQModel first:\n"
            "  uv pip install git+https://github.com/ModelCloud/GPTQModel.git\n"
            f"detail: {e}"
        )

    print("[1/4] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/4] building GPTQ config (GPTQModel backend)...")
    quant_cfg = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        desc_act=args.desc_act,
        tokenizer=tokenizer,
        dataset="c4",
    )

    print("[3/4] quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        quantization_config=quant_cfg,
        trust_remote_code=args.trust_remote_code,
    )

    print("[4/4] saving model/tokenizer...")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    print("done")
    print(f"model_id={args.model_id}")
    print(f"bits={args.bits}, group_size={args.group_size}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
