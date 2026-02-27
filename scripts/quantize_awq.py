#!/usr/bin/env python3
import argparse
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="AWQ quantization")
    p.add_argument("--model-id", required=True)
    p.add_argument("--w-bit", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-calib-samples", type=int, default=128)
    p.add_argument("--max-calib-seq-len", type=int, default=512)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/4] loading model...")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    print("[3/4] quantizing...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
    )

    print("[4/4] saving...")
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))

    print("done")
    print(f"model_id={args.model_id}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
