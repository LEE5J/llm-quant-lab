#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Quantize model with llm-compressor GPTQ (W4A16)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--num-calibration-samples", type=int, default=64)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    p.add_argument("--splits", default="train")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from llmcompressor import oneshot

    recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: "W4A16"
      ignore: ["lm_head"]
"""

    oneshot(
        model=args.model_id,
        recipe=recipe,
        trust_remote_code_model=args.trust_remote_code,
        dataset=args.dataset,
        dataset_config_name=args.dataset_config_name,
        splits=args.splits,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        output_dir=str(out),
    )

    print(f"done: {out}")


if __name__ == "__main__":
    main()
