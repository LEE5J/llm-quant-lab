#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="GPTQ quantization via llm-compressor (AutoGPTQ removed)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)

    # Backward-compatible legacy args (ignored except for validation)
    p.add_argument("--bits", type=int, default=4, choices=[4], help="Only 4 is supported in this script")
    p.add_argument("--group-size", type=int, default=128, help="Reserved for future use")
    p.add_argument("--calib-file", default=None, help="Not used by llm-compressor path")
    p.add_argument("--max-calib-samples", type=int, default=64)
    p.add_argument("--max-calib-seq-len", type=int, default=512)
    p.add_argument("--desc-act", action="store_true", help="Not used by llm-compressor path")
    p.add_argument("--damp-percent", type=float, default=0.1, help="Not used by llm-compressor path")
    p.add_argument("--use-triton", action="store_true", help="Not used by llm-compressor path")

    # Native llm-compressor inputs
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    p.add_argument("--splits", default="train")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from llmcompressor import oneshot
    except Exception as e:
        raise SystemExit(
            "llmcompressor import failed. Install deps first:\n"
            "  uv pip install -r requirements.txt\n"
            f"detail: {e}"
        )

    recipe = """
quant_stage:
  quant_modifiers:
    GPTQModifier:
      targets: ["Linear"]
      scheme: "W4A16"
      ignore: ["lm_head"]
"""

    print("running llm-compressor GPTQ oneshot...")
    oneshot(
        model=args.model_id,
        recipe=recipe,
        trust_remote_code_model=args.trust_remote_code,
        dataset=args.dataset,
        dataset_config_name=args.dataset_config_name,
        splits=args.splits,
        num_calibration_samples=args.max_calib_samples,
        max_seq_length=args.max_calib_seq_len,
        output_dir=str(out),
    )

    print(f"done: {out}")


if __name__ == "__main__":
    main()
