#!/usr/bin/env python3
import argparse
from pathlib import Path

from scripts.quantization.calibration_profiles import (
    KOREAN_MIX_V1,
    build_calibration_dataset,
    profile_description,
    profile_names,
)


def main():
    p = argparse.ArgumentParser(description="AWQ quantization via llm-compressor (W4A16)")
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--num-calibration-samples", type=int, default=64)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    p.add_argument("--splits", default="train")
    p.add_argument("--calibration-profile", default=KOREAN_MIX_V1, choices=profile_names() + ["none"])
    p.add_argument("--seed", type=int, default=42)
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
    AWQModifier:
      ignore: ['lm_head']
      config_groups:
        group_0:
          targets: ['Linear']
          input_activations: null
          output_activations: null
          weights:
            num_bits: 4
            type: int
            symmetric: false
            strategy: group
            group_size: 128
"""

    oneshot_kwargs = dict(
        model=args.model_id,
        recipe=recipe,
        trust_remote_code_model=args.trust_remote_code,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        output_dir=str(out),
    )

    if args.calibration_profile != 'none':
        dataset_arg = build_calibration_dataset(
            profile_name=args.calibration_profile,
            num_samples=args.num_calibration_samples,
            seed=args.seed,
        )
        print(f'using calibration profile: {args.calibration_profile}')
        print(profile_description(args.calibration_profile))
    else:
        dataset_arg = args.dataset
        oneshot_kwargs.update(
            dataset_config_name=args.dataset_config_name,
            splits=args.splits,
        )
        print(f'using legacy dataset: {args.dataset}/{args.dataset_config_name} split={args.splits}')

    print("running llm-compressor oneshot (AWQ path)...")
    oneshot(dataset=dataset_arg, **oneshot_kwargs)

    print(f"done: {out}")


if __name__ == "__main__":
    main()
