#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.quantization.calibration_profiles import (
    KOREAN_MIX_V1,
    build_calibration_dataset,
    build_chat_templated_calibration_dataset,
    profile_description,
    profile_names,
)


def _build_default_recipe() -> str:
    return """
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


def _build_exaone45_recipe() -> str:
    import yaml

    mappings = []
    for i in range(64):
        base = f"model.language_model.layers.{i}"
        mappings.append({
            "smooth_layer": f"{base}.post_attention_layernorm",
            "balance_layers": [f"{base}.mlp.gate_proj", f"{base}.mlp.up_proj"],
        })
        mappings.append({
            "smooth_layer": f"{base}.mlp.up_proj",
            "balance_layers": [f"{base}.mlp.down_proj"],
        })

    recipe_obj = {
        "quant_stage": {
            "quant_modifiers": {
                "AWQModifier": {
                    "ignore": [
                        "lm_head",
                        r"re:model\\.visual\\..*",
                        r"re:model\\.multi_modal_projector\\..*",
                        r"re:model\\.image_newline.*",
                        r"re:mtp\\..*",
                    ],
                    "offload_device": "cpu",
                    "duo_scaling": False,
                    "mappings": mappings,
                    "config_groups": {
                        "group_0": {
                            "targets": ["Linear"],
                            "input_activations": None,
                            "output_activations": None,
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "symmetric": False,
                                "strategy": "group",
                                "group_size": 128,
                            },
                        }
                    },
                }
            }
        }
    }
    return yaml.safe_dump(recipe_obj, sort_keys=False)


def _load_exaone45_model_and_tokenizer(model_id: str):
    import copy
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.exaone4_5 import Exaone4_5_ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    orig_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()
    source_snapshot = Path(
        snapshot_download(
            model_id,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ],
        )
    )
    model = Exaone4_5_ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "24GiB", "cpu": "110GiB"},
        offload_folder="/mnt/quant-data/work/offload/exaone45-awq",
        low_cpu_mem_usage=True,
    )
    model.config.num_nextn_predict_layers = 0
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.num_nextn_predict_layers = 0
    return model, tokenizer, copy.deepcopy(orig_config), source_snapshot


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

    if args.model_id != "LGAI-EXAONE/EXAONE-4.5-33B":
        if args.calibration_profile != "none":
            dataset_arg = build_calibration_dataset(
                profile_name=args.calibration_profile,
                num_samples=args.num_calibration_samples,
                seed=args.seed,
            )
            print(f"using calibration profile: {args.calibration_profile}")
            print(profile_description(args.calibration_profile))
        else:
            dataset_arg = args.dataset

    if args.model_id == "LGAI-EXAONE/EXAONE-4.5-33B":
        import json
        import shutil

        recipe = _build_exaone45_recipe()
        model, tokenizer, orig_config, source_snapshot = _load_exaone45_model_and_tokenizer(args.model_id)
        if args.calibration_profile != "none":
            dataset_arg = build_chat_templated_calibration_dataset(
                tokenizer=tokenizer,
                profile_name=args.calibration_profile,
                num_samples=args.num_calibration_samples,
                seed=args.seed,
                max_length=args.max_seq_length,
            )
            print(f"using chat-templated calibration profile: {args.calibration_profile}")
            print(profile_description(args.calibration_profile))
        else:
            dataset_arg = args.dataset
        print("running llm-compressor oneshot (AWQ path, EXAONE 4.5 custom recipe)...")
        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe,
            dataset=dataset_arg,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
            output_dir=str(out),
            sequential_offload_device="cpu",
        )

        cfg_path = out / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["architectures"] = orig_config.get("architectures", cfg.get("architectures"))
        if "vision_config" in orig_config:
            cfg["vision_config"] = orig_config["vision_config"]
        text_cfg = cfg.get("text_config", {})
        orig_text_cfg = orig_config.get("text_config", {})
        for key in [
            "architectures",
            "model_type",
            "num_hidden_layers",
            "num_nextn_predict_layers",
            "layer_types",
        ]:
            if key in orig_text_cfg:
                text_cfg[key] = orig_text_cfg[key]
        cfg["text_config"] = text_cfg
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

        for name in [
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
        ]:
            src = source_snapshot / name
            dst = out / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
    else:
        recipe = _build_default_recipe()
        oneshot_kwargs = dict(
            model=args.model_id,
            recipe=recipe,
            trust_remote_code_model=args.trust_remote_code,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
            output_dir=str(out),
        )
        if args.calibration_profile == "none":
            oneshot_kwargs.update(
                dataset_config_name=args.dataset_config_name,
                splits=args.splits,
            )
            print(f"using legacy dataset: {args.dataset}/{args.dataset_config_name} split={args.splits}")
        print("running llm-compressor oneshot (AWQ path)...")
        oneshot(dataset=dataset_arg, **oneshot_kwargs)

    print(f"done: {out}")


if __name__ == "__main__":
    main()
