#!/usr/bin/env python3
import argparse
from pathlib import Path


DEFAULT_IGNORE = [
    'lm_head',
    r're:model\.visual\..*',
    r're:model\.merger\..*',
]


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_recipe(ignore: list[str], scheme: str) -> str:
    ignore_yaml = '[' + ', '.join(_yaml_quote(name) for name in ignore) + ']'
    return f"""
quant_stage:
  quant_modifiers:
    GPTQModifier:
      targets: ['Linear']
      scheme: '{scheme}'
      ignore: {ignore_yaml}
"""


def load_model(model_id: str, trust_remote_code: bool):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )

    if hasattr(model.config, 'num_nextn_predict_layers'):
        model.config.num_nextn_predict_layers = 0
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_nextn_predict_layers'):
        model.config.text_config.num_nextn_predict_layers = 0

    return model


def main():
    p = argparse.ArgumentParser(
        description='GPTQ quantization via llmcompressor for EXAONE 4.5 multimodal model'
    )
    p.add_argument('--model-id', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--scheme', default='W4A16', choices=['W4A16', 'W8A16'])
    p.add_argument('--max-seq-length', type=int, default=512)
    p.add_argument('--num-calibration-samples', type=int, default=64)
    p.add_argument('--dataset', default='wikitext')
    p.add_argument('--dataset-config-name', default='wikitext-2-raw-v1')
    p.add_argument('--splits', default='train')
    p.add_argument('--preprocessing-num-workers', type=int, default=8)
    p.add_argument('--dataloader-num-workers', type=int, default=2)
    p.add_argument('--ignore', nargs='*', default=DEFAULT_IGNORE)
    p.add_argument('--trust-remote-code', action='store_true')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from llmcompressor import oneshot
    except Exception as e:
        raise SystemExit(
            'llmcompressor import failed. Install compatible deps first:\n'
            '  python -m pip install -r requirements.txt\n'
            f'detail: {e}'
        )

    recipe = build_recipe(args.ignore, args.scheme)

    print('loading model...')
    model = load_model(args.model_id, args.trust_remote_code)

    print('running llmcompressor oneshot (GPTQ path)...')
    print(f'scheme: {args.scheme}')
    print('ignore patterns:')
    for item in args.ignore:
        print(f'  - {item}')

    oneshot(
        model=model,
        processor=args.model_id,
        recipe=recipe,
        trust_remote_code_model=args.trust_remote_code,
        dataset=args.dataset,
        dataset_config_name=args.dataset_config_name,
        splits=args.splits,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        dataloader_num_workers=args.dataloader_num_workers,
        output_dir=str(out),
    )

    print(f'done: {out}')


if __name__ == '__main__':
    main()
