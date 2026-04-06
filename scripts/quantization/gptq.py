#!/usr/bin/env python3
import argparse
from pathlib import Path


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(description='GPTQ quantization via llmcompressor (W4A16)')
    p.add_argument('--model-id', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-seq-length', type=int, default=512)
    p.add_argument('--num-calibration-samples', type=int, default=64)
    p.add_argument('--dataset', default='wikitext')
    p.add_argument('--dataset-config-name', default='wikitext-2-raw-v1')
    p.add_argument('--splits', default='train')
    p.add_argument('--preprocessing-num-workers', type=int, default=8)
    p.add_argument('--dataloader-num-workers', type=int, default=2)
    p.add_argument(
        '--ignore',
        nargs='*',
        default=[
            'lm_head',
            're:model\\.vision_tower\\..*',
            're:model\\.embed_vision\\..*',
            're:embed_vision\\..*',
            're:model\\.multi_modal_projector\\..*',
            're:multi_modal_projector\\..*',
        ],
    )
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

    ignore_yaml = '[' + ', '.join(_yaml_quote(name) for name in args.ignore) + ']'
    recipe = f"""
quant_stage:
  quant_modifiers:
    GPTQModifier:
      targets: ['Linear']
      scheme: 'W4A16'
      ignore: {ignore_yaml}
"""

    print('running llmcompressor oneshot (GPTQ path)...')
    oneshot(
        model=args.model_id,
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
