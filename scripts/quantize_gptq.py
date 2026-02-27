#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def load_calib_texts(calib_file: str | None, max_samples: int) -> list[str]:
    if not calib_file:
        # Generic fallback prompts (KR/EN mixed) for architecture-agnostic quick calibration
        defaults = [
            "안녕하세요. 오늘 날씨를 한 문장으로 설명해 주세요.",
            "다음 문장을 영어로 번역해 주세요: 저는 양자화 실험을 하고 있습니다.",
            "Write a short Python function to compute fibonacci numbers.",
            "Summarize the benefits and risks of model quantization.",
            "한국어와 영어를 번갈아 한 문장씩 작성해 주세요.",
            "Give me three bullet points about GPU memory optimization.",
            "다음 텍스트를 2문장으로 요약해 주세요: 대규모 언어 모델의 추론 비용은 ...",
            "Explain attention mechanism in 5 sentences.",
        ]
        return defaults[:max_samples]

    p = Path(calib_file)
    if not p.exists():
        raise FileNotFoundError(f"calibration file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        texts = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if text:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break
        return texts

    # plain txt: one sample per line
    texts = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return texts[:max_samples]


def build_examples(tokenizer, texts: list[str], max_seq_len: int):
    examples = []
    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        examples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    return examples


def main():
    p = argparse.ArgumentParser(description="GPTQ quantization via AutoGPTQ")
    p.add_argument("--model-id", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calib-file", default=None, help="txt(jsonl) file for calibration samples")
    p.add_argument("--max-calib-samples", type=int, default=64)
    p.add_argument("--max-calib-seq-len", type=int, default=512)
    p.add_argument("--desc-act", action="store_true", help="enable desc_act")
    p.add_argument("--damp-percent", type=float, default=0.1)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-triton", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/5] preparing calibration dataset...")
    texts = load_calib_texts(args.calib_file, args.max_calib_samples)
    if not texts:
        raise RuntimeError("no calibration texts loaded")
    examples = build_examples(tokenizer, texts, args.max_calib_seq_len)
    print(f"calib_samples={len(examples)}")

    print("[3/5] loading model...")
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        damp_percent=args.damp_percent,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_id,
        quantize_config=quantize_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )

    print("[4/5] quantizing...")
    model.quantize(examples, use_triton=args.use_triton)

    print("[5/5] saving...")
    model.save_quantized(str(out), use_safetensors=True)
    tokenizer.save_pretrained(str(out))

    print("done")
    print(f"model_id={args.model_id}")
    print(f"bits={args.bits}, group_size={args.group_size}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
