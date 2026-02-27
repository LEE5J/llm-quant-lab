#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_with_autoawq(model_path: str, tokenizer, prompt: str, max_new_tokens: int):
    from awq import AutoAWQForCausalLM

    print("[AutoAWQ] loading quantized model...")
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=False,
        safetensors=True,
        trust_remote_code=True,
        device_map="cpu",
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_with_transformers(
    model_path: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    offload_dir: str,
    trust_remote_code: bool,
    force_cpu: bool,
):
    print("[Transformers] selecting device map...")
    if force_cpu or not torch.cuda.is_available():
        device_map = {"": "cpu"}
        dtype = torch.float32
    else:
        device_map = "auto"
        dtype = torch.float16

    print(f"device_map={device_map}, dtype={dtype}")
    print("[Transformers] loading model with offload options...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        device_map=device_map,
        offload_folder=offload_dir,
        offload_state_dict=True,
        dtype=dtype,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    exec_device = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = {k: v.to(exec_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="CPU-offload smoke inference for safetensors models")
    p.add_argument("--model-path", required=True, help="local model dir")
    p.add_argument("--prompt", default="안녕하세요. 한 줄 자기소개를 해주세요.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--offload-dir", default="offload")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--force-cpu", action="store_true", help="force full CPU (no GPU)")
    args = p.parse_args()

    model_path = str(Path(args.model_path))
    offload_dir = Path(args.offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)

    print("[2/3] trying AutoAWQ path first...")
    try:
        text = run_with_autoawq(model_path, tokenizer, args.prompt, args.max_new_tokens)
        print("[3/3] done (AutoAWQ)")
        print("=== OUTPUT ===")
        print(text)
        return
    except Exception as e:
        print(f"AutoAWQ path failed: {e}")
        print("Falling back to Transformers path...")

    text = run_with_transformers(
        model_path=model_path,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        offload_dir=str(offload_dir),
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
    )
    print("[3/3] done (Transformers)")
    print("=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()
