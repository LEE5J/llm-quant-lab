#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="CPU-offload smoke inference for safetensors models")
    p.add_argument("--model-path", required=True, help="local model dir")
    p.add_argument("--prompt", default="안녕하세요. 한 줄 자기소개를 해주세요.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--offload-dir", default="offload")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--force-cpu", action="store_true", help="force full CPU (no GPU)")
    args = p.parse_args()

    model_path = Path(args.model_path)
    offload_dir = Path(args.offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )

    print("[2/4] selecting device map...")
    if args.force_cpu or not torch.cuda.is_available():
        device_map = {"": "cpu"}
        dtype = torch.float32
    else:
        device_map = "auto"
        dtype = torch.float16

    print(f"device_map={device_map}, dtype={dtype}")

    print("[3/4] loading model with offload options...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        device_map=device_map,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        torch_dtype=dtype,
    )

    print("[4/4] running smoke generation...")
    inputs = tokenizer(args.prompt, return_tensors="pt")

    # put inputs on the first available execution device
    if hasattr(model, "device"):
        exec_device = model.device
    else:
        exec_device = torch.device("cpu")

    inputs = {k: v.to(exec_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()
