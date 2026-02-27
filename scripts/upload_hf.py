#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description="Upload quantized model folder to Hugging Face")
    p.add_argument("--local-dir", required=True)
    p.add_argument("--repo-id", required=True, help="e.g. username/model-name-w4a16")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None)
    args = p.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"local dir not found: {local_dir}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=args.repo_id)

    print(f"uploaded: {local_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
