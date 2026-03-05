#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Usage:
#   ./scripts/export_gguf.sh <hf_model_dir_or_repo> <output_file.gguf> [qtype]

MODEL_SRC=${1:-}
OUT_FILE=${2:-}
QTYPE=${3:-Q4_K_M}

if [[ -z "$MODEL_SRC" || -z "$OUT_FILE" ]]; then
  echo "Usage: $0 <hf_model_dir_or_repo> <output_file.gguf> [qtype]"
  exit 1
fi

python scripts/quantization/gguf.py \
  --model-src "$MODEL_SRC" \
  --output-file "$OUT_FILE" \
  --qtype "$QTYPE"
