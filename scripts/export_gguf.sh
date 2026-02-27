#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/export_gguf.sh <hf_model_dir_or_repo> <output_file.gguf> [qtype]
# Example qtype: Q4_K_M, Q5_K_M, Q8_0

MODEL_SRC=${1:-}
OUT_FILE=${2:-}
QTYPE=${3:-Q4_K_M}

if [[ -z "$MODEL_SRC" || -z "$OUT_FILE" ]]; then
  echo "Usage: $0 <hf_model_dir_or_repo> <output_file.gguf> [qtype]"
  exit 1
fi

# TODO: llama.cpp 경로/스크립트 경로 환경에 맞게 설정
# python /path/to/llama.cpp/convert_hf_to_gguf.py "$MODEL_SRC" --outfile "$OUT_FILE" --outtype "$QTYPE"

echo "[DRY] GGUF export placeholder"
echo "MODEL_SRC=$MODEL_SRC"
echo "OUT_FILE=$OUT_FILE"
echo "QTYPE=$QTYPE"
