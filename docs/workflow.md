# Workflow

## 1) 후보 모델 조사
- MODEL_TRACKER.md에 후보 추가
- GGUF / GPTQ / AWQ 존재 여부 체크

## 2) 양자화
- llm-compressor 양자화:
  ```bash
  python scripts/quantize_awq.py \
    --model-id <repo> \
    --output-dir results/<name>-llmc-w4a16 \
    --trust-remote-code
  ```
- GPTQ:
  ```bash
  python scripts/quantize_gptq.py \
    --model-id <repo> \
    --bits 4 \
    --group-size 128 \
    --max-calib-samples 64 \
    --max-calib-seq-len 512 \
    --calib-file <optional_txt_or_jsonl> \
    --trust-remote-code \
    --output-dir results/<name>-gptq4
  ```

## 3) GGUF 변환 (가능 시)
```bash
bash scripts/export_gguf.sh <local_model_path> results/<name>.gguf Q4_K_M
```

## 4) HF 업로드
```bash
python scripts/upload_hf.py --local-dir results/<artifact> --repo-id <hf_user>/<repo-name>
```

## 5) 검증 기록
- 샘플 프롬프트 응답
- VRAM/속도
- 품질 저하 체감 포인트
