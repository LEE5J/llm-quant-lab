# Workflow

## 1) 후보 모델 조사
- MODEL_TRACKER.md에 후보 추가
- GGUF / GPTQ / AWQ 존재 여부 체크

## 2) 양자화
- llm-compressor 양자화:
  ```bash
  python scripts/quantization/awq.py \
    --model-id <repo> \
    --output-dir results/<name>-w4a16 \
    --trust-remote-code
  ```
- 기본 캘리브레이션 프로필은 `korean-mix-v1` 입니다.
- 상세한 데이터셋 구성은 `docs/calibration.md` 참고
- GPTQ (llmcompressor):
  ```bash
  python scripts/quantization/gptq.py \
    --model-id <repo> \
    --num-calibration-samples 64 \
    --max-seq-length 512 \
    --trust-remote-code \
    --output-dir results/<name>-gptq-w4a16
  ```
- 기본 캘리브레이션 프로필은 `korean-mix-v1` 입니다.
- 상세한 데이터셋 구성은 `docs/calibration.md` 참고

## 3) GGUF 변환 (가능 시)
```bash
python scripts/quantization/gguf.py --model-src <local_model_path> --output-file results/<name>.gguf --qtype Q4_K_M
```

## 4) 네이밍 정규화
- 형식: `<model>-<release>-awq-w4a16`

## 5) HF 업로드
```bash
python scripts/upload_hf.py --local-dir results/<artifact> --repo-id <hf_user>/<repo-name>
```

## 6) 검증 기록
- 샘플 프롬프트 응답
- VRAM/속도
- 품질 저하 체감 포인트
