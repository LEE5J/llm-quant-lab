# llm-quant-lab

최근 공개 LLM 중에서 **GGUF 미지원** 또는 **GPTQ/AWQ 4bit/8bit 미제공** 모델을 대상으로,
직접 양자화하고 Hugging Face에 배포하기 위한 실험 레포입니다.

## 목표

1. 타깃 모델 선별 (양자화 공백 확인)
2. AWQ / GPTQ / GGUF 변환 파이프라인 실행
3. 품질 점검 (간단 벤치 + 샘플 추론)
4. Hugging Face 업로드 자동화

---

## 빠른 시작

```bash
git clone https://github.com/LEE5J/llm-quant-lab.git
cd llm-quant-lab
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

환경변수 설정:

```bash
cp .env.example .env
# .env 파일에 HF_TOKEN, CUDA_DEVICE 등 필요 값 입력
```

---

## 구조

- `configs/models.yaml` : 타깃 모델 목록/상태
- `scripts/quantization/awq.py` : AWQ 양자화
- `scripts/quantization/gptq.py` : GPTQ 양자화 (llmcompressor 기반)
- `scripts/quantization/gguf.py` : GGUF 변환
- `scripts/benchmark/smoke_infer.py` : 벤치마크/스모크 추론
- `scripts/upload_hf.py` : HF 업로드

(호환용 래퍼: `scripts/quantize_awq.py`, `scripts/quantize_gptq.py`, `scripts/quantize_gptq_llmc.py`, `scripts/export_gguf.sh`, `scripts/smoke_infer_cpu_offload.py`)

권장 산출물 네이밍: `<model>-<release>-awq-w4a16` (예: `kanana-2-30b-a3b-instruct-2601-awq-w4a16`)
- `MODEL_TRACKER.md` : 모델별 진행 현황

---

## 주의

- 원본 모델 라이선스를 반드시 준수하세요.
- 모델 카드에 원본 레포 링크/라이선스/변경사항을 명시하세요.
- 일부 최신 아키텍처는 GGUF 변환이 아직 불가능할 수 있습니다.


## 환경 호환성

- GPTQ/AWQ 경로는 `llmcompressor` 기준으로 관리합니다.
- 현재 검증 대상 버전 범위는 `transformers<=4.57.6`, `accelerate<=1.12.0`, `torch<=2.10.0` 입니다.
- `requirements-gptq.txt`는 별도 백엔드 의존성 파일이 아니라, GPTQ도 `requirements.txt` 하나로 처리한다는 안내만 남겨둡니다.
