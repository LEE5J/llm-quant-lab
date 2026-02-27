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
# (optional) GPTQ
# uv pip install -r requirements-gptq.txt
```

환경변수 설정:

```bash
cp .env.example .env
# .env 파일에 HF_TOKEN, CUDA_DEVICE 등 필요 값 입력
```

---

## 구조

- `configs/models.yaml` : 타깃 모델 목록/상태
- `scripts/quantize_awq.py` : AWQ 양자화
- `scripts/quantize_gptq.py` : GPTQ 양자화
- `scripts/export_gguf.sh` : GGUF 변환(지원 아키텍처)
- `scripts/upload_hf.py` : HF 업로드
- `MODEL_TRACKER.md` : 모델별 진행 현황

---

## 주의

- 원본 모델 라이선스를 반드시 준수하세요.
- 모델 카드에 원본 레포 링크/라이선스/변경사항을 명시하세요.
- 일부 최신 아키텍처는 GGUF 변환이 아직 불가능할 수 있습니다.
