# Calibration profiles

## 기본 프로필: `korean-mix-v1`

앞으로 기본 캘리브레이션 프로필은 `korean-mix-v1` 입니다.
이 프로필은 한국어 양자화 품질을 높이기 위해 아래 3개 도메인을 균등 혼합합니다.

1. `ko_wiki_sentences_100000`
2. `KoAlpaca-RealQA` (Q&A 형식)
3. `korean_textbooks` (장문·지식)

## 실제 기본 소스

원 요청 이름과 접근성/데이터 상태를 반영해 아래 소스를 기본 사용합니다.

- `ko_wiki_sentences_100000`
  - 기본 소스: `wikimedia/wikipedia` / config `20231101.ko`
  - 이유: 공개 HF 상의 `jonghwi/ko_wiki_sentences_1000000` 는 현재 dataset_size=0 상태라 실사용 불가
- `KoAlpaca-RealQA`
  - 기본 소스: `juyoung-trl/KoAlpaca-RealQA`
  - 이유: 원본 `beomi/KoAlpaca-RealQA` 는 gated 접근 제한
- `korean_textbooks`
  - 기본 소스: `maywell/korean_textbooks` / config `tiny-textbooks`
  - 이유: 장문·지식 성격의 텍스트 비중이 높아 캘리브레이션용으로 적합

## 샘플링 정책

- 3개 도메인 균등 혼합
- `--num-calibration-samples N` 이면 각 도메인에 대해 대략 `N/3` 개씩 수집
- 최종 텍스트는 `datasets.Dataset` 의 `text` 컬럼으로 합쳐서 `llmcompressor.oneshot()` 에 전달
- 한국어가 포함된 텍스트만 사용

## 사용 예시

### GPTQ

```bash
python scripts/quantization/gptq.py \
  --model-id LGAI-EXAONE/EXAONE-4.5-33B \
  --output-dir results/exaone45-33b-gptq-w4a16 \
  --num-calibration-samples 96 \
  --max-seq-length 512 \
  --trust-remote-code
```

### AWQ

```bash
python scripts/quantization/awq.py \
  --model-id LGAI-EXAONE/EXAONE-4.5-33B \
  --output-dir results/exaone45-33b-awq-w4a16 \
  --num-calibration-samples 96 \
  --max-seq-length 512 \
  --trust-remote-code
```

### EXAONE 4.5 GPTQ 전용 경로

```bash
python scripts/quantization/gptq_exaone4.5.py \
  --model-id LGAI-EXAONE/EXAONE-4.5-33B \
  --output-dir results/exaone45-33b-gptq-w4a16 \
  --num-calibration-samples 96 \
  --max-seq-length 512 \
  --trust-remote-code
```

## 레거시 데이터셋으로 강제 전환

기존 단일 데이터셋 방식이 필요하면 아래처럼 profile 을 끌 수 있습니다.

```bash
python scripts/quantization/gptq.py \
  --calibration-profile none \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --splits train \
  ...
```
