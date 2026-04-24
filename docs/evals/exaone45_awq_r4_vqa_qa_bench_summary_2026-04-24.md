# EXAONE-4.5-33B-AWQ-R4 QA 중심 VQA 벤치마크 서브셋 요약

- 실행일: 2026-04-24
- 모델: `EXAONE-4.5-33B-AWQ-R4`
- 산출물: `/mnt/quant-data/work/awq/exaone45-33b-awq-4bit-fullstruct-mtp-r4-chat128-l512`
- 실행 도구: `scripts/benchmark/vqa_benchmark_subset.py`
- 실행 방식: vLLM OpenAI-compatible `/v1/chat/completions`, image_url 입력, `temperature=0`
- 샘플 수: 각 태스크 5개, 총 45개

## 실행한 태스크
요청 목록 중 QA/VQA 성격이 강한 항목을 우선 실행했다.

- MMMU
- MMMU-Pro
- MathVista-mini
- MathVision
- WeMath
- LogicVista
- Charxiv-RQ
- K-Viscuit
- KRETA

BLINK는 public split에서 answer가 hidden인 구성이 있어 이번 자동 채점 대상에서는 제외했다.

## 결과 표

| Task | N | Correct | Accuracy |
|---|---:|---:|---:|
| MMMU | 5 | 0 | 0.00 |
| MMMU-Pro | 5 | 1 | 0.20 |
| MathVista-mini | 5 | 0 | 0.00 |
| MathVision | 5 | 1 | 0.20 |
| WeMath | 5 | 0 | 0.00 |
| LogicVista | 5 | 1 | 0.20 |
| Charxiv-RQ | 5 | 0 | 0.00 |
| K-Viscuit | 5 | 1 | 0.20 |
| KRETA | 5 | 0 | 0.00 |

총합으로는 45개 중 4개 정답으로, 단순 정확도는 약 8.9%다.

## 정성 관찰

1. 모델이 지시한 "정답 letter만 출력"을 잘 따르지 않는다.
   - 많은 샘플에서 정답 대신 긴 reasoning/explanation을 생성한다.
   - 이 때문에 multiple-choice 태스크에서 answer extraction이 실패하는 경우가 많았다.

2. 이미지 이해 자체도 불안정하다.
   - K-Viscuit, KRETA 같은 비교적 단순한 이미지 선택형에서도 오답이 많았다.
   - KRETA는 OCR/한국어 시각 정보가 필요한데 거의 맞추지 못했다.

3. 수학/차트 계열은 특히 약하다.
   - MathVista-mini, WeMath, Charxiv-RQ는 이번 5샘플 서브셋에서 0점이다.
   - 출력은 문제를 해석하려고 하지만 최종값까지 안정적으로 도달하지 못했다.

4. 이전 정성평가와 같은 반복/형식 붕괴 경향이 남아 있다.
   - 단, 이번에는 image QA 프롬프트라 긴 반복보다는 "설명만 하고 final answer를 내지 않는" 형태가 더 두드러졌다.

## 해석상 주의

- 각 태스크 5개만 본 빠른 서브셋이므로 공식 점수는 아니다.
- 다만 모든 태스크에서 일관되게 낮은 결과가 나왔기 때문에, 현재 R4 산출물은 QA/VQA 벤치마크용 품질로 보기 어렵다.
- especially QA 관련 calibration을 넣었지만, 이 결과만 보면 multimodal QA/general VQA 성능 개선으로 이어졌다고 보기는 어렵다.

## 다음 권장 실험

1. 같은 스크립트로 원본 FP/BF16 모델을 같은 45샘플에 돌려 baseline을 만든다.
2. R4와 원본 baseline 차이를 비교해 양자화 손실인지, 현재 fork/vLLM/프롬프트 문제인지 분리한다.
3. decoding을 `max_tokens=16`, stop/regex post-processing, "letter only" chat template 방식으로 바꿔 answer extraction 실패를 줄인다.
4. 이후 성능이 의미 있게 나오면 샘플 수를 5 -> 50 또는 100으로 늘린다.

## 파일

- raw JSON: `docs/evals/exaone45_awq_r4_vqa_qa_bench_subset_2026-04-24.json`
- 상세 MD: `docs/evals/exaone45_awq_r4_vqa_qa_bench_subset_2026-04-24.md`
- 요약 MD: `docs/evals/exaone45_awq_r4_vqa_qa_bench_summary_2026-04-24.md`
