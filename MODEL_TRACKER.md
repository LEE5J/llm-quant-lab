# MODEL TRACKER

한국 기업(조직) 공개 모델 중심으로 후보를 추립니다.

Legend: ✅ 있음 / ❌ 검색상 미확인 / ⚠️ 일부(비공식/부분)

| Model | Org (KR) | GGUF | GPTQ 4bit | GPTQ 8bit | AWQ 4bit | AWQ 8bit | Priority | Notes |
|---|---|---|---|---|---|---|---|---|
| LGAI-EXAONE/K-EXAONE-236B-A23B | LG AI Research | ✅ | ❌ | ❌ | ❌ | ❌ | P2 | GGUF는 존재. GPTQ/AWQ 공백 큼(초대형이라 비용 큼) |
| kakaocorp/kanana-2-30b-a3b-instruct-2601 | Kakao | ❌ | ❌ | ❌ | ⚠️ | ❌ | P1 | AWQ는 비공식 1건 확인, GGUF/GPTQ 미확인 |
| kakaocorp/kanana-2-30b-a3b-thinking-2601 | Kakao | ❌ | ❌ | ❌ | ❌ | ❌ | P1 | 최신 계열, 양자화 공백 가능성 높음 |
| kakaocorp/kanana-2-30b-a3b-base-2601 | Kakao | ❌ | ❌ | ❌ | ❌ | ❌ | P1 | base 계열 공백 타깃 |
| NCSOFT/VARCO-VISION-2.0-14B | NCSOFT | ❌ | ❌ | ❌ | ❌ | ❌ | P2 | VLM 계열(텍스트 전용 파이프라인과 분리 필요) |
| NCSOFT/Llama-VARCO-8B-Instruct | NCSOFT | ❌ | ❌ | ❌ | ❌ | ❌ | P1 | 8B라 실험/배포 난이도 적당 |
| skt/A.X-4.0-Light | SKT | ✅ | ❌ | ❌ | ⚠️ | ❌ | P2 | GGUF 다수, AWQ 일부. GPTQ 공백 가능 |
| skt/A.X-4.0 | SKT | ❌ | ❌ | ❌ | ❌ | ❌ | P2 | 대형 모델, 자원 요구 높음 |
| LGAI-EXAONE/EXAONE-4.0-1.2B | LG AI Research | ⚠️ | ❌ | ❌ | ✅ | ❌ | P3 | 공식 AWQ 존재. 추가 가치 낮음 |

## 조사 메모
- `upstage/Solar-Open-100B`는 한국 기업 모델이지만, 기존 양자화(예: 8bit 공개) 이력이 있어 우선순위에서 제외.
- `kakaocorp`의 Kanana 2 계열이 현재 타깃으로 가장 유리(최근/공백 큼/관심도 높음).

## 다음 실행 제안
1. P1부터 실제 상세 검색 재검증 (모델명 + AWQ/GPTQ/GGUF)
2. 1차 실행 타깃: `kakaocorp/kanana-2-30b-a3b-base-2601`
3. 산출물 우선순위: AWQ 4bit → GPTQ 4bit → (가능 시) GGUF
