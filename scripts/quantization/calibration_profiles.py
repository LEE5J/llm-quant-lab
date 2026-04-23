#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

KOREAN_MIX_V1 = 'korean-mix-v1'


@dataclass(frozen=True)
class DomainSpec:
    label: str
    dataset_id: str
    config_name: str | None
    split: str
    text_builder: Callable[[dict], list[str]]
    note: str


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _has_hangul(text: str) -> bool:
    return bool(re.search(r'[가-힣]', text))


def _sentence_chunks(text: str, min_chars: int = 40, max_chars: int = 220) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []

    pieces = re.split(r'(?<=[.!?。！？])\s+|\n+', text)
    results: list[str] = []
    carry = ''
    for piece in pieces:
        piece = _normalize_text(piece)
        if not piece:
            continue
        candidate = piece if not carry else f'{carry} {piece}'
        if len(candidate) < min_chars:
            carry = candidate
            continue
        if len(candidate) <= max_chars:
            if _has_hangul(candidate):
                results.append(candidate)
            carry = ''
            continue
        if carry and _has_hangul(carry):
            results.append(carry)
        carry = ''
        if _has_hangul(piece):
            results.append(piece[:max_chars])
    if carry and len(carry) >= min_chars and _has_hangul(carry):
        results.append(carry)
    return results


def _build_wiki(row: dict) -> list[str]:
    text = row.get('text') or row.get('sentence') or ''
    return _sentence_chunks(text)


def _build_qa(row: dict) -> list[str]:
    question = row.get('question') or row.get('instruction') or row.get('prompt') or ''
    answer = row.get('answer') or row.get('response') or row.get('output') or ''
    text = f'질문: {_normalize_text(question)}\n\n답변: {_normalize_text(answer)}'.strip()
    return [text] if len(text) >= 40 and _has_hangul(text) else []


def _build_text(row: dict) -> list[str]:
    text = row.get('text') or row.get('sentence') or ''
    text = _normalize_text(text)
    return [text] if len(text) >= 80 and _has_hangul(text) else []


PROFILE_SPECS: dict[str, list[DomainSpec]] = {
    KOREAN_MIX_V1: [
        DomainSpec(
            label='ko_wiki_sentences_100000',
            dataset_id='wikimedia/wikipedia',
            config_name='20231101.ko',
            split='train',
            text_builder=_build_wiki,
            note='요청한 ko_wiki_sentences_100000 도메인 대체 소스로 한국어 위키피디아 문장을 사용',
        ),
        DomainSpec(
            label='KoAlpaca-RealQA',
            dataset_id='juyoung-trl/KoAlpaca-RealQA',
            config_name=None,
            split='train',
            text_builder=_build_qa,
            note='beomi/KoAlpaca-RealQA가 gated라서 공개 미러를 기본 소스로 사용',
        ),
        DomainSpec(
            label='korean_textbooks',
            dataset_id='maywell/korean_textbooks',
            config_name='tiny-textbooks',
            split='train',
            text_builder=_build_text,
            note='장문·지식 성격의 tiny-textbooks config를 기본 사용',
        ),
    ]
}


def profile_names() -> list[str]:
    return sorted(PROFILE_SPECS)


def profile_description(name: str) -> str:
    domains = PROFILE_SPECS[name]
    return '; '.join(
        f"{domain.label} <- {domain.dataset_id}{'/' + domain.config_name if domain.config_name else ''}"
        for domain in domains
    )


def _collect_profile_texts(profile_name: str, num_samples: int, seed: int = 42):
    from datasets import load_dataset

    if profile_name not in PROFILE_SPECS:
        raise ValueError(f'Unknown calibration profile: {profile_name}. available={profile_names()}')
    if num_samples <= 0:
        raise ValueError('num_samples must be > 0')

    domains = PROFILE_SPECS[profile_name]
    counts = [num_samples // len(domains)] * len(domains)
    for idx in range(num_samples % len(domains)):
        counts[idx] += 1

    all_texts: list[str] = []
    debug_rows: list[tuple[str, int, str]] = []

    for index, (domain, target) in enumerate(zip(domains, counts, strict=True)):
        ds = load_dataset(
            domain.dataset_id,
            domain.config_name,
            split=domain.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=seed + index, buffer_size=max(1000, target * 50))
        gathered: list[str] = []
        scanned = 0
        for row in ds:
            scanned += 1
            for text in domain.text_builder(row):
                if text not in gathered:
                    gathered.append(text)
                    if len(debug_rows) < 12:
                        debug_rows.append((domain.label, len(text), text[:120]))
                    if len(gathered) >= target:
                        break
            if len(gathered) >= target:
                break
            if scanned > max(target * 500, 5000):
                break
        if len(gathered) < target:
            raise RuntimeError(
                f'Calibration profile {profile_name} failed to collect enough samples for {domain.label}: '
                f'wanted={target} got={len(gathered)} source={domain.dataset_id}'
            )
        all_texts.extend(gathered)

    return all_texts, debug_rows


def build_calibration_dataset(profile_name: str, num_samples: int, seed: int = 42):
    from datasets import Dataset

    all_texts, debug_rows = _collect_profile_texts(profile_name, num_samples=num_samples, seed=seed)
    dataset = Dataset.from_dict({'text': all_texts})
    dataset = dataset.shuffle(seed=seed)
    dataset.info.description = f'calibration profile: {profile_name}'
    for label, size, preview in debug_rows[:12]:
        print(f'[calibration] {label} ({size} chars): {preview}')
    return dataset


def _to_chat_messages(text: str) -> list[dict]:
    if text.startswith('질문:') and '\n\n답변:' in text:
        question, answer = text.split('\n\n답변:', 1)
        question = question.removeprefix('질문:').strip()
        answer = answer.strip()
        return [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ]

    return [
        {'role': 'system', 'content': '당신은 한국어 텍스트를 정확히 이해하고 응답하는 AI 어시스턴트입니다.'},
        {'role': 'user', 'content': text},
    ]


def build_chat_templated_calibration_dataset(tokenizer, profile_name: str, num_samples: int, seed: int = 42, max_length: int = 512):
    from datasets import Dataset

    raw_texts, debug_rows = _collect_profile_texts(profile_name, num_samples=num_samples, seed=seed)
    rendered_texts: list[str] = []

    for text in raw_texts:
        messages = _to_chat_messages(text)
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(tokenized, 'get'):
            token_ids = tokenized.get('input_ids', tokenized)
        elif hasattr(tokenized, 'ids'):
            token_ids = tokenized.ids
        else:
            token_ids = tokenized
        token_ids = token_ids[:max_length]
        rendered_texts.append(tokenizer.decode(token_ids, skip_special_tokens=False))

    dataset = Dataset.from_dict({'text': rendered_texts})
    dataset = dataset.shuffle(seed=seed)
    dataset.info.description = f'chat-templated calibration profile: {profile_name}'
    print(f'[calibration] chat template applied: {getattr(tokenizer, "chat_template", None) is not None}')
    print(f'[calibration] max_length={max_length}, num_samples={num_samples}')
    for label, size, preview in debug_rows[:12]:
        print(f'[calibration-raw] {label} ({size} chars): {preview}')
    for sample in rendered_texts[:6]:
        print(f'[calibration-rendered] {sample[:180]}')
    return dataset
