#!/usr/bin/env python3
import argparse
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9_.-]+", text)


def _repetition_score(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) < 6:
        return 0.0
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    return most_common / max(len(tokens), 1)


def _has_bad_loop(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return True
    patterns = [
        r"(\b\w+\b)(?:\s+\1){3,}",
        r"([가-힣A-Za-z]{2,})(?:\s*,?\s*\1){3,}",
    ]
    return any(re.search(p, normalized) for p in patterns)


def _judge(sample: dict, text: str) -> dict:
    tokens = _tokenize(text)
    repetition = _repetition_score(text)
    hits = [kw for kw in sample.get("expected_keywords", []) if kw in text]
    expected = sample.get("expected_keywords", [])
    keyword_ratio = (len(hits) / len(expected)) if expected else 1.0
    empty = len(text.strip()) == 0
    bad_loop = _has_bad_loop(text)

    if empty or bad_loop or repetition > 0.24:
        grade = "fail"
    elif keyword_ratio >= 0.5 and len(tokens) >= 12:
        grade = "pass"
    else:
        grade = "mixed"

    return {
        "grade": grade,
        "keyword_hits": hits,
        "keyword_ratio": round(keyword_ratio, 3),
        "token_count": len(tokens),
        "repetition_score": round(repetition, 3),
        "bad_loop": bad_loop,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    args = p.parse_args()

    samples = json.loads(Path(args.samples).read_text())
    rows = []
    summary = Counter()

    for sample in samples:
        payload = {
            "model": args.model,
            "prompt": sample["prompt"],
            "max_tokens": sample.get("max_tokens", 128),
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        resp = _post_json(f"{args.base_url}/completions", payload)
        text = resp["choices"][0].get("text", "")
        metrics = _judge(sample, text)
        row = {
            **sample,
            "output": text,
            "metrics": metrics,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "usage": resp.get("usage", {}),
        }
        rows.append(row)
        summary[metrics["grade"]] += 1

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "summary": dict(summary),
        "samples": rows,
    }
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    lines = [
        f"# EXAONE-4.5 AWQ 정성평가 ({args.model})",
        "",
        f"- generated_at: {result['generated_at']}",
        f"- summary: pass={summary['pass']}, mixed={summary['mixed']}, fail={summary['fail']}",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['id']} [{row['domain']}] - {row['metrics']['grade']}",
            f"- focus: {row['focus']}",
            f"- keyword_hits: {', '.join(row['metrics']['keyword_hits']) or '-'}",
            f"- repetition_score: {row['metrics']['repetition_score']}",
            f"- finish_reason: {row['finish_reason']}",
            "- prompt:",
            "```",
            row["prompt"],
            "```",
            "- output:",
            "```",
            row["output"].strip(),
            "```",
            "",
        ])
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
