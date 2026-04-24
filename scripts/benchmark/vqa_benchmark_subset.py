#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from PIL import Image

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Example:
    task: str
    sample_id: str
    question: str
    answer: str | None
    options: list[str]
    images: list[Any]
    answer_type: str = "choice"  # choice | text
    meta: dict[str, Any] | None = None


def pil_to_data_url(img: Image.Image, max_side: int = 512) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def maybe_base64_image_to_data_url(s: str) -> str:
    # KRETA stores raw base64 JPEG strings without a data URL prefix.
    if s.startswith("data:image"):
        raw = s.split(",", 1)[1]
    else:
        raw = s
    img = Image.open(BytesIO(base64.b64decode(raw)))
    return pil_to_data_url(img)


def parse_options_string(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    # e.g. "A. 45; B. 60; C. 72"
    parts = re.split(r"\s*;\s*", text)
    cleaned = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"^[A-Z][\).]\s*", "", p)
        if p:
            cleaned.append(p)
    return cleaned


def choice_prompt(question: str, options: list[str]) -> str:
    opt_text = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    if opt_text:
        return (
            "Answer the visual question. Choose exactly one option letter. "
            "Do not explain.\n"
            f"Question: {question}\nOptions:\n{opt_text}\nFinal answer:"
        )
    return (
        "Answer the visual question as briefly as possible. Do not explain.\n"
        f"Question: {question}\nFinal answer:"
    )


def extract_choice(text: str, n_options: int) -> str | None:
    text = text.strip()
    # Prefer explicit "Final answer: X" or a standalone leading letter.
    m = re.search(r"(?:final answer|answer)\s*[:：]?\s*\(?([A-Z])\)?", text, re.I)
    if m:
        c = m.group(1).upper()
        if LETTERS.index(c) < n_options:
            return c
    m = re.search(r"^\s*\(?([A-Z])\)?(?:[\).\s]|$)", text, re.I)
    if m:
        c = m.group(1).upper()
        if LETTERS.index(c) < n_options:
            return c
    # Fall back to any single option letter mentioned early.
    for c in LETTERS[:n_options]:
        if re.search(rf"\b{c}\b", text[:80], re.I):
            return c
    return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def score_answer(example: Example, output: str) -> dict[str, Any]:
    if example.answer is None or str(example.answer).lower() in {"", "hidden", "?"}:
        return {"scored": False, "correct": None, "prediction": None, "target": example.answer}
    if example.answer_type == "choice" and example.options:
        pred = extract_choice(output, len(example.options))
        target = str(example.answer).strip().upper()
        if target.isdigit() and int(target) < len(example.options):
            target = LETTERS[int(target)]
        # Some benchmark rows have options but store a free-form numeric answer.
        # If the target is not a plausible option letter, fall back to text containment.
        if len(target) != 1 or target not in LETTERS[:len(example.options)]:
            pred_norm = normalize_text(output[:200])
            target_norm = normalize_text(example.answer)
            return {
                "scored": True,
                "correct": bool(target_norm and target_norm in pred_norm),
                "prediction": output.strip()[:120],
                "target": example.answer,
            }
        return {"scored": True, "correct": pred == target, "prediction": pred, "target": target}
    pred_norm = normalize_text(output[:200])
    target_norm = normalize_text(example.answer)
    return {
        "scored": True,
        "correct": bool(target_norm and target_norm in pred_norm),
        "prediction": output.strip()[:120],
        "target": example.answer,
    }


def post_chat(base_url: str, model: str, prompt: str, images: list[Any], max_tokens: int) -> str:
    content = [{"type": "text", "text": prompt}]
    for img in images:
        if isinstance(img, Image.Image):
            url = pil_to_data_url(img)
        elif isinstance(img, str):
            url = maybe_base64_image_to_data_url(img)
        else:
            continue
        content.append({"type": "image_url", "image_url": {"url": url}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def examples_mmmu(task: str, dataset_id: str, config: str, split: str, limit: int):
    ds = load_dataset(dataset_id, name=config, split=split)
    for row in ds.select(range(min(limit, len(ds)))):
        options = parse_options_string(row.get("options"))
        images = [row.get(f"image_{i}") for i in range(1, 8) if row.get(f"image_{i}") is not None]
        yield Example(task, row.get("id", ""), row["question"], row.get("answer"), options, images, "choice", {"config": config})


def examples_mathvista(limit: int):
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    for row in ds.select(range(min(limit, len(ds)))):
        options = row.get("choices") or []
        ans_type = "choice" if options else "text"
        yield Example("MathVista-mini", str(row["pid"]), row.get("query") or row["question"], row.get("answer"), options, [row["decoded_image"]], ans_type)


def examples_mathvision(limit: int):
    ds = load_dataset("MathLLMs/MathVision", split="testmini")
    for row in ds.select(range(min(limit, len(ds)))):
        opts = row.get("options") or []
        yield Example("MathVision", str(row["id"]), row["question"], row.get("answer"), opts, [row["decoded_image"]], "choice" if opts else "text")


def examples_wemath(limit: int):
    ds = load_dataset("We-Math/We-Math", split="testmini")
    for row in ds.select(range(min(limit, len(ds)))):
        opts = parse_options_string(row.get("option"))
        yield Example("WeMath", row["ID"], row["question"], row.get("answer"), opts, [row["image_path"]], "choice")


def examples_logicvista(limit: int):
    ds = load_dataset("lscpku/LogicVista", split="test")
    for row in ds.select(range(min(limit, len(ds)))):
        opts = ["A", "B", "C", "D"]
        yield Example("LogicVista", row["id"], row["question"], row.get("answer"), opts, [row["image"]], "choice")


def examples_charxiv(limit: int):
    ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    for i, row in enumerate(ds.select(range(min(limit, len(ds))))):
        yield Example("Charxiv-RQ", row.get("original_id") or str(i), row["reasoning_q"], row.get("reasoning_a"), [], [row["image"]], "text")


def examples_kviscuit(limit: int):
    ds = load_dataset("ddehun/k-viscuit", split="test")
    for row in ds.select(range(min(limit, len(ds)))):
        opts = row.get("options") or []
        ans = str(row.get("answer")) if row.get("answer") is not None else None
        yield Example("K-Viscuit", row["id_"], row["question"], ans, opts, [row["image"]], "choice", {"category": row.get("category")})


def examples_kreta(limit: int):
    ds = load_dataset("tabtoyou/KRETA", split="test")
    for row in ds.select(range(min(limit, len(ds)))):
        opts = [row["A"], row["B"], row["C"], row["D"]]
        yield Example("KRETA", row["id"], row["question"], row.get("answer"), opts, [row["image"]], "choice", {"category": row.get("category")})


def iter_examples(tasks: list[str], limit: int):
    for task in tasks:
        if task == "MMMU":
            yield from examples_mmmu("MMMU", "MMMU/MMMU", "Accounting", "validation", limit)
        elif task == "MMMU-Pro":
            yield from examples_mmmu("MMMU-Pro", "MMMU/MMMU_Pro", "standard (4 options)", "test", limit)
        elif task == "MathVista-mini":
            yield from examples_mathvista(limit)
        elif task == "MathVision":
            yield from examples_mathvision(limit)
        elif task == "WeMath":
            yield from examples_wemath(limit)
        elif task == "LogicVista":
            yield from examples_logicvista(limit)
        elif task == "Charxiv-RQ":
            yield from examples_charxiv(limit)
        elif task == "K-Viscuit":
            yield from examples_kviscuit(limit)
        elif task == "KRETA":
            yield from examples_kreta(limit)
        else:
            raise ValueError(f"Unknown task: {task}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="EXAONE-4.5-33B-AWQ-R4")
    p.add_argument("--tasks", default="MMMU,MMMU-Pro,MathVista-mini,MathVision,WeMath,LogicVista,Charxiv-RQ,K-Viscuit,KRETA")
    p.add_argument("--limit-per-task", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    args = p.parse_args()

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    results = []
    for ex in iter_examples(tasks, args.limit_per_task):
        prompt = choice_prompt(ex.question, ex.options)
        start = time.time()
        error = None
        output = ""
        try:
            output = post_chat(args.base_url, args.model, prompt, ex.images, args.max_tokens)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        latency = time.time() - start
        score = score_answer(ex, output) if not error else {"scored": False, "correct": None, "prediction": None, "target": ex.answer}
        row = {
            "task": ex.task,
            "id": ex.sample_id,
            "question": ex.question,
            "options": ex.options,
            "answer": ex.answer,
            "output": output,
            "score": score,
            "latency_sec": round(latency, 3),
            "error": error,
            "meta": ex.meta or {},
        }
        print(json.dumps({k: row[k] for k in ["task", "id", "score", "error"]}, ensure_ascii=False), flush=True)
        results.append(row)

    summary = {}
    for task in tasks:
        rows = [r for r in results if r["task"] == task]
        scored = [r for r in rows if r["score"].get("scored")]
        correct = [r for r in scored if r["score"].get("correct")]
        summary[task] = {
            "n": len(rows),
            "scored": len(scored),
            "correct": len(correct),
            "accuracy": (len(correct) / len(scored)) if scored else None,
            "errors": sum(1 for r in rows if r.get("error")),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "tasks": tasks,
        "limit_per_task": args.limit_per_task,
        "summary": summary,
        "results": results,
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# EXAONE-4.5 AWQ-R4 QA-centered VQA Benchmark Subset",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- model: {args.model}",
        f"- limit_per_task: {args.limit_per_task}",
        "",
        "## Summary",
        "",
        "| Task | N | Scored | Correct | Accuracy | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, s in summary.items():
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.3f}"
        lines.append(f"| {task} | {s['n']} | {s['scored']} | {s['correct']} | {acc} | {s['errors']} |")
    lines.extend(["", "## Samples", ""])
    for r in results:
        lines.extend([
            f"### {r['task']} / {r['id']}",
            f"- target: {r['score'].get('target')}",
            f"- prediction: {r['score'].get('prediction')}",
            f"- correct: {r['score'].get('correct')}",
            f"- latency_sec: {r['latency_sec']}",
            f"- error: {r['error'] or '-'}",
            "- question:",
            "```",
            r["question"],
            "```",
            "- output:",
            "```",
            (r["output"] or "").strip(),
            "```",
            "",
        ])
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
