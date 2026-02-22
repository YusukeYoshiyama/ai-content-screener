#!/usr/bin/env python3
"""
Build a labeled local test dataset (AI vs Human) from a parquet file.

Input dataset expected columns:
- prompt
- Human_story
- one or more AI model output columns
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq


DEFAULT_AI_COLUMNS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def to_records(
    parquet_path: Path,
    ai_columns: List[str],
    min_chars: int,
    max_chars: int,
) -> Tuple[List[Dict], List[Dict]]:
    table = pq.read_table(parquet_path)
    columns = set(table.column_names)

    required = {"prompt", "Human_story"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unknown_ai_cols = [col for col in ai_columns if col not in columns]
    if unknown_ai_cols:
        raise ValueError(f"Missing AI columns: {unknown_ai_cols}")

    data = table.to_pydict()
    total_rows = len(data["prompt"])

    human_pool: List[Dict] = []
    ai_pool: List[Dict] = []

    seen_human_hash: set[str] = set()
    seen_ai_hash: set[str] = set()

    for row_idx in range(total_rows):
        prompt = (data["prompt"][row_idx] or "").strip()
        human_text = (data["Human_story"][row_idx] or "").strip()
        if len(normalize_text(human_text)) >= min_chars:
            human_trimmed = human_text[:max_chars]
            h = text_hash(human_text)
            if h not in seen_human_hash:
                seen_human_hash.add(h)
                human_pool.append(
                    {
                        "source_row": row_idx,
                        "prompt": prompt,
                        "model": "human",
                        "label": "Human",
                        "label_confidence": 0.98,
                        "label_reason": "Human_story column (dataset-provided human text)",
                        "text": human_trimmed,
                        "text_hash": h,
                        "original_text_length": len(human_text),
                        "text_length": len(human_trimmed),
                    }
                )

        for col in ai_columns:
            ai_text = (data[col][row_idx] or "").strip()
            if len(normalize_text(ai_text)) < min_chars:
                continue
            ai_trimmed = ai_text[:max_chars]
            h = text_hash(ai_text)
            if h in seen_ai_hash:
                continue
            seen_ai_hash.add(h)
            ai_pool.append(
                {
                    "source_row": row_idx,
                    "prompt": prompt,
                    "model": col,
                    "label": "AI",
                    "label_confidence": 0.98,
                    "label_reason": f"{col} column (dataset-provided model output)",
                    "text": ai_trimmed,
                    "text_hash": h,
                    "original_text_length": len(ai_text),
                    "text_length": len(ai_trimmed),
                }
            )

    return human_pool, ai_pool


def pick_balanced_ai(ai_pool: List[Dict], n_ai: int, seed: int) -> List[Dict]:
    by_model: Dict[str, List[Dict]] = {}
    for item in ai_pool:
        by_model.setdefault(item["model"], []).append(item)

    rng = random.Random(seed)
    for items in by_model.values():
        rng.shuffle(items)

    model_names = sorted(by_model.keys())
    if not model_names:
        return []

    result: List[Dict] = []
    model_index = 0
    while len(result) < n_ai:
        model = model_names[model_index % len(model_names)]
        pool = by_model[model]
        if pool:
            result.append(pool.pop())
        model_index += 1

        if all(len(items) == 0 for items in by_model.values()):
            break

    return result[:n_ai]


def sample_dataset(
    human_pool: List[Dict],
    ai_pool: List[Dict],
    size: int,
    seed: int,
    source_dataset: str,
) -> List[Dict]:
    n_human = size // 2
    n_ai = size - n_human

    if len(human_pool) < n_human:
        raise ValueError(f"Not enough human texts: need {n_human}, got {len(human_pool)}")
    if len(ai_pool) < n_ai:
        raise ValueError(f"Not enough AI texts: need {n_ai}, got {len(ai_pool)}")

    rng = random.Random(seed)

    human_candidates = list(human_pool)
    rng.shuffle(human_candidates)
    chosen_human = human_candidates[:n_human]

    chosen_ai = pick_balanced_ai(ai_pool, n_ai=n_ai, seed=seed)
    if len(chosen_ai) < n_ai:
        raise ValueError(f"Not enough balanced AI texts: need {n_ai}, got {len(chosen_ai)}")

    now = datetime.now(timezone.utc).isoformat()
    merged = chosen_human + chosen_ai
    rng.shuffle(merged)

    output: List[Dict] = []
    for idx, item in enumerate(merged, start=1):
        output.append(
            {
                "id": f"sample-{idx:04d}",
                "label": item["label"],
                "label_confidence": item["label_confidence"],
                "label_reason": item["label_reason"],
                "source_dataset": source_dataset,
                "source_row": item["source_row"],
                "prompt": item["prompt"],
                "model": item["model"],
                "original_text_length": item["original_text_length"],
                "text_length": item["text_length"],
                "text_hash": item["text_hash"],
                "retrieved_at": now,
                "text": item["text"],
            }
        )
    return output


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "label",
        "label_confidence",
        "label_reason",
        "source_dataset",
        "source_row",
        "prompt",
        "model",
        "original_text_length",
        "text_length",
        "text_hash",
        "retrieved_at",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/gsingh_train.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/processed/articles_labeled_500.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/articles_labeled_500.csv",
        help="Output CSV path",
    )
    parser.add_argument("--size", type=int, default=500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum normalized character count",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum character count stored in output text",
    )
    parser.add_argument(
        "--ai-columns",
        default=",".join(DEFAULT_AI_COLUMNS),
        help="Comma-separated AI columns",
    )
    parser.add_argument(
        "--source-dataset",
        default="gsingh1-py/train",
        help="Dataset identifier for metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ai_columns = [x.strip() for x in args.ai_columns.split(",") if x.strip()]
    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    human_pool, ai_pool = to_records(
        parquet_path=input_path,
        ai_columns=ai_columns,
        min_chars=args.min_chars,
        max_chars=max(200, int(args.max_chars)),
    )
    records = sample_dataset(
        human_pool=human_pool,
        ai_pool=ai_pool,
        size=args.size,
        seed=args.seed,
        source_dataset=args.source_dataset,
    )
    write_jsonl(records, output_jsonl)
    write_csv(records, output_csv)

    human_count = sum(1 for x in records if x["label"] == "Human")
    ai_count = sum(1 for x in records if x["label"] == "AI")
    model_counts: Dict[str, int] = {}
    for x in records:
        if x["label"] == "AI":
            model_counts[x["model"]] = model_counts.get(x["model"], 0) + 1

    print("Done")
    print(f"input={input_path}")
    print(f"jsonl={output_jsonl}")
    print(f"csv={output_csv}")
    print(f"rows={len(records)} human={human_count} ai={ai_count}")
    print(f"ai_models={json.dumps(model_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
