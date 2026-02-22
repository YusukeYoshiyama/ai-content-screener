#!/usr/bin/env python3
"""
Build a single unified dataset with only:
  - text
  - label (AI | Human)

Output format:
  - parquet (single file)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


GSINGH_AI_COLUMNS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_assistant_text(messages) -> str:
    if not isinstance(messages, list):
        return ""

    assistant_parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        content = str(msg.get("content") or "").strip()
        if role == "assistant" and content:
            assistant_parts.append(content)
    if assistant_parts:
        return "\n\n".join(assistant_parts)

    fallback_parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            fallback_parts.append(content)
    return "\n\n".join(fallback_parts)


class UnifiedWriter:
    def __init__(self, output_path: Path, batch_size: int = 5000):
        self.output_path = output_path
        self.batch_size = max(1, batch_size)
        self.schema = pa.schema([("text", pa.string()), ("label", pa.string())])
        self.writer = pq.ParquetWriter(output_path, self.schema, compression="zstd")
        self.texts: List[str] = []
        self.labels: List[str] = []
        self.count = 0

    def add(self, text: str, label: str):
        self.texts.append(text)
        self.labels.append(label)
        if len(self.texts) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.texts:
            return
        table = pa.Table.from_arrays(
            [pa.array(self.texts, type=pa.string()), pa.array(self.labels, type=pa.string())],
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.count += len(self.texts)
        self.texts.clear()
        self.labels.clear()

    def close(self):
        self.flush()
        self.writer.close()


def sanitize(
    text: Optional[str],
    min_chars: int,
    max_chars: int,
) -> Optional[str]:
    if not isinstance(text, str):
        return None
    value = text.strip()
    if not value:
        return None
    normalized = normalize_text(value)
    if len(normalized) < min_chars:
        return None
    if max_chars > 0:
        value = value[:max_chars]
    return value


def iter_parquet_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.parquet") if p.is_file()])


def add_gsingh(writer: UnifiedWriter, parquet_path: Path, min_chars: int, max_chars: int) -> int:
    added = 0
    pf = pq.ParquetFile(parquet_path)
    columns = ["Human_story"] + GSINGH_AI_COLUMNS
    for batch in pf.iter_batches(columns=columns, batch_size=2048):
        rows = batch.to_pylist()
        for row in rows:
            human = sanitize(row.get("Human_story"), min_chars, max_chars)
            if human:
                writer.add(human, "Human")
                added += 1
            for col in GSINGH_AI_COLUMNS:
                ai = sanitize(row.get(col), min_chars, max_chars)
                if ai:
                    writer.add(ai, "AI")
                    added += 1
    return added


def add_dmitva(writer: UnifiedWriter, parquet_dir: Path, min_chars: int, max_chars: int) -> int:
    added = 0
    for p in iter_parquet_files(parquet_dir):
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["human_text", "ai_text"], batch_size=4096):
            rows = batch.to_pylist()
            for row in rows:
                human = sanitize(row.get("human_text"), min_chars, max_chars)
                if human:
                    writer.add(human, "Human")
                    added += 1
                ai = sanitize(row.get("ai_text"), min_chars, max_chars)
                if ai:
                    writer.add(ai, "AI")
                    added += 1
    return added


def add_japanese_human(writer: UnifiedWriter, parquet_paths: List[Path], min_chars: int, max_chars: int) -> int:
    added = 0
    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["text"], batch_size=4096):
            rows = batch.to_pylist()
            for row in rows:
                text = sanitize(row.get("text"), min_chars, max_chars)
                if not text:
                    continue
                writer.add(text, "Human")
                added += 1
    return added


def add_japanese_ai(writer: UnifiedWriter, min_chars: int, max_chars: int) -> int:
    added = 0

    # messages-based synthetic datasets
    for p in [
        Path("data/raw/Aratako__Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted/0000__0000.parquet"),
        Path("data/raw/Aratako__Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted/0000__0000.parquet"),
    ]:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["messages"], batch_size=1024):
            rows = batch.to_pylist()
            for row in rows:
                text = extract_assistant_text(row.get("messages"))
                value = sanitize(text, min_chars, max_chars)
                if not value:
                    continue
                writer.add(value, "AI")
                added += 1

    # instruction/output dataset
    inst_path = Path("data/raw/CausalLM__GPT-4-Self-Instruct-Japanese/0000__0000.parquet")
    pf = pq.ParquetFile(inst_path)
    for batch in pf.iter_batches(columns=["instruction", "output"], batch_size=2048):
        rows = batch.to_pylist()
        for row in rows:
            text = str(row.get("output") or "").strip() or str(row.get("instruction") or "").strip()
            value = sanitize(text, min_chars, max_chars)
            if not value:
                continue
            writer.add(value, "AI")
            added += 1

    return added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/unified_text_label.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="0 means no truncation",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = UnifiedWriter(output, batch_size=args.batch_size)
    min_chars = max(1, int(args.min_chars))
    max_chars = int(args.max_chars)

    stats = {}
    stats["gsingh"] = add_gsingh(
        writer,
        parquet_path=Path("data/raw/gsingh1-py__train/0000__0000.parquet"),
        min_chars=min_chars,
        max_chars=max_chars,
    )
    stats["dmitva"] = add_dmitva(
        writer,
        parquet_dir=Path("data/raw/dmitva__human_ai_generated_text"),
        min_chars=min_chars,
        max_chars=max_chars,
    )
    stats["ja_human"] = add_japanese_human(
        writer,
        parquet_paths=[
            Path("data/raw/hpprc__jawiki-news-paragraphs/0000__0000.parquet"),
            Path("data/raw/hpprc__jawiki-books-paragraphs/0000__0000.parquet"),
        ],
        min_chars=min_chars,
        max_chars=max_chars,
    )
    stats["ja_ai"] = add_japanese_ai(
        writer,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    writer.close()

    summary = {
        "output": str(output),
        "rows": writer.count,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "components": stats,
    }
    summary_path = Path("data/processed/unified_text_label_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
