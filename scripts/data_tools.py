#!/usr/bin/env python3
"""
Unified entrypoint for dataset download, dataset build, and detector evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_AI_COLUMNS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]
DEFAULT_MODEL_PATH = "data/processed/hash_nb_model_4096_sampled.json"
DEFAULT_HUMAN_THRESHOLD = 0.45
DEFAULT_AI_THRESHOLD = 0.55
GSINGH_PARQUET_PATH = Path("data/raw/gsingh1-py__train/0000__0000.parquet")
DMITVA_PARQUET_DIR = Path("data/raw/dmitva__human_ai_generated_text")
JAPANESE_HUMAN_PARQUET_PATHS = [
    Path("data/raw/hpprc__jawiki-news-paragraphs/0000__0000.parquet"),
    Path("data/raw/hpprc__jawiki-books-paragraphs/0000__0000.parquet"),
]
JAPANESE_AI_MESSAGE_PARQUET_PATHS = [
    Path("data/raw/Aratako__Synthetic-Japanese-Roleplay-NSFW-gpt-5-chat-5k-formatted/0000__0000.parquet"),
    Path("data/raw/Aratako__Synthetic-Japanese-Roleplay-NSFW-Claude-4.5s-3.5k-formatted/0000__0000.parquet"),
]
JAPANESE_AI_INSTRUCTION_PARQUET_PATH = Path("data/raw/CausalLM__GPT-4-Self-Instruct-Japanese/0000__0000.parquet")

WORKER_MODEL: Optional[Dict] = None
WORKER_MODEL_JA: Optional[Dict] = None
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u00a0", " ")).strip()


def detector_normalize_text(value: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", str(value or "").replace("\u00a0", " ")).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(collapse_whitespace(text).encode("utf-8")).hexdigest()


def fetch_json(url: str) -> Dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sanitize_dataset_name(name: str) -> str:
    return name.replace("/", "__")


def ensure_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.rename(out_path)


def command_download_hf(args: argparse.Namespace) -> None:
    dataset_name = args.dataset.strip()
    encoded_name = urllib.parse.quote(dataset_name, safe="")
    parquet_api = f"https://datasets-server.huggingface.co/parquet?dataset={encoded_name}"
    splits_api = f"https://datasets-server.huggingface.co/splits?dataset={encoded_name}"

    print(f"[info] dataset={dataset_name}")
    parquet_info = fetch_json(parquet_api)
    splits_info = fetch_json(splits_api)

    files: List[Dict] = parquet_info.get("parquet_files") or []
    if not files:
        raise ValueError("no parquet files found")

    dataset_dir = Path(args.out_dir) / sanitize_dataset_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    manifest_files = []
    for idx, item in enumerate(files):
        url = item["url"]
        filename = item.get("filename") or os.path.basename(urllib.parse.urlparse(url).path)
        out_name = f"{idx:04d}__{filename}"
        out_path = dataset_dir / out_name
        ensure_download(url, out_path)
        size = out_path.stat().st_size
        total_bytes += size
        manifest_files.append(
            {
                "index": idx,
                "config": item.get("config"),
                "split": item.get("split"),
                "source_url": url,
                "source_filename": item.get("filename"),
                "downloaded_path": str(out_path),
                "size_bytes": size,
            }
        )
        print(f"[ok] {idx + 1}/{len(files)} {out_name} ({size} bytes)")

    manifest = {
        "dataset": dataset_name,
        "splits": splits_info.get("splits", []),
        "files": manifest_files,
        "total_files": len(manifest_files),
        "total_bytes": total_bytes,
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest={manifest_path} total_bytes={total_bytes}")


def to_test_records(
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

    missing_ai_columns = [column for column in ai_columns if column not in columns]
    if missing_ai_columns:
        raise ValueError(f"Missing AI columns: {missing_ai_columns}")

    data = table.to_pydict()
    total_rows = len(data["prompt"])
    human_pool: List[Dict] = []
    ai_pool: List[Dict] = []
    seen_human_hash: set[str] = set()
    seen_ai_hash: set[str] = set()

    for row_index in range(total_rows):
        prompt = (data["prompt"][row_index] or "").strip()
        human_text = (data["Human_story"][row_index] or "").strip()
        if len(collapse_whitespace(human_text)) >= min_chars:
            human_trimmed = human_text[:max_chars]
            human_hash = text_hash(human_text)
            if human_hash not in seen_human_hash:
                seen_human_hash.add(human_hash)
                human_pool.append(
                    {
                        "source_row": row_index,
                        "prompt": prompt,
                        "model": "human",
                        "label": "Human",
                        "label_confidence": 0.98,
                        "label_reason": "Human_story column (dataset-provided human text)",
                        "text": human_trimmed,
                        "text_hash": human_hash,
                        "original_text_length": len(human_text),
                        "text_length": len(human_trimmed),
                    }
                )

        for column in ai_columns:
            ai_text = (data[column][row_index] or "").strip()
            if len(collapse_whitespace(ai_text)) < min_chars:
                continue
            ai_trimmed = ai_text[:max_chars]
            ai_hash = text_hash(ai_text)
            if ai_hash in seen_ai_hash:
                continue
            seen_ai_hash.add(ai_hash)
            ai_pool.append(
                {
                    "source_row": row_index,
                    "prompt": prompt,
                    "model": column,
                    "label": "AI",
                    "label_confidence": 0.98,
                    "label_reason": f"{column} column (dataset-provided model output)",
                    "text": ai_trimmed,
                    "text_hash": ai_hash,
                    "original_text_length": len(ai_text),
                    "text_length": len(ai_trimmed),
                }
            )

    return human_pool, ai_pool


def pick_balanced_ai(ai_pool: List[Dict], target_count: int, seed: int) -> List[Dict]:
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
    while len(result) < target_count:
        model_name = model_names[model_index % len(model_names)]
        pool = by_model[model_name]
        if pool:
            result.append(pool.pop())
        model_index += 1
        if all(len(items) == 0 for items in by_model.values()):
            break

    return result[:target_count]


def sample_test_dataset(
    human_pool: List[Dict],
    ai_pool: List[Dict],
    size: int,
    seed: int,
    source_dataset: str,
) -> List[Dict]:
    human_count = size // 2
    ai_count = size - human_count

    if len(human_pool) < human_count:
        raise ValueError(f"Not enough human texts: need {human_count}, got {len(human_pool)}")
    if len(ai_pool) < ai_count:
        raise ValueError(f"Not enough AI texts: need {ai_count}, got {len(ai_pool)}")

    rng = random.Random(seed)
    human_candidates = list(human_pool)
    rng.shuffle(human_candidates)
    chosen_human = human_candidates[:human_count]
    chosen_ai = pick_balanced_ai(ai_pool, target_count=ai_count, seed=seed)
    if len(chosen_ai) < ai_count:
        raise ValueError(f"Not enough balanced AI texts: need {ai_count}, got {len(chosen_ai)}")

    now = datetime.now(timezone.utc).isoformat()
    merged = chosen_human + chosen_ai
    rng.shuffle(merged)

    output: List[Dict] = []
    for index, item in enumerate(merged, start=1):
        output.append(
            {
                "id": f"sample-{index:04d}",
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
    with path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(item)


def command_build_test(args: argparse.Namespace) -> None:
    ai_columns = [value.strip() for value in args.ai_columns.split(",") if value.strip()]
    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    human_pool, ai_pool = to_test_records(
        parquet_path=input_path,
        ai_columns=ai_columns,
        min_chars=args.min_chars,
        max_chars=max(200, int(args.max_chars)),
    )
    records = sample_test_dataset(
        human_pool=human_pool,
        ai_pool=ai_pool,
        size=args.size,
        seed=args.seed,
        source_dataset=args.source_dataset,
    )
    write_jsonl(records, output_jsonl)
    write_csv(records, output_csv)

    model_counts: Dict[str, int] = {}
    for item in records:
        if item["label"] == "AI":
            model_counts[item["model"]] = model_counts.get(item["model"], 0) + 1

    print("Done")
    print(f"input={input_path}")
    print(f"jsonl={output_jsonl}")
    print(f"csv={output_csv}")
    print(f"rows={len(records)} human={sum(1 for x in records if x['label'] == 'Human')} ai={sum(1 for x in records if x['label'] == 'AI')}")
    print(f"ai_models={json.dumps(model_counts, ensure_ascii=False)}")


def extract_assistant_text(messages) -> str:
    if not isinstance(messages, list):
        return ""

    assistant_parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = str(message.get("content") or "").strip()
        if role == "assistant" and content:
            assistant_parts.append(content)
    if assistant_parts:
        return "\n\n".join(assistant_parts)

    fallback_parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
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

    def add(self, text: str, label: str) -> None:
        self.texts.append(text)
        self.labels.append(label)
        if len(self.texts) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
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

    def close(self) -> None:
        self.flush()
        self.writer.close()


def sanitize_dataset_text(text: Optional[str], min_chars: int, max_chars: int) -> Optional[str]:
    if not isinstance(text, str):
        return None
    value = text.strip()
    if not value:
        return None
    normalized = collapse_whitespace(value)
    if len(normalized) < min_chars:
        return None
    if max_chars > 0:
        value = value[:max_chars]
    return value


def iter_parquet_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    return sorted([path for path in dir_path.glob("*.parquet") if path.is_file()])


def add_gsingh(writer: UnifiedWriter, parquet_path: Path, min_chars: int, max_chars: int) -> int:
    added = 0
    parquet_file = pq.ParquetFile(parquet_path)
    columns = ["Human_story"] + DEFAULT_AI_COLUMNS
    for batch in parquet_file.iter_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            human_text = sanitize_dataset_text(row.get("Human_story"), min_chars, max_chars)
            if human_text:
                writer.add(human_text, "Human")
                added += 1
            for column in DEFAULT_AI_COLUMNS:
                ai_text = sanitize_dataset_text(row.get(column), min_chars, max_chars)
                if ai_text:
                    writer.add(ai_text, "AI")
                    added += 1
    return added


def add_dmitva(writer: UnifiedWriter, parquet_dir: Path, min_chars: int, max_chars: int) -> int:
    added = 0
    for parquet_path in iter_parquet_files(parquet_dir):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["human_text", "ai_text"], batch_size=4096):
            for row in batch.to_pylist():
                human_text = sanitize_dataset_text(row.get("human_text"), min_chars, max_chars)
                if human_text:
                    writer.add(human_text, "Human")
                    added += 1
                ai_text = sanitize_dataset_text(row.get("ai_text"), min_chars, max_chars)
                if ai_text:
                    writer.add(ai_text, "AI")
                    added += 1
    return added


def add_japanese_human(writer: UnifiedWriter, parquet_paths: List[Path], min_chars: int, max_chars: int) -> int:
    added = 0
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=4096):
            for row in batch.to_pylist():
                text = sanitize_dataset_text(row.get("text"), min_chars, max_chars)
                if not text:
                    continue
                writer.add(text, "Human")
                added += 1
    return added


def add_japanese_ai(writer: UnifiedWriter, min_chars: int, max_chars: int) -> int:
    added = 0
    for parquet_path in JAPANESE_AI_MESSAGE_PARQUET_PATHS:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["messages"], batch_size=1024):
            for row in batch.to_pylist():
                text = extract_assistant_text(row.get("messages"))
                value = sanitize_dataset_text(text, min_chars, max_chars)
                if not value:
                    continue
                writer.add(value, "AI")
                added += 1

    instruction_file = pq.ParquetFile(JAPANESE_AI_INSTRUCTION_PARQUET_PATH)
    for batch in instruction_file.iter_batches(columns=["instruction", "output"], batch_size=2048):
        for row in batch.to_pylist():
            text = str(row.get("output") or "").strip() or str(row.get("instruction") or "").strip()
            value = sanitize_dataset_text(text, min_chars, max_chars)
            if not value:
                continue
            writer.add(value, "AI")
            added += 1

    return added


def command_build_unified(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = UnifiedWriter(output_path, batch_size=args.batch_size)
    min_chars = max(1, int(args.min_chars))
    max_chars = int(args.max_chars)

    stats = {
        "gsingh": add_gsingh(writer, GSINGH_PARQUET_PATH, min_chars=min_chars, max_chars=max_chars),
        "dmitva": add_dmitva(writer, DMITVA_PARQUET_DIR, min_chars=min_chars, max_chars=max_chars),
        "ja_human": add_japanese_human(
            writer,
            parquet_paths=JAPANESE_HUMAN_PARQUET_PATHS,
            min_chars=min_chars,
            max_chars=max_chars,
        ),
        "ja_ai": add_japanese_ai(writer, min_chars=min_chars, max_chars=max_chars),
    }
    writer.close()

    summary = {
        "output": str(output_path),
        "rows": writer.count,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "components": stats,
    }
    summary_path = Path("data/processed/unified_text_label_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


def clamp(value: float, min_value: float, max_value: float) -> float:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return min_value
    return max(min_value, min(max_value, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def hash_trigram(text: str, start: int, dim: int) -> int:
    h = 2166136261
    for index in range(3):
        h ^= ord(text[start + index])
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def compute_score(text: str, model: Dict) -> float:
    dim = max(1, int(model["dim"]))
    max_chars = max(200, int(model["max_chars"]))
    model_type = str(model.get("type") or "naive_bayes_hash3")

    normalized = detector_normalize_text(text)
    if len(normalized) < 3:
        return 0.5

    limit = min(len(normalized), max_chars)
    if model_type == "logistic_hash3":
        weights = model["weights"]
        bias = float(model["bias"])
        trigram_count = max(1, limit - 2)
        bucket_counts: Dict[int, int] = {}
        for index in range(limit - 2):
            bucket = hash_trigram(normalized, index, dim)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        logit = bias
        for bucket, count in bucket_counts.items():
            logit += float(weights[bucket]) * (count / trigram_count)
        return sigmoid(logit)

    delta = model["delta"]
    prior = float(model["prior_logit"])
    logit = prior
    for index in range(limit - 2):
        bucket = hash_trigram(normalized, index, dim)
        logit += float(delta[bucket])
    return sigmoid(logit)


def is_likely_japanese_text(text: str) -> bool:
    sample = str(text or "")[:2400]
    if not sample:
        return False
    jp_count = len(JP_CHAR_RE.findall(sample))
    if jp_count >= 40:
        return True
    return jp_count >= 8 and (jp_count / max(1, len(sample))) >= 0.03


def predict_judge(score: float, human_threshold: float, ai_threshold: float) -> str:
    if score < human_threshold:
        return "Human"
    if score < ai_threshold:
        return "Unknown"
    return "AI"


def load_model(path: Path) -> Dict:
    model = json.loads(path.read_text(encoding="utf-8"))
    dim = int(model.get("dim") or 0)
    if dim <= 0:
        raise ValueError("Invalid model: dim must be > 0")

    model_type = str(model.get("type") or "naive_bayes_hash3")
    base = {
        "name": str(model.get("name") or "hash_model"),
        "type": model_type,
        "dim": dim,
        "max_chars": int(model.get("max_chars") or 1200),
        "thresholds": model.get("thresholds") or {},
    }
    if model_type == "logistic_hash3":
        weights = model.get("weights") or []
        if not isinstance(weights, list) or len(weights) != dim:
            raise ValueError("Invalid logistic model: weights length mismatch")
        return {
            **base,
            "bias": float(model.get("bias") or 0.0),
            "weights": [float(value) for value in weights],
        }

    delta = model.get("delta") or []
    if not isinstance(delta, list) or len(delta) != dim:
        raise ValueError("Invalid NB model: delta length mismatch")
    return {
        **base,
        "prior_logit": float(model.get("prior_logit") or 0.0),
        "delta": [float(value) for value in delta],
    }


@dataclass
class EvalState:
    total: int = 0
    skipped: int = 0
    strict_correct: int = 0
    decided_rows: int = 0
    decided_correct: int = 0
    gt_counts: Counter = field(default_factory=Counter)
    pred_counts: Counter = field(default_factory=Counter)
    conf: Counter = field(default_factory=Counter)
    score_sum: float = 0.0
    jp_rows: int = 0
    jp_strict_correct: int = 0
    non_jp_rows: int = 0
    non_jp_strict_correct: int = 0


def init_worker(model: Dict, model_ja: Optional[Dict]) -> None:
    global WORKER_MODEL
    global WORKER_MODEL_JA
    WORKER_MODEL = model
    WORKER_MODEL_JA = model_ja


def evaluate_valid_rows(
    rows: List[Tuple[str, str]],
    human_threshold: float,
    ai_threshold: float,
    model: Optional[Dict] = None,
    model_ja: Optional[Dict] = None,
    human_threshold_ja: Optional[float] = None,
    ai_threshold_ja: Optional[float] = None,
) -> Dict[str, float]:
    actual_model = model if model is not None else WORKER_MODEL
    actual_model_ja = model_ja if model_ja is not None else WORKER_MODEL_JA
    if actual_model is None:
        raise RuntimeError("Model is not initialized")

    summary = {
        "total": 0,
        "strict_correct": 0,
        "decided_rows": 0,
        "decided_correct": 0,
        "score_sum": 0.0,
        "gt_ai": 0,
        "gt_human": 0,
        "pred_ai": 0,
        "pred_human": 0,
        "pred_unknown": 0,
        "ai_ai": 0,
        "ai_human": 0,
        "ai_unknown": 0,
        "human_ai": 0,
        "human_human": 0,
        "human_unknown": 0,
        "jp_rows": 0,
        "jp_strict_correct": 0,
        "non_jp_rows": 0,
        "non_jp_strict_correct": 0,
    }

    for ground_truth, text in rows:
        row_model = actual_model
        row_human_threshold = human_threshold
        row_ai_threshold = ai_threshold

        is_japanese = is_likely_japanese_text(text)
        if actual_model_ja is not None and is_japanese:
            row_model = actual_model_ja
            row_human_threshold = human_threshold if human_threshold_ja is None else float(human_threshold_ja)
            row_ai_threshold = ai_threshold if ai_threshold_ja is None else float(ai_threshold_ja)

        score = compute_score(text, row_model)
        predicted = predict_judge(score, human_threshold=row_human_threshold, ai_threshold=row_ai_threshold)

        summary["total"] += 1
        summary["score_sum"] += score
        if predicted == ground_truth:
            summary["strict_correct"] += 1

        if is_japanese:
            summary["jp_rows"] += 1
            if predicted == ground_truth:
                summary["jp_strict_correct"] += 1
        else:
            summary["non_jp_rows"] += 1
            if predicted == ground_truth:
                summary["non_jp_strict_correct"] += 1

        if predicted != "Unknown":
            summary["decided_rows"] += 1
            if predicted == ground_truth:
                summary["decided_correct"] += 1

        if ground_truth == "AI":
            summary["gt_ai"] += 1
            if predicted == "AI":
                summary["ai_ai"] += 1
            elif predicted == "Human":
                summary["ai_human"] += 1
            else:
                summary["ai_unknown"] += 1
        else:
            summary["gt_human"] += 1
            if predicted == "AI":
                summary["human_ai"] += 1
            elif predicted == "Human":
                summary["human_human"] += 1
            else:
                summary["human_unknown"] += 1

        if predicted == "AI":
            summary["pred_ai"] += 1
        elif predicted == "Human":
            summary["pred_human"] += 1
        else:
            summary["pred_unknown"] += 1

    return summary


def merge_partial_result(state: EvalState, part: Dict[str, float]) -> None:
    state.total += int(part["total"])
    state.strict_correct += int(part["strict_correct"])
    state.decided_rows += int(part["decided_rows"])
    state.decided_correct += int(part["decided_correct"])
    state.score_sum += float(part["score_sum"])
    state.gt_counts["AI"] += int(part["gt_ai"])
    state.gt_counts["Human"] += int(part["gt_human"])
    state.pred_counts["AI"] += int(part["pred_ai"])
    state.pred_counts["Human"] += int(part["pred_human"])
    state.pred_counts["Unknown"] += int(part["pred_unknown"])
    state.conf[("AI", "AI")] += int(part["ai_ai"])
    state.conf[("AI", "Human")] += int(part["ai_human"])
    state.conf[("AI", "Unknown")] += int(part["ai_unknown"])
    state.conf[("Human", "AI")] += int(part["human_ai"])
    state.conf[("Human", "Human")] += int(part["human_human"])
    state.conf[("Human", "Unknown")] += int(part["human_unknown"])
    state.jp_rows += int(part["jp_rows"])
    state.jp_strict_correct += int(part["jp_strict_correct"])
    state.non_jp_rows += int(part["non_jp_rows"])
    state.non_jp_strict_correct += int(part["non_jp_strict_correct"])


def select_valid_rows(rows: List[Dict], max_rows: int, accepted_rows: int) -> Tuple[List[Tuple[str, str]], int, int, bool]:
    valid_rows: List[Tuple[str, str]] = []
    skipped = 0
    stop = False

    for row in rows:
        ground_truth_raw = str(row.get("label") or "").strip().lower()
        if ground_truth_raw not in ("ai", "human"):
            skipped += 1
            continue
        ground_truth = "AI" if ground_truth_raw == "ai" else "Human"
        text = str(row.get("text") or "")
        if not detector_normalize_text(text):
            skipped += 1
            continue

        valid_rows.append((ground_truth, text))
        accepted_rows += 1
        if max_rows > 0 and accepted_rows >= max_rows:
            stop = True
            break

    return valid_rows, accepted_rows, skipped, stop


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_evaluation(
    parquet_path: Path,
    model: Dict,
    model_ja: Optional[Dict],
    human_threshold: float,
    ai_threshold: float,
    human_threshold_ja: float,
    ai_threshold_ja: float,
    max_rows: int,
    workers: int,
) -> Dict:
    state = EvalState(gt_counts=Counter(), pred_counts=Counter(), conf=Counter())
    parquet_file = pq.ParquetFile(parquet_path)
    worker_count = max(1, int(workers))
    accepted_rows = 0

    if worker_count == 1:
        for batch in parquet_file.iter_batches(columns=["text", "label"], batch_size=2048):
            valid_rows, accepted_rows, skipped, should_stop = select_valid_rows(batch.to_pylist(), max_rows, accepted_rows)
            state.skipped += skipped
            if valid_rows:
                merge_partial_result(
                    state,
                    evaluate_valid_rows(
                        valid_rows,
                        human_threshold=human_threshold,
                        ai_threshold=ai_threshold,
                        model=model,
                        model_ja=model_ja,
                        human_threshold_ja=human_threshold_ja,
                        ai_threshold_ja=ai_threshold_ja,
                    ),
                )
            if should_stop:
                break
    else:
        max_inflight = max(2, worker_count * 3)
        pending: List[concurrent.futures.Future] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(model, model_ja),
        ) as executor:
            for batch in parquet_file.iter_batches(columns=["text", "label"], batch_size=2048):
                valid_rows, accepted_rows, skipped, should_stop = select_valid_rows(batch.to_pylist(), max_rows, accepted_rows)
                state.skipped += skipped

                if valid_rows:
                    pending.append(
                        executor.submit(
                            evaluate_valid_rows,
                            valid_rows,
                            human_threshold,
                            ai_threshold,
                            None,
                            None,
                            human_threshold_ja,
                            ai_threshold_ja,
                        )
                    )

                if len(pending) >= max_inflight:
                    done, not_done = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    pending = list(not_done)
                    for future in done:
                        merge_partial_result(state, future.result())

                if should_stop:
                    break

            for future in concurrent.futures.as_completed(pending):
                merge_partial_result(state, future.result())

    tp_ai = state.conf[("AI", "AI")]
    fp_ai = state.conf[("Human", "AI")]
    fn_ai = state.conf[("AI", "Human")] + state.conf[("AI", "Unknown")]
    precision_ai, recall_ai, f1_ai = precision_recall_f1(tp_ai, fp_ai, fn_ai)

    tp_human = state.conf[("Human", "Human")]
    fp_human = state.conf[("AI", "Human")]
    fn_human = state.conf[("Human", "AI")] + state.conf[("Human", "Unknown")]
    precision_human, recall_human, f1_human = precision_recall_f1(tp_human, fp_human, fn_human)

    strict_accuracy = state.strict_correct / state.total if state.total else 0.0
    decided_accuracy = state.decided_correct / state.decided_rows if state.decided_rows else 0.0
    coverage = state.decided_rows / state.total if state.total else 0.0
    unknown_rate = state.pred_counts["Unknown"] / state.total if state.total else 0.0
    average_score = state.score_sum / state.total if state.total else 0.0
    jp_strict_accuracy = state.jp_strict_correct / state.jp_rows if state.jp_rows else 0.0
    non_jp_strict_accuracy = state.non_jp_strict_correct / state.non_jp_rows if state.non_jp_rows else 0.0

    return {
        "input": str(parquet_path),
        "thresholds": {"human_max": round(human_threshold, 4), "ai_min": round(ai_threshold, 4)},
        "thresholds_ja": {"human_max": round(human_threshold_ja, 4), "ai_min": round(ai_threshold_ja, 4)},
        "max_rows": max_rows,
        "processed_rows": state.total,
        "skipped_rows": state.skipped,
        "strict_accuracy": round(strict_accuracy, 6),
        "decided_accuracy": round(decided_accuracy, 6),
        "coverage": round(coverage, 6),
        "unknown_rate": round(unknown_rate, 6),
        "avg_score": round(average_score, 4),
        "language_segments": {
            "jp_rows": state.jp_rows,
            "jp_strict_accuracy": round(jp_strict_accuracy, 6),
            "non_jp_rows": state.non_jp_rows,
            "non_jp_strict_accuracy": round(non_jp_strict_accuracy, 6),
        },
        "ground_truth_counts": dict(state.gt_counts),
        "prediction_counts": dict(state.pred_counts),
        "confusion_matrix": {
            "AI->AI": state.conf[("AI", "AI")],
            "AI->Human": state.conf[("AI", "Human")],
            "AI->Unknown": state.conf[("AI", "Unknown")],
            "Human->AI": state.conf[("Human", "AI")],
            "Human->Human": state.conf[("Human", "Human")],
            "Human->Unknown": state.conf[("Human", "Unknown")],
        },
        "metrics": {
            "AI": {"precision": round(precision_ai, 6), "recall": round(recall_ai, 6), "f1": round(f1_ai, 6)},
            "Human": {
                "precision": round(precision_human, 6),
                "recall": round(recall_human, 6),
                "f1": round(f1_human, 6),
            },
        },
        "model": {"name": model["name"], "dim": model["dim"], "max_chars": model["max_chars"]},
        "model_ja": (
            {"name": model_ja["name"], "dim": model_ja["dim"], "max_chars": model_ja["max_chars"]}
            if model_ja is not None
            else None
        ),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    model_ja = load_model(Path(args.model_ja)) if args.model_ja else None

    model_human = clamp(float(model.get("thresholds", {}).get("human_max", DEFAULT_HUMAN_THRESHOLD)), 0.0, 1.0)
    model_ai = clamp(float(model.get("thresholds", {}).get("ai_min", DEFAULT_AI_THRESHOLD)), 0.0, 1.0)
    human_threshold = model_human if args.human_threshold < 0 else clamp(args.human_threshold, 0.0, 1.0)
    ai_threshold = model_ai if args.ai_threshold < 0 else clamp(args.ai_threshold, 0.0, 1.0)
    if ai_threshold <= human_threshold:
        raise ValueError("--ai-threshold must be greater than --human-threshold")

    if model_ja is None:
        human_threshold_ja = human_threshold
        ai_threshold_ja = ai_threshold
    else:
        model_human_ja = clamp(
            float(model_ja.get("thresholds", {}).get("human_max", DEFAULT_HUMAN_THRESHOLD)),
            0.0,
            1.0,
        )
        model_ai_ja = clamp(
            float(model_ja.get("thresholds", {}).get("ai_min", DEFAULT_AI_THRESHOLD)),
            0.0,
            1.0,
        )
        human_threshold_ja = (
            model_human_ja if args.human_threshold_ja < 0 else clamp(args.human_threshold_ja, 0.0, 1.0)
        )
        ai_threshold_ja = model_ai_ja if args.ai_threshold_ja < 0 else clamp(args.ai_threshold_ja, 0.0, 1.0)
        if ai_threshold_ja <= human_threshold_ja:
            raise ValueError("--ai-threshold-ja must be greater than --human-threshold-ja")

    result = run_evaluation(
        parquet_path=Path(args.input),
        model=model,
        model_ja=model_ja,
        human_threshold=human_threshold,
        ai_threshold=ai_threshold,
        human_threshold_ja=human_threshold_ja,
        ai_threshold_ja=ai_threshold_ja,
        max_rows=max(0, int(args.max_rows)),
        workers=max(1, int(args.workers)),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"summary={output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project data and evaluation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-hf", help="Download all parquet shards from a Hugging Face dataset")
    download_parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. org/name")
    download_parser.add_argument("--out-dir", default="data/raw", help="Base output directory")
    download_parser.set_defaults(func=command_download_hf)

    build_unified_parser = subparsers.add_parser("build-unified", help="Build unified parquet dataset with text and label columns")
    build_unified_parser.add_argument(
        "--output",
        default="data/processed/unified_text_label.parquet",
        help="Output parquet path",
    )
    build_unified_parser.add_argument("--min-chars", type=int, default=1)
    build_unified_parser.add_argument("--max-chars", type=int, default=8000, help="0 means no truncation")
    build_unified_parser.add_argument("--batch-size", type=int, default=5000)
    build_unified_parser.set_defaults(func=command_build_unified)

    build_test_parser = subparsers.add_parser("build-test", help="Build a labeled local test dataset from a parquet source")
    build_test_parser.add_argument(
        "--input",
        default=str(GSINGH_PARQUET_PATH),
        help="Path to parquet dataset",
    )
    build_test_parser.add_argument(
        "--output-jsonl",
        default="data/processed/articles_labeled_500.jsonl",
        help="Output JSONL path",
    )
    build_test_parser.add_argument(
        "--output-csv",
        default="data/processed/articles_labeled_500.csv",
        help="Output CSV path",
    )
    build_test_parser.add_argument("--size", type=int, default=500, help="Number of samples")
    build_test_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    build_test_parser.add_argument("--min-chars", type=int, default=500, help="Minimum normalized character count")
    build_test_parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum character count stored in output text",
    )
    build_test_parser.add_argument(
        "--ai-columns",
        default=",".join(DEFAULT_AI_COLUMNS),
        help="Comma-separated AI columns",
    )
    build_test_parser.add_argument(
        "--source-dataset",
        default="gsingh1-py/train",
        help="Dataset identifier for metadata",
    )
    build_test_parser.set_defaults(func=command_build_test)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the detector against a text/label parquet dataset")
    evaluate_parser.add_argument(
        "--input",
        default="data/processed/unified_text_label.parquet",
        help="Unified parquet path (text,label)",
    )
    evaluate_parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model json path")
    evaluate_parser.add_argument("--model-ja", default="", help="Japanese model json path (optional)")
    evaluate_parser.add_argument(
        "--human-threshold",
        type=float,
        default=-1.0,
        help="score < human-threshold -> Human (negative means use model default)",
    )
    evaluate_parser.add_argument(
        "--ai-threshold",
        type=float,
        default=-1.0,
        help="score >= ai-threshold -> AI (between is Unknown; negative means use model default)",
    )
    evaluate_parser.add_argument(
        "--human-threshold-ja",
        type=float,
        default=-1.0,
        help="JP: score < human-threshold-ja -> Human (negative means use JA model default)",
    )
    evaluate_parser.add_argument(
        "--ai-threshold-ja",
        type=float,
        default=-1.0,
        help="JP: score >= ai-threshold-ja -> AI (negative means use JA model default)",
    )
    evaluate_parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    evaluate_parser.add_argument(
        "--output",
        default="data/processed/detector_eval_summary.json",
        help="Output JSON summary path",
    )
    evaluate_parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel worker count (1 disables parallel execution)",
    )
    evaluate_parser.set_defaults(func=command_evaluate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
