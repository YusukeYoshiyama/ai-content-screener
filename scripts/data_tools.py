#!/usr/bin/env python3
"""Unified entrypoint for dataset download, dataset build, and detector evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import sys
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from data_tools_lib.text_pipeline import (
    DEFAULT_AI_COLUMNS,
    DEFAULT_AI_THRESHOLD,
    DEFAULT_COLLECT_LIVE_OUTPUT,
    DEFAULT_HUMAN_THRESHOLD,
    DEFAULT_HYBRID_MODEL_JA_PATH,
    DEFAULT_HYBRID_MODEL_PATH,
    DEFAULT_LIVE_CACHE_DIR,
    DEFAULT_MODEL_JS_JA_PATH,
    DEFAULT_MODEL_JS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REGRESSION_CASES_MANIFEST,
    DEFAULT_SERP_AUDIT_MANIFEST,
    DEFAULT_WEB_AI_PAGE_MANIFEST,
    DEFAULT_WEB_AI_SITE_MANIFEST,
    DEFAULT_WEB_HUMAN_MANIFEST,
    DMITVA_PARQUET_DIR,
    FEATURE_NAMES,
    GSINGH_PARQUET_PATH,
    JAPANESE_AI_INSTRUCTION_PARQUET_PATH,
    JAPANESE_AI_MESSAGE_PARQUET_PATHS,
    JAPANESE_HUMAN_PARQUET_PATHS,
    Payload,
    LiveRecord,
    build_live_records_from_specs,
    build_text_metrics,
    clamp,
    clamp01,
    collapse_whitespace,
    detector_normalize_text,
    ensure_download,
    fetch_and_extract_live_payload,
    fetch_json,
    is_likely_japanese_text,
    manifest_hash_for_payload,
    mean_or_zero,
    normalize_live_value,
    read_live_manifest,
    sanitize_dataset_name,
    text_hash,
    write_live_manifest,
)

WORKER_MODEL: Optional[Dict] = None
WORKER_MODEL_JA: Optional[Dict] = None

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


def compute_hash_score(text: str, model: Dict) -> float:
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


def feature_vector_from_metrics(base_hash_score: float, metrics: Dict[str, float]) -> List[float]:
    return [
        clamp01(base_hash_score),
        math.log1p(max(0.0, metrics.get("text_length", 0.0))),
        float(metrics.get("sentence_count", 0.0)),
        float(metrics.get("avg_sentence_length", 0.0)),
        float(metrics.get("sentence_length_std", 0.0)),
        float(metrics.get("line_count", 0.0)),
        float(metrics.get("short_line_ratio", 0.0)),
        float(metrics.get("repeat_line_ratio", 0.0)),
        float(metrics.get("unique_line_ratio", 0.0)),
        float(metrics.get("bullet_line_ratio", 0.0)),
        float(metrics.get("heading_body_ratio", 0.0)),
        float(metrics.get("numeric_char_ratio", 0.0)),
        float(metrics.get("date_like_ratio", 0.0)),
        float(metrics.get("currency_like_ratio", 0.0)),
        float(metrics.get("product_code_ratio", 0.0)),
        float(metrics.get("external_link_density", 0.0)),
        float(metrics.get("punctuation_char_ratio", 0.0)),
        float(metrics.get("symbol_char_ratio", 0.0)),
        float(metrics.get("quality_score", 0.0)),
        float(metrics.get("explicit_ai_disclosure", 0.0)),
        float(metrics.get("ai_route_hint", 0.0)),
        float(metrics.get("shell_page_ratio", 0.0)),
        float(metrics.get("official_guard", 0.0)),
        float(metrics.get("meta_body_gap", 0.0)),
        float(metrics.get("content_generation_cue", 0.0)),
        float(metrics.get("template_footprint", 0.0)),
        float(metrics.get("title_body_consistency", 0.0)),
        float(metrics.get("meta_body_consistency", 0.0)),
        float(metrics.get("source_disagreement", 0.0)),
        float(metrics.get("disclaimer_density", 0.0)),
        float(metrics.get("short_shell_guard", 0.0)),
        float(metrics.get("source_body", 0.0)),
        float(metrics.get("source_meta", 0.0)),
        float(metrics.get("source_snippet", 0.0)),
        float(metrics.get("jp_char_ratio", 0.0)),
        float(metrics.get("hiragana_ratio", 0.0)),
        float(metrics.get("katakana_ratio", 0.0)),
        float(metrics.get("kanji_ratio", 0.0)),
        float(metrics.get("latin_ratio", 0.0)),
        float(metrics.get("stopword_ratio", 0.0)),
        float(metrics.get("vocab_richness", 0.0)),
    ]


def apply_calibration(base_hash_score: float, metrics: Dict[str, float], calibration: Optional[Dict]) -> float:
    if not calibration:
        return clamp01(base_hash_score)

    vector = feature_vector_from_metrics(base_hash_score, metrics)
    means = calibration.get("means") or [0.0] * len(vector)
    scales = calibration.get("scales") or [1.0] * len(vector)
    weights = calibration.get("weights") or [0.0] * len(vector)
    bias = float(calibration.get("bias") or 0.0)

    logit = bias
    for index, value in enumerate(vector):
        mean = float(means[index]) if index < len(means) else 0.0
        scale = float(scales[index]) if index < len(scales) else 1.0
        weight = float(weights[index]) if index < len(weights) else 0.0
        safe_scale = scale if abs(scale) > 1e-9 else 1.0
        logit += ((value - mean) / safe_scale) * weight
    calibrated = sigmoid(logit)
    return blend_score_with_quality(calibrated, metrics.get("quality_score", 0.0), metrics)


def blend_score_with_quality(score: float, quality_score: float, metrics: Optional[Dict[str, float]] = None) -> float:
    source_penalty = 0.0
    if metrics:
        source_penalty = 0.10 * float(metrics.get("source_snippet", 0.0)) + 0.04 * float(metrics.get("source_meta", 0.0))
    confidence = clamp01(float(quality_score) - source_penalty)
    blend = 0.25 + 0.75 * confidence
    return clamp01(0.5 + (float(score) - 0.5) * blend)


def compute_score(
    text: str,
    model: Dict,
    *,
    headings_text: str = "",
    external_link_count: int = 0,
    source: str = "body",
) -> float:
    base_hash_score = compute_hash_score(text, model)
    metrics = build_text_metrics(
        text,
        headings_text=headings_text,
        external_link_count=external_link_count,
        source=source,
    )
    return apply_calibration(base_hash_score, metrics, model.get("calibration"))


def compute_payload_score(payload: Payload, model: Dict) -> float:
    metrics = payload.metrics or build_text_metrics(
        payload.text,
        headings_text=payload.headings_text,
        external_link_count=payload.external_link_count,
        source=payload.source,
    )
    base_hash_score = compute_hash_score(payload.text, model)
    return apply_calibration(base_hash_score, metrics, model.get("calibration"))


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

    calibration = None
    raw_calibration = model.get("calibration")
    if raw_calibration:
        weights = raw_calibration.get("weights") or []
        means = raw_calibration.get("means") or []
        scales = raw_calibration.get("scales") or []
        feature_names = raw_calibration.get("feature_names") or FEATURE_NAMES[: len(weights)]
        if len(means) != len(weights) or len(scales) != len(weights):
            raise ValueError("Invalid hybrid calibration: feature vector length mismatch")
        calibration = {
            "feature_names": list(feature_names),
            "bias": float(raw_calibration.get("bias") or 0.0),
            "weights": [float(value) for value in weights],
            "means": [float(value) for value in means],
            "scales": [float(value) for value in scales],
        }

    delta = model.get("delta") or []
    if not isinstance(delta, list) or len(delta) != dim:
        raise ValueError("Invalid NB model: delta length mismatch")
    return {
        **base,
        "prior_logit": float(model.get("prior_logit") or 0.0),
        "delta": [float(value) for value in delta],
        "calibration": calibration,
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


def resolve_model_thresholds(model: Dict) -> Tuple[float, float]:
    thresholds = model.get("thresholds") or {}
    human = clamp(float(thresholds.get("human_max", DEFAULT_HUMAN_THRESHOLD)), 0.0, 1.0)
    ai = clamp(float(thresholds.get("ai_min", DEFAULT_AI_THRESHOLD)), 0.0, 1.0)
    if ai <= human:
        return DEFAULT_HUMAN_THRESHOLD, DEFAULT_AI_THRESHOLD
    return human, ai


def score_live_payload(payload: Payload, model: Dict, model_ja: Optional[Dict]) -> Dict[str, object]:
    is_japanese = is_likely_japanese_text(payload.text)
    active_model = model_ja if (model_ja is not None and is_japanese) else model
    score = compute_payload_score(payload, active_model)
    human_threshold, ai_threshold = resolve_model_thresholds(active_model)
    judge = predict_judge(score, human_threshold=human_threshold, ai_threshold=ai_threshold)
    return {
        "score": score,
        "judge": judge,
        "is_japanese": is_japanese,
        "model_name": active_model["name"],
    }


def summarize_live_bucket(values: Dict[str, float]) -> Dict[str, object]:
    return {
        "rows": int(values["rows"]),
        "strict_accuracy": round(values["strict_correct"] / values["rows"] if values["rows"] else 0.0, 6),
        "human_false_positive_rate": round(values["human_ai"] / values["human_rows"] if values["human_rows"] else 0.0, 6),
        "ai_recall": round(values["ai_ai"] / values["ai_rows"] if values["ai_rows"] else 0.0, 6),
    }


def summarize_live_predictions(predictions: Sequence[Dict[str, object]], input_path: Path) -> Dict[str, object]:
    total = len(predictions)
    strict_correct = sum(1 for item in predictions if item["label"] == item["judge"])
    decided_rows = sum(1 for item in predictions if item["judge"] != "Unknown")
    decided_correct = sum(1 for item in predictions if item["judge"] != "Unknown" and item["label"] == item["judge"])
    unknown_rows = sum(1 for item in predictions if item["judge"] == "Unknown")
    human_rows = [item for item in predictions if item["label"] == "Human"]
    ai_rows = [item for item in predictions if item["label"] == "AI"]
    hard_negative_rows = [
        item for item in predictions
        if item["label"] == "Human" and "hard-negative" in normalize_live_value(item.get("label_reason")).lower()
    ]

    by_domain: Dict[str, Dict[str, float]] = {}
    by_source: Dict[str, Dict[str, float]] = {}
    by_lang: Dict[str, Dict[str, float]] = {}
    by_query: Dict[str, Dict[str, float]] = {}
    query_order: Dict[str, int] = {}
    query_top2_human_rows = 0
    query_top2_human_fp = 0

    for item in predictions:
        for bucket_map, key in (
            (by_domain, str(item["domain_type"] or "unknown")),
            (by_source, str(item["source"] or "unknown")),
            (by_lang, "jp" if item["is_japanese"] else "non_jp"),
        ):
            bucket = bucket_map.setdefault(
                key,
                {"rows": 0, "strict_correct": 0, "human_rows": 0, "human_ai": 0, "ai_rows": 0, "ai_ai": 0},
            )
            bucket["rows"] += 1
            if item["label"] == item["judge"]:
                bucket["strict_correct"] += 1
            if item["label"] == "Human":
                bucket["human_rows"] += 1
                if item["judge"] == "AI":
                    bucket["human_ai"] += 1
            if item["label"] == "AI":
                bucket["ai_rows"] += 1
                if item["judge"] == "AI":
                    bucket["ai_ai"] += 1

        query = str(item["query"] or "")
        if query:
            query_bucket = by_query.setdefault(
                query,
                {"rows": 0, "strict_correct": 0, "human_rows": 0, "human_ai": 0, "ai_rows": 0, "ai_ai": 0},
            )
            query_bucket["rows"] += 1
            if item["label"] == item["judge"]:
                query_bucket["strict_correct"] += 1
            if item["label"] == "Human":
                query_bucket["human_rows"] += 1
                if item["judge"] == "AI":
                    query_bucket["human_ai"] += 1
            if item["label"] == "AI":
                query_bucket["ai_rows"] += 1
                if item["judge"] == "AI":
                    query_bucket["ai_ai"] += 1
            query_order[query] = query_order.get(query, 0) + 1
            if query_order[query] <= 2 and item["label"] == "Human":
                query_top2_human_rows += 1
                if item["judge"] == "AI":
                    query_top2_human_fp += 1

    result = {
        "input": str(input_path),
        "processed_rows": total,
        "strict_accuracy": round(strict_correct / total if total else 0.0, 6),
        "decided_accuracy": round(decided_correct / decided_rows if decided_rows else 0.0, 6),
        "coverage": round(decided_rows / total if total else 0.0, 6),
        "unknown_rate": round(unknown_rows / total if total else 0.0, 6),
        "human_false_positive_rate": round(
            sum(1 for item in human_rows if item["judge"] == "AI") / len(human_rows) if human_rows else 0.0,
            6,
        ),
        "ai_recall": round(sum(1 for item in ai_rows if item["judge"] == "AI") / len(ai_rows) if ai_rows else 0.0, 6),
        "hard_negative_human_false_positive_rate": round(
            sum(1 for item in hard_negative_rows if item["judge"] == "AI") / len(hard_negative_rows)
            if hard_negative_rows
            else 0.0,
            6,
        ),
        "top2_human_false_positive_rate": round(query_top2_human_fp / query_top2_human_rows if query_top2_human_rows else 0.0, 6),
        "by_domain_type": {key: summarize_live_bucket(value) for key, value in sorted(by_domain.items())},
        "by_source": {key: summarize_live_bucket(value) for key, value in sorted(by_source.items())},
        "by_lang": {key: summarize_live_bucket(value) for key, value in sorted(by_lang.items())},
        "by_query": {key: summarize_live_bucket(value) for key, value in sorted(by_query.items())},
    }
    return result


def evaluate_live_manifest(
    manifest_path: Path,
    model: Dict,
    model_ja: Optional[Dict],
    cache_dir: Path,
    workers: int,
    include_predictions: bool,
) -> Dict[str, object]:
    records = read_live_manifest(manifest_path)
    predictions: List[Optional[Dict[str, object]]] = [None] * len(records)

    def evaluate_record(record: LiveRecord) -> Dict[str, object]:
        try:
            payload, final_url = fetch_and_extract_live_payload(record, cache_dir)
        except Exception as error:  # noqa: BLE001
            return {
                "query": record.query,
                "url": record.url,
                "final_url": record.url,
                "domain_type": record.domain_type,
                "label": record.label,
                "label_reason": record.label_reason,
                "judge": "Unknown",
                "score": 0.5,
                "source": "error",
                "is_japanese": record.lang.lower() == "ja",
                "quality_score": 0.0,
                "template_footprint": 0.0,
                "title_body_consistency": 0.0,
                "meta_body_consistency": 0.0,
                "source_disagreement": 0.0,
                "disclaimer_density": 0.0,
                "short_shell_guard": 0.0,
                "content_generation_cue": 0.0,
                "official_guard": 0.0,
                "error": str(error),
            }

        scored = score_live_payload(payload, model, model_ja)
        metrics = payload.metrics or {}
        return {
            "query": record.query,
            "url": record.url,
            "final_url": final_url,
            "domain_type": record.domain_type,
            "label": record.label,
            "label_reason": record.label_reason,
            "judge": scored["judge"],
            "score": round(float(scored["score"]), 6),
            "source": payload.source,
            "is_japanese": bool(scored["is_japanese"]),
            "quality_score": round(float(metrics.get("quality_score") or 0.0), 6),
            "template_footprint": round(float(metrics.get("template_footprint") or 0.0), 6),
            "title_body_consistency": round(float(metrics.get("title_body_consistency") or 0.0), 6),
            "meta_body_consistency": round(float(metrics.get("meta_body_consistency") or 0.0), 6),
            "source_disagreement": round(float(metrics.get("source_disagreement") or 0.0), 6),
            "disclaimer_density": round(float(metrics.get("disclaimer_density") or 0.0), 6),
            "short_shell_guard": round(float(metrics.get("short_shell_guard") or 0.0), 6),
            "content_generation_cue": round(float(metrics.get("content_generation_cue") or 0.0), 6),
            "official_guard": round(float(metrics.get("official_guard") or 0.0), 6),
            "error": "",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(evaluate_record, record): index for index, record in enumerate(records)}
        for future in concurrent.futures.as_completed(futures):
            predictions[futures[future]] = future.result()

    finalized_predictions = [item for item in predictions if item is not None]
    summary = summarize_live_predictions(finalized_predictions, manifest_path)
    summary["predictions"] = finalized_predictions if include_predictions else []
    return summary


def command_verify_live(args: argparse.Namespace) -> None:
    manifest_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    records = read_live_manifest(manifest_path)
    verified: List[Optional[LiveRecord]] = [None] * len(records)
    errors = []
    error_summary = {"fetch_failed": 0, "non_html": 0, "extract_failed": 0}

    def verify_record(record: LiveRecord) -> Tuple[LiveRecord, Optional[Dict[str, str]]]:
        try:
            payload, _final_url = fetch_and_extract_live_payload(record, cache_dir)
            return (
                LiveRecord(
                    query=record.query,
                    url=record.url,
                    lang=record.lang,
                    domain_type=record.domain_type,
                    label=record.label,
                    label_confidence=record.label_confidence,
                    label_reason=record.label_reason,
                    last_verified_hash=manifest_hash_for_payload(payload),
                ),
                None,
            )
        except Exception as error:  # noqa: BLE001
            return record, {"url": record.url, "error": str(error)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_record, record): index for index, record in enumerate(records)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            verified_record, error = future.result()
            verified[index] = verified_record
            if error:
                errors.append(error)
                message = str(error.get("error") or "")
                if "Non-HTML response" in message:
                    error_summary["non_html"] += 1
                elif "timed out" in message.lower() or "http error" in message.lower() or "urlopen" in message.lower():
                    error_summary["fetch_failed"] += 1
                else:
                    error_summary["extract_failed"] += 1

    verified_records = [record if record is not None else original for record, original in zip(verified, records)]

    if args.write_manifest:
        write_live_manifest(manifest_path, verified_records)

    output = {
        "input": str(manifest_path),
        "records": len(records),
        "updated_hashes": sum(1 for old, new in zip(records, verified_records) if old.last_verified_hash != new.last_verified_hash),
        "error_summary": error_summary,
        "errors": errors,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"summary={output_path}")


def command_evaluate_live(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    model_ja = load_model(Path(args.model_ja)) if args.model_ja else None
    manifest_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    summary = evaluate_live_manifest(
        manifest_path,
        model,
        model_ja,
        cache_dir,
        max(1, int(args.workers)),
        args.include_predictions,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={output_path}")


def command_evaluate_live_suite(args: argparse.Namespace) -> None:
    model = load_model(Path(args.model))
    model_ja = load_model(Path(args.model_ja)) if args.model_ja else None
    cache_dir = Path(args.cache_dir)

    suite_inputs = {
        "web_human": Path(args.web_human_manifest),
        "web_ai_page": Path(args.web_ai_page_manifest),
        "web_ai_site": Path(args.web_ai_site_manifest),
        "serp_audit": Path(args.serp_audit_manifest),
        "regression_cases": Path(args.regression_manifest),
    }
    suite_results: Dict[str, object] = {}

    for key, manifest_path in suite_inputs.items():
        if not manifest_path.exists():
            continue
        suite_results[key] = evaluate_live_manifest(
            manifest_path,
            model,
            model_ja,
            cache_dir,
            max(1, int(args.workers)),
            args.include_predictions,
        )

    output = {
        "model": str(Path(args.model)),
        "model_ja": str(Path(args.model_ja)) if args.model_ja else "",
        "cache_dir": str(cache_dir),
        "suite": suite_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"summary={output_path}")


def classify_live_failure(item: Dict[str, object]) -> Optional[str]:
    label = normalize_live_value(item.get("label"))
    judge = normalize_live_value(item.get("judge"))
    source = normalize_live_value(item.get("source"))
    if source == "error":
        return "fetch_or_extract_error"
    if label == "Human" and judge == "AI":
        return "Human -> AI"
    if label == "AI" and judge == "Human":
        return "AI -> Human"
    if judge == "Unknown":
        return "AI/Human -> Unknown"
    return None


def analyze_live_failures(predictions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    failed = []
    for item in predictions:
        failure_type = classify_live_failure(item)
        if not failure_type:
            continue
        enriched = dict(item)
        enriched["failure_type"] = failure_type
        if normalize_live_value(item.get("source")) != "error":
            template = float(item.get("template_footprint") or 0.0)
            shell = float(item.get("short_shell_guard") or 0.0)
            disagreement = float(item.get("source_disagreement") or 0.0)
            if disagreement >= 0.55:
                enriched["failure_bucket"] = "source mis-selection"
            elif template >= 0.52:
                enriched["failure_bucket"] = "template pollution"
            elif shell >= 0.5:
                enriched["failure_bucket"] = "short/shell page"
            else:
                enriched["failure_bucket"] = "model score miss"
        else:
            enriched["failure_bucket"] = "fetch_or_extract_error"
        failed.append(enriched)

    def counter_by(key: str) -> Dict[str, int]:
        return dict(Counter(normalize_live_value(item.get(key)) or "(empty)" for item in failed))

    return {
        "rows": len(failed),
        "by_failure_type": counter_by("failure_type"),
        "by_failure_bucket": counter_by("failure_bucket"),
        "by_domain_type": counter_by("domain_type"),
        "by_lang": dict(Counter("jp" if item.get("is_japanese") else "non_jp" for item in failed)),
        "by_source": counter_by("source"),
        "by_query": counter_by("query"),
        "examples": failed[: min(60, len(failed))],
    }


def command_analyze_live_failures(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    suite = data.get("suite")
    if not isinstance(suite, dict):
        raise ValueError("Input must be evaluate-live-suite output JSON")

    output = {
        "input": str(input_path),
        "analysis": {
            key: analyze_live_failures(value.get("predictions") or [])
            for key, value in suite.items()
            if isinstance(value, dict)
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"summary={output_path}")


def command_collect_live_seed(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise ValueError("Spec file must be a JSON array")

    records = build_live_records_from_specs(specs)
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(records)

    limit = max(0, int(args.limit))
    output_records = records[:limit] if limit else records
    output_path = Path(args.output)
    write_live_manifest(output_path, output_records)

    summary = {
        "spec": str(spec_path),
        "output": str(output_path),
        "records": len(output_records),
        "by_label": dict(Counter(record.label for record in output_records)),
        "by_lang": dict(Counter(record.lang for record in output_records)),
        "by_domain_type": dict(Counter(record.domain_type for record in output_records)),
    }
    output_summary = Path(args.summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={output_summary}")


def make_payload_from_text(text: str, source: str = "body") -> Payload:
    normalized = normalize_live_value(text)
    metrics = build_text_metrics(normalized, headings_text="", external_link_count=0, source=source)
    return Payload(
        text=normalized,
        headings_text="",
        meta_description="",
        external_link_count=0,
        source=source,
        url="",
        quality_score=metrics["quality_score"],
        metrics=metrics,
    )


def reservoir_add(reservoir: List[Dict[str, object]], limit: int, seen_count: int, item: Dict[str, object], rng: random.Random) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    choice = rng.randrange(seen_count)
    if choice < limit:
        reservoir[choice] = item


def is_hard_negative_reason(label_reason: str) -> bool:
    return "hard-negative" in normalize_live_value(label_reason).lower()


def sample_unified_examples(
    parquet_path: Path,
    *,
    target_japanese: bool,
    train_per_label: int,
    validation_per_label: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    limit_per_label = max(1, train_per_label + validation_per_label)
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, object]]] = {"Human": [], "AI": []}
    seen_per_label = {"Human": 0, "AI": 0}
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(columns=["text", "label"], batch_size=2048):
        for row in batch.to_pylist():
            label_raw = normalize_live_value(row.get("label")).lower()
            if label_raw not in {"human", "ai"}:
                continue
            label = "Human" if label_raw == "human" else "AI"
            text = normalize_live_value(row.get("text"))
            if not text:
                continue
            row_is_japanese = is_likely_japanese_text(text)
            if row_is_japanese != target_japanese:
                continue

            seen_per_label[label] += 1
            reservoir_add(
                reservoirs[label],
                limit=limit_per_label,
                seen_count=seen_per_label[label],
                item={"label": label, "text": text, "sample_weight": 1.0},
                rng=rng,
            )

    train_examples: List[Dict[str, object]] = []
    validation_examples: List[Dict[str, object]] = []
    for label, rows in reservoirs.items():
        rng.shuffle(rows)
        for row in rows[:validation_per_label]:
            validation_examples.append(
                {
                    "label": label,
                    "payload": make_payload_from_text(str(row["text"]), source="body"),
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                }
            )
        for row in rows[validation_per_label:validation_per_label + train_per_label]:
            train_examples.append(
                {
                    "label": label,
                    "payload": make_payload_from_text(str(row["text"]), source="body"),
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                }
            )

    rng.shuffle(train_examples)
    rng.shuffle(validation_examples)
    return train_examples, validation_examples


def load_live_examples(
    manifest_path: Path,
    cache_dir: Path,
    *,
    target_japanese: bool,
    label_filter: Optional[str],
    sample_weight: float,
) -> List[Dict[str, object]]:
    if not manifest_path.exists():
        return []

    def build_example(record: LiveRecord) -> Optional[Dict[str, object]]:
        if label_filter and normalize_live_value(record.label).lower() != label_filter.lower():
            return None
        try:
            payload, _final_url = fetch_and_extract_live_payload(record, cache_dir)
        except Exception:  # noqa: BLE001
            return None
        if is_likely_japanese_text(payload.text) != target_japanese:
            return None
        label = "AI" if normalize_live_value(record.label).lower() == "ai" else "Human"
        effective_weight = float(sample_weight)
        if label == "Human" and is_hard_negative_reason(record.label_reason):
            effective_weight *= 2.0
        return {
            "label": label,
            "payload": payload,
            "query": record.query,
            "domain_type": record.domain_type,
            "label_reason": record.label_reason,
            "url": record.url,
            "sample_weight": effective_weight,
        }

    records = read_live_manifest(manifest_path)
    examples: List[Optional[Dict[str, object]]] = [None] * len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(build_example, record): index for index, record in enumerate(records)}
        for future in concurrent.futures.as_completed(futures):
            examples[futures[future]] = future.result()
    return [example for example in examples if example is not None]


def example_feature_vector(example: Dict[str, object], base_model: Dict) -> List[float]:
    payload = example["payload"]
    assert isinstance(payload, Payload)
    base_hash_score = compute_hash_score(payload.text, base_model)
    return feature_vector_from_metrics(base_hash_score, payload.metrics)


def fit_calibration_model(train_examples: Sequence[Dict[str, object]], base_model: Dict) -> Dict[str, object]:
    if not train_examples:
        raise ValueError("No training examples")

    matrix = [example_feature_vector(example, base_model) for example in train_examples]
    labels = [1 if example["label"] == "AI" else 0 for example in train_examples]
    weights = [float(example.get("sample_weight", 1.0)) for example in train_examples]

    columns = list(zip(*matrix))
    means = [mean_or_zero(column) for column in columns]
    scales = [stdev_or_zero(column) if stdev_or_zero(column) > 1e-9 else 1.0 for column in columns]

    standardized = []
    for row in matrix:
        standardized.append([(value - means[index]) / scales[index] for index, value in enumerate(row)])

    estimator = LogisticRegression(
        solver="liblinear",
        max_iter=600,
        random_state=42,
    )
    estimator.fit(standardized, labels, sample_weight=weights)

    return {
        "feature_names": FEATURE_NAMES,
        "bias": float(estimator.intercept_[0]),
        "weights": [float(value) for value in estimator.coef_[0]],
        "means": [float(value) for value in means],
        "scales": [float(value) for value in scales],
    }


def score_example_with_calibration(example: Dict[str, object], base_model: Dict, calibration: Dict[str, object]) -> float:
    payload = example["payload"]
    assert isinstance(payload, Payload)
    base_hash_score = compute_hash_score(payload.text, base_model)
    return apply_calibration(base_hash_score, payload.metrics, calibration)


def summarize_validation_scores(
    validation_scores: Sequence[Tuple[str, float]],
    human_threshold: float,
    ai_threshold: float,
) -> Dict[str, float]:
    strict_correct = 0
    decided_rows = 0
    decided_correct = 0
    unknown_rows = 0

    for label, score in validation_scores:
        judge = predict_judge(score, human_threshold=human_threshold, ai_threshold=ai_threshold)
        if label == judge:
            strict_correct += 1
        if judge != "Unknown":
            decided_rows += 1
            if label == judge:
                decided_correct += 1
        else:
            unknown_rows += 1

    total = len(validation_scores)
    return {
        "strict_accuracy": strict_correct / max(1, total),
        "decided_accuracy": decided_correct / max(1, decided_rows),
        "unknown_rate": unknown_rows / max(1, total),
    }


def score_live_examples_with_calibration(
    examples: Sequence[Dict[str, object]],
    base_model: Dict,
    calibration: Dict[str, object],
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    for example in examples:
        payload = example["payload"]
        assert isinstance(payload, Payload)
        scored.append(
            {
                "label": str(example["label"]),
                "query": str(example.get("query") or ""),
                "domain_type": str(example.get("domain_type") or ""),
                "label_reason": str(example.get("label_reason") or ""),
                "source": payload.source,
                "is_japanese": is_likely_japanese_text(payload.text),
                "score": score_example_with_calibration(example, base_model, calibration),
            }
        )
    return scored


def summarize_scored_live_examples(
    scored_examples: Sequence[Dict[str, object]],
    human_threshold: float,
    ai_threshold: float,
    name: str,
) -> Dict[str, object]:
    predictions = [
        {
            "query": str(example.get("query") or ""),
            "url": "",
            "final_url": "",
            "domain_type": str(example.get("domain_type") or ""),
            "label": str(example["label"]),
            "label_reason": str(example.get("label_reason") or ""),
            "judge": predict_judge(float(example["score"]), human_threshold=human_threshold, ai_threshold=ai_threshold),
            "score": round(float(example["score"]), 6),
            "source": str(example.get("source") or "body"),
            "is_japanese": bool(example.get("is_japanese")),
            "error": "",
        }
        for example in scored_examples
    ]
    summary = summarize_live_predictions(predictions, Path(name))
    summary["predictions"] = predictions
    return summary


def max_human_false_positive(bucket_summary: Dict[str, Dict[str, object]]) -> float:
    if not bucket_summary:
        return 0.0
    return max(float(value.get("human_false_positive_rate") or 0.0) for value in bucket_summary.values())


def summary_has_rows(summary: Dict[str, object]) -> bool:
    return int(summary.get("processed_rows") or 0) > 0


def has_regression_failure(summary: Dict[str, object]) -> bool:
    predictions = summary.get("predictions") or []
    return any(item.get("label") == "Human" and item.get("judge") != "Human" for item in predictions)


def tune_thresholds(
    validation_examples: Sequence[Dict[str, object]],
    live_human_examples: Sequence[Dict[str, object]],
    live_ai_page_examples: Sequence[Dict[str, object]],
    live_ai_site_examples: Sequence[Dict[str, object]],
    serp_examples: Sequence[Dict[str, object]],
    regression_examples: Sequence[Dict[str, object]],
    base_model: Dict,
    calibration: Dict[str, object],
) -> Dict[str, object]:
    if not validation_examples:
        return {
            "thresholds": (DEFAULT_HUMAN_THRESHOLD, DEFAULT_AI_THRESHOLD),
            "metrics": {},
        }

    validation_scores = [
        (str(example["label"]), score_example_with_calibration(example, base_model, calibration))
        for example in validation_examples
    ]
    scored_web_human = score_live_examples_with_calibration(live_human_examples, base_model, calibration)
    scored_web_ai_page = score_live_examples_with_calibration(live_ai_page_examples, base_model, calibration)
    scored_web_ai_site = score_live_examples_with_calibration(live_ai_site_examples, base_model, calibration)
    scored_serp = score_live_examples_with_calibration(serp_examples, base_model, calibration)
    scored_regression = score_live_examples_with_calibration(regression_examples, base_model, calibration)

    best_thresholds = (DEFAULT_HUMAN_THRESHOLD, DEFAULT_AI_THRESHOLD)
    best_metrics: Dict[str, object] = {}
    best_key = None

    for human_threshold_index in range(40, 56):
        human_threshold = human_threshold_index / 100.0
        for ai_threshold_index in range(max(human_threshold_index + 6, 54), 79):
            ai_threshold = ai_threshold_index / 100.0
            if ai_threshold <= human_threshold:
                continue

            validation_summary = summarize_validation_scores(validation_scores, human_threshold, ai_threshold)
            web_human_summary = summarize_scored_live_examples(
                scored_web_human,
                human_threshold,
                ai_threshold,
                "web_human",
            )
            web_ai_page_summary = summarize_scored_live_examples(
                scored_web_ai_page,
                human_threshold,
                ai_threshold,
                "web_ai_page",
            )
            web_ai_site_summary = summarize_scored_live_examples(
                scored_web_ai_site,
                human_threshold,
                ai_threshold,
                "web_ai_site",
            )
            serp_summary = summarize_scored_live_examples(scored_serp, human_threshold, ai_threshold, "serp_audit")
            regression_summary = summarize_scored_live_examples(scored_regression, human_threshold, ai_threshold, "regression")

            violations = []
            if validation_summary["strict_accuracy"] < 0.96:
                violations.append(0.96 - validation_summary["strict_accuracy"])
            if validation_summary["decided_accuracy"] < 0.962:
                violations.append(0.962 - validation_summary["decided_accuracy"])
            if validation_summary["unknown_rate"] > 0.03:
                violations.append(validation_summary["unknown_rate"] - 0.03)
            if summary_has_rows(web_human_summary) and float(web_human_summary["human_false_positive_rate"]) > 0.03:
                violations.append(float(web_human_summary["human_false_positive_rate"]) - 0.03)
            if summary_has_rows(web_human_summary) and float(web_human_summary["hard_negative_human_false_positive_rate"]) > 0.05:
                violations.append(float(web_human_summary["hard_negative_human_false_positive_rate"]) - 0.05)
            if summary_has_rows(web_human_summary) and max_human_false_positive(web_human_summary["by_domain_type"]) > 0.12:
                violations.append(max_human_false_positive(web_human_summary["by_domain_type"]) - 0.12)
            if summary_has_rows(web_ai_page_summary) and float(web_ai_page_summary["ai_recall"]) < 0.96:
                violations.append(0.96 - float(web_ai_page_summary["ai_recall"]))
            if summary_has_rows(web_ai_page_summary) and float(web_ai_page_summary["unknown_rate"]) > 0.05:
                violations.append(float(web_ai_page_summary["unknown_rate"]) - 0.05)
            if summary_has_rows(web_ai_site_summary) and float(web_ai_site_summary["ai_recall"]) < 0.93:
                violations.append(0.93 - float(web_ai_site_summary["ai_recall"]))
            if summary_has_rows(web_ai_site_summary) and float(web_ai_site_summary["unknown_rate"]) > 0.08:
                violations.append(float(web_ai_site_summary["unknown_rate"]) - 0.08)
            if summary_has_rows(serp_summary) and float(serp_summary["ai_recall"]) < 0.95:
                violations.append(0.95 - float(serp_summary["ai_recall"]))
            if summary_has_rows(serp_summary) and float(serp_summary["human_false_positive_rate"]) > 0.04:
                violations.append(float(serp_summary["human_false_positive_rate"]) - 0.04)
            if summary_has_rows(serp_summary) and float(serp_summary["top2_human_false_positive_rate"]) > 0.02:
                violations.append(float(serp_summary["top2_human_false_positive_rate"]) - 0.02)
            official_bucket = serp_summary["by_domain_type"].get("official", {})
            official_human_fp = float(official_bucket.get("human_false_positive_rate") or 0.0)
            if summary_has_rows(serp_summary) and official_human_fp > 0.07:
                violations.append(official_human_fp - 0.07)
            if summary_has_rows(regression_summary) and has_regression_failure(regression_summary):
                violations.append(1.0)

            avg_unknown = mean_or_zero([
                float(validation_summary["unknown_rate"]),
                float(web_human_summary["unknown_rate"]),
                float(web_ai_page_summary["unknown_rate"]),
                float(web_ai_site_summary["unknown_rate"]),
                float(serp_summary["unknown_rate"]),
            ])
            distance_penalty = abs(human_threshold - DEFAULT_HUMAN_THRESHOLD) + abs(ai_threshold - DEFAULT_AI_THRESHOLD)
            key = (
                len(violations),
                round(sum(violations), 6),
                round(float(web_human_summary["human_false_positive_rate"]), 6),
                round(float(serp_summary["top2_human_false_positive_rate"]), 6),
                -round(float(web_ai_page_summary["ai_recall"]), 6),
                -round(float(web_ai_site_summary["ai_recall"]), 6),
                -round(float(serp_summary["ai_recall"]), 6),
                round(avg_unknown, 6),
                round(distance_penalty, 6),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = (human_threshold, ai_threshold)
                best_metrics = {
                    "validation": validation_summary,
                    "web_human": web_human_summary,
                    "web_ai_page": web_ai_page_summary,
                    "web_ai_site": web_ai_site_summary,
                    "serp_audit": serp_summary,
                    "regression": regression_summary,
                    "violations": len(violations),
                    "violation_sum": round(sum(violations), 6),
                }

    return {
        "thresholds": best_thresholds,
        "metrics": best_metrics,
    }


def merge_hybrid_model(base_model_path: Path, calibration: Dict[str, object], thresholds: Tuple[float, float]) -> Dict[str, object]:
    base_model = json.loads(base_model_path.read_text(encoding="utf-8"))
    base_model["type"] = "hybrid_hash3_calibrated"
    base_model["calibration"] = calibration
    base_model["thresholds"] = {
        "human_max": round(float(thresholds[0]), 4),
        "ai_min": round(float(thresholds[1]), 4),
    }
    return base_model


def write_model_js(path: Path, global_name: str, model: Dict[str, object]) -> None:
    payload = json.dumps(model, ensure_ascii=False)
    path.write_text(
        "\"use strict\";\n\n(() => {\n  globalThis.%s = %s;\n})();\n" % (global_name, payload),
        encoding="utf-8",
    )


def strip_predictions(summary: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in summary.items():
        if key == "predictions":
            continue
        if isinstance(value, dict):
            cleaned[key] = strip_predictions(value)
        else:
            cleaned[key] = value
    return cleaned


def command_train_hybrid(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    live_human_manifest = Path(args.live_human_manifest) if args.live_human_manifest else None
    live_ai_page_manifest = Path(args.live_ai_page_manifest) if args.live_ai_page_manifest else None
    live_ai_site_manifest = Path(args.live_ai_site_manifest) if args.live_ai_site_manifest else None
    serp_audit_manifest = Path(args.serp_audit_manifest) if args.serp_audit_manifest else None
    regression_manifest = Path(args.regression_manifest) if args.regression_manifest else None

    base_model_non_ja = load_model(Path(args.base_model))
    base_model_ja = load_model(Path(args.base_model_ja))

    train_non_ja, validation_non_ja = sample_unified_examples(
        input_path,
        target_japanese=False,
        train_per_label=max(1000, int(args.train_per_label)),
        validation_per_label=max(200, int(args.validation_per_label)),
        seed=42,
    )
    train_ja, validation_ja = sample_unified_examples(
        input_path,
        target_japanese=True,
        train_per_label=max(1000, int(args.train_per_label_ja)),
        validation_per_label=max(200, int(args.validation_per_label_ja)),
        seed=142,
    )

    live_human_non_ja = (
        load_live_examples(live_human_manifest, cache_dir, target_japanese=False, label_filter="Human", sample_weight=args.live_weight)
        if live_human_manifest
        else []
    )
    live_human_ja = (
        load_live_examples(live_human_manifest, cache_dir, target_japanese=True, label_filter="Human", sample_weight=args.live_weight)
        if live_human_manifest
        else []
    )
    live_ai_page_non_ja = (
        load_live_examples(
            live_ai_page_manifest,
            cache_dir,
            target_japanese=False,
            label_filter="AI",
            sample_weight=args.live_ai_page_weight,
        )
        if live_ai_page_manifest and live_ai_page_manifest.exists()
        else []
    )
    live_ai_page_ja = (
        load_live_examples(
            live_ai_page_manifest,
            cache_dir,
            target_japanese=True,
            label_filter="AI",
            sample_weight=args.live_ai_page_weight,
        )
        if live_ai_page_manifest and live_ai_page_manifest.exists()
        else []
    )
    live_ai_site_non_ja = (
        load_live_examples(
            live_ai_site_manifest,
            cache_dir,
            target_japanese=False,
            label_filter="AI",
            sample_weight=args.live_ai_site_weight,
        )
        if live_ai_site_manifest and live_ai_site_manifest.exists()
        else []
    )
    live_ai_site_ja = (
        load_live_examples(
            live_ai_site_manifest,
            cache_dir,
            target_japanese=True,
            label_filter="AI",
            sample_weight=args.live_ai_site_weight,
        )
        if live_ai_site_manifest and live_ai_site_manifest.exists()
        else []
    )
    serp_non_ja = (
        load_live_examples(serp_audit_manifest, cache_dir, target_japanese=False, label_filter=None, sample_weight=1.0)
        if serp_audit_manifest and serp_audit_manifest.exists()
        else []
    )
    serp_ja = (
        load_live_examples(serp_audit_manifest, cache_dir, target_japanese=True, label_filter=None, sample_weight=1.0)
        if serp_audit_manifest and serp_audit_manifest.exists()
        else []
    )
    regression_non_ja = (
        load_live_examples(regression_manifest, cache_dir, target_japanese=False, label_filter=None, sample_weight=1.0)
        if regression_manifest and regression_manifest.exists()
        else []
    )
    regression_ja = (
        load_live_examples(regression_manifest, cache_dir, target_japanese=True, label_filter=None, sample_weight=1.0)
        if regression_manifest and regression_manifest.exists()
        else []
    )

    calibration_non_ja = fit_calibration_model(
        train_non_ja + live_human_non_ja + live_ai_page_non_ja + live_ai_site_non_ja,
        base_model_non_ja,
    )
    calibration_ja = fit_calibration_model(
        train_ja + live_human_ja + live_ai_page_ja + live_ai_site_ja,
        base_model_ja,
    )

    thresholds_non_ja_info = tune_thresholds(
        validation_non_ja,
        live_human_non_ja,
        live_ai_page_non_ja,
        live_ai_site_non_ja,
        serp_non_ja,
        regression_non_ja,
        base_model_non_ja,
        calibration_non_ja,
    )
    thresholds_ja_info = tune_thresholds(
        validation_ja,
        live_human_ja,
        live_ai_page_ja,
        live_ai_site_ja,
        serp_ja,
        regression_ja,
        base_model_ja,
        calibration_ja,
    )
    thresholds_non_ja = thresholds_non_ja_info["thresholds"]
    thresholds_ja = thresholds_ja_info["thresholds"]

    hybrid_non_ja = merge_hybrid_model(Path(args.base_model), calibration_non_ja, thresholds_non_ja)
    hybrid_ja = merge_hybrid_model(Path(args.base_model_ja), calibration_ja, thresholds_ja)

    output_model = Path(args.output_model)
    output_model_ja = Path(args.output_model_ja)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model_ja.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(hybrid_non_ja, ensure_ascii=False, indent=2), encoding="utf-8")
    output_model_ja.write_text(json.dumps(hybrid_ja, ensure_ascii=False, indent=2), encoding="utf-8")

    write_model_js(Path(args.output_js), "AI_SCREENER_HASH_MODEL", hybrid_non_ja)
    write_model_js(Path(args.output_js_ja), "AI_SCREENER_HASH_MODEL_JA", hybrid_ja)

    summary = {
        "input": str(input_path),
        "output_model": str(output_model),
        "output_model_ja": str(output_model_ja),
        "output_js": str(args.output_js),
        "output_js_ja": str(args.output_js_ja),
        "train_non_ja": len(train_non_ja),
        "train_ja": len(train_ja),
        "live_human_non_ja": len(live_human_non_ja),
        "live_human_ja": len(live_human_ja),
        "live_ai_page_non_ja": len(live_ai_page_non_ja),
        "live_ai_page_ja": len(live_ai_page_ja),
        "live_ai_site_non_ja": len(live_ai_site_non_ja),
        "live_ai_site_ja": len(live_ai_site_ja),
        "serp_non_ja": len(serp_non_ja),
        "serp_ja": len(serp_ja),
        "regression_non_ja": len(regression_non_ja),
        "regression_ja": len(regression_ja),
        "thresholds_non_ja": {"human_max": thresholds_non_ja[0], "ai_min": thresholds_non_ja[1]},
        "thresholds_ja": {"human_max": thresholds_ja[0], "ai_min": thresholds_ja[1]},
        "threshold_metrics_non_ja": strip_predictions(dict(thresholds_non_ja_info["metrics"])),
        "threshold_metrics_ja": strip_predictions(dict(thresholds_ja_info["metrics"])),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={output_summary}")


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
    evaluate_parser.add_argument("--model", default=str(DEFAULT_HYBRID_MODEL_PATH), help="Model json path")
    evaluate_parser.add_argument("--model-ja", default=str(DEFAULT_HYBRID_MODEL_JA_PATH), help="Japanese model json path (optional)")
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

    verify_live_parser = subparsers.add_parser("verify-live", help="Fetch live URLs, extract text, and refresh verification hashes")
    verify_live_parser.add_argument("--input", default=str(DEFAULT_WEB_HUMAN_MANIFEST), help="Live manifest CSV path")
    verify_live_parser.add_argument("--cache-dir", default=str(DEFAULT_LIVE_CACHE_DIR), help="Local ignored cache directory")
    verify_live_parser.add_argument(
        "--output",
        default="data/live_eval/verify_live_summary.json",
        help="Verification summary JSON path",
    )
    verify_live_parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write refreshed last_verified_hash back to the input manifest",
    )
    verify_live_parser.set_defaults(func=command_verify_live)

    collect_live_parser = subparsers.add_parser("collect-live-seed", help="Build a live manifest from sitemap / RSS / fixed URL list specs")
    collect_live_parser.add_argument("--spec", required=True, help="JSON array of seed specs")
    collect_live_parser.add_argument("--output", default=str(DEFAULT_COLLECT_LIVE_OUTPUT), help="Output manifest CSV path")
    collect_live_parser.add_argument("--summary", default="data/live_eval/collect_live_seed_summary.json", help="Output summary JSON path")
    collect_live_parser.add_argument("--limit", type=int, default=0, help="Optional max number of output rows")
    collect_live_parser.add_argument("--shuffle", action="store_true", help="Shuffle records before truncation")
    collect_live_parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    collect_live_parser.set_defaults(func=command_collect_live_seed)

    evaluate_live_parser = subparsers.add_parser("evaluate-live", help="Evaluate the detector against a live URL manifest")
    evaluate_live_parser.add_argument("--input", default=str(DEFAULT_WEB_HUMAN_MANIFEST), help="Live manifest CSV path")
    evaluate_live_parser.add_argument("--model", default=str(DEFAULT_HYBRID_MODEL_PATH), help="Model json path")
    evaluate_live_parser.add_argument("--model-ja", default=str(DEFAULT_HYBRID_MODEL_JA_PATH), help="Japanese model json path")
    evaluate_live_parser.add_argument("--cache-dir", default=str(DEFAULT_LIVE_CACHE_DIR), help="Local ignored cache directory")
    evaluate_live_parser.add_argument("--workers", type=int, default=8, help="Parallel worker count")
    evaluate_live_parser.add_argument(
        "--output",
        default="data/live_eval/evaluate_live_summary.json",
        help="Evaluation summary JSON path",
    )
    evaluate_live_parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Embed per-URL predictions in the output JSON",
    )
    evaluate_live_parser.set_defaults(func=command_evaluate_live)

    evaluate_live_suite_parser = subparsers.add_parser("evaluate-live-suite", help="Evaluate all live manifests and emit one JSON summary")
    evaluate_live_suite_parser.add_argument("--model", default=str(DEFAULT_HYBRID_MODEL_PATH), help="Model json path")
    evaluate_live_suite_parser.add_argument("--model-ja", default=str(DEFAULT_HYBRID_MODEL_JA_PATH), help="Japanese model json path")
    evaluate_live_suite_parser.add_argument("--cache-dir", default=str(DEFAULT_LIVE_CACHE_DIR), help="Local ignored cache directory")
    evaluate_live_suite_parser.add_argument("--workers", type=int, default=8, help="Parallel worker count")
    evaluate_live_suite_parser.add_argument("--web-human-manifest", default=str(DEFAULT_WEB_HUMAN_MANIFEST), help="Human live manifest CSV path")
    evaluate_live_suite_parser.add_argument("--web-ai-page-manifest", default=str(DEFAULT_WEB_AI_PAGE_MANIFEST), help="AI page-level live manifest CSV path")
    evaluate_live_suite_parser.add_argument("--web-ai-site-manifest", default=str(DEFAULT_WEB_AI_SITE_MANIFEST), help="AI site-level live manifest CSV path")
    evaluate_live_suite_parser.add_argument("--serp-audit-manifest", default=str(DEFAULT_SERP_AUDIT_MANIFEST), help="SERP audit manifest CSV path")
    evaluate_live_suite_parser.add_argument("--regression-manifest", default=str(DEFAULT_REGRESSION_CASES_MANIFEST), help="Regression cases manifest CSV path")
    evaluate_live_suite_parser.add_argument("--output", default="data/live_eval/evaluate_live_suite_summary.json", help="Evaluation summary JSON path")
    evaluate_live_suite_parser.add_argument("--include-predictions", action="store_true", help="Embed per-URL predictions in the output JSON")
    evaluate_live_suite_parser.set_defaults(func=command_evaluate_live_suite)

    analyze_live_failures_parser = subparsers.add_parser(
        "analyze-live-failures",
        help="Analyze failure buckets from evaluate-live-suite JSON output",
    )
    analyze_live_failures_parser.add_argument("--input", required=True, help="evaluate-live-suite JSON path")
    analyze_live_failures_parser.add_argument("--output", default="data/live_eval/live_failure_analysis.json", help="Failure analysis JSON path")
    analyze_live_failures_parser.set_defaults(func=command_analyze_live_failures)

    train_hybrid_parser = subparsers.add_parser("train-hybrid", help="Train lightweight hybrid calibration layers and export extension models")
    train_hybrid_parser.add_argument("--input", default="data/processed/unified_text_label.parquet", help="Unified parquet path")
    train_hybrid_parser.add_argument("--base-model", default=DEFAULT_MODEL_PATH, help="Base non-JP model json path")
    train_hybrid_parser.add_argument("--base-model-ja", default="data/processed/hash_nb_model_4096_ja.json", help="Base JP model json path")
    train_hybrid_parser.add_argument("--output-model", default=str(DEFAULT_HYBRID_MODEL_PATH), help="Hybrid non-JP model json path")
    train_hybrid_parser.add_argument("--output-model-ja", default=str(DEFAULT_HYBRID_MODEL_JA_PATH), help="Hybrid JP model json path")
    train_hybrid_parser.add_argument("--output-js", default=str(DEFAULT_MODEL_JS_PATH), help="Content-script non-JP model JS path")
    train_hybrid_parser.add_argument("--output-js-ja", default=str(DEFAULT_MODEL_JS_JA_PATH), help="Content-script JP model JS path")
    train_hybrid_parser.add_argument(
        "--output-summary",
        default="data/processed/train_hybrid_summary.json",
        help="Training summary JSON path",
    )
    train_hybrid_parser.add_argument("--cache-dir", default=str(DEFAULT_LIVE_CACHE_DIR), help="Local ignored cache directory")
    train_hybrid_parser.add_argument("--live-human-manifest", default=str(DEFAULT_WEB_HUMAN_MANIFEST), help="Optional human live manifest CSV path")
    train_hybrid_parser.add_argument("--live-ai-page-manifest", default=str(DEFAULT_WEB_AI_PAGE_MANIFEST), help="Optional page-level AI live manifest CSV path")
    train_hybrid_parser.add_argument("--live-ai-site-manifest", default=str(DEFAULT_WEB_AI_SITE_MANIFEST), help="Optional site-level AI live manifest CSV path")
    train_hybrid_parser.add_argument("--serp-audit-manifest", default=str(DEFAULT_SERP_AUDIT_MANIFEST), help="Optional SERP audit manifest CSV path")
    train_hybrid_parser.add_argument("--regression-manifest", default=str(DEFAULT_REGRESSION_CASES_MANIFEST), help="Optional regression cases manifest CSV path")
    train_hybrid_parser.add_argument("--train-per-label", type=int, default=30000, help="Non-JP training rows per label")
    train_hybrid_parser.add_argument("--validation-per-label", type=int, default=5000, help="Non-JP validation rows per label")
    train_hybrid_parser.add_argument("--train-per-label-ja", type=int, default=12000, help="JP training rows per label")
    train_hybrid_parser.add_argument("--validation-per-label-ja", type=int, default=2000, help="JP validation rows per label")
    train_hybrid_parser.add_argument("--live-weight", type=float, default=8.0, help="Sample weight for human live rows")
    train_hybrid_parser.add_argument("--live-ai-page-weight", type=float, default=10.0, help="Sample weight for explicit page-level AI rows")
    train_hybrid_parser.add_argument("--live-ai-site-weight", type=float, default=6.0, help="Sample weight for site-level AI rows")
    train_hybrid_parser.set_defaults(func=command_train_hybrid)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
