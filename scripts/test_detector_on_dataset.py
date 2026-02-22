#!/usr/bin/env python3
"""
Evaluate the local hash-NB detector against a text/label dataset.

Input dataset schema:
  - text: string
  - label: "AI" | "Human"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq


DEFAULT_MODEL_PATH = "data/processed/hash_nb_model_4096_sampled.json"
DEFAULT_HUMAN_THRESHOLD = 0.45
DEFAULT_AI_THRESHOLD = 0.55

WORKER_MODEL: Optional[Dict] = None
WORKER_MODEL_JA: Optional[Dict] = None
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def normalize_text(value: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", str(value or "").replace("\u00a0", " ")).strip().lower()


def clamp(value: float, min_value: float, max_value: float) -> float:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return min_value
    return max(min_value, min(max_value, float(value)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def hash_trigram(text: str, start: int, dim: int) -> int:
    h = 2166136261
    for i in range(3):
        h ^= ord(text[start + i])
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def compute_score(text: str, model: Dict) -> float:
    dim = max(1, int(model["dim"]))
    max_chars = max(200, int(model["max_chars"]))
    model_type = str(model.get("type") or "naive_bayes_hash3")

    normalized = normalize_text(text)
    if len(normalized) < 3:
        return 0.5

    limit = min(len(normalized), max_chars)
    if model_type == "logistic_hash3":
        weights = model["weights"]
        bias = float(model["bias"])
        trigram_count = max(1, limit - 2)
        bucket_counts: Dict[int, int] = {}
        for i in range(limit - 2):
            idx = hash_trigram(normalized, i, dim)
            bucket_counts[idx] = bucket_counts.get(idx, 0) + 1
        logit = bias
        for idx, count in bucket_counts.items():
            logit += float(weights[idx]) * (count / trigram_count)
        return sigmoid(logit)

    delta = model["delta"]
    prior = float(model["prior_logit"])
    logit = prior
    for i in range(limit - 2):
        idx = hash_trigram(normalized, i, dim)
        logit += float(delta[idx])
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
            "weights": [float(x) for x in weights],
        }

    delta = model.get("delta") or []
    if not isinstance(delta, list) or len(delta) != dim:
        raise ValueError("Invalid NB model: delta length mismatch")
    return {
        **base,
        "prior_logit": float(model.get("prior_logit") or 0.0),
        "delta": [float(x) for x in delta],
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
    conf: Counter = field(default_factory=Counter)  # key=(gt,pred)
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

    total = 0
    strict_correct = 0
    decided_rows = 0
    decided_correct = 0
    score_sum = 0.0
    gt_ai = 0
    gt_human = 0
    pred_ai = 0
    pred_human = 0
    pred_unknown = 0
    ai_ai = 0
    ai_human = 0
    ai_unknown = 0
    human_ai = 0
    human_human = 0
    human_unknown = 0
    jp_rows = 0
    jp_strict_correct = 0
    non_jp_rows = 0
    non_jp_strict_correct = 0

    for gt, text in rows:
        row_model = actual_model
        row_human_threshold = human_threshold
        row_ai_threshold = ai_threshold

        is_japanese = is_likely_japanese_text(text)
        if actual_model_ja is not None and is_japanese:
            row_model = actual_model_ja
            row_human_threshold = human_threshold if human_threshold_ja is None else float(human_threshold_ja)
            row_ai_threshold = ai_threshold if ai_threshold_ja is None else float(ai_threshold_ja)

        score = compute_score(text, row_model)
        pred = predict_judge(score, human_threshold=row_human_threshold, ai_threshold=row_ai_threshold)

        total += 1
        score_sum += score
        if pred == gt:
            strict_correct += 1

        if is_japanese:
            jp_rows += 1
            if pred == gt:
                jp_strict_correct += 1
        else:
            non_jp_rows += 1
            if pred == gt:
                non_jp_strict_correct += 1

        if pred != "Unknown":
            decided_rows += 1
            if pred == gt:
                decided_correct += 1

        if gt == "AI":
            gt_ai += 1
            if pred == "AI":
                ai_ai += 1
            elif pred == "Human":
                ai_human += 1
            else:
                ai_unknown += 1
        else:
            gt_human += 1
            if pred == "AI":
                human_ai += 1
            elif pred == "Human":
                human_human += 1
            else:
                human_unknown += 1

        if pred == "AI":
            pred_ai += 1
        elif pred == "Human":
            pred_human += 1
        else:
            pred_unknown += 1

    return {
        "total": total,
        "strict_correct": strict_correct,
        "decided_rows": decided_rows,
        "decided_correct": decided_correct,
        "score_sum": score_sum,
        "gt_ai": gt_ai,
        "gt_human": gt_human,
        "pred_ai": pred_ai,
        "pred_human": pred_human,
        "pred_unknown": pred_unknown,
        "ai_ai": ai_ai,
        "ai_human": ai_human,
        "ai_unknown": ai_unknown,
        "human_ai": human_ai,
        "human_human": human_human,
        "human_unknown": human_unknown,
        "jp_rows": jp_rows,
        "jp_strict_correct": jp_strict_correct,
        "non_jp_rows": non_jp_rows,
        "non_jp_strict_correct": non_jp_strict_correct,
    }


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


def select_valid_rows(
    rows: List[Dict],
    max_rows: int,
    accepted_rows: int,
) -> Tuple[List[Tuple[str, str]], int, int, bool]:
    valid_rows: List[Tuple[str, str]] = []
    skipped = 0
    stop = False

    for row in rows:
        gt_raw = str(row.get("label") or "").strip().lower()
        if gt_raw not in ("ai", "human"):
            skipped += 1
            continue
        gt = "AI" if gt_raw == "ai" else "Human"
        text = str(row.get("text") or "")
        if not normalize_text(text):
            skipped += 1
            continue

        valid_rows.append((gt, text))
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
    pf = pq.ParquetFile(parquet_path)
    worker_count = max(1, int(workers))
    accepted_rows = 0
    should_stop = False

    if worker_count == 1:
        for batch in pf.iter_batches(columns=["text", "label"], batch_size=2048):
            rows = batch.to_pylist()
            valid_rows, accepted_rows, skipped, should_stop = select_valid_rows(rows, max_rows, accepted_rows)
            state.skipped += skipped
            if valid_rows:
                part = evaluate_valid_rows(
                    valid_rows,
                    human_threshold=human_threshold,
                    ai_threshold=ai_threshold,
                    model=model,
                    model_ja=model_ja,
                    human_threshold_ja=human_threshold_ja,
                    ai_threshold_ja=ai_threshold_ja,
                )
                merge_partial_result(state, part)
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
            for batch in pf.iter_batches(columns=["text", "label"], batch_size=2048):
                rows = batch.to_pylist()
                valid_rows, accepted_rows, skipped, should_stop = select_valid_rows(rows, max_rows, accepted_rows)
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
                    for fut in done:
                        merge_partial_result(state, fut.result())

                if should_stop:
                    break

            for fut in concurrent.futures.as_completed(pending):
                merge_partial_result(state, fut.result())

    tp_ai = state.conf[("AI", "AI")]
    fp_ai = state.conf[("Human", "AI")]
    fn_ai = state.conf[("AI", "Human")] + state.conf[("AI", "Unknown")]
    p_ai, r_ai, f1_ai = precision_recall_f1(tp_ai, fp_ai, fn_ai)

    tp_h = state.conf[("Human", "Human")]
    fp_h = state.conf[("AI", "Human")]
    fn_h = state.conf[("Human", "AI")] + state.conf[("Human", "Unknown")]
    p_h, r_h, f1_h = precision_recall_f1(tp_h, fp_h, fn_h)

    strict_accuracy = state.strict_correct / state.total if state.total else 0.0
    decided_accuracy = state.decided_correct / state.decided_rows if state.decided_rows else 0.0
    coverage = state.decided_rows / state.total if state.total else 0.0
    unknown_rate = state.pred_counts["Unknown"] / state.total if state.total else 0.0
    avg_score = state.score_sum / state.total if state.total else 0.0
    jp_strict_accuracy = state.jp_strict_correct / state.jp_rows if state.jp_rows else 0.0
    non_jp_strict_accuracy = state.non_jp_strict_correct / state.non_jp_rows if state.non_jp_rows else 0.0

    return {
        "input": str(parquet_path),
        "thresholds": {
            "human_max": round(human_threshold, 4),
            "ai_min": round(ai_threshold, 4),
        },
        "thresholds_ja": {
            "human_max": round(human_threshold_ja, 4),
            "ai_min": round(ai_threshold_ja, 4),
        },
        "max_rows": max_rows,
        "processed_rows": state.total,
        "skipped_rows": state.skipped,
        "strict_accuracy": round(strict_accuracy, 6),
        "decided_accuracy": round(decided_accuracy, 6),
        "coverage": round(coverage, 6),
        "unknown_rate": round(unknown_rate, 6),
        "avg_score": round(avg_score, 4),
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
            "AI": {"precision": round(p_ai, 6), "recall": round(r_ai, 6), "f1": round(f1_ai, 6)},
            "Human": {"precision": round(p_h, 6), "recall": round(r_h, 6), "f1": round(f1_h, 6)},
        },
        "model": {
            "name": model["name"],
            "dim": model["dim"],
            "max_chars": model["max_chars"],
        },
        "model_ja": (
            {
                "name": model_ja["name"],
                "dim": model_ja["dim"],
                "max_chars": model_ja["max_chars"],
            }
            if model_ja is not None
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/unified_text_label.parquet",
        help="Unified parquet path (text,label)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Model json path",
    )
    parser.add_argument(
        "--model-ja",
        default="",
        help="Japanese model json path (optional)",
    )
    parser.add_argument(
        "--human-threshold",
        type=float,
        default=-1.0,
        help="score < human-threshold -> Human (negative means use model default)",
    )
    parser.add_argument(
        "--ai-threshold",
        type=float,
        default=-1.0,
        help="score >= ai-threshold -> AI (between is Unknown; negative means use model default)",
    )
    parser.add_argument(
        "--human-threshold-ja",
        type=float,
        default=-1.0,
        help="JP: score < human-threshold-ja -> Human (negative means use JA model default)",
    )
    parser.add_argument(
        "--ai-threshold-ja",
        type=float,
        default=-1.0,
        help="JP: score >= ai-threshold-ja -> AI (negative means use JA model default)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--output",
        default="data/processed/detector_eval_summary.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel worker count (1 disables parallel execution)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        model_ai_ja = clamp(float(model_ja.get("thresholds", {}).get("ai_min", DEFAULT_AI_THRESHOLD)), 0.0, 1.0)
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

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"summary={out}")


if __name__ == "__main__":
    main()
