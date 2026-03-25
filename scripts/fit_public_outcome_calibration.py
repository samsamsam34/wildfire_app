#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.version import (  # noqa: E402
    API_VERSION,
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    SCORING_MODEL_VERSION,
)
from backend.public_outcome_governance import (  # noqa: E402
    build_calibration_comparison_markdown,
    build_calibration_run_comparison,
    list_public_outcome_runs,
    resolve_baseline_run_id,
)

DEFAULT_EVALUATION_DATASET_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_CALIBRATION_OUTPUT_ROOT = Path("benchmark/public_outcomes/calibration")


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    score: float
    label: int
    confidence_tier: str
    evidence_tier: str
    join_confidence_tier: str
    evidence_group: str
    fallback_weight_fraction: float


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_damage_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"destroyed", "total_loss"}:
        return "destroyed"
    if text in {"major", "major_damage", "severe", "severe_damage"}:
        return "major_damage"
    if text in {"minor", "minor_damage", "affected"}:
        return "minor_damage"
    if text in {"none", "no_damage", "no_known_damage", "undamaged"}:
        return "no_damage"
    return "unknown"


def _infer_label_from_outcome_class(label: str) -> int | None:
    normalized = _normalize_damage_class(label)
    if normalized in {"major_damage", "destroyed"}:
        return 1
    if normalized in {"minor_damage", "no_damage"}:
        return 0
    return None


def _normalize_evidence_group(*, evidence_tier: str, fallback_factor_count: int, missing_factor_count: int, fallback_weight_fraction: float) -> str:
    if evidence_tier in {"low", "preliminary"}:
        return "fallback_heavy"
    if evidence_tier == "high" and fallback_factor_count == 0 and missing_factor_count <= 1:
        return "high_evidence"
    if fallback_factor_count >= 2 or missing_factor_count >= 3 or fallback_weight_fraction >= 0.5:
        return "fallback_heavy"
    return "mixed_evidence"


def _load_raw_rows(dataset_path: Path) -> list[dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [row for row in payload["rows"] if isinstance(row, dict)]
        if isinstance(payload.get("records"), list):
            return [row for row in payload["records"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("Unsupported dataset format. Expected JSON rows/records, JSONL, or CSV.")


def _sample_from_joined_row(row: dict[str, Any]) -> CalibrationSample | None:
    event = row.get("event") if isinstance(row.get("event"), dict) else {}
    feature = row.get("feature") if isinstance(row.get("feature"), dict) else {}
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    confidence = row.get("confidence") if isinstance(row.get("confidence"), dict) else {}
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    evidence_summary = (
        evidence.get("evidence_quality_summary")
        if isinstance(evidence.get("evidence_quality_summary"), dict)
        else {}
    )
    join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}

    score = _safe_float(scores.get("wildfire_risk_score"))
    label = _safe_int(outcome.get("structure_loss_or_major_damage"))
    if label not in {0, 1}:
        label = _infer_label_from_outcome_class(
            outcome.get("damage_severity_class") or outcome.get("damage_label")
        )
    if score is None or label not in {0, 1}:
        return None

    event_id = str(event.get("event_id") or "unknown_event").strip()
    record_id = str(feature.get("record_id") or outcome.get("record_id") or "unknown_record").strip()
    confidence_tier = str(confidence.get("confidence_tier") or "unknown").strip().lower() or "unknown"
    evidence_tier = str(evidence.get("evidence_quality_tier") or "unknown").strip().lower() or "unknown"
    join_confidence_tier = str(join_meta.get("join_confidence_tier") or "unknown").strip().lower() or "unknown"
    fallback_factor_count = int(_safe_int(evidence_summary.get("fallback_factor_count")) or 0)
    missing_factor_count = int(_safe_int(evidence_summary.get("missing_factor_count")) or 0)
    fallback_weight_fraction = float(_safe_float(evidence_summary.get("fallback_weight_fraction")) or 0.0)
    evidence_group = _normalize_evidence_group(
        evidence_tier=evidence_tier,
        fallback_factor_count=fallback_factor_count,
        missing_factor_count=missing_factor_count,
        fallback_weight_fraction=fallback_weight_fraction,
    )
    return CalibrationSample(
        sample_id=f"{event_id}::{record_id}",
        score=float(score),
        label=int(label),
        confidence_tier=confidence_tier,
        evidence_tier=evidence_tier,
        join_confidence_tier=join_confidence_tier,
        evidence_group=evidence_group,
        fallback_weight_fraction=fallback_weight_fraction,
    )


def _sample_from_flat_row(row: dict[str, Any]) -> CalibrationSample | None:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    score = _safe_float(row.get("wildfire_risk_score"))
    if score is None:
        score = _safe_float(scores.get("wildfire_risk_score"))

    label = _safe_int(row.get("adverse"))
    if label not in {0, 1}:
        label = _safe_int(row.get("structure_loss_or_major_damage"))
    if label not in {0, 1}:
        label = _infer_label_from_outcome_class(row.get("outcome_label"))
    if score is None or label not in {0, 1}:
        return None

    fallback_flags = row.get("fallback_default_flags") if isinstance(row.get("fallback_default_flags"), dict) else {}
    confidence_tier = str(row.get("confidence_tier") or "unknown").strip().lower() or "unknown"
    evidence_tier = str(row.get("evidence_quality_tier") or "unknown").strip().lower() or "unknown"
    join_confidence_tier = str(row.get("join_confidence_tier") or "unknown").strip().lower() or "unknown"
    fallback_factor_count = int(_safe_int(fallback_flags.get("fallback_factor_count")) or 0)
    missing_factor_count = int(_safe_int(fallback_flags.get("missing_factor_count")) or 0)
    fallback_weight_fraction = float(_safe_float(fallback_flags.get("fallback_weight_fraction")) or 0.0)
    evidence_group = _normalize_evidence_group(
        evidence_tier=evidence_tier,
        fallback_factor_count=fallback_factor_count,
        missing_factor_count=missing_factor_count,
        fallback_weight_fraction=fallback_weight_fraction,
    )
    event_id = str(row.get("event_id") or "unknown_event").strip()
    record_id = str(row.get("record_id") or "unknown_record").strip()
    return CalibrationSample(
        sample_id=f"{event_id}::{record_id}",
        score=float(score),
        label=int(label),
        confidence_tier=confidence_tier,
        evidence_tier=evidence_tier,
        join_confidence_tier=join_confidence_tier,
        evidence_group=evidence_group,
        fallback_weight_fraction=fallback_weight_fraction,
    )


def _extract_samples(raw_rows: list[dict[str, Any]]) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    for row in raw_rows:
        if "outcome" in row and "scores" in row:
            sample = _sample_from_joined_row(row)
        else:
            sample = _sample_from_flat_row(row)
        if sample is not None:
            samples.append(sample)
    samples.sort(key=lambda item: item.sample_id)
    return samples


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _fit_logistic(scores: list[float], labels: list[int], *, epochs: int = 2500, lr: float = 0.35) -> tuple[float, float]:
    x = [max(0.0, min(1.0, s / 100.0)) for s in scores]
    y = [int(v) for v in labels]
    b0 = 0.0
    b1 = 1.0
    n = float(len(x))
    for _ in range(epochs):
        grad0 = 0.0
        grad1 = 0.0
        for xi, yi in zip(x, y):
            pred = _sigmoid(b0 + (b1 * xi))
            err = pred - float(yi)
            grad0 += err
            grad1 += err * xi
        b0 -= lr * (grad0 / n)
        b1 -= lr * (grad1 / n)
    return b0, b1


def _predict_logistic(scores: list[float], intercept: float, slope: float) -> list[float]:
    out: list[float] = []
    for score in scores:
        x = max(0.0, min(1.0, score / 100.0))
        out.append(_sigmoid(intercept + (slope * x)))
    return out


def _fit_isotonic_points(scores: list[float], labels: list[int]) -> list[list[float]]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    if not pairs:
        return []
    # PAV block merge.
    blocks: list[dict[str, float]] = []
    for score, label in pairs:
        blocks.append({"weight": 1.0, "sum_y": float(label), "sum_x": float(score)})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["sum_y"] / left["weight"]
            right_mean = right["sum_y"] / right["weight"]
            if left_mean <= right_mean:
                break
            merged = {
                "weight": left["weight"] + right["weight"],
                "sum_y": left["sum_y"] + right["sum_y"],
                "sum_x": left["sum_x"] + right["sum_x"],
            }
            blocks.pop()
            blocks.pop()
            blocks.append(merged)

    points: list[list[float]] = []
    for block in blocks:
        mean_x = block["sum_x"] / block["weight"]
        mean_y = block["sum_y"] / block["weight"]
        points.append([round(mean_x, 6), round(max(0.0, min(1.0, mean_y)), 8)])
    if len(points) < 2:
        mean_rate = sum(labels) / float(len(labels))
        low = min(scores)
        high = max(scores)
        return [[round(low, 6), round(mean_rate, 8)], [round(high, 6), round(mean_rate, 8)]]
    return points


def _predict_piecewise(scores: list[float], points: list[list[float]]) -> list[float]:
    if len(points) < 2:
        return [0.5 for _ in scores]
    parsed = sorted(
        [(float(x), max(0.0, min(1.0, float(y)))) for x, y in points],
        key=lambda item: item[0],
    )
    out: list[float] = []
    for score in scores:
        if score <= parsed[0][0]:
            out.append(parsed[0][1])
            continue
        if score >= parsed[-1][0]:
            out.append(parsed[-1][1])
            continue
        placed = False
        for idx in range(1, len(parsed)):
            x0, y0 = parsed[idx - 1]
            x1, y1 = parsed[idx]
            if x0 <= score <= x1:
                if x1 == x0:
                    out.append((y0 + y1) / 2.0)
                else:
                    ratio = (score - x0) / (x1 - x0)
                    out.append(y0 + (ratio * (y1 - y0)))
                placed = True
                break
        if not placed:
            out.append(parsed[-1][1])
    return out


def _fit_bin_rate_table(
    scores: list[float],
    labels: list[int],
    *,
    bin_count: int = 8,
    smoothing_strength: float = 2.0,
) -> list[dict[str, float | int]]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    if not pairs:
        return []
    total = len(pairs)
    global_rate = (sum(int(y) for _, y in pairs) / float(total)) if total > 0 else 0.5
    bins: list[dict[str, float | int]] = []
    bins_use = max(2, int(bin_count))
    for idx in range(bins_use):
        start = int((idx / bins_use) * total)
        end = int(((idx + 1) / bins_use) * total)
        if end <= start:
            continue
        bucket = pairs[start:end]
        if not bucket:
            continue
        bucket_scores = [float(score) for score, _ in bucket]
        positives = sum(int(label) for _, label in bucket)
        count = len(bucket)
        smoothed_rate = (
            (float(positives) + (float(smoothing_strength) * float(global_rate)))
            / (float(count) + float(smoothing_strength))
        )
        bins.append(
            {
                "bin": int(idx + 1),
                "count": int(count),
                "positives": int(positives),
                "score_min": float(min(bucket_scores)),
                "score_max": float(max(bucket_scores)),
                "score_center": float(sum(bucket_scores) / float(count)),
                "probability": float(max(0.0, min(1.0, smoothed_rate))),
            }
        )
    if not bins:
        return []
    # Enforce monotonic non-decreasing probabilities to avoid erratic steps.
    running = 0.0
    for row in bins:
        prob = float(row.get("probability") or 0.0)
        running = max(running, prob)
        row["probability"] = float(max(0.0, min(1.0, running)))
    return bins


def _predict_bin_rate_table(scores: list[float], bin_table: list[dict[str, Any]]) -> list[float]:
    if not bin_table:
        return [0.5 for _ in scores]
    parsed: list[tuple[float, float, float]] = []
    for row in bin_table:
        if not isinstance(row, dict):
            continue
        s_min = _safe_float(row.get("score_min"))
        s_max = _safe_float(row.get("score_max"))
        prob = _safe_float(row.get("probability"))
        if s_min is None or s_max is None or prob is None:
            continue
        lo = min(float(s_min), float(s_max))
        hi = max(float(s_min), float(s_max))
        parsed.append((lo, hi, max(0.0, min(1.0, float(prob)))))
    if not parsed:
        return [0.5 for _ in scores]
    parsed.sort(key=lambda item: (item[0], item[1]))
    out: list[float] = []
    for score in scores:
        matched = None
        for lo, hi, prob in parsed:
            if lo <= float(score) <= hi:
                matched = prob
                break
        if matched is not None:
            out.append(matched)
            continue
        if float(score) < parsed[0][0]:
            out.append(parsed[0][2])
            continue
        out.append(parsed[-1][2])
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = _mean(x)
    my = _mean(y)
    if mx is None or my is None:
        return None
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    return _pearson(_rank(x), _rank(y))


def _roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) != len(y_score) or len(y_true) < 3:
        return None
    pos = sum(1 for y in y_true if y == 1)
    neg = sum(1 for y in y_true if y == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = _rank(y_score)
    pos_rank_sum = sum(rank for rank, y in zip(ranks, y_true) if y == 1)
    auc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return max(0.0, min(1.0, auc))


def _pr_auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) != len(y_score) or len(y_true) < 3:
        return None
    pos_total = sum(1 for y in y_true if y == 1)
    if pos_total <= 0:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    curve: list[tuple[float, float]] = [(0.0, 1.0)]
    for _, truth in pairs:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / float(pos_total)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
        curve.append((recall, precision))
    area = 0.0
    prev_recall, prev_precision = curve[0]
    for recall, precision in curve[1:]:
        area += max(0.0, recall - prev_recall) * ((precision + prev_precision) / 2.0)
        prev_recall = recall
        prev_precision = precision
    return max(0.0, min(1.0, area))


def _brier(y_true: list[int], probs: list[float]) -> float | None:
    if len(y_true) != len(probs) or not y_true:
        return None
    return sum((float(y) - float(p)) ** 2 for y, p in zip(y_true, probs)) / float(len(y_true))


def _log_loss(y_true: list[int], probs: list[float]) -> float | None:
    if len(y_true) != len(probs) or not y_true:
        return None
    eps = 1e-12
    total = 0.0
    for y, p in zip(y_true, probs):
        p = min(1.0 - eps, max(eps, float(p)))
        total += -(float(y) * math.log(p) + (1.0 - float(y)) * math.log(1.0 - p))
    return total / float(len(y_true))


def _calibration_bins(y_true: list[int], probs: list[float], bins: int = 10) -> dict[str, Any]:
    if len(y_true) != len(probs) or not y_true:
        return {"bins": [], "expected_calibration_error": None}
    ranked = sorted(zip(probs, y_true), key=lambda item: item[0])
    n = len(ranked)
    out: list[dict[str, Any]] = []
    ece = 0.0
    for idx in range(max(1, int(bins))):
        start = int((idx / bins) * n)
        end = int(((idx + 1) / bins) * n)
        if end <= start:
            continue
        bucket = ranked[start:end]
        bucket_probs = [float(p) for p, _ in bucket]
        bucket_truth = [int(y) for _, y in bucket]
        mean_pred = _mean(bucket_probs)
        observed = _mean([float(v) for v in bucket_truth])
        if mean_pred is None or observed is None:
            continue
        count = len(bucket)
        ece += (count / float(n)) * abs(mean_pred - observed)
        out.append(
            {
                "bin": idx + 1,
                "count": count,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
                "probability_min": min(bucket_probs),
                "probability_max": max(bucket_probs),
            }
        )
    return {"bins": out, "expected_calibration_error": ece}


def _compute_metrics(y_true: list[int], raw_scores: list[float], probs: list[float]) -> dict[str, Any]:
    score_probs = [max(0.0, min(1.0, score / 100.0)) for score in raw_scores]
    return {
        "row_count": len(y_true),
        "positive_rate": _mean([float(v) for v in y_true]),
        "roc_auc_raw_score": _roc_auc(y_true, score_probs),
        "roc_auc_probability": _roc_auc(y_true, probs),
        "pr_auc_probability": _pr_auc(y_true, probs),
        "brier_probability": _brier(y_true, probs),
        "log_loss_probability": _log_loss(y_true, probs),
        "spearman_score_vs_label": _spearman(raw_scores, [float(v) for v in y_true]),
        "calibration": _calibration_bins(y_true, probs, bins=10),
    }


def _is_validation_bucket(sample_id: str) -> bool:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    token = int(digest[:8], 16) % 100
    return token < 20


def _split_samples(samples: list[CalibrationSample]) -> tuple[list[CalibrationSample], list[CalibrationSample]]:
    train: list[CalibrationSample] = []
    validation: list[CalibrationSample] = []
    for sample in samples:
        if _is_validation_bucket(sample.sample_id):
            validation.append(sample)
        else:
            train.append(sample)
    if len(validation) < 8:
        return samples, []
    val_labels = {sample.label for sample in validation}
    if len(val_labels) < 2:
        return samples, []
    return train, validation


def _evaluate_candidate(samples: list[CalibrationSample], probs: list[float]) -> dict[str, Any]:
    y = [sample.label for sample in samples]
    scores = [sample.score for sample in samples]
    return _compute_metrics(y, scores, probs)


def _choose_method(
    *,
    train_samples: list[CalibrationSample],
    validation_samples: list[CalibrationSample],
    method_preference: str,
    min_rows_for_isotonic: int,
    min_unique_scores_for_isotonic: int,
    min_rows_for_binned: int,
    min_unique_scores_for_binned: int,
    binned_bin_count: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    warnings: list[str] = []
    train_scores = [sample.score for sample in train_samples]
    train_labels = [sample.label for sample in train_samples]
    val_base = validation_samples if validation_samples else train_samples
    val_scores = [sample.score for sample in val_base]
    val_labels = [sample.label for sample in val_base]
    val_raw_probs = [max(0.0, min(1.0, score / 100.0)) for score in val_scores]
    raw_validation_metrics = _evaluate_candidate(val_base, val_raw_probs)

    logistic_intercept, logistic_slope = _fit_logistic(train_scores, train_labels)
    logistic_probs_val = _predict_logistic(val_scores, logistic_intercept, logistic_slope)
    logistic_eval = _evaluate_candidate(val_base, logistic_probs_val)
    candidates: dict[str, dict[str, Any]] = {
        "logistic": {
            "method": "logistic",
            "parameters": {
                "intercept": round(logistic_intercept, 8),
                "slope": round(logistic_slope, 8),
                "x_scale": 100.0,
            },
            "validation_metrics": logistic_eval,
            "train_predictions": _predict_logistic(train_scores, logistic_intercept, logistic_slope),
            "predictor": {"intercept": logistic_intercept, "slope": logistic_slope},
        }
    }
    logistic_raw_spearman = _spearman(
        [max(0.0, min(1.0, score / 100.0)) for score in train_scores],
        [float(label) for label in train_labels],
    )
    if logistic_raw_spearman is not None and logistic_raw_spearman <= 0.05:
        warnings.append(
            "Logistic calibration assumption warning: score-to-label monotonic signal is weak in training data."
        )

    unique_scores = len({round(score, 6) for score in train_scores})
    isotonic_allowed = (
        method_preference in {"auto", "isotonic"}
        and len(train_samples) >= int(min_rows_for_isotonic)
        and unique_scores >= int(min_unique_scores_for_isotonic)
    )
    if isotonic_allowed:
        points = _fit_isotonic_points(train_scores, train_labels)
        isotonic_probs_val = _predict_piecewise(val_scores, points)
        isotonic_eval = _evaluate_candidate(val_base, isotonic_probs_val)
        candidates["isotonic"] = {
            "method": "piecewise_linear",
            "fit_family": "isotonic_regression",
            "parameters": {"points": points},
            "validation_metrics": isotonic_eval,
            "train_predictions": _predict_piecewise(train_scores, points),
            "predictor": {"points": points},
        }
    elif method_preference == "isotonic":
        warnings.append(
            "Requested isotonic calibration but data volume/score diversity is insufficient; using logistic fallback."
        )

    binned_allowed = (
        method_preference in {"auto", "binned"}
        and len(train_samples) >= int(min_rows_for_binned)
        and unique_scores >= int(min_unique_scores_for_binned)
    )
    if binned_allowed:
        bin_table = _fit_bin_rate_table(
            train_scores,
            train_labels,
            bin_count=max(3, int(binned_bin_count)),
            smoothing_strength=2.0,
        )
        binned_probs_val = _predict_bin_rate_table(val_scores, bin_table)
        binned_eval = _evaluate_candidate(val_base, binned_probs_val)
        candidates["binned"] = {
            "method": "bin_rate_table",
            "fit_family": "binned_reliability",
            "parameters": {
                "bin_count": max(3, int(binned_bin_count)),
                "bin_table": bin_table,
                "smoothing_strength": 2.0,
            },
            "validation_metrics": binned_eval,
            "train_predictions": _predict_bin_rate_table(train_scores, bin_table),
            "predictor": {"bin_table": bin_table},
        }
    elif method_preference == "binned":
        warnings.append(
            "Requested binned calibration but data volume/score diversity is insufficient; using logistic fallback."
        )

    if method_preference == "logistic":
        selected_key = "logistic"
    elif method_preference == "isotonic" and "isotonic" in candidates:
        selected_key = "isotonic"
    elif method_preference == "binned" and "binned" in candidates:
        selected_key = "binned"
    elif method_preference == "auto":
        # Auto: prefer lower validation Brier, but require a margin to switch from logistic.
        baseline = "logistic"
        baseline_brier = _safe_float(
            ((candidates[baseline].get("validation_metrics") or {}).get("brier_probability"))
        )
        selected_key = baseline
        selected_brier = baseline_brier
        for key in sorted(candidates.keys()):
            metrics = candidates[key].get("validation_metrics") if isinstance(candidates[key], dict) else {}
            brier = _safe_float((metrics or {}).get("brier_probability"))
            if brier is None:
                continue
            if selected_brier is None:
                selected_key = key
                selected_brier = brier
                continue
            if key == baseline:
                continue
            # Cautious method switching: require material validation gain over logistic.
            if baseline_brier is not None and brier + 0.002 < baseline_brier:
                if selected_brier is None or brier < selected_brier:
                    selected_key = key
                    selected_brier = brier
    else:
        selected_key = "logistic"

    selected = dict(candidates[selected_key])
    selected["selected_key"] = selected_key
    selected["selection_reason"] = (
        f"selected_{selected_key}_from_{','.join(sorted(candidates.keys()))}"
    )
    diagnostics = {
        "raw_validation_metrics": raw_validation_metrics,
        "candidate_methods": {
            key: {
                "method": value.get("method"),
                "fit_family": value.get("fit_family", key),
                "validation_metrics": value.get("validation_metrics"),
                "brier_delta_vs_raw": (
                    _safe_float(((value.get("validation_metrics") or {}).get("brier_probability")))
                    - _safe_float((raw_validation_metrics or {}).get("brier_probability"))
                    if _safe_float(((value.get("validation_metrics") or {}).get("brier_probability"))) is not None
                    and _safe_float((raw_validation_metrics or {}).get("brier_probability")) is not None
                    else None
                ),
            }
            for key, value in candidates.items()
        },
        "validation_split_used": bool(validation_samples),
        "train_row_count": len(train_samples),
        "validation_row_count": len(validation_samples),
        "class_balance": {
            "train_positive_rate": _mean([float(v) for v in train_labels]),
            "validation_positive_rate": _mean([float(v) for v in val_labels]),
        },
        "selection_reason": selected.get("selection_reason"),
    }
    return selected, warnings, diagnostics


def _pre_post_metrics(samples: list[CalibrationSample], calibrated_probs: list[float]) -> dict[str, Any]:
    y = [sample.label for sample in samples]
    scores = [sample.score for sample in samples]
    raw_probs = [max(0.0, min(1.0, score / 100.0)) for score in scores]
    pre = _compute_metrics(y, scores, raw_probs)
    post = _compute_metrics(y, scores, calibrated_probs)

    def _slice(name: str, selector: Any) -> dict[str, Any]:
        chosen = [sample for sample in samples if selector(sample)]
        if not chosen:
            return {"row_count": 0}
        idx = {sample.sample_id for sample in chosen}
        y_sub = [sample.label for sample in samples if sample.sample_id in idx]
        scores_sub = [sample.score for sample in samples if sample.sample_id in idx]
        raw_sub = [max(0.0, min(1.0, score / 100.0)) for score in scores_sub]
        post_sub = [prob for sample, prob in zip(samples, calibrated_probs) if sample.sample_id in idx]
        return {
            "row_count": len(y_sub),
            "pre": _compute_metrics(y_sub, scores_sub, raw_sub),
            "post": _compute_metrics(y_sub, scores_sub, post_sub),
        }

    slices = {
        "high_evidence": _slice("high_evidence", lambda sample: sample.evidence_group == "high_evidence"),
        "fallback_heavy": _slice("fallback_heavy", lambda sample: sample.evidence_group == "fallback_heavy"),
        "high_join_confidence": _slice("high_join_confidence", lambda sample: sample.join_confidence_tier == "high"),
        "low_join_confidence": _slice("low_join_confidence", lambda sample: sample.join_confidence_tier == "low"),
    }
    return {
        "pre": pre,
        "post": post,
        "delta": {
            "brier_improvement": (
                _safe_float(pre.get("brier_probability")) - _safe_float(post.get("brier_probability"))
                if _safe_float(pre.get("brier_probability")) is not None
                and _safe_float(post.get("brier_probability")) is not None
                else None
            ),
            "log_loss_improvement": (
                _safe_float(pre.get("log_loss_probability")) - _safe_float(post.get("log_loss_probability"))
                if _safe_float(pre.get("log_loss_probability")) is not None
                and _safe_float(post.get("log_loss_probability")) is not None
                else None
            ),
        },
        "slices": slices,
    }


def _quality_warnings(samples: list[CalibrationSample]) -> list[str]:
    warnings: list[str] = []
    if not samples:
        return warnings
    n = float(len(samples))
    low_confidence = sum(1 for sample in samples if sample.confidence_tier in {"low", "unknown"}) / n
    low_join = sum(1 for sample in samples if sample.join_confidence_tier in {"low", "unknown"}) / n
    fallback_heavy = sum(1 for sample in samples if sample.evidence_group == "fallback_heavy") / n
    if low_confidence >= 0.5:
        warnings.append("More than half of training rows are low-confidence assessments.")
    if low_join >= 0.4:
        warnings.append("A large share of training rows have low/unknown join-confidence tiers.")
    if fallback_heavy >= 0.5:
        warnings.append("Fallback-heavy evidence dominates this calibration dataset.")
    return warnings


def _resolve_latest_dataset_jsonl(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise ValueError(
            f"Evaluation dataset root does not exist: {dataset_root}. "
            "Run scripts/build_public_outcome_evaluation_dataset.py first."
        )
    run_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidate = run_dir / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
    raise ValueError(
        f"No evaluation_dataset.jsonl files found under {dataset_root}. "
        "Run scripts/build_public_outcome_evaluation_dataset.py first."
    )


def _resolve_dataset_path(
    *,
    dataset_path: Path | None,
    dataset_root: Path,
    dataset_run_id: str | None,
) -> Path:
    if dataset_path is not None:
        resolved = dataset_path.expanduser()
        if not resolved.exists():
            raise ValueError(f"Dataset not found: {resolved}")
        return resolved
    if dataset_run_id:
        run_dir = dataset_root / str(dataset_run_id)
        candidate_jsonl = run_dir / "evaluation_dataset.jsonl"
        candidate_json = run_dir / "evaluation_dataset.json"
        if candidate_jsonl.exists():
            return candidate_jsonl
        if candidate_json.exists():
            return candidate_json
        raise ValueError(
            f"Dataset run '{dataset_run_id}' not found under {dataset_root} or missing evaluation_dataset file."
        )
    return _resolve_latest_dataset_jsonl(dataset_root)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_feature_input_versions(raw_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    observed: dict[str, set[str]] = {
        "scoring_model_versions": set(),
        "factor_schema_versions": set(),
        "rules_logic_versions": set(),
        "region_data_versions": set(),
        "data_bundle_versions": set(),
    }
    for row in raw_rows:
        governance = None
        if isinstance(row.get("model_governance"), dict):
            governance = row.get("model_governance")
        elif isinstance(row.get("provenance"), dict) and isinstance((row.get("provenance") or {}).get("model_governance"), dict):
            governance = (row.get("provenance") or {}).get("model_governance")
        governance = governance if isinstance(governance, dict) else {}
        for key, field in (
            ("scoring_model_versions", "scoring_model_version"),
            ("factor_schema_versions", "factor_schema_version"),
            ("rules_logic_versions", "rules_logic_version"),
            ("region_data_versions", "region_data_version"),
            ("data_bundle_versions", "data_bundle_version"),
        ):
            text = str(governance.get(field) or "").strip()
            if text:
                observed[key].add(text)
    return {key: sorted(values) for key, values in observed.items() if values}


def _dataset_governance(dataset_path: Path, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_manifest_path = dataset_path.parent / "manifest.json"
    dataset_manifest = _safe_load_json(dataset_manifest_path) if dataset_manifest_path.exists() else None
    dataset_inputs = dataset_manifest.get("inputs") if isinstance(dataset_manifest, dict) else {}
    outcomes_path = (
        Path(str(dataset_inputs.get("normalized_outcomes_path"))).expanduser()
        if isinstance(dataset_inputs, dict) and dataset_inputs.get("normalized_outcomes_path")
        else None
    )
    outcomes_manifest_path = outcomes_path.parent / "manifest.json" if outcomes_path else None
    outcomes_manifest = (
        _safe_load_json(outcomes_manifest_path)
        if outcomes_manifest_path is not None and outcomes_manifest_path.exists()
        else None
    )
    return {
        "evaluation_dataset_run_id": dataset_path.parent.name,
        "evaluation_dataset_schema_version": (
            dataset_manifest.get("schema_version")
            if isinstance(dataset_manifest, dict)
            else None
        ),
        "evaluation_dataset_manifest_path": (
            str(dataset_manifest_path) if dataset_manifest_path.exists() else None
        ),
        "outcome_dataset_run_id": (
            outcomes_manifest.get("run_id")
            if isinstance(outcomes_manifest, dict)
            else None
        ),
        "outcome_dataset_schema_version": (
            outcomes_manifest.get("schema_version")
            if isinstance(outcomes_manifest, dict)
            else None
        ),
        "outcome_dataset_manifest_path": (
            str(outcomes_manifest_path)
            if outcomes_manifest_path is not None and outcomes_manifest_path.exists()
            else None
        ),
        "feature_input_versions": _extract_feature_input_versions(raw_rows),
    }


def _build_summary_markdown(
    *,
    run_id: str,
    generated_at: str,
    dataset_path: Path,
    fitted: bool,
    artifact: dict[str, Any] | None,
    pre_post: dict[str, Any],
    warnings: list[str],
    selection_diagnostics: dict[str, Any] | None = None,
) -> str:
    pre = pre_post.get("pre") if isinstance(pre_post.get("pre"), dict) else {}
    post = pre_post.get("post") if isinstance(pre_post.get("post"), dict) else {}
    delta = pre_post.get("delta") if isinstance(pre_post.get("delta"), dict) else {}
    lines = [
        "# Public Outcome Calibration",
        "",
        "Calibration maps raw wildfire score to a public-observed adverse-outcome probability proxy.",
        "This is directional and does not represent insurer claims truth.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Dataset: `{dataset_path}`",
        f"- Fitted: `{fitted}`",
    ]
    if artifact:
        lines.append(f"- Method: `{artifact.get('method')}`")
        lines.append(f"- Calibrated field semantic: `{artifact.get('calibrated_semantic_name')}`")
        guardrail_decision = artifact.get("guardrail_decision") if isinstance(artifact.get("guardrail_decision"), dict) else {}
        if guardrail_decision:
            lines.append(f"- Guardrail applied calibration: `{guardrail_decision.get('applied')}`")
    lines.extend(
        [
            "",
            "## Pre vs Post",
            f"- Pre Brier: `{pre.get('brier_probability')}`",
            f"- Post Brier: `{post.get('brier_probability')}`",
            f"- Brier improvement: `{delta.get('brier_improvement')}`",
            f"- Pre Log Loss: `{pre.get('log_loss_probability')}`",
            f"- Post Log Loss: `{post.get('log_loss_probability')}`",
            f"- Log Loss improvement: `{delta.get('log_loss_improvement')}`",
            "",
            "## Caveats",
            "- Raw deterministic wildfire scores are preserved; calibration is additive and optional.",
            "- Calibration target is public observed adverse outcome (`major_damage` or `destroyed`).",
            "- Public outcomes are incomplete and not equivalent to carrier claims labels.",
            "",
            "## Guardrails",
        ]
    )
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No additional guardrail warnings.")
    if isinstance(selection_diagnostics, dict) and selection_diagnostics:
        candidate_methods = (
            selection_diagnostics.get("candidate_methods")
            if isinstance(selection_diagnostics.get("candidate_methods"), dict)
            else {}
        )
        lines.append("")
        lines.append("## Candidate Method Comparison")
        raw_val = (
            selection_diagnostics.get("raw_validation_metrics")
            if isinstance(selection_diagnostics.get("raw_validation_metrics"), dict)
            else {}
        )
        lines.append(f"- Raw validation Brier: `{raw_val.get('brier_probability')}`")
        lines.append(f"- Validation split used: `{selection_diagnostics.get('validation_split_used')}`")
        for key in sorted(candidate_methods.keys()):
            detail = candidate_methods[key] if isinstance(candidate_methods[key], dict) else {}
            vm = detail.get("validation_metrics") if isinstance(detail.get("validation_metrics"), dict) else {}
            lines.append(
                f"- `{key}` ({detail.get('method')}): "
                f"brier={vm.get('brier_probability')}, "
                f"ece={(vm.get('calibration') or {}).get('expected_calibration_error') if isinstance(vm.get('calibration'), dict) else None}, "
                f"delta_vs_raw={detail.get('brier_delta_vs_raw')}"
            )
    return "\n".join(lines) + "\n"


def _base_artifact(*, generated_at: str, dataset_path: Path, row_count: int, positive_rate: float | None) -> dict[str, Any]:
    return {
        "artifact_version": "2.0.0",
        "generated_at": generated_at,
        "method": None,
        "calibrated_semantic_name": "calibrated_adverse_outcome_probability_public",
        "output_fields": {
            "calibrated_damage_likelihood": "backward_compatible_runtime_field",
            "empirical_damage_likelihood_proxy": "backward_compatible_runtime_field",
            "empirical_loss_likelihood_proxy": "backward_compatible_runtime_field",
        },
        "dataset": {
            "input_path": str(dataset_path),
            "row_count": row_count,
            "adverse_rate": positive_rate,
            "outcome_definition": "structure_loss_or_major_damage (major_damage or destroyed = 1)",
            "basis": "public_observed_outcomes",
        },
        "scope": {},
        "notes": [
            "Public-outcome calibration is optional and additive.",
            "Calibration does not replace the deterministic wildfire scoring engine.",
            "Calibration is based on public observed outcomes, not carrier claims truth.",
        ],
    }


def run_public_outcome_calibration(
    *,
    dataset_path: Path | None = None,
    dataset_root: Path = DEFAULT_EVALUATION_DATASET_ROOT,
    dataset_run_id: str | None = None,
    output_root: Path = DEFAULT_CALIBRATION_OUTPUT_ROOT,
    run_id: str | None = None,
    method: str = "auto",
    min_rows: int = 25,
    min_positive: int = 8,
    min_negative: int = 8,
    min_rows_for_isotonic: int = 80,
    min_unique_scores_for_isotonic: int = 25,
    min_rows_for_binned: int = 60,
    min_unique_scores_for_binned: int = 20,
    binned_bin_count: int = 8,
    max_allowed_brier_worsening: float = 0.0,
    baseline_run_id: str | None = None,
    overwrite: bool = False,
    export_artifact_path: Path | None = None,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = str(run_id) if run_id else _utc_now_iso()
    run_dir = output_root.expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset = _resolve_dataset_path(
        dataset_path=dataset_path,
        dataset_root=dataset_root.expanduser(),
        dataset_run_id=dataset_run_id,
    )
    raw_rows = _load_raw_rows(resolved_dataset)
    samples = _extract_samples(raw_rows)
    labels = [sample.label for sample in samples]
    scores = [sample.score for sample in samples]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_rate = (positive_count / float(len(labels))) if labels else None
    warnings = _quality_warnings(samples)
    fitted = False

    artifact = _base_artifact(
        generated_at=generated_at,
        dataset_path=resolved_dataset,
        row_count=len(samples),
        positive_rate=positive_rate,
    )
    pre_probs = [max(0.0, min(1.0, score / 100.0)) for score in scores]
    calibrated_probs = list(pre_probs)
    selection_diagnostics: dict[str, Any] = {}

    if len(samples) < int(min_rows) or positive_count < int(min_positive) or negative_count < int(min_negative):
        warnings.append(
            "Calibration not fit: insufficient label support "
            f"(rows={len(samples)}, positive={positive_count}, negative={negative_count})."
        )
    else:
        train_samples, validation_samples = _split_samples(samples)
        selected, method_warnings, diagnostics = _choose_method(
            train_samples=train_samples,
            validation_samples=validation_samples,
            method_preference=str(method).strip().lower(),
            min_rows_for_isotonic=int(min_rows_for_isotonic),
            min_unique_scores_for_isotonic=int(min_unique_scores_for_isotonic),
            min_rows_for_binned=int(min_rows_for_binned),
            min_unique_scores_for_binned=int(min_unique_scores_for_binned),
            binned_bin_count=int(binned_bin_count),
        )
        warnings.extend(method_warnings)
        selection_diagnostics = diagnostics

        if selected.get("method") == "logistic":
            params = selected.get("parameters") if isinstance(selected.get("parameters"), dict) else {}
            intercept = float(params.get("intercept"))
            slope = float(params.get("slope"))
            calibrated_probs = _predict_logistic(scores, intercept, slope)
            artifact["method"] = "logistic"
            artifact["parameters"] = params
        elif selected.get("method") == "bin_rate_table":
            params = selected.get("parameters") if isinstance(selected.get("parameters"), dict) else {}
            bin_table = params.get("bin_table") if isinstance(params.get("bin_table"), list) else []
            calibrated_probs = _predict_bin_rate_table(scores, bin_table)
            artifact["method"] = "bin_rate_table"
            artifact["fit_family"] = "binned_reliability"
            artifact["parameters"] = params
        else:
            params = selected.get("parameters") if isinstance(selected.get("parameters"), dict) else {}
            points = params.get("points") if isinstance(params.get("points"), list) else []
            calibrated_probs = _predict_piecewise(scores, points)
            artifact["method"] = "piecewise_linear"
            artifact["fit_family"] = "isotonic_regression"
            artifact["points"] = points

        fitted = True

    candidate_pre_post: dict[str, Any] | None = None
    if fitted:
        candidate_pre_post = _pre_post_metrics(samples, calibrated_probs)
        candidate_pre = candidate_pre_post.get("pre") if isinstance(candidate_pre_post.get("pre"), dict) else {}
        candidate_post = candidate_pre_post.get("post") if isinstance(candidate_pre_post.get("post"), dict) else {}
        candidate_pre_brier = _safe_float(candidate_pre.get("brier_probability"))
        candidate_post_brier = _safe_float(candidate_post.get("brier_probability"))
        raw_validation_metrics = (
            selection_diagnostics.get("raw_validation_metrics")
            if isinstance(selection_diagnostics.get("raw_validation_metrics"), dict)
            else {}
        ) if isinstance(selection_diagnostics, dict) else {}
        raw_validation_brier = _safe_float(raw_validation_metrics.get("brier_probability"))
        selected_key = str((selection_diagnostics.get("selection_reason") or "") if isinstance(selection_diagnostics, dict) else "")
        selected_candidate_id = str((selected.get("selected_key") if isinstance(selected, dict) else "") or "")
        selected_validation_metrics = (
            (((selection_diagnostics.get("candidate_methods") or {}).get(selected_candidate_id) or {}).get("validation_metrics"))
            if isinstance(selection_diagnostics, dict)
            else {}
        )
        selected_validation_brier = _safe_float(
            (selected_validation_metrics or {}).get("brier_probability")
            if isinstance(selected_validation_metrics, dict)
            else None
        )
        worsen_limit = float(max(0.0, max_allowed_brier_worsening))
        degrade_train = (
            candidate_pre_brier is not None
            and candidate_post_brier is not None
            and candidate_post_brier > candidate_pre_brier + worsen_limit
        )
        degrade_validation = (
            raw_validation_brier is not None
            and selected_validation_brier is not None
            and selected_validation_brier > raw_validation_brier + worsen_limit
        )
        if degrade_train or degrade_validation:
            fitted = False
            calibrated_probs = list(pre_probs)
            warnings.append(
                "Calibration skipped by guardrail because candidate worsened Brier relative to raw baseline."
            )
            artifact.pop("method", None)
            artifact.pop("parameters", None)
            artifact.pop("fit_family", None)
            artifact["guardrail_decision"] = {
                "applied": False,
                "reason": "brier_worsened_vs_raw",
                "max_allowed_brier_worsening": worsen_limit,
                "candidate_pre_brier": candidate_pre_brier,
                "candidate_post_brier": candidate_post_brier,
                "raw_validation_brier": raw_validation_brier,
                "candidate_validation_brier": selected_validation_brier,
                "selection_reason": selected_key,
            }

    pre_post = _pre_post_metrics(samples, calibrated_probs)
    pre_brier = _safe_float(((pre_post.get("pre") or {}).get("brier_probability")))
    post_brier = _safe_float(((pre_post.get("post") or {}).get("brier_probability")))
    if fitted and pre_brier is not None and post_brier is not None and post_brier > pre_brier + 0.01:
        warnings.append("Calibrated Brier score is worse than raw baseline on this dataset; review fit scope.")
    if fitted:
        artifact["metrics"] = {
            "pre": pre_post.get("pre"),
            "post": pre_post.get("post"),
            "delta": pre_post.get("delta"),
        }
    elif candidate_pre_post is not None:
        artifact["candidate_metrics_not_applied"] = {
            "pre": candidate_pre_post.get("pre"),
            "post_if_applied": candidate_pre_post.get("post"),
            "delta_if_applied": candidate_pre_post.get("delta"),
        }

    artifact["limitations"] = sorted(set(warnings))
    artifact_path = run_dir / "calibration_model.json"
    _write_json(artifact_path, artifact)

    calibration_config = {
        "artifact_path": str(artifact_path),
        "env_var": "WF_PUBLIC_CALIBRATION_ARTIFACT",
        "calibrated_semantic_name": "calibrated_adverse_outcome_probability_public",
        "target_definition": "structure_loss_or_major_damage (major_damage or destroyed = 1)",
        "runtime_backward_compatible_fields": [
            "calibrated_damage_likelihood",
            "empirical_damage_likelihood_proxy",
            "empirical_loss_likelihood_proxy",
        ],
        "calibration_optional": True,
    }
    calibration_config_path = run_dir / "calibration_config.json"
    _write_json(calibration_config_path, calibration_config)

    pre_post_metrics_path = run_dir / "pre_vs_post_metrics.json"
    _write_json(
        pre_post_metrics_path,
        {
            **pre_post,
            "selection_diagnostics": selection_diagnostics,
            "fitted": fitted,
            "warnings": sorted(set(warnings)),
        },
    )
    calibration_curve_path = run_dir / "calibration_curve.json"
    _write_json(
        calibration_curve_path,
        {
            "raw": ((pre_post.get("pre") or {}).get("calibration")),
            "calibrated": ((pre_post.get("post") or {}).get("calibration")),
            "fitted": fitted,
        },
    )
    summary_path = run_dir / "summary.md"
    summary_path.write_text(
        _build_summary_markdown(
            run_id=run_token,
            generated_at=generated_at,
            dataset_path=resolved_dataset,
            fitted=fitted,
            artifact=artifact,
            pre_post=pre_post,
            warnings=sorted(set(warnings)),
            selection_diagnostics=selection_diagnostics,
        ),
        encoding="utf-8",
    )

    listing = list_public_outcome_runs(artifact_root=output_root)
    ordered_ids = [
        str(item.get("run_id"))
        for item in (listing.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    ]
    selected_baseline_run = resolve_baseline_run_id(
        ordered_run_ids=ordered_ids,
        current_run_id=run_token,
        baseline_run_id=baseline_run_id,
    )
    baseline_manifest: dict[str, Any] | None = None
    baseline_pre_post: dict[str, Any] | None = None
    if selected_baseline_run:
        baseline_dir = output_root.expanduser() / selected_baseline_run
        baseline_manifest = _safe_load_json(baseline_dir / "manifest.json")
        baseline_pre_post = _safe_load_json(baseline_dir / "pre_vs_post_metrics.json")
    comparison_payload = build_calibration_run_comparison(
        current_run_id=run_token,
        current_manifest={
            "versions": {
                "product_version": PRODUCT_VERSION,
                "api_version": API_VERSION,
                "scoring_model_version": SCORING_MODEL_VERSION,
                "rules_logic_version": RULESET_LOGIC_VERSION,
                "factor_schema_version": FACTOR_SCHEMA_VERSION,
                "benchmark_pack_version": BENCHMARK_PACK_VERSION,
                "calibration_version": CALIBRATION_VERSION,
            },
            "inputs": {"dataset_path": str(resolved_dataset)},
        },
        current_pre_post=pre_post,
        baseline_run_id=selected_baseline_run,
        baseline_manifest=baseline_manifest,
        baseline_pre_post=baseline_pre_post,
    )
    comparison_json_path = run_dir / "comparison_to_previous.json"
    _write_json(comparison_json_path, comparison_payload)
    comparison_md_path = run_dir / "comparison_to_previous.md"
    comparison_md_path.write_text(
        build_calibration_comparison_markdown(comparison_payload),
        encoding="utf-8",
    )

    dataset_governance = _dataset_governance(resolved_dataset, raw_rows)

    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_token,
        "generated_at": generated_at,
        "fitted": fitted,
        "inputs": {
            "dataset_path": str(resolved_dataset),
            "dataset_root": str(dataset_root.expanduser()),
            "dataset_run_id": dataset_run_id,
            "method_preference": method,
            "min_rows": int(min_rows),
            "min_positive": int(min_positive),
            "min_negative": int(min_negative),
            "min_rows_for_isotonic": int(min_rows_for_isotonic),
            "min_unique_scores_for_isotonic": int(min_unique_scores_for_isotonic),
            "min_rows_for_binned": int(min_rows_for_binned),
            "min_unique_scores_for_binned": int(min_unique_scores_for_binned),
            "binned_bin_count": int(binned_bin_count),
            "max_allowed_brier_worsening": float(max_allowed_brier_worsening),
            "baseline_run_id_requested": baseline_run_id,
        },
        "versions": {
            "product_version": PRODUCT_VERSION,
            "api_version": API_VERSION,
            "scoring_model_version": SCORING_MODEL_VERSION,
            "rules_logic_version": RULESET_LOGIC_VERSION,
            "factor_schema_version": FACTOR_SCHEMA_VERSION,
            "benchmark_pack_version": BENCHMARK_PACK_VERSION,
            "calibration_version": CALIBRATION_VERSION,
        },
        "governance": {
            "model_version": SCORING_MODEL_VERSION,
            "score_logic_version": RULESET_LOGIC_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            **dataset_governance,
            "command_config": {
                "script": "scripts/fit_public_outcome_calibration.py",
                "method_preference": method,
                "min_rows": int(min_rows),
                "min_positive": int(min_positive),
                "min_negative": int(min_negative),
                "min_rows_for_isotonic": int(min_rows_for_isotonic),
                "min_unique_scores_for_isotonic": int(min_unique_scores_for_isotonic),
                "min_rows_for_binned": int(min_rows_for_binned),
                "min_unique_scores_for_binned": int(min_unique_scores_for_binned),
                "binned_bin_count": int(binned_bin_count),
                "max_allowed_brier_worsening": float(max_allowed_brier_worsening),
                "baseline_run_id": selected_baseline_run,
            },
        },
        "caveat": (
            "Calibration is based on public observed wildfire outcomes and is directional. "
            "It is not carrier claims validation."
        ),
        "artifacts": {
            "calibration_model": str(artifact_path),
            "calibration_config": str(calibration_config_path),
            "pre_vs_post_metrics": str(pre_post_metrics_path),
            "calibration_curve": str(calibration_curve_path),
            "comparison_to_previous_json": str(comparison_json_path),
            "comparison_to_previous_markdown": str(comparison_md_path),
            "summary_markdown": str(summary_path),
        },
        "warnings": sorted(set(warnings)),
        "raw_score_integrity": {
            "raw_wildfire_risk_score_preserved": True,
            "note": "Calibration is additive and optional; raw model score remains intact.",
        },
        "comparison_to_previous": {
            "available": bool(comparison_payload.get("available")),
            "baseline_run_id": comparison_payload.get("baseline_run_id"),
            "overall_direction_signals": comparison_payload.get("overall_direction_signals"),
            "likely_change_drivers": comparison_payload.get("likely_change_drivers"),
        },
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    if export_artifact_path is not None:
        export_path = export_artifact_path.expanduser()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_token,
        "run_dir": str(run_dir),
        "fitted": fitted,
        "calibration_model_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "comparison_json_path": str(comparison_json_path),
        "comparison_markdown_path": str(comparison_md_path),
        "warnings": sorted(set(warnings)),
    }


def fit_calibration(*, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    # Backward-compatible helper used by existing tests and scripts.
    result = run_public_outcome_calibration(
        dataset_path=dataset_path,
        output_root=output_path.parent / ".tmp_public_outcome_calibration_runs",
        run_id="direct_fit",
        method="auto",
        min_rows=10,
        min_positive=2,
        min_negative=2,
        min_rows_for_isotonic=30,
        min_unique_scores_for_isotonic=10,
        overwrite=True,
        export_artifact_path=output_path,
    )
    artifact = json.loads(Path(result["calibration_model_path"]).read_text(encoding="utf-8"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an optional, versioned public-outcome calibration artifact from labeled evaluation data."
        )
    )
    parser.add_argument(
        "--dataset",
        default="",
        help=(
            "Path to labeled evaluation dataset (.json/.jsonl/.csv). "
            "If omitted, the latest run in --dataset-root is used."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_EVALUATION_DATASET_ROOT),
        help="Root containing timestamped labeled evaluation dataset runs.",
    )
    parser.add_argument(
        "--dataset-run-id",
        default="",
        help="Optional dataset run id under --dataset-root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_CALIBRATION_OUTPUT_ROOT),
        help="Root for calibration bundle outputs.",
    )
    parser.add_argument("--run-id", default="", help="Optional deterministic run id.")
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "logistic", "isotonic", "binned"],
        help="Calibration method preference.",
    )
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--min-positive", type=int, default=8)
    parser.add_argument("--min-negative", type=int, default=8)
    parser.add_argument("--min-rows-for-isotonic", type=int, default=80)
    parser.add_argument("--min-unique-scores-for-isotonic", type=int, default=25)
    parser.add_argument("--min-rows-for-binned", type=int, default=60)
    parser.add_argument("--min-unique-scores-for-binned", type=int, default=20)
    parser.add_argument("--binned-bin-count", type=int, default=8)
    parser.add_argument(
        "--max-allowed-brier-worsening",
        type=float,
        default=0.0,
        help="Guardrail: skip calibration if Brier worsens beyond this amount vs raw baseline.",
    )
    parser.add_argument(
        "--baseline-run-id",
        default="",
        help="Optional baseline calibration run id to compare against. Defaults to previous run.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to copy/export calibration_model.json for runtime use.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = run_public_outcome_calibration(
        dataset_path=(Path(args.dataset).expanduser() if args.dataset else None),
        dataset_root=Path(args.dataset_root).expanduser(),
        dataset_run_id=(args.dataset_run_id or None),
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        method=args.method,
        min_rows=max(5, int(args.min_rows)),
        min_positive=max(1, int(args.min_positive)),
        min_negative=max(1, int(args.min_negative)),
        min_rows_for_isotonic=max(10, int(args.min_rows_for_isotonic)),
        min_unique_scores_for_isotonic=max(5, int(args.min_unique_scores_for_isotonic)),
        min_rows_for_binned=max(10, int(args.min_rows_for_binned)),
        min_unique_scores_for_binned=max(5, int(args.min_unique_scores_for_binned)),
        binned_bin_count=max(3, int(args.binned_bin_count)),
        max_allowed_brier_worsening=max(0.0, float(args.max_allowed_brier_worsening)),
        baseline_run_id=(args.baseline_run_id or None),
        overwrite=bool(args.overwrite),
        export_artifact_path=(Path(args.output).expanduser() if args.output else None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
