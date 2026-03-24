from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
LEAKAGE_TOKENS = (
    "outcome",
    "damage",
    "destroy",
    "loss",
    "major_damage",
    "structure_loss_or_major_damage",
)
OUTCOME_RANKS = {
    "unknown": 0,
    "no_damage": 1,
    "no_known_damage": 1,
    "minor_damage": 2,
    "major_damage": 3,
    "destroyed": 4,
}


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = _mean(values) or 0.0
    return math.sqrt(sum((v - mu) ** 2 for v in values) / float(len(values)))


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
    mean_x = _mean(x)
    mean_y = _mean(y)
    if mean_x is None or mean_y is None:
        return None
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
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


def _brier(y_true: list[int], probs: list[float]) -> float | None:
    if len(y_true) != len(probs) or not y_true:
        return None
    total = 0.0
    for truth, prob in zip(y_true, probs):
        total += (float(truth) - float(prob)) ** 2
    return total / float(len(y_true))


def _confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _precision_recall(conf: dict[str, int]) -> dict[str, float | None]:
    tp = int(conf.get("tp") or 0)
    fp = int(conf.get("fp") or 0)
    fn = int(conf.get("fn") or 0)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else None
    recall = tp / float(tp + fn) if (tp + fn) > 0 else None
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if isinstance(precision, float) and isinstance(recall, float) and (precision + recall) > 0.0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


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


def _calibration_table(
    *,
    y_true: list[int],
    probs: list[float],
    bins: int,
) -> tuple[list[dict[str, Any]], float | None]:
    if len(y_true) != len(probs) or not y_true:
        return [], None
    ranked = sorted(zip(probs, y_true), key=lambda item: item[0])
    n = len(ranked)
    table: list[dict[str, Any]] = []
    ece_total = 0.0
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
        ece_total += (count / float(n)) * abs(mean_pred - observed)
        table.append(
            {
                "bin": idx + 1,
                "count": count,
                "mean_predicted": mean_pred,
                "observed_rate": observed,
                "probability_min": min(bucket_probs),
                "probability_max": max(bucket_probs),
            }
        )
    return table, ece_total


def _normalize_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in OUTCOME_RANKS:
        return text
    if "destroy" in text:
        return "destroyed"
    if "major" in text or "severe" in text:
        return "major_damage"
    if "minor" in text or "affected" in text:
        return "minor_damage"
    if text in {"none", "undamaged", "no known damage"}:
        return "no_damage"
    return "unknown"


def _extract_score(row: dict[str, Any], key: str) -> float | None:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    if key in scores:
        return _safe_float(scores.get(key))
    return _safe_float(row.get(key))


def _extract_confidence_tier(row: dict[str, Any]) -> str:
    direct = str(row.get("confidence_tier") or "").strip().lower()
    if direct:
        return direct
    confidence = row.get("confidence") if isinstance(row.get("confidence"), dict) else {}
    tier = str(confidence.get("confidence_tier") or "").strip().lower()
    return tier or "unknown"


def _extract_evidence_tier(row: dict[str, Any]) -> str:
    direct = str(row.get("evidence_quality_tier") or "").strip().lower()
    if direct:
        return direct
    eq = row.get("evidence_quality_summary") if isinstance(row.get("evidence_quality_summary"), dict) else {}
    tier = str(eq.get("evidence_tier") or "").strip().lower()
    return tier or "unknown"


def _extract_region_id(row: dict[str, Any]) -> str:
    for candidate in (
        row.get("region_id"),
        row.get("resolved_region_id"),
        ((row.get("property_level_context") or {}).get("region_id") if isinstance(row.get("property_level_context"), dict) else None),
        ((row.get("property_level_context") or {}).get("resolved_region_id") if isinstance(row.get("property_level_context"), dict) else None),
        ((row.get("model_governance") or {}).get("region_data_version") if isinstance(row.get("model_governance"), dict) else None),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return "unknown"


def _extract_join_confidence_tier(row: dict[str, Any]) -> str:
    direct = str(row.get("join_confidence_tier") or "").strip().lower()
    if direct:
        return direct
    join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    tier = str(join_meta.get("join_confidence_tier") or "").strip().lower()
    return tier or "unknown"


def _extract_join_confidence_score(row: dict[str, Any]) -> float | None:
    direct = _safe_float(row.get("join_confidence_score"))
    if direct is not None:
        return direct
    join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
    return _safe_float(join_meta.get("join_confidence_score"))


def _extract_fallback_flags(row: dict[str, Any]) -> dict[str, float | int]:
    flags = row.get("fallback_default_flags") if isinstance(row.get("fallback_default_flags"), dict) else {}
    return {
        "fallback_factor_count": int(_safe_int(flags.get("fallback_factor_count")) or 0),
        "missing_factor_count": int(_safe_int(flags.get("missing_factor_count")) or 0),
        "inferred_factor_count": int(_safe_int(flags.get("inferred_factor_count")) or 0),
        "coverage_failed_count": int(_safe_int(flags.get("coverage_failed_count")) or 0),
        "coverage_fallback_count": int(_safe_int(flags.get("coverage_fallback_count")) or 0),
        "fallback_weight_fraction": _safe_float(flags.get("fallback_weight_fraction")) or 0.0,
    }


def _derive_evidence_group(*, evidence_tier: str, fallback_flags: dict[str, float | int]) -> str:
    if evidence_tier in {"low", "preliminary"}:
        return "fallback_heavy"
    if evidence_tier == "high":
        return "high_evidence"
    if (
        fallback_flags["coverage_failed_count"] > 0
        or fallback_flags["fallback_factor_count"] >= 2
        or fallback_flags["coverage_fallback_count"] >= 1
        or fallback_flags["missing_factor_count"] >= 3
    ):
        return "fallback_heavy"
    if (
        fallback_flags["fallback_factor_count"] == 0
        and fallback_flags["missing_factor_count"] <= 1
        and fallback_flags["coverage_failed_count"] == 0
    ):
        return "high_evidence"
    return "mixed_evidence"


def _derive_validation_confidence_tier(
    *,
    confidence_tier: str,
    evidence_group: str,
    join_confidence_tier: str,
    fallback_flags: dict[str, float | int],
    existing_row_tier: str,
) -> str:
    existing = str(existing_row_tier or "").strip().lower()
    if existing in {"high-confidence", "medium-confidence", "low-confidence"}:
        return existing
    conf = str(confidence_tier or "").strip().lower()
    evidence = str(evidence_group or "").strip().lower()
    join = str(join_confidence_tier or "").strip().lower()
    fallback_weight = float(fallback_flags.get("fallback_weight_fraction") or 0.0)
    if conf in {"low", "preliminary", "unknown"} or join == "low" or evidence == "fallback_heavy" or fallback_weight >= 0.5:
        return "low-confidence"
    if conf in {"high", "moderate"} and join == "high" and evidence == "high_evidence":
        return "high-confidence"
    return "medium-confidence"


def _compute_subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "positive_rate": None,
            "wildfire_risk_score_auc": None,
            "wildfire_risk_score_pr_auc": None,
            "wildfire_risk_score_brier": None,
        }
    y_true = [int(row["structure_loss_or_major_damage"]) for row in rows]
    probs = [max(0.0, min(1.0, float(row["wildfire_risk_score"]) / 100.0)) for row in rows]
    return {
        "count": len(rows),
        "positive_rate": _mean([float(v) for v in y_true]),
        "wildfire_risk_score_auc": _roc_auc(y_true, probs),
        "wildfire_risk_score_pr_auc": _pr_auc(y_true, probs),
        "wildfire_risk_score_brier": _brier(y_true, probs),
    }


def _top_factor_contributions(row: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    factors = row.get("factor_contribution_breakdown")
    if not isinstance(factors, dict):
        return []
    rows: list[dict[str, Any]] = []
    for factor, detail in factors.items():
        if not isinstance(detail, dict):
            continue
        contrib = _safe_float(detail.get("contribution"))
        if contrib is None:
            continue
        rows.append({"factor": str(factor), "contribution": contrib})
    rows.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return rows[: max(1, int(limit))]


def _factor_summary_by_confusion_class(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = str(row.get("_confusion_class") or "unknown")
        factors = row.get("factor_contribution_breakdown")
        if not isinstance(factors, dict):
            continue
        for factor, detail in factors.items():
            if not isinstance(detail, dict):
                continue
            contrib = _safe_float(detail.get("contribution"))
            if contrib is None:
                continue
            grouped[label][str(factor)].append(contrib)

    out: dict[str, Any] = {}
    for label, factor_map in grouped.items():
        rows_out = []
        for factor, values in factor_map.items():
            rows_out.append(
                {
                    "factor": factor,
                    "mean_contribution": _mean(values),
                    "stddev_contribution": _stddev(values),
                    "count": len(values),
                }
            )
        rows_out.sort(key=lambda item: abs(float(item.get("mean_contribution") or 0.0)), reverse=True)
        out[label] = rows_out[:12]
    return out


def _distribution_by_outcome(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = _extract_score(row, score_key)
        if score is None:
            continue
        grouped[str(row.get("outcome_label") or "unknown")].append(score)
    out: dict[str, Any] = {}
    for label, values in grouped.items():
        out[label] = {
            "count": len(values),
            "mean": _mean(values),
            "stddev": _stddev(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return out


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


def _flatten_joined_labeled_row(row: dict[str, Any]) -> dict[str, Any]:
    # Accept rows from scripts/build_public_outcome_evaluation_dataset.py.
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
    coverage_summary = evidence.get("coverage_summary") if isinstance(evidence.get("coverage_summary"), dict) else {}
    feature_snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else {}
    join_meta = row.get("join_metadata") if isinstance(row.get("join_metadata"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    governance = provenance.get("model_governance") if isinstance(provenance.get("model_governance"), dict) else {}

    outcome_label = _normalize_damage_class(
        outcome.get("damage_label") or outcome.get("damage_severity_class")
    )
    target = _safe_int(outcome.get("structure_loss_or_major_damage"))
    if target not in {0, 1}:
        target = 1 if outcome_label in {"major_damage", "destroyed"} else (0 if outcome_label in {"minor_damage", "no_damage"} else None)

    fallback_default_flags = {
        "fallback_factor_count": int(_safe_int(evidence_summary.get("fallback_factor_count")) or 0),
        "missing_factor_count": int(_safe_int(evidence_summary.get("missing_factor_count")) or 0),
        "inferred_factor_count": int(_safe_int(evidence_summary.get("inferred_factor_count")) or 0),
        "coverage_failed_count": int(_safe_int(coverage_summary.get("failed_count")) or 0),
        "coverage_fallback_count": int(_safe_int(coverage_summary.get("fallback_count")) or 0),
        "fallback_weight_fraction": _safe_float(evidence_summary.get("fallback_weight_fraction")) or 0.0,
    }

    return {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "record_id": feature.get("record_id") or outcome.get("record_id"),
        "source_record_id": feature.get("source_record_id") or outcome.get("source_record_id"),
        "address_text": feature.get("address_text") or outcome.get("address_text"),
        "latitude": _safe_float(feature.get("latitude") or outcome.get("latitude")),
        "longitude": _safe_float(feature.get("longitude") or outcome.get("longitude")),
        "region_id": governance.get("region_data_version"),
        "outcome_label": outcome_label,
        "outcome_rank": OUTCOME_RANKS.get(outcome_label, 0),
        "structure_loss_or_major_damage": target,
        "scores": {
            "wildfire_risk_score": _safe_float(scores.get("wildfire_risk_score")),
            "site_hazard_score": _safe_float(scores.get("site_hazard_score")),
            "home_ignition_vulnerability_score": _safe_float(scores.get("home_ignition_vulnerability_score")),
            "insurance_readiness_score": _safe_float(scores.get("insurance_readiness_score")),
            "calibrated_damage_likelihood": _safe_float(scores.get("calibrated_damage_likelihood")),
        },
        "confidence_tier": str(confidence.get("confidence_tier") or "unknown").strip().lower(),
        "confidence_score": _safe_float(confidence.get("confidence_score")),
        "evidence_quality_tier": str(evidence.get("evidence_quality_tier") or "unknown").strip().lower(),
        "evidence_quality_summary": evidence_summary,
        "coverage_summary": coverage_summary,
        "fallback_default_flags": fallback_default_flags,
        "raw_feature_vector": (
            feature_snapshot.get("raw_feature_vector")
            if isinstance(feature_snapshot.get("raw_feature_vector"), dict)
            else {}
        ),
        "transformed_feature_vector": (
            feature_snapshot.get("transformed_feature_vector")
            if isinstance(feature_snapshot.get("transformed_feature_vector"), dict)
            else {}
        ),
        "factor_contribution_breakdown": (
            feature_snapshot.get("factor_contribution_breakdown")
            if isinstance(feature_snapshot.get("factor_contribution_breakdown"), dict)
            else {}
        ),
        "compression_flags": (
            feature_snapshot.get("compression_flags")
            if isinstance(feature_snapshot.get("compression_flags"), list)
            else []
        ),
        "join_metadata": join_meta,
        "join_method": join_meta.get("join_method"),
        "join_confidence_tier": join_meta.get("join_confidence_tier"),
        "join_confidence_score": join_meta.get("join_confidence_score"),
        "row_confidence_tier": evaluation.get("row_confidence_tier"),
        "caveat_flags": row.get("caveat_flags") if isinstance(row.get("caveat_flags"), list) else [],
        "leakage_flags": row.get("leakage_flags") if isinstance(row.get("leakage_flags"), list) else [],
        "model_governance": governance,
    }


def _load_rows_from_dataset_file(dataset_path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    continue
                rows.append(
                    _flatten_joined_labeled_row(payload)
                    if ("outcome" in payload and "scores" in payload)
                    else payload
                )
        return rows, "jsonl"

    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [dict(row) for row in reader]
        return rows, "csv"

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [row for row in payload["rows"] if isinstance(row, dict)], "json_rows"
        if isinstance(payload.get("records"), list):
            rows = [row for row in payload["records"] if isinstance(row, dict)]
            if rows and isinstance(rows[0], dict) and ("outcome" in rows[0] and "scores" in rows[0]):
                return [_flatten_joined_labeled_row(row) for row in rows], "json_records_joined"
            return rows, "json_records"
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        if rows and ("outcome" in rows[0] and "scores" in rows[0]):
            return [_flatten_joined_labeled_row(row) for row in rows], "json_list_joined"
        return rows, "json_list"
    raise ValueError(
        "Unsupported dataset format. Expected JSON with rows/records, JSONL, or CSV."
    )


def _slice_metrics(
    rows: list[dict[str, Any]],
    *,
    slice_key: str,
    min_slice_size: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(slice_key) or "unknown")].append(row)
    out: dict[str, Any] = {}
    for group_name in sorted(groups.keys()):
        bucket = groups[group_name]
        y_true = [int(row["structure_loss_or_major_damage"]) for row in bucket]
        probs = [max(0.0, min(1.0, float(row["wildfire_risk_score"]) / 100.0)) for row in bucket]
        auc = _roc_auc(y_true, probs)
        out[group_name] = {
            "count": len(bucket),
            "positive_rate": _mean([float(v) for v in y_true]),
            "wildfire_risk_score_auc": auc,
            "wildfire_risk_score_pr_auc": _pr_auc(y_true, probs),
            "wildfire_risk_score_brier": _brier(y_true, probs),
            "wildfire_risk_score_mean": _mean([float(row["wildfire_risk_score"]) for row in bucket]),
            "wildfire_risk_score_stddev": _stddev([float(row["wildfire_risk_score"]) for row in bucket]),
            "fallback_heavy_rate": _mean([1.0 if str(row.get("evidence_group")) == "fallback_heavy" else 0.0 for row in bucket]),
            "small_sample_warning": (len(bucket) < max(1, int(min_slice_size))),
        }
    return out


def _detect_data_leakage_risks(rows: list[dict[str, Any]], wildfire_auc: float | None) -> dict[str, Any]:
    suspicious_feature_keys: Counter[str] = Counter()
    conflicting_labels: list[str] = []
    dedupe: dict[tuple[str, str], int] = {}

    for row in rows:
        for container_key in ("raw_feature_vector", "transformed_feature_vector", "factor_contribution_breakdown"):
            container = row.get(container_key)
            if not isinstance(container, dict):
                continue
            for key in container.keys():
                text = str(key).strip().lower()
                if any(token in text for token in LEAKAGE_TOKENS):
                    suspicious_feature_keys[text] += 1

        pair = (str(row.get("event_id") or ""), str(row.get("record_id") or ""))
        label = int(row.get("structure_loss_or_major_damage") or 0)
        if pair in dedupe and dedupe[pair] != label:
            conflicting_labels.append(f"{pair[0]}::{pair[1]}")
        else:
            dedupe[pair] = label

    warnings: list[str] = []
    if suspicious_feature_keys:
        warnings.append("Feature vectors contain outcome-like tokens; verify no label leakage into model inputs.")
    if conflicting_labels:
        warnings.append("Same event_id/record_id appears with conflicting labels in the joined dataset.")
    if wildfire_auc is not None and wildfire_auc > 0.98 and len(rows) < 80:
        warnings.append("AUC is near-perfect on a small sample; verify split strategy and leakage controls.")

    return {
        "warnings": warnings,
        "suspicious_feature_keys": [
            {"key": key, "count": count}
            for key, count in suspicious_feature_keys.most_common(25)
        ],
        "conflicting_event_record_labels": sorted(set(conflicting_labels))[:50],
    }


def _build_guardrail_warnings(
    *,
    row_count: int,
    positive_rate: float | None,
    fallback_heavy_fraction: float,
    leak_warnings: list[str],
) -> list[str]:
    warnings: list[str] = []
    if row_count < 25:
        warnings.append("Very small labeled sample (<25); discrimination and calibration estimates are unstable.")
    elif row_count < 100:
        warnings.append("Small labeled sample (<100); treat calibration fit decisions as provisional.")
    if positive_rate is not None and (positive_rate < 0.05 or positive_rate > 0.95):
        warnings.append("Class balance is highly skewed; threshold and calibration estimates may be unstable.")
    if fallback_heavy_fraction >= 0.5:
        warnings.append("Fallback-heavy rows dominate this dataset; avoid fitting production calibration without better coverage.")
    warnings.extend(leak_warnings)
    return warnings


def _false_review_sets(
    *,
    rows: list[dict[str, Any]],
    false_low_max_score: float,
    false_high_min_score: float,
) -> dict[str, Any]:
    false_low: list[dict[str, Any]] = []
    false_high: list[dict[str, Any]] = []
    unstable_positive: list[dict[str, Any]] = []
    low_confidence_positive: list[dict[str, Any]] = []
    low_factor_counter: Counter[str] = Counter()
    high_factor_counter: Counter[str] = Counter()

    for row in rows:
        score = float(row["wildfire_risk_score"])
        label = int(row["structure_loss_or_major_damage"])
        caveat_flags = row.get("caveat_flags") if isinstance(row.get("caveat_flags"), list) else []
        join_tier = str(row.get("join_confidence_tier") or "unknown")
        review_entry = {
            "event_id": row.get("event_id"),
            "record_id": row.get("record_id"),
            "region_id": row.get("region_id"),
            "outcome_label": row.get("outcome_label"),
            "wildfire_risk_score": score,
            "confidence_tier": row.get("confidence_tier"),
            "evidence_group": row.get("evidence_group"),
            "join_confidence_tier": join_tier,
            "fallback_default_flags": row.get("fallback_default_flags"),
            "caveat_flags": caveat_flags,
            "top_factor_contributions": _top_factor_contributions(row, limit=5),
        }
        if label == 1 and score < false_low_max_score:
            false_low.append(review_entry)
            for factor in review_entry["top_factor_contributions"]:
                low_factor_counter[str(factor.get("factor"))] += 1
        if label == 0 and score >= false_high_min_score:
            false_high.append(review_entry)
            for factor in review_entry["top_factor_contributions"]:
                high_factor_counter[str(factor.get("factor"))] += 1
        if label == 1 and (
            join_tier == "low"
            or "low_confidence_join" in caveat_flags
            or "high_join_distance" in caveat_flags
            or str(row.get("evidence_group")) == "fallback_heavy"
        ):
            unstable_positive.append(review_entry)
        if label == 1 and str(row.get("confidence_tier") or "unknown") in {"low", "unknown"}:
            low_confidence_positive.append(review_entry)

    false_low.sort(key=lambda row: float(row.get("wildfire_risk_score") or 0.0))
    false_high.sort(key=lambda row: float(row.get("wildfire_risk_score") or 0.0), reverse=True)
    unstable_positive.sort(key=lambda row: float(row.get("wildfire_risk_score") or 0.0))
    low_confidence_positive.sort(key=lambda row: float(row.get("wildfire_risk_score") or 0.0))
    return {
        "false_low_max_score": float(false_low_max_score),
        "false_high_min_score": float(false_high_min_score),
        "false_low_count": len(false_low),
        "false_high_count": len(false_high),
        "unstable_positive_count": len(unstable_positive),
        "low_confidence_positive_count": len(low_confidence_positive),
        "false_low_examples": false_low[:100],
        "false_high_examples": false_high[:100],
        "unstable_positive_examples": unstable_positive[:100],
        "low_confidence_positive_examples": low_confidence_positive[:100],
        "common_top_factors_false_low": low_factor_counter.most_common(12),
        "common_top_factors_false_high": high_factor_counter.most_common(12),
    }


def _prepare_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    missing_required = Counter()
    invalid_examples: list[dict[str, Any]] = []

    for row in rows:
        target_int = _safe_int(row.get("structure_loss_or_major_damage"))
        if target_int not in {0, 1}:
            missing_required["structure_loss_or_major_damage"] += 1
            if len(invalid_examples) < 12:
                invalid_examples.append(
                    {"record_id": row.get("record_id"), "reason": "missing_or_invalid_structure_loss_or_major_damage"}
                )
            continue
        wildfire_score = _extract_score(row, "wildfire_risk_score")
        if wildfire_score is None:
            missing_required["scores.wildfire_risk_score"] += 1
            if len(invalid_examples) < 12:
                invalid_examples.append({"record_id": row.get("record_id"), "reason": "missing_scores.wildfire_risk_score"})
            continue

        label = _normalize_label(row.get("outcome_label"))
        outcome_rank = _safe_int(row.get("outcome_rank"))
        if outcome_rank is None:
            outcome_rank = OUTCOME_RANKS.get(label, 0)

        fallback_flags = _extract_fallback_flags(row)
        evidence_tier = _extract_evidence_tier(row)
        confidence_tier = _extract_confidence_tier(row)
        join_tier = _extract_join_confidence_tier(row)
        join_score = _extract_join_confidence_score(row)
        evidence_group = _derive_evidence_group(
            evidence_tier=evidence_tier,
            fallback_flags=fallback_flags,
        )
        prepared_row = dict(row)
        prepared_row["outcome_label"] = label
        prepared_row["outcome_rank"] = int(outcome_rank)
        prepared_row["structure_loss_or_major_damage"] = int(target_int)
        prepared_row["wildfire_risk_score"] = float(wildfire_score)
        prepared_row["wildfire_probability_proxy"] = max(0.0, min(1.0, float(wildfire_score) / 100.0))
        prepared_row["calibrated_damage_likelihood"] = _extract_score(row, "calibrated_damage_likelihood")
        prepared_row["confidence_tier"] = confidence_tier
        prepared_row["evidence_quality_tier"] = evidence_tier
        prepared_row["evidence_group"] = evidence_group
        prepared_row["region_id"] = _extract_region_id(row)
        prepared_row["fallback_default_flags"] = fallback_flags
        prepared_row["join_confidence_tier"] = join_tier
        prepared_row["join_confidence_score"] = join_score
        prepared_row["join_method"] = (
            str(row.get("join_method") or "").strip()
            or str(((row.get("join_metadata") or {}).get("join_method") if isinstance(row.get("join_metadata"), dict) else "")).strip()
            or "unknown"
        )
        prepared_row["validation_confidence_tier"] = _derive_validation_confidence_tier(
            confidence_tier=confidence_tier,
            evidence_group=evidence_group,
            join_confidence_tier=join_tier,
            fallback_flags=fallback_flags,
            existing_row_tier=str(row.get("row_confidence_tier") or ""),
        )
        prepared_row["fallback_status"] = "fallback_heavy" if evidence_group == "fallback_heavy" else "not_fallback_heavy"
        prepared.append(prepared_row)

    return prepared, {
        "missing_required_fields": dict(missing_required),
        "invalid_row_examples": invalid_examples,
    }


def evaluate_public_outcome_dataset_rows(
    *,
    rows: list[dict[str, Any]],
    thresholds: list[float] | None = None,
    bins: int = 10,
    min_slice_size: int = 20,
    false_low_max_score: float = 40.0,
    false_high_min_score: float = 70.0,
    min_labeled_rows: int = 1,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_rows, validation = _prepare_rows(rows)
    if len(prepared_rows) < max(1, int(min_labeled_rows)):
        missing = validation.get("missing_required_fields") or {}
        raise ValueError(
            "Not enough usable labeled rows for evaluation. "
            f"usable_rows={len(prepared_rows)} min_labeled_rows={max(1, int(min_labeled_rows))} "
            f"missing_required_counts={missing}"
        )

    y_true = [int(row["structure_loss_or_major_damage"]) for row in prepared_rows]
    raw_probs = [float(row["wildfire_probability_proxy"]) for row in prepared_rows]
    raw_scores = [float(row["wildfire_risk_score"]) for row in prepared_rows]
    outcome_ranks = [float(row["outcome_rank"]) for row in prepared_rows]
    positive_rate = _mean([float(v) for v in y_true])

    thresholds_use = [float(v) for v in (thresholds or list(DEFAULT_THRESHOLDS))]
    threshold_metrics: dict[str, Any] = {}
    for threshold in thresholds_use:
        preds = [1 if score >= threshold else 0 for score in raw_scores]
        conf = _confusion(y_true, preds)
        threshold_metrics[str(int(round(threshold)))] = {
            "threshold": float(threshold),
            "confusion_matrix": conf,
            **_precision_recall(conf),
        }
    default_threshold = 70.0
    default_preds = [1 if score >= default_threshold else 0 for score in raw_scores]
    default_conf = _confusion(y_true, default_preds)

    for row, pred, truth in zip(prepared_rows, default_preds, y_true):
        if truth == 1 and pred == 1:
            row["_confusion_class"] = "tp"
        elif truth == 0 and pred == 1:
            row["_confusion_class"] = "fp"
        elif truth == 1 and pred == 0:
            row["_confusion_class"] = "fn"
        else:
            row["_confusion_class"] = "tn"

    calibrated_pairs = [
        (int(row["structure_loss_or_major_damage"]), float(row["calibrated_damage_likelihood"]))
        for row in prepared_rows
        if row.get("calibrated_damage_likelihood") is not None
    ]
    calibrated_y = [pair[0] for pair in calibrated_pairs]
    calibrated_probs = [pair[1] for pair in calibrated_pairs]

    raw_bins, raw_ece = _calibration_table(y_true=y_true, probs=raw_probs, bins=bins)
    calibrated_bins, calibrated_ece = _calibration_table(
        y_true=calibrated_y,
        probs=calibrated_probs,
        bins=bins,
    )

    by_event: dict[str, Any] = {}
    by_region: dict[str, Any] = {}
    by_join_confidence_tier: dict[str, Any] = {}
    by_validation_confidence_tier: dict[str, Any] = {}
    event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    region_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    join_tier_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation_tier_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared_rows:
        event_groups[str(row.get("event_id") or "unknown")].append(row)
        region_groups[str(row.get("region_id") or "unknown")].append(row)
        join_tier_groups[str(row.get("join_confidence_tier") or "unknown")].append(row)
        validation_tier_groups[str(row.get("validation_confidence_tier") or "unknown")].append(row)
    for key in sorted(event_groups.keys()):
        bucket = event_groups[key]
        by_event[key] = {
            "count": len(bucket),
            "positive_rate": _mean([float(row["structure_loss_or_major_damage"]) for row in bucket]),
        }
    for key in sorted(region_groups.keys()):
        bucket = region_groups[key]
        by_region[key] = {
            "count": len(bucket),
            "positive_rate": _mean([float(row["structure_loss_or_major_damage"]) for row in bucket]),
        }
    for key in sorted(join_tier_groups.keys()):
        bucket = join_tier_groups[key]
        by_join_confidence_tier[key] = {
            "count": len(bucket),
            "positive_rate": _mean([float(row["structure_loss_or_major_damage"]) for row in bucket]),
        }
    for key in sorted(validation_tier_groups.keys()):
        bucket = validation_tier_groups[key]
        by_validation_confidence_tier[key] = {
            "count": len(bucket),
            "positive_rate": _mean([float(row["structure_loss_or_major_damage"]) for row in bucket]),
        }

    fallback_heavy_count = sum(1 for row in prepared_rows if str(row.get("evidence_group")) == "fallback_heavy")
    fallback_heavy_fraction = fallback_heavy_count / float(len(prepared_rows))
    fallback_stats = {
        "rows_with_any_fallback_flag": sum(
            1
            for row in prepared_rows
            if int((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0) > 0
            or int((row.get("fallback_default_flags") or {}).get("coverage_fallback_count") or 0) > 0
        ),
        "mean_fallback_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0) for row in prepared_rows]
        ),
        "mean_missing_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("missing_factor_count") or 0) for row in prepared_rows]
        ),
        "fallback_heavy_count": fallback_heavy_count,
        "fallback_heavy_fraction": fallback_heavy_fraction,
    }

    raw_auc = _roc_auc(y_true, raw_probs)
    leak_risks = _detect_data_leakage_risks(prepared_rows, raw_auc)
    guardrail_warnings = _build_guardrail_warnings(
        row_count=len(prepared_rows),
        positive_rate=positive_rate,
        fallback_heavy_fraction=fallback_heavy_fraction,
        leak_warnings=list(leak_risks.get("warnings") or []),
    )
    review_sets = _false_review_sets(
        rows=prepared_rows,
        false_low_max_score=float(false_low_max_score),
        false_high_min_score=float(false_high_min_score),
    )
    high_confidence_rows = [row for row in prepared_rows if str(row.get("validation_confidence_tier")) == "high-confidence"]
    high_evidence_rows = [row for row in prepared_rows if str(row.get("evidence_group")) == "high_evidence"]
    if not high_confidence_rows:
        guardrail_warnings.append("No high-confidence rows in this run; prioritize stronger join/feature coverage before calibration decisions.")
    if not high_evidence_rows:
        guardrail_warnings.append("No high-evidence rows in this run; results are dominated by inferred/fallback-heavy evidence.")

    directional_predictive_value = bool(raw_auc is not None and raw_auc >= 0.6)
    calibration_recommendation = (
        "candidate"
        if directional_predictive_value
        and raw_ece is not None
        and raw_ece <= 0.12
        and fallback_heavy_fraction < 0.5
        and len(prepared_rows) >= 100
        else "not_recommended_yet"
    )

    report = {
        "schema_version": "1.1.0",
        "generated_at": generated_at or _utc_now_iso(),
        "row_count_labeled": len(prepared_rows),
        "sample_counts": {
            "row_count_total": len(rows),
            "row_count_usable": len(prepared_rows),
            "positive_count": int(sum(y_true)),
            "negative_count": int(len(y_true) - sum(y_true)),
            "positive_rate": positive_rate,
            "fallback_heavy_count": fallback_heavy_count,
            "fallback_heavy_fraction": fallback_heavy_fraction,
            "by_event": by_event,
            "by_region": by_region,
            "by_join_confidence_tier": by_join_confidence_tier,
            "by_validation_confidence_tier": by_validation_confidence_tier,
            "validation_exclusions": validation,
        },
        "discrimination_metrics": {
            "wildfire_risk_score_auc": raw_auc,
            "wildfire_risk_score_pr_auc": _pr_auc(y_true, raw_probs),
            "site_hazard_score_auc": _roc_auc(
                y_true,
                [max(0.0, min(1.0, (_extract_score(row, "site_hazard_score") or 0.0) / 100.0)) for row in prepared_rows],
            ),
            "home_ignition_vulnerability_score_auc": _roc_auc(
                y_true,
                [
                    max(
                        0.0,
                        min(1.0, (_extract_score(row, "home_ignition_vulnerability_score") or 0.0) / 100.0),
                    )
                    for row in prepared_rows
                ],
            ),
            "calibrated_damage_likelihood_auc": (
                _roc_auc(calibrated_y, calibrated_probs) if calibrated_y and calibrated_probs else None
            ),
            "wildfire_vs_outcome_rank_spearman": _spearman(raw_scores, outcome_ranks),
        },
        "threshold_metrics_wildfire_risk_score": threshold_metrics,
        "default_threshold_70": {
            "confusion_matrix": default_conf,
            **_precision_recall(default_conf),
        },
        "brier_scores": {
            "wildfire_probability_proxy": _brier(y_true, raw_probs),
            "calibrated_damage_likelihood": (
                _brier(calibrated_y, calibrated_probs) if calibrated_y and calibrated_probs else None
            ),
        },
        "score_distributions_by_outcome": {
            "wildfire_risk_score": _distribution_by_outcome(prepared_rows, "wildfire_risk_score"),
            "site_hazard_score": _distribution_by_outcome(prepared_rows, "site_hazard_score"),
            "home_ignition_vulnerability_score": _distribution_by_outcome(
                prepared_rows, "home_ignition_vulnerability_score"
            ),
            "insurance_readiness_score": _distribution_by_outcome(prepared_rows, "insurance_readiness_score"),
        },
        "calibration_metrics": {
            "wildfire_risk_score": {
                "bins": raw_bins,
                "expected_calibration_error": raw_ece,
            },
            "calibrated_damage_likelihood": {
                "bins": calibrated_bins,
                "expected_calibration_error": calibrated_ece,
                "row_count": len(calibrated_pairs),
            },
        },
        "calibration_table_wildfire_risk_score": raw_bins,
        "slice_metrics": {
            "by_confidence_tier": _slice_metrics(
                prepared_rows,
                slice_key="confidence_tier",
                min_slice_size=min_slice_size,
            ),
            "by_evidence_quality_tier": _slice_metrics(
                prepared_rows,
                slice_key="evidence_quality_tier",
                min_slice_size=min_slice_size,
            ),
            "by_evidence_group": _slice_metrics(
                prepared_rows,
                slice_key="evidence_group",
                min_slice_size=min_slice_size,
            ),
            "by_join_confidence_tier": _slice_metrics(
                prepared_rows,
                slice_key="join_confidence_tier",
                min_slice_size=min_slice_size,
            ),
            "by_validation_confidence_tier": _slice_metrics(
                prepared_rows,
                slice_key="validation_confidence_tier",
                min_slice_size=min_slice_size,
            ),
            "by_fallback_status": _slice_metrics(
                prepared_rows,
                slice_key="fallback_status",
                min_slice_size=min_slice_size,
            ),
        },
        "subset_metrics": {
            "full_dataset": _compute_subset_metrics(prepared_rows),
            "high_confidence_subset": _compute_subset_metrics(high_confidence_rows),
            "high_evidence_subset": _compute_subset_metrics(high_evidence_rows),
        },
        "fallback_diagnostics": fallback_stats,
        "factor_contribution_summary_by_confusion_class": _factor_summary_by_confusion_class(prepared_rows),
        "false_review_sets": review_sets,
        "data_leakage_risks": leak_risks,
        "guardrails": {
            "warnings": guardrail_warnings,
            "small_sample_warning": any("sample" in w.lower() for w in guardrail_warnings),
            "fallback_heavy_warning": fallback_heavy_fraction >= 0.5,
            "leakage_warning": bool(leak_risks.get("warnings")),
        },
        "directional_predictive_value": directional_predictive_value,
        "calibration_artifact_recommendation": calibration_recommendation,
    }
    return report, prepared_rows


def evaluate_public_outcome_dataset_file(
    *,
    dataset_path: Path,
    thresholds: list[float] | None = None,
    bins: int = 10,
    min_slice_size: int = 20,
    false_low_max_score: float = 40.0,
    false_high_min_score: float = 70.0,
    min_labeled_rows: int = 1,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_rows, dataset_format = _load_rows_from_dataset_file(dataset_path)
    report, eval_rows = evaluate_public_outcome_dataset_rows(
        rows=clean_rows,
        thresholds=thresholds,
        bins=bins,
        min_slice_size=min_slice_size,
        false_low_max_score=false_low_max_score,
        false_high_min_score=false_high_min_score,
        min_labeled_rows=min_labeled_rows,
        generated_at=generated_at,
    )
    report["dataset_path"] = str(dataset_path)
    report["dataset_format"] = dataset_format
    if dataset_path.suffix.lower() == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            report["dataset_schema_version"] = payload.get("schema_version")
    return report, eval_rows


def trace_public_outcome_dataset_flow(
    *,
    dataset_path: Path,
) -> dict[str, Any]:
    rows, dataset_format = _load_rows_from_dataset_file(dataset_path)
    prepared_rows, validation = _prepare_rows(rows)
    missing = validation.get("missing_required_fields") if isinstance(validation, dict) else {}
    invalid = validation.get("invalid_row_examples") if isinstance(validation, dict) else []
    return {
        "dataset_path": str(dataset_path),
        "dataset_format": dataset_format,
        "loaded_rows": len(rows),
        "prepared_rows": len(prepared_rows),
        "dropped_rows": max(0, len(rows) - len(prepared_rows)),
        "missing_required_fields": (missing if isinstance(missing, dict) else {}),
        "invalid_row_examples": (invalid if isinstance(invalid, list) else []),
    }


def write_evaluation_rows_csv(*, rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "event_id",
                "record_id",
                "region_id",
                "outcome_label",
                "structure_loss_or_major_damage",
                "wildfire_risk_score",
                "wildfire_probability_proxy",
                "calibrated_damage_likelihood",
                "confidence_tier",
                "evidence_quality_tier",
                "evidence_group",
                "join_confidence_tier",
                "join_confidence_score",
                "validation_confidence_tier",
                "join_method",
                "confusion_class_default_threshold_70",
                "fallback_factor_count",
                "missing_factor_count",
                "inferred_factor_count",
                "coverage_failed_count",
                "coverage_fallback_count",
                "fallback_weight_fraction",
            ],
        )
        writer.writeheader()
        for row in rows:
            flags = row.get("fallback_default_flags") if isinstance(row.get("fallback_default_flags"), dict) else {}
            writer.writerow(
                {
                    "event_id": row.get("event_id"),
                    "record_id": row.get("record_id"),
                    "region_id": row.get("region_id"),
                    "outcome_label": row.get("outcome_label"),
                    "structure_loss_or_major_damage": row.get("structure_loss_or_major_damage"),
                    "wildfire_risk_score": row.get("wildfire_risk_score"),
                    "wildfire_probability_proxy": row.get("wildfire_probability_proxy"),
                    "calibrated_damage_likelihood": row.get("calibrated_damage_likelihood"),
                    "confidence_tier": row.get("confidence_tier"),
                    "evidence_quality_tier": row.get("evidence_quality_tier"),
                    "evidence_group": row.get("evidence_group"),
                    "join_confidence_tier": row.get("join_confidence_tier"),
                    "join_confidence_score": row.get("join_confidence_score"),
                    "validation_confidence_tier": row.get("validation_confidence_tier"),
                    "join_method": row.get("join_method"),
                    "confusion_class_default_threshold_70": row.get("_confusion_class"),
                    "fallback_factor_count": flags.get("fallback_factor_count"),
                    "missing_factor_count": flags.get("missing_factor_count"),
                    "inferred_factor_count": flags.get("inferred_factor_count"),
                    "coverage_failed_count": flags.get("coverage_failed_count"),
                    "coverage_fallback_count": flags.get("coverage_fallback_count"),
                    "fallback_weight_fraction": flags.get("fallback_weight_fraction"),
                }
            )
