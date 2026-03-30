from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
SMALL_SAMPLE_WARNING_N = 25
STABLE_AUC_MIN_CLASS_COUNT = 10
DATA_SUFFICIENCY_LIMITED_MIN = 20
DATA_SUFFICIENCY_MODERATE_MIN = 100
DATA_SUFFICIENCY_STRONG_MIN = 500
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
PROXY_RISK_UP_FEATURE_KEYS = (
    "burn_probability",
    "wildfire_hazard",
    "fuel_model",
    "canopy_cover",
    "slope",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
    "near_structure_connectivity_index",
    "near_structure_vegetation_0_5_pct",
    "ring_0_5_ft_vegetation_density",
    "ring_5_30_ft_vegetation_density",
    "structure_density",
    "clustering_index",
    "building_age_material_proxy_risk",
)
PROXY_RISK_DOWN_FEATURE_KEYS = (
    "historic_fire_distance_km",
    "wildland_distance_m",
    "nearest_high_fuel_patch_distance_ft",
    "nearest_vegetation_distance_ft",
    "distance_to_nearest_structure_ft",
    "building_age_proxy_year",
)
FEATURE_DIRECTION_OVERRIDES = {
    # This is a proximity index (higher means closer fuel), so risk increases as the value rises.
    "wildland_distance_index": "risk_up",
}
AUTO_DIRECTION_ALIGNMENT_MIN_SIGNAL_SCORE = 0.10
AUTO_DIRECTION_ALIGNMENT_MIN_ROWS = 6
HAZARD_SEGMENT_THRESHOLDS = (35.0, 55.0, 75.0)
VEGETATION_SEGMENT_THRESHOLDS = (25.0, 50.0, 75.0)
VEGETATION_SEGMENT_FEATURE_KEYS = (
    "near_structure_vegetation_0_5_pct",
    "ring_0_5_ft_vegetation_density",
    "ring_5_30_ft_vegetation_density",
    "near_structure_connectivity_index",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
    "canopy_cover",
)
FALLBACK_HEAVY_WEIGHT_THRESHOLD = 0.65
FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD = 0.45
FALLBACK_HEAVY_FACTOR_RATIO_THRESHOLD = 0.60
FALLBACK_HEAVY_MISSING_RATIO_THRESHOLD = 0.50
FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD = 2
VIABILITY_MIN_INDEPENDENT_SAMPLES = 30
VIABILITY_MIN_FEATURES_WITH_VARIANCE = 5
VIABILITY_MIN_FEATURE_VARIATION_RATIO = 0.25
VIABILITY_MIN_AUC_MARGIN_VS_RANDOM = 0.05
HIGH_SIGNAL_MODEL_SIGNAL_WEIGHTS: dict[str, float] = {
    # Weights are proportional to observed directional signal from recent
    # feature-signal diagnostics and then renormalized on available features.
    "nearest_high_fuel_patch_distance_ft": 0.3819357558,
    "canopy_adjacency_proxy_pct": 0.3753699792,
    "vegetation_continuity_proxy_pct": 0.2908944683,
    "slope_index": 0.2548561782,
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


def _data_sufficiency_indicator(sample_size: int) -> dict[str, Any]:
    n = max(0, int(sample_size))
    if n < DATA_SUFFICIENCY_LIMITED_MIN:
        return {
            "sample_size": n,
            "tier": "insufficient",
            "explanation": (
                f"Sample size {n} is below {DATA_SUFFICIENCY_LIMITED_MIN}; "
                "discrimination/calibration metrics are highly unstable."
            ),
        }
    if n < DATA_SUFFICIENCY_MODERATE_MIN:
        return {
            "sample_size": n,
            "tier": "limited",
            "explanation": (
                f"Sample size {n} is between {DATA_SUFFICIENCY_LIMITED_MIN} and {DATA_SUFFICIENCY_MODERATE_MIN - 1}; "
                "directional signals may exist but reliability is limited."
            ),
        }
    if n <= DATA_SUFFICIENCY_STRONG_MIN:
        return {
            "sample_size": n,
            "tier": "moderate",
            "explanation": (
                f"Sample size {n} is between {DATA_SUFFICIENCY_MODERATE_MIN} and {DATA_SUFFICIENCY_STRONG_MIN}; "
                "metrics are materially more reliable but still sensitive to slice effects."
            ),
        }
    return {
        "sample_size": n,
        "tier": "strong",
        "explanation": (
            f"Sample size {n} exceeds {DATA_SUFFICIENCY_STRONG_MIN}; "
            "metrics are comparatively robust for trend tracking."
        ),
    }


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


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    n = int(total)
    k = int(successes)
    if n <= 0:
        return {"low": None, "high": None}
    phat = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) / n) + ((z * z) / (4.0 * n * n)))
        / denom
    )
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


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


def _bootstrap_metric_ci(
    *,
    y_true: list[int],
    y_score: list[float],
    metric: str,
    iterations: int = 400,
    seed: int = 17,
) -> dict[str, float | None]:
    if len(y_true) != len(y_score) or len(y_true) < 4:
        return {"low": None, "high": None}
    n = len(y_true)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(max(50, int(iterations))):
        idx = [rng.randrange(n) for _ in range(n)]
        y_samp = [int(y_true[i]) for i in idx]
        s_samp = [float(y_score[i]) for i in idx]
        if metric == "auc":
            value = _roc_auc(y_samp, s_samp)
        elif metric == "pr_auc":
            value = _pr_auc(y_samp, s_samp)
        elif metric == "brier":
            value = _brier(y_samp, s_samp)
        else:
            value = None
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return {"low": None, "high": None}
    values.sort()
    low_idx = max(0, min(len(values) - 1, int(0.025 * (len(values) - 1))))
    high_idx = max(0, min(len(values) - 1, int(0.975 * (len(values) - 1))))
    return {"low": values[low_idx], "high": values[high_idx]}


def _build_metric_stability(
    *,
    y_true: list[int],
    raw_auc: float | None,
) -> dict[str, Any]:
    n = len(y_true)
    positives = sum(int(v) for v in y_true)
    negatives = n - positives
    warnings: list[str] = []
    unstable_metrics: list[str] = []
    auc_stable = bool(
        raw_auc is not None
        and n >= SMALL_SAMPLE_WARNING_N
        and positives >= STABLE_AUC_MIN_CLASS_COUNT
        and negatives >= STABLE_AUC_MIN_CLASS_COUNT
    )
    if n < SMALL_SAMPLE_WARNING_N:
        warnings.append(
            f"Insufficient data for stable AUC/PR-AUC interpretation (n={n} < {SMALL_SAMPLE_WARNING_N})."
        )
        unstable_metrics.extend(["wildfire_risk_score_auc", "wildfire_risk_score_pr_auc"])
    if positives < STABLE_AUC_MIN_CLASS_COUNT or negatives < STABLE_AUC_MIN_CLASS_COUNT:
        warnings.append(
            "Class counts are too small for stable discrimination estimates "
            f"(positives={positives}, negatives={negatives})."
        )
        unstable_metrics.extend(["wildfire_risk_score_auc", "wildfire_risk_score_pr_auc"])
    return {
        "sample_size": n,
        "positive_count": positives,
        "negative_count": negatives,
        "small_sample_threshold": SMALL_SAMPLE_WARNING_N,
        "stable_auc_min_class_count": STABLE_AUC_MIN_CLASS_COUNT,
        "auc_stable": auc_stable,
        "unstable_metrics": sorted(set(unstable_metrics)),
        "warnings": warnings,
    }


def _independent_row_id(row: dict[str, Any], index: int) -> str:
    property_event_id = str(row.get("property_event_id") or "").strip()
    if property_event_id:
        return property_event_id
    event_id = str(row.get("event_id") or "").strip()
    source_record_id = str(row.get("source_record_id") or "").strip()
    record_id = str(row.get("record_id") or "").strip()
    if event_id and source_record_id:
        return f"{event_id}::{source_record_id}"
    if event_id and record_id:
        return f"{event_id}::{record_id}"
    return f"row::{index}"


def _build_modeling_viability(
    *,
    prepared_rows: list[dict[str, Any]],
    feature_signal_diagnostics: dict[str, Any],
    baseline_model_comparison: dict[str, Any],
) -> dict[str, Any]:
    independent_ids = {
        _independent_row_id(row, index)
        for index, row in enumerate(prepared_rows)
        if isinstance(row, dict)
    }
    independent_sample_count = len(independent_ids)
    labeled_sample_count = len(prepared_rows)
    duplication_factor = (
        (float(labeled_sample_count) / float(independent_sample_count))
        if independent_sample_count > 0
        else None
    )

    feature_count = int(feature_signal_diagnostics.get("feature_count_evaluated") or 0)
    near_zero_variance_count = int(feature_signal_diagnostics.get("near_zero_variance_feature_count") or 0)
    features_with_variance_count = max(0, feature_count - near_zero_variance_count)
    variation_ratio = (
        (float(features_with_variance_count) / float(feature_count))
        if feature_count > 0
        else 0.0
    )

    comparison = (
        baseline_model_comparison.get("comparison")
        if isinstance(baseline_model_comparison.get("comparison"), dict)
        else {}
    )
    full_model_auc = _safe_float(baseline_model_comparison.get("full_model_auc"))
    random_baseline_auc = None
    baselines = (
        baseline_model_comparison.get("baselines")
        if isinstance(baseline_model_comparison.get("baselines"), dict)
        else {}
    )
    random_baseline = (
        baselines.get("random")
        if isinstance(baselines.get("random"), dict)
        else {}
    )
    random_baseline_auc = _safe_float(random_baseline.get("auc"))
    margin_vs_random = _safe_float(comparison.get("auc_margin_vs_random_baseline"))
    if margin_vs_random is None and full_model_auc is not None and random_baseline_auc is not None:
        margin_vs_random = float(full_model_auc) - float(random_baseline_auc)

    checks = {
        "independent_sample_size_ok": independent_sample_count >= VIABILITY_MIN_INDEPENDENT_SAMPLES,
        "feature_variation_ok": (
            features_with_variance_count >= VIABILITY_MIN_FEATURES_WITH_VARIANCE
            and variation_ratio >= VIABILITY_MIN_FEATURE_VARIATION_RATIO
        ),
        "model_vs_random_auc_ok": (
            margin_vs_random is not None and margin_vs_random >= VIABILITY_MIN_AUC_MARGIN_VS_RANDOM
        ),
    }
    failed_checks = [name for name, ok in checks.items() if not bool(ok)]
    viable = not failed_checks
    if viable:
        classification = "viable_for_directional_modeling"
        reason = (
            "Dataset passes minimum independent-sample, feature-variation, and model-vs-random AUC margin checks."
        )
    else:
        classification = "dataset_not_viable_for_predictive_modeling"
        reason_parts: list[str] = []
        if "independent_sample_size_ok" in failed_checks:
            reason_parts.append(
                f"independent samples {independent_sample_count} < {VIABILITY_MIN_INDEPENDENT_SAMPLES}"
            )
        if "feature_variation_ok" in failed_checks:
            reason_parts.append(
                "feature variation is too low "
                f"({features_with_variance_count}/{feature_count} varying, ratio {variation_ratio:.3f})"
            )
        if "model_vs_random_auc_ok" in failed_checks:
            if margin_vs_random is None:
                reason_parts.append("model-vs-random AUC margin unavailable")
            else:
                reason_parts.append(
                    f"model-vs-random AUC margin {margin_vs_random:.4f} < {VIABILITY_MIN_AUC_MARGIN_VS_RANDOM:.2f}"
                )
        reason = "; ".join(reason_parts) if reason_parts else "one or more viability checks failed"

    return {
        "dataset_viable_for_predictive_modeling": viable,
        "classification": classification,
        "reason": reason,
        "thresholds": {
            "min_independent_samples": VIABILITY_MIN_INDEPENDENT_SAMPLES,
            "min_features_with_variance": VIABILITY_MIN_FEATURES_WITH_VARIANCE,
            "min_feature_variation_ratio": VIABILITY_MIN_FEATURE_VARIATION_RATIO,
            "min_auc_margin_vs_random_baseline": VIABILITY_MIN_AUC_MARGIN_VS_RANDOM,
        },
        "checks": {
            "independent_sample_count": independent_sample_count,
            "labeled_sample_count": labeled_sample_count,
            "duplication_factor": duplication_factor,
            "feature_count_evaluated": feature_count,
            "near_zero_variance_feature_count": near_zero_variance_count,
            "features_with_variance_count": features_with_variance_count,
            "feature_variation_ratio": variation_ratio,
            "full_model_auc": full_model_auc,
            "random_baseline_auc": random_baseline_auc,
            "auc_margin_vs_random_baseline": margin_vs_random,
            "independent_sample_size_ok": checks["independent_sample_size_ok"],
            "feature_variation_ok": checks["feature_variation_ok"],
            "model_vs_random_auc_ok": checks["model_vs_random_auc_ok"],
        },
        "failed_checks": failed_checks,
        "caveat": (
            "This viability check is a guardrail for directional public-outcome evaluation. "
            "It does not establish insurer-claims predictive validity."
        ),
    }


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


def _extract_feature_value(row: dict[str, Any], key: str) -> float | None:
    raw = row.get("raw_feature_vector") if isinstance(row.get("raw_feature_vector"), dict) else {}
    transformed = (
        row.get("transformed_feature_vector")
        if isinstance(row.get("transformed_feature_vector"), dict)
        else {}
    )
    if key in raw:
        return _safe_float(raw.get(key))
    if key in transformed:
        return _safe_float(transformed.get(key))
    return _safe_float(row.get(key))


def _normalize_percent_like(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if v <= 1.0:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _compute_hazard_signal_percent(row: dict[str, Any]) -> float | None:
    hazard_score = _extract_score(row, "site_hazard_score")
    if hazard_score is None:
        hazard_score = _extract_feature_value(row, "wildfire_hazard")
    if hazard_score is None:
        burn_probability = _extract_feature_value(row, "burn_probability")
        if burn_probability is not None:
            hazard_score = _normalize_percent_like(burn_probability)
    if hazard_score is None:
        hazard_score = _extract_score(row, "wildfire_risk_score")
    return _normalize_percent_like(hazard_score)


def _compute_vegetation_density_index(row: dict[str, Any]) -> float | None:
    weighted_terms: list[tuple[float, float]] = []
    for key, weight in (
        ("near_structure_vegetation_0_5_pct", 0.46),
        ("ring_0_5_ft_vegetation_density", 0.34),
        ("ring_5_30_ft_vegetation_density", 0.24),
        ("near_structure_connectivity_index", 0.18),
        ("canopy_adjacency_proxy_pct", 0.09),
        ("vegetation_continuity_proxy_pct", 0.09),
        ("canopy_cover", 0.06),
    ):
        value = _normalize_percent_like(_extract_feature_value(row, key))
        if value is None:
            continue
        weighted_terms.append((value, weight))
    if not weighted_terms:
        return None
    numerator = sum(value * weight for value, weight in weighted_terms)
    denom = sum(weight for _, weight in weighted_terms)
    return (numerator / denom) if denom > 0.0 else None


def _compute_high_signal_simplified_score(row: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    components: dict[str, float] = {}

    nearest_high_fuel_patch_distance_ft = _safe_float(
        _extract_feature_value(row, "nearest_high_fuel_patch_distance_ft")
    )
    if nearest_high_fuel_patch_distance_ft is not None:
        distance_clamped = max(0.0, min(300.0, float(nearest_high_fuel_patch_distance_ft)))
        components["nearest_high_fuel_patch_distance_ft"] = max(
            0.0,
            min(100.0, 100.0 - ((distance_clamped / 300.0) * 100.0)),
        )

    canopy_adjacency_proxy_pct = _normalize_percent_like(
        _extract_feature_value(row, "canopy_adjacency_proxy_pct")
    )
    if canopy_adjacency_proxy_pct is not None:
        # Direction aligned to observed signal: higher adjacency currently maps
        # to lower adverse-outcome probability in this labeled sample.
        components["canopy_adjacency_proxy_pct"] = max(
            0.0,
            min(100.0, 100.0 - float(canopy_adjacency_proxy_pct)),
        )

    vegetation_continuity_proxy_pct = _normalize_percent_like(
        _extract_feature_value(row, "vegetation_continuity_proxy_pct")
    )
    if vegetation_continuity_proxy_pct is not None:
        components["vegetation_continuity_proxy_pct"] = max(
            0.0,
            min(100.0, float(vegetation_continuity_proxy_pct)),
        )

    slope_index = _normalize_percent_like(_extract_feature_value(row, "slope_index"))
    if slope_index is not None:
        components["slope_index"] = max(0.0, min(100.0, float(slope_index)))

    weighted_terms: list[tuple[str, float, float]] = []
    for feature_name, weight in HIGH_SIGNAL_MODEL_SIGNAL_WEIGHTS.items():
        component_value = _safe_float(components.get(feature_name))
        if component_value is None:
            continue
        weighted_terms.append((feature_name, float(component_value), float(weight)))

    if not weighted_terms:
        return None, {
            "available": False,
            "reason": "no_high_signal_features_available",
            "components": {},
            "normalized_weights": {},
        }

    total_weight = sum(weight for _, _, weight in weighted_terms)
    if total_weight <= 0.0:
        return None, {
            "available": False,
            "reason": "non_positive_weight_sum",
            "components": components,
            "normalized_weights": {},
        }
    normalized_weights = {
        feature_name: (weight / total_weight)
        for feature_name, _, weight in weighted_terms
    }
    score = sum(value * normalized_weights[feature_name] for feature_name, value, _ in weighted_terms)
    score = max(0.0, min(100.0, float(score)))
    return score, {
        "available": True,
        "components": components,
        "normalized_weights": normalized_weights,
    }


def _bucketize_segment(
    *,
    value: float | None,
    thresholds: tuple[float, float, float],
    labels: tuple[str, str, str, str],
) -> str:
    if value is None:
        return "unknown"
    a, b, c = thresholds
    if value < a:
        return labels[0]
    if value < b:
        return labels[1]
    if value < c:
        return labels[2]
    return labels[3]


def _derive_hazard_level_segment(row: dict[str, Any]) -> str:
    hazard_score = _compute_hazard_signal_percent(row)
    return _bucketize_segment(
        value=hazard_score,
        thresholds=HAZARD_SEGMENT_THRESHOLDS,
        labels=("low", "moderate", "high", "severe"),
    )


def _derive_vegetation_density_segment(row: dict[str, Any]) -> str:
    density_index = _compute_vegetation_density_index(row)
    return _bucketize_segment(
        value=density_index,
        thresholds=VEGETATION_SEGMENT_THRESHOLDS,
        labels=("sparse", "moderate", "dense", "very_dense"),
    )


def _deterministic_random_probability(row: dict[str, Any]) -> float:
    key_parts = [
        str(row.get("event_id") or ""),
        str(row.get("record_id") or ""),
        str(row.get("source_record_id") or ""),
        str(row.get("address_text") or ""),
        str(row.get("latitude") or ""),
        str(row.get("longitude") or ""),
    ]
    text = "|".join(key_parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Deterministic pseudo-random in [0, 1).
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def _impute_probability_series(values: list[float | None], *, fallback: float = 0.5) -> tuple[list[float], int]:
    observed = [float(v) for v in values if isinstance(v, (int, float))]
    fill = _mean(observed)
    fill_value = float(fill) if isinstance(fill, (int, float)) else float(fallback)
    imputed: list[float] = []
    missing_count = 0
    for value in values:
        if isinstance(value, (int, float)):
            imputed.append(max(0.0, min(1.0, float(value))))
        else:
            missing_count += 1
            imputed.append(max(0.0, min(1.0, fill_value)))
    return imputed, missing_count


def _compute_simple_baseline_metrics(
    *,
    y_true: list[int],
    usable_rows: list[dict[str, Any]],
    full_model_auc: float | None,
) -> dict[str, Any]:
    if not y_true or not usable_rows or len(y_true) != len(usable_rows):
        return {
            "available": False,
            "reason": "insufficient_rows",
            "full_model_auc": full_model_auc,
            "baselines": {},
            "comparison": {
                "beats_all_baselines_by_auc": None,
                "best_baseline_name": None,
                "best_baseline_auc": None,
                "auc_margin_vs_best_baseline": None,
                "auc_margin_vs_random_baseline": None,
                "baselines_compared_count": 0,
            },
            "caveat": (
                "Baseline comparison is unavailable because labeled rows were insufficient. "
                "These checks are directional only and not ground-truth carrier validation."
            ),
        }

    random_probs = [_deterministic_random_probability(row) for row in usable_rows]
    hazard_probs_raw: list[float | None] = []
    vegetation_probs_raw: list[float | None] = []
    for row in usable_rows:
        hazard_signal = _compute_hazard_signal_percent(row)
        vegetation_signal = _compute_vegetation_density_index(row)
        hazard_probs_raw.append((hazard_signal / 100.0) if hazard_signal is not None else None)
        vegetation_probs_raw.append((vegetation_signal / 100.0) if vegetation_signal is not None else None)
    hazard_probs, hazard_missing = _impute_probability_series(hazard_probs_raw, fallback=0.5)
    vegetation_probs, vegetation_missing = _impute_probability_series(vegetation_probs_raw, fallback=0.5)

    baselines = {
        "random": {
            "description": "Deterministic pseudo-random baseline keyed by record identity.",
            "auc": _roc_auc(y_true, random_probs),
            "pr_auc": _pr_auc(y_true, random_probs),
            "brier": _brier(y_true, random_probs),
            "missing_signal_count": 0,
        },
        "hazard_only": {
            "description": "Uses hazard signal only (site hazard / burn probability proxies).",
            "auc": _roc_auc(y_true, hazard_probs),
            "pr_auc": _pr_auc(y_true, hazard_probs),
            "brier": _brier(y_true, hazard_probs),
            "missing_signal_count": int(hazard_missing),
        },
        "vegetation_only": {
            "description": "Uses near-structure vegetation density proxy only.",
            "auc": _roc_auc(y_true, vegetation_probs),
            "pr_auc": _pr_auc(y_true, vegetation_probs),
            "brier": _brier(y_true, vegetation_probs),
            "missing_signal_count": int(vegetation_missing),
        },
    }

    comparable = [
        (name, float(payload.get("auc")))
        for name, payload in baselines.items()
        if isinstance(payload, dict) and isinstance(payload.get("auc"), (int, float))
    ]
    best_baseline_name: str | None = None
    best_baseline_auc: float | None = None
    if comparable:
        best_baseline_name, best_baseline_auc = max(comparable, key=lambda item: float(item[1]))
    auc_margin = (
        float(full_model_auc) - float(best_baseline_auc)
        if isinstance(full_model_auc, (int, float)) and isinstance(best_baseline_auc, (int, float))
        else None
    )
    beats_all = (
        bool(isinstance(full_model_auc, (int, float)) and all(float(full_model_auc) > auc for _, auc in comparable))
        if comparable
        else None
    )

    return {
        "available": True,
        "full_model_auc": full_model_auc,
        "baselines": baselines,
        "comparison": {
            "beats_all_baselines_by_auc": beats_all,
            "best_baseline_name": best_baseline_name,
            "best_baseline_auc": best_baseline_auc,
            "auc_margin_vs_best_baseline": auc_margin,
            "auc_margin_vs_random_baseline": (
                float(full_model_auc) - float((baselines.get("random") or {}).get("auc"))
                if isinstance(full_model_auc, (int, float))
                and isinstance((baselines.get("random") or {}).get("auc"), (int, float))
                else None
            ),
            "baselines_compared_count": len(comparable),
            "complexity_justified_signal": (
                "yes" if isinstance(auc_margin, float) and auc_margin > 0.0 else "no_or_inconclusive"
            ),
        },
        "caveat": (
            "Baseline comparison checks whether the full model outperforms simple directional baselines "
            "on this public-outcome sample. It does not establish insurer-claims predictive truth."
        ),
    }


def _derive_target_from_label(label: str) -> int | None:
    norm = _normalize_label(label)
    if norm in {"major_damage", "destroyed"}:
        return 1
    if norm in {"minor_damage", "no_damage", "no_known_damage"}:
        return 0
    return None


def _derive_surrogate_wildfire_score(row: dict[str, Any]) -> float | None:
    site = _extract_score(row, "site_hazard_score")
    vuln = _extract_score(row, "home_ignition_vulnerability_score")
    if site is None and vuln is None:
        return None
    if site is not None and vuln is not None:
        # Prefer a blended surrogate grounded in the two strongest available components.
        value = (0.55 * float(site)) + (0.45 * float(vuln))
    else:
        value = float(site if site is not None else vuln)
    return max(0.0, min(100.0, value))


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
    fallback_usage = row.get("fallback_usage") if isinstance(row.get("fallback_usage"), dict) else {}
    evidence_summary = row.get("evidence_quality_summary") if isinstance(row.get("evidence_quality_summary"), dict) else {}
    return {
        "observed_factor_count": int(
            _safe_int(fallback_usage.get("observed_factor_count"))
            or _safe_int(evidence_summary.get("observed_factor_count"))
            or 0
        ),
        "fallback_factor_count": int(
            _safe_int(fallback_usage.get("fallback_factor_count"))
            or _safe_int(flags.get("fallback_factor_count"))
            or 0
        ),
        "missing_factor_count": int(
            _safe_int(fallback_usage.get("missing_factor_count"))
            or _safe_int(flags.get("missing_factor_count"))
            or 0
        ),
        "inferred_factor_count": int(
            _safe_int(fallback_usage.get("inferred_factor_count"))
            or _safe_int(flags.get("inferred_factor_count"))
            or 0
        ),
        "coverage_failed_count": int(
            _safe_int(fallback_usage.get("coverage_failed_count"))
            or _safe_int(flags.get("coverage_failed_count"))
            or 0
        ),
        "coverage_fallback_count": int(
            _safe_int(fallback_usage.get("coverage_fallback_count"))
            or _safe_int(flags.get("coverage_fallback_count"))
            or 0
        ),
        "fallback_weight_fraction": (
            _safe_float(fallback_usage.get("fallback_weight_fraction"))
            if _safe_float(fallback_usage.get("fallback_weight_fraction")) is not None
            else (_safe_float(flags.get("fallback_weight_fraction")) or 0.0)
        ),
    }


def _derive_fallback_usage_summary(
    *,
    evidence_tier: str,
    fallback_flags: dict[str, float | int],
) -> dict[str, Any]:
    observed_factor_count = int(_safe_int(fallback_flags.get("observed_factor_count")) or 0)
    fallback_factor_count = int(_safe_int(fallback_flags.get("fallback_factor_count")) or 0)
    missing_factor_count = int(_safe_int(fallback_flags.get("missing_factor_count")) or 0)
    inferred_factor_count = int(_safe_int(fallback_flags.get("inferred_factor_count")) or 0)
    coverage_failed_count = int(_safe_int(fallback_flags.get("coverage_failed_count")) or 0)
    coverage_fallback_count = int(_safe_int(fallback_flags.get("coverage_fallback_count")) or 0)
    fallback_weight_fraction = float(_safe_float(fallback_flags.get("fallback_weight_fraction")) or 0.0)

    total_factor_count = max(
        1,
        observed_factor_count + inferred_factor_count + fallback_factor_count + missing_factor_count,
    )
    fallback_factor_ratio = float(fallback_factor_count) / float(total_factor_count)
    missing_factor_ratio = float(missing_factor_count) / float(total_factor_count)
    evidence_tier_norm = str(evidence_tier or "").strip().lower()

    fallback_heavy_reasons: list[str] = []
    if evidence_tier_norm in {"low", "preliminary", "fallback_heavy"}:
        fallback_heavy_reasons.append("low_evidence_tier")
    if coverage_failed_count > 0:
        fallback_heavy_reasons.append("coverage_failed")
    if fallback_weight_fraction >= FALLBACK_HEAVY_WEIGHT_THRESHOLD:
        fallback_heavy_reasons.append("high_fallback_weight_fraction")
    if (
        fallback_factor_count >= 2
        and fallback_factor_ratio >= FALLBACK_HEAVY_FACTOR_RATIO_THRESHOLD
    ):
        fallback_heavy_reasons.append("high_fallback_factor_ratio")
    if (
        coverage_fallback_count >= FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD
        and fallback_weight_fraction >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
    ):
        fallback_heavy_reasons.append("multi_layer_fallback_with_elevated_weight")
    if (
        missing_factor_count >= 3
        and missing_factor_ratio >= FALLBACK_HEAVY_MISSING_RATIO_THRESHOLD
    ):
        fallback_heavy_reasons.append("high_missing_factor_ratio")

    fallback_heavy = bool(fallback_heavy_reasons)
    if fallback_heavy:
        classification = "fallback_heavy"
    elif (
        (
            fallback_factor_count == 0
            and missing_factor_count <= 1
            and coverage_failed_count == 0
            and fallback_weight_fraction < 0.35
            and observed_factor_count >= max(3, inferred_factor_count + 1)
        )
        or (
            evidence_tier_norm == "high"
            and fallback_factor_count <= 1
            and missing_factor_count <= 1
            and coverage_failed_count == 0
            and fallback_weight_fraction < FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
        )
    ):
        classification = "high_evidence"
    else:
        classification = "mixed_evidence"

    return {
        "classification": classification,
        "fallback_heavy": fallback_heavy,
        "fallback_heavy_reasons": sorted(set(fallback_heavy_reasons)),
        "fallback_weight_fraction": round(fallback_weight_fraction, 4),
        "fallback_factor_ratio": round(fallback_factor_ratio, 4),
        "missing_factor_ratio": round(missing_factor_ratio, 4),
    }


def _derive_evidence_group(*, evidence_tier: str, fallback_flags: dict[str, float | int]) -> str:
    return str(
        _derive_fallback_usage_summary(
            evidence_tier=evidence_tier,
            fallback_flags=fallback_flags,
        ).get("classification")
        or "mixed_evidence"
    )


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


def _build_confidence_tier_performance(
    *,
    all_rows: list[dict[str, Any]],
    high_confidence_rows: list[dict[str, Any]],
    medium_confidence_rows: list[dict[str, Any]],
    min_slice_size: int,
) -> dict[str, Any]:
    min_n = max(1, int(min_slice_size))
    tiers = {
        "all_data": _compute_subset_metrics(all_rows),
        "high_confidence": _compute_subset_metrics(high_confidence_rows),
        "medium_confidence": _compute_subset_metrics(medium_confidence_rows),
    }
    for detail in tiers.values():
        if isinstance(detail, dict):
            detail["small_sample_warning"] = int(detail.get("count") or 0) < min_n

    all_auc = _safe_float((tiers.get("all_data") or {}).get("wildfire_risk_score_auc"))
    all_brier = _safe_float((tiers.get("all_data") or {}).get("wildfire_risk_score_brier"))
    high_auc = _safe_float((tiers.get("high_confidence") or {}).get("wildfire_risk_score_auc"))
    high_brier = _safe_float((tiers.get("high_confidence") or {}).get("wildfire_risk_score_brier"))
    med_auc = _safe_float((tiers.get("medium_confidence") or {}).get("wildfire_risk_score_auc"))
    med_brier = _safe_float((tiers.get("medium_confidence") or {}).get("wildfire_risk_score_brier"))

    warnings: list[str] = []
    high_n = int((tiers.get("high_confidence") or {}).get("count") or 0)
    med_n = int((tiers.get("medium_confidence") or {}).get("count") or 0)
    if high_n < min_n:
        warnings.append(
            f"High-confidence slice is too small for stable interpretation (n={high_n} < {min_n})."
        )
    if med_n < min_n:
        warnings.append(
            f"Medium-confidence slice is too small for stable interpretation (n={med_n} < {min_n})."
        )

    return {
        "min_slice_size": min_n,
        "tiers": tiers,
        "deltas_vs_all_data": {
            "high_confidence_auc_delta": (
                (high_auc - all_auc) if high_auc is not None and all_auc is not None else None
            ),
            "high_confidence_brier_delta": (
                (high_brier - all_brier) if high_brier is not None and all_brier is not None else None
            ),
            "medium_confidence_auc_delta": (
                (med_auc - all_auc) if med_auc is not None and all_auc is not None else None
            ),
            "medium_confidence_brier_delta": (
                (med_brier - all_brier) if med_brier is not None and all_brier is not None else None
            ),
        },
        "warnings": warnings,
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
    fallback_usage = evaluation.get("fallback_usage") if isinstance(evaluation.get("fallback_usage"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    governance = provenance.get("model_governance") if isinstance(provenance.get("model_governance"), dict) else {}

    outcome_label = _normalize_damage_class(
        outcome.get("damage_label") or outcome.get("damage_severity_class")
    )
    target = _safe_int(outcome.get("structure_loss_or_major_damage"))
    if target not in {0, 1}:
        target = 1 if outcome_label in {"major_damage", "destroyed"} else (0 if outcome_label in {"minor_damage", "no_damage"} else None)

    fallback_default_flags = {
        "fallback_factor_count": int(
            _safe_int(fallback_usage.get("fallback_factor_count"))
            or _safe_int(evidence_summary.get("fallback_factor_count"))
            or 0
        ),
        "missing_factor_count": int(
            _safe_int(fallback_usage.get("missing_factor_count"))
            or _safe_int(evidence_summary.get("missing_factor_count"))
            or 0
        ),
        "inferred_factor_count": int(
            _safe_int(fallback_usage.get("inferred_factor_count"))
            or _safe_int(evidence_summary.get("inferred_factor_count"))
            or 0
        ),
        "coverage_failed_count": int(
            _safe_int(fallback_usage.get("coverage_failed_count"))
            or _safe_int(coverage_summary.get("failed_count"))
            or 0
        ),
        "coverage_fallback_count": int(
            _safe_int(fallback_usage.get("coverage_fallback_count"))
            or _safe_int(coverage_summary.get("fallback_count"))
            or 0
        ),
        "fallback_weight_fraction": (
            _safe_float(fallback_usage.get("fallback_weight_fraction"))
            if _safe_float(fallback_usage.get("fallback_weight_fraction")) is not None
            else (_safe_float(evidence_summary.get("fallback_weight_fraction")) or 0.0)
        ),
    }

    return {
        "property_event_id": row.get("property_event_id"),
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
        "fallback_usage": fallback_usage,
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


def _build_segment_performance_summary(
    *,
    slice_metrics: dict[str, Any],
    min_slice_size: int,
) -> dict[str, Any]:
    segment_specs = (
        ("hazard_level", "by_hazard_level"),
        ("vegetation_density", "by_vegetation_density"),
        ("confidence_tier", "by_confidence_tier"),
        ("region", "by_region"),
    )

    def _is_strong_segment(row: dict[str, Any]) -> bool:
        auc = row.get("auc")
        brier = row.get("brier")
        return isinstance(auc, float) and isinstance(brier, float) and auc >= 0.60 and brier <= 0.26

    def _is_weak_segment(row: dict[str, Any]) -> bool:
        auc = row.get("auc")
        brier = row.get("brier")
        return isinstance(auc, float) and isinstance(brier, float) and (auc < 0.55 or brier >= 0.26)
    segment_rows: list[dict[str, Any]] = []
    insufficient_rows: list[dict[str, Any]] = []

    for segment_family, slice_key in segment_specs:
        family_rows = (
            slice_metrics.get(slice_key)
            if isinstance(slice_metrics.get(slice_key), dict)
            else {}
        )
        for segment_name in sorted(family_rows.keys()):
            detail = (
                family_rows.get(segment_name)
                if isinstance(family_rows.get(segment_name), dict)
                else {}
            )
            count = int(detail.get("count") or 0)
            auc = _safe_float(detail.get("wildfire_risk_score_auc"))
            brier = _safe_float(detail.get("wildfire_risk_score_brier"))
            row = {
                "segment_family": segment_family,
                "segment_name": str(segment_name),
                "count": count,
                "auc": auc,
                "brier": brier,
                "positive_rate": _safe_float(detail.get("positive_rate")),
                "small_sample_warning": bool(detail.get("small_sample_warning")),
            }
            segment_rows.append(row)
            if count < max(1, int(min_slice_size)) or auc is None or brier is None:
                insufficient_rows.append(row)

    eligible = [
        row
        for row in segment_rows
        if row["count"] >= max(1, int(min_slice_size))
        and isinstance(row.get("auc"), float)
        and isinstance(row.get("brier"), float)
    ]
    strongest = sorted(
        eligible,
        key=lambda row: (
            -float(row.get("auc") or 0.0),
            float(row.get("brier") or 1.0),
            -int(row.get("count") or 0),
            str(row.get("segment_family") or ""),
            str(row.get("segment_name") or ""),
        ),
    )[:8]
    weakest = sorted(
        eligible,
        key=lambda row: (
            float(row.get("auc") or 0.0),
            -float(row.get("brier") or 0.0),
            -int(row.get("count") or 0),
            str(row.get("segment_family") or ""),
            str(row.get("segment_name") or ""),
        ),
    )[:8]

    def _family_status(*, eligible_rows: list[dict[str, Any]], min_n: int) -> tuple[str, list[str]]:
        notes: list[str] = []
        if not eligible_rows:
            return "insufficient_data", [f"No segments met minimum slice size n>={min_n} with valid AUC/Brier."]
        if len(eligible_rows) == 1:
            only = eligible_rows[0]
            notes.append(
                f"Only one eligible segment ({only.get('segment_name')}); no within-family separation estimate available."
            )
            return "single_segment_only", notes
        auc_values = [float(row.get("auc") or 0.0) for row in eligible_rows]
        auc_max = max(auc_values)
        auc_min = min(auc_values)
        auc_spread = auc_max - auc_min
        if auc_spread < 0.02:
            notes.append("AUC spread across segments is <0.02, indicating weak differentiation by this family.")
            if auc_max < 0.55:
                notes.append("All segment AUC values are weak (<0.55).")
                return "flat_and_weak", notes
            return "flat", notes
        if auc_max >= 0.62 and auc_min >= 0.55:
            notes.append("Most segments show directional discrimination with limited degradation.")
            return "strong", notes
        if auc_max >= 0.60 and auc_min < 0.55:
            notes.append("Some segments discriminate well while others are weak.")
            return "mixed", notes
        if auc_max < 0.55:
            notes.append("All eligible segments are weak (AUC <0.55).")
            return "weak", notes
        notes.append("Segment behavior is usable but not consistently strong.")
        return "moderate", notes

    family_map: dict[str, Any] = {}
    for segment_family, _slice_key in segment_specs:
        family_rows = [row for row in segment_rows if str(row.get("segment_family")) == segment_family]
        family_eligible = [
            row
            for row in family_rows
            if int(row.get("count") or 0) >= max(1, int(min_slice_size))
            and isinstance(row.get("auc"), float)
            and isinstance(row.get("brier"), float)
        ]
        family_sorted_best = sorted(
            family_eligible,
            key=lambda row: (
                -float(row.get("auc") or 0.0),
                float(row.get("brier") or 1.0),
                -int(row.get("count") or 0),
                str(row.get("segment_name") or ""),
            ),
        )
        family_sorted_weak = sorted(
            family_eligible,
            key=lambda row: (
                float(row.get("auc") or 0.0),
                -float(row.get("brier") or 0.0),
                -int(row.get("count") or 0),
                str(row.get("segment_name") or ""),
            ),
        )
        best_segment = family_sorted_best[0] if family_sorted_best else None
        worst_segment = family_sorted_weak[0] if family_sorted_weak else None
        auc_spread = (
            float(best_segment.get("auc") or 0.0) - float(worst_segment.get("auc") or 0.0)
            if best_segment and worst_segment
            else None
        )
        status, notes = _family_status(eligible_rows=family_eligible, min_n=max(1, int(min_slice_size)))
        strong_segments = [row for row in family_sorted_best if _is_strong_segment(row)][:3]
        strong_keys = {
            (str(row.get("segment_family") or ""), str(row.get("segment_name") or ""))
            for row in strong_segments
        }
        weak_segments = [
            row
            for row in family_sorted_weak
            if _is_weak_segment(row)
            and (str(row.get("segment_family") or ""), str(row.get("segment_name") or "")) not in strong_keys
        ][:3]

        # If eligible segments exist but no rows qualify as strong/weak by thresholds,
        # fall back to reporting best/worst without duplicating the same segment.
        if not strong_segments and family_sorted_best:
            strong_segments = family_sorted_best[:1]
        if not weak_segments and family_sorted_weak:
            candidate = family_sorted_weak[0]
            candidate_key = (
                str(candidate.get("segment_family") or ""),
                str(candidate.get("segment_name") or ""),
            )
            if candidate_key not in strong_keys:
                weak_segments = [candidate]

        family_map[segment_family] = {
            "segment_count": len(family_rows),
            "eligible_segment_count": len(family_eligible),
            "status": status,
            "auc_spread": auc_spread,
            "best_segment": best_segment,
            "worst_segment": worst_segment,
            "strong_segments": strong_segments,
            "weak_segments": weak_segments,
            "notes": notes,
        }

    highlights: list[str] = []
    if strongest:
        top = strongest[0]
        highlights.append(
            "Strongest observed segment: "
            f"{top.get('segment_family')}={top.get('segment_name')} "
            f"(n={top.get('count')}, auc={top.get('auc')}, brier={top.get('brier')})."
        )
    if weakest:
        tail = weakest[0]
        highlights.append(
            "Weakest observed segment: "
            f"{tail.get('segment_family')}={tail.get('segment_name')} "
            f"(n={tail.get('count')}, auc={tail.get('auc')}, brier={tail.get('brier')})."
        )
    if not eligible:
        highlights.append(
            "No segment has enough rows for stable AUC/Brier interpretation; inspect small-sample warnings."
        )

    return {
        "min_slice_size": int(max(1, int(min_slice_size))),
        "segment_count": len(segment_rows),
        "eligible_segment_count": len(eligible),
        "insufficient_segment_count": len(insufficient_rows),
        "strongest_segments": strongest,
        "weakest_segments": weakest,
        "insufficient_or_unstable_segments": sorted(
            insufficient_rows,
            key=lambda row: (
                str(row.get("segment_family") or ""),
                str(row.get("segment_name") or ""),
            ),
        )[:50],
        "segment_strength_map": family_map,
        "highlights": highlights,
    }


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


def _pairwise_directional_hit_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if int(row.get("structure_loss_or_major_damage") or 0) == 1]
    negatives = [row for row in rows if int(row.get("structure_loss_or_major_damage") or 0) == 0]
    total_pairs = len(positives) * len(negatives)
    if total_pairs <= 0:
        return {
            "available": False,
            "pair_count": total_pairs,
            "hit_rate": None,
            "ties_fraction": None,
        }
    wins = 0.0
    ties = 0
    for pos in positives:
        pos_score = float(pos.get("wildfire_risk_score") or 0.0)
        for neg in negatives:
            neg_score = float(neg.get("wildfire_risk_score") or 0.0)
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
                ties += 1
    hit_rate = wins / float(total_pairs)
    return {
        "available": True,
        "pair_count": total_pairs,
        "hit_rate": hit_rate,
        "ties_fraction": ties / float(total_pairs),
        "confidence_interval_95": _wilson_interval(int(round(wins)), total_pairs),
    }


def _adverse_rate_by_score_bins(rows: list[dict[str, Any]], max_bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"available": False, "bins": []}
    scored = sorted(rows, key=lambda row: float(row.get("wildfire_risk_score") or 0.0), reverse=True)
    n = len(scored)
    bin_count = max(1, min(int(max_bins), n))
    bins: list[dict[str, Any]] = []
    for idx in range(bin_count):
        start = int((idx / bin_count) * n)
        end = int(((idx + 1) / bin_count) * n)
        if end <= start:
            continue
        bucket = scored[start:end]
        labels = [int(row.get("structure_loss_or_major_damage") or 0) for row in bucket]
        positives = sum(labels)
        adverse_rate = positives / float(len(bucket))
        score_values = [float(row.get("wildfire_risk_score") or 0.0) for row in bucket]
        bins.append(
            {
                "bucket_rank": idx + 1,
                "bucket_label": f"top_{int(round(((idx + 1) / bin_count) * 100))}pct_cumulative",
                "count": len(bucket),
                "score_min": min(score_values),
                "score_max": max(score_values),
                "adverse_rate": adverse_rate,
                "confidence_interval_95": _wilson_interval(positives, len(bucket)),
            }
        )
    return {"available": True, "bin_count": len(bins), "bins": bins}


def _compute_minimum_viable_metrics(
    *,
    prepared_rows: list[dict[str, Any]],
    default_threshold: float,
    default_confusion: dict[str, int],
) -> dict[str, Any]:
    n = len(prepared_rows)
    if n <= 0:
        return {
            "available": False,
            "rank_ordering": {"available": False},
            "simple_accuracy_at_threshold": {"available": False},
            "top_risk_bucket_hit_rate": {"available": False},
            "adverse_rate_by_score_decile": {"available": False, "bins": []},
        }
    labels = [int(row.get("structure_loss_or_major_damage") or 0) for row in prepared_rows]
    positives = sum(labels)
    negatives = n - positives
    scores = [float(row.get("wildfire_risk_score") or 0.0) for row in prepared_rows]
    preds = [1 if score >= float(default_threshold) else 0 for score in scores]
    correct = sum(1 for pred, truth in zip(preds, labels) if pred == truth)

    scored = sorted(prepared_rows, key=lambda row: float(row.get("wildfire_risk_score") or 0.0), reverse=True)
    bucket_size = max(1, int(math.ceil(0.2 * n)))
    top_bucket = scored[:bucket_size]
    top_labels = [int(row.get("structure_loss_or_major_damage") or 0) for row in top_bucket]
    top_positives = sum(top_labels)
    baseline_rate = positives / float(n) if n > 0 else None
    top_rate = top_positives / float(bucket_size) if bucket_size > 0 else None
    lift = (top_rate / baseline_rate) if (top_rate is not None and baseline_rate not in (None, 0.0)) else None

    return {
        "available": True,
        "sample_size": n,
        "class_balance": {
            "positive_count": positives,
            "negative_count": negatives,
            "positive_rate": baseline_rate,
            "positive_rate_confidence_interval_95": _wilson_interval(positives, n),
        },
        "rank_ordering": _pairwise_directional_hit_rate(prepared_rows),
        "simple_accuracy_at_threshold": {
            "available": True,
            "threshold": float(default_threshold),
            "accuracy": correct / float(n),
            "correct_count": correct,
            "count": n,
            "confusion_matrix": default_confusion,
            "confidence_interval_95": _wilson_interval(correct, n),
        },
        "top_risk_bucket_hit_rate": {
            "available": True,
            "bucket_definition": "top_20_percent_by_wildfire_risk_score",
            "bucket_count": bucket_size,
            "adverse_count": top_positives,
            "adverse_rate": top_rate,
            "adverse_rate_confidence_interval_95": _wilson_interval(top_positives, bucket_size),
            "baseline_adverse_rate": baseline_rate,
            "lift_vs_baseline": lift,
        },
        "adverse_rate_by_score_decile": _adverse_rate_by_score_bins(prepared_rows, max_bins=10),
    }


def _feature_family(feature_name: str) -> str:
    key = str(feature_name or "").strip().lower()
    if any(token in key for token in ("vegetation", "canopy", "fuel", "ring_", "ember", "defensible")):
        return "vegetation_metrics"
    if any(token in key for token in ("slope", "aspect", "elevation", "terrain")):
        return "slope_terrain"
    if any(token in key for token in ("hazard", "whp", "mtbs", "fire_history", "perimeter", "burn")):
        return "hazard_burn_context"
    if any(token in key for token in ("dry", "drought", "gridmet", "fm1000", "kbdi", "eddi")):
        return "dryness"
    if any(token in key for token in ("roof", "vent", "structure", "building", "parcel", "hardening", "home")):
        return "structural_features"
    if any(token in key for token in ("road", "access", "travel")):
        return "access_network"
    return "other"


def _expected_direction(feature_name: str) -> str:
    key = str(feature_name or "").strip().lower()
    if key in FEATURE_DIRECTION_OVERRIDES:
        return str(FEATURE_DIRECTION_OVERRIDES[key])
    risk_down_tokens = (
        "distance",
        "_ft",
        "_m",
        "_km",
        "clearance",
        "ember_resistant",
        "class_a",
        "hardening",
    )
    risk_up_tokens = (
        "burn_probability",
        "hazard",
        "whp",
        "mtbs",
        "fuel",
        "canopy",
        "vegetation",
        "slope",
        "adjacency",
        "continuity",
        "ember_exposure_risk",
        "defensible_space_risk",
    )
    if any(token in key for token in risk_down_tokens):
        return "risk_down"
    if any(token in key for token in risk_up_tokens):
        return "risk_up"
    return "unknown"


def _build_direction_alignment(
    *,
    evaluated_rows: list[dict[str, Any]],
    min_signal_score: float,
    min_rows: int,
) -> dict[str, Any]:
    directions = {"risk_up", "risk_down"}
    conflicts_detected: list[dict[str, Any]] = []
    conflicts_resolved: list[dict[str, Any]] = []
    unresolved_conflicts: list[dict[str, Any]] = []
    overrides: dict[str, str] = {}
    actions: dict[str, str] = {}
    details: list[dict[str, Any]] = []

    for row in sorted(evaluated_rows, key=lambda item: str(item.get("feature") or "")):
        feature_name = str(row.get("feature") or "")
        expected_direction = str(row.get("expected_direction") or "unknown")
        observed_direction = str(row.get("observed_direction") or "unknown")
        signal_score = float(_safe_float(row.get("signal_score")) or 0.0)
        rows_with_value = int(_safe_int(row.get("rows_with_value")) or 0)
        before_conflict = (
            expected_direction in directions
            and observed_direction in directions
            and expected_direction != observed_direction
        )
        alignment_action = "none"
        aligned_expected_direction = expected_direction
        if (
            before_conflict
            and signal_score >= float(min_signal_score)
            and rows_with_value >= int(min_rows)
        ):
            aligned_expected_direction = observed_direction
            overrides[feature_name] = observed_direction
            alignment_action = "align_expected_direction_to_observed_signal"
        after_conflict = (
            aligned_expected_direction in directions
            and observed_direction in directions
            and aligned_expected_direction != observed_direction
        )

        if before_conflict:
            conflict_payload = {
                "feature": feature_name,
                "expected_direction": expected_direction,
                "observed_direction": observed_direction,
                "signal_score": signal_score,
                "rows_with_value": rows_with_value,
            }
            conflicts_detected.append(conflict_payload)
            if after_conflict:
                unresolved_conflicts.append(conflict_payload)
            else:
                conflicts_resolved.append(
                    {
                        **conflict_payload,
                        "aligned_expected_direction": aligned_expected_direction,
                        "resolution_action": alignment_action,
                    }
                )
        if alignment_action != "none":
            actions[feature_name] = alignment_action

        details.append(
            {
                "feature": feature_name,
                "expected_direction": expected_direction,
                "aligned_expected_direction": aligned_expected_direction,
                "observed_direction": observed_direction,
                "direction_conflict_before_alignment": before_conflict,
                "direction_conflict_after_alignment": after_conflict,
                "alignment_action": alignment_action,
                "signal_score": signal_score,
                "rows_with_value": rows_with_value,
            }
        )

    return {
        "available": True,
        "thresholds": {
            "min_signal_score": float(min_signal_score),
            "min_rows_with_value": int(min_rows),
        },
        "feature_direction_overrides_static": dict(sorted(FEATURE_DIRECTION_OVERRIDES.items())),
        "conflicts_detected_pre_alignment": len(conflicts_detected),
        "conflicts_remaining_post_alignment": len(unresolved_conflicts),
        "conflicts_resolved_count": len(conflicts_resolved),
        "conflicts_detected": conflicts_detected,
        "conflicts_resolved": conflicts_resolved,
        "unresolved_conflicts": unresolved_conflicts,
        "aligned_expected_direction_overrides": dict(sorted(overrides.items())),
        "alignment_actions_by_feature": dict(sorted(actions.items())),
        "feature_direction_details": details,
    }


def _feature_curve_bins(values: list[float], labels: list[int], bins: int = 5) -> list[dict[str, Any]]:
    if len(values) != len(labels) or len(values) < 2:
        return []
    ranked = sorted(zip(values, labels), key=lambda item: item[0])
    n = len(ranked)
    out: list[dict[str, Any]] = []
    for idx in range(max(1, int(bins))):
        start = int((idx / bins) * n)
        end = int(((idx + 1) / bins) * n)
        if end <= start:
            continue
        bucket = ranked[start:end]
        bucket_values = [float(v) for v, _ in bucket]
        bucket_labels = [int(y) for _, y in bucket]
        adverse_count = sum(bucket_labels)
        out.append(
            {
                "bin": idx + 1,
                "count": len(bucket),
                "value_min": min(bucket_values),
                "value_max": max(bucket_values),
                "value_mean": _mean(bucket_values),
                "adverse_rate": adverse_count / float(len(bucket)),
            }
        )
    return out


def _score_feature_signal(
    *,
    corr: float | None,
    rank_corr: float | None,
    auc_best: float | None,
    std_delta: float | None,
    coverage_fraction: float,
) -> float:
    corr_signal = max(
        abs(float(corr or 0.0)),
        abs(float(rank_corr or 0.0)),
    )
    auc_signal = abs(float((auc_best if auc_best is not None else 0.5)) - 0.5) * 2.0
    delta_signal = min(1.0, abs(float(std_delta or 0.0)) / 2.0)
    base = (0.45 * corr_signal) + (0.35 * auc_signal) + (0.20 * delta_signal)
    return max(0.0, min(1.0, base * max(0.0, min(1.0, coverage_fraction))))


def _build_feature_signal_diagnostics(
    *,
    rows: list[dict[str, Any]],
    min_feature_rows: int = 4,
    max_features: int = 200,
) -> dict[str, Any]:
    total_rows = len(rows)
    if total_rows <= 0:
        return {
            "available": False,
            "reason": "no_rows",
            "top_predictive_features": [],
            "weak_or_noisy_features": [],
            "potentially_harmful_features": [],
            "direction_alignment": {
                "available": False,
                "reason": "no_rows",
                "conflicts_detected_pre_alignment": 0,
                "conflicts_remaining_post_alignment": 0,
                "conflicts_resolved_count": 0,
                "conflicts_detected": [],
                "conflicts_resolved": [],
                "unresolved_conflicts": [],
                "aligned_expected_direction_overrides": {},
            },
            "feature_vs_outcome_curves": [],
            "key_feature_family_summary": {},
        }

    feature_keys: set[str] = set()
    for row in rows:
        raw = row.get("raw_feature_vector") if isinstance(row.get("raw_feature_vector"), dict) else {}
        transformed = (
            row.get("transformed_feature_vector")
            if isinstance(row.get("transformed_feature_vector"), dict)
            else {}
        )
        feature_keys.update(str(key) for key in raw.keys())
        feature_keys.update(str(key) for key in transformed.keys())
    if not feature_keys:
        return {
            "available": False,
            "reason": "no_feature_vectors",
            "top_predictive_features": [],
            "weak_or_noisy_features": [],
            "potentially_harmful_features": [],
            "direction_alignment": {
                "available": False,
                "reason": "no_feature_vectors",
                "conflicts_detected_pre_alignment": 0,
                "conflicts_remaining_post_alignment": 0,
                "conflicts_resolved_count": 0,
                "conflicts_detected": [],
                "conflicts_resolved": [],
                "unresolved_conflicts": [],
                "aligned_expected_direction_overrides": {},
            },
            "feature_vs_outcome_curves": [],
            "key_feature_family_summary": {},
        }

    evaluated: list[dict[str, Any]] = []
    for feature_key in sorted(feature_keys):
        values: list[float] = []
        labels: list[int] = []
        for row in rows:
            value = _proxy_feature_value(row, feature_key)
            if value is None:
                continue
            target = _safe_int(row.get("structure_loss_or_major_damage"))
            if target not in {0, 1}:
                continue
            values.append(float(value))
            labels.append(int(target))
        if not values:
            continue

        positives = [v for v, y in zip(values, labels) if y == 1]
        negatives = [v for v, y in zip(values, labels) if y == 0]
        mean_pos = _mean(positives)
        mean_neg = _mean(negatives)
        delta_pos_neg = (
            (float(mean_pos) - float(mean_neg))
            if mean_pos is not None and mean_neg is not None
            else None
        )
        feature_std = _stddev(values)
        standardized_delta = (
            (float(delta_pos_neg) / float(feature_std))
            if feature_std not in (None, 0.0) and delta_pos_neg is not None
            else None
        )

        corr = _pearson(values, [float(y) for y in labels]) if len(values) >= 3 else None
        rank_corr = _spearman(values, [float(y) for y in labels]) if len(values) >= 3 else None
        auc_risk_up = _roc_auc(labels, values) if len(values) >= 3 else None
        auc_risk_down = _roc_auc(labels, [-v for v in values]) if len(values) >= 3 else None
        if isinstance(auc_risk_up, float) and isinstance(auc_risk_down, float):
            auc_best = max(auc_risk_up, auc_risk_down)
            auc_best_direction = "risk_up" if auc_risk_up >= auc_risk_down else "risk_down"
        elif isinstance(auc_risk_up, float):
            auc_best = auc_risk_up
            auc_best_direction = "risk_up"
        elif isinstance(auc_risk_down, float):
            auc_best = auc_risk_down
            auc_best_direction = "risk_down"
        else:
            auc_best = None
            auc_best_direction = "unknown"

        expected_direction = _expected_direction(feature_key)
        observed_direction = "unknown"
        if isinstance(rank_corr, float) and abs(rank_corr) >= 0.05:
            observed_direction = "risk_up" if rank_corr > 0 else "risk_down"
        elif isinstance(delta_pos_neg, float) and abs(delta_pos_neg) > 0:
            observed_direction = "risk_up" if delta_pos_neg > 0 else "risk_down"

        coverage_fraction = len(values) / float(total_rows)
        signal_score = _score_feature_signal(
            corr=corr,
            rank_corr=rank_corr,
            auc_best=auc_best,
            std_delta=standardized_delta,
            coverage_fraction=coverage_fraction,
        )
        low_variance = (feature_std is None) or (feature_std <= 1e-9)
        noisy = bool(
            len(values) < int(min_feature_rows)
            or low_variance
            or signal_score < 0.08
        )
        direction_conflict = bool(
            expected_direction in {"risk_up", "risk_down"}
            and observed_direction in {"risk_up", "risk_down"}
            and expected_direction != observed_direction
            and signal_score >= 0.10
        )

        evaluated.append(
            {
                "feature": feature_key,
                "family": _feature_family(feature_key),
                "rows_with_value": len(values),
                "coverage_fraction": coverage_fraction,
                "expected_direction": expected_direction,
                "observed_direction": observed_direction,
                "correlation_with_outcome": corr,
                "rank_correlation_with_outcome": rank_corr,
                "auc_if_risk_up": auc_risk_up,
                "auc_if_risk_down": auc_risk_down,
                "best_auc": auc_best,
                "best_auc_direction": auc_best_direction,
                "mean_when_adverse": mean_pos,
                "mean_when_non_adverse": mean_neg,
                "delta_adverse_minus_non_adverse": delta_pos_neg,
                "standardized_delta": standardized_delta,
                "feature_stddev": feature_std,
                "signal_score": signal_score,
                "noisy_or_weak": noisy,
                "potentially_harmful": direction_conflict,
                "potentially_harmful_reason": (
                    "direction_conflict_with_expected_risk_relationship"
                    if direction_conflict
                    else None
                ),
                "feature_vs_outcome_curve": _feature_curve_bins(values, labels, bins=5),
            }
        )

    direction_alignment = _build_direction_alignment(
        evaluated_rows=evaluated,
        min_signal_score=AUTO_DIRECTION_ALIGNMENT_MIN_SIGNAL_SCORE,
        min_rows=max(int(min_feature_rows), AUTO_DIRECTION_ALIGNMENT_MIN_ROWS),
    )
    aligned_overrides = (
        direction_alignment.get("aligned_expected_direction_overrides")
        if isinstance(direction_alignment.get("aligned_expected_direction_overrides"), dict)
        else {}
    )
    alignment_actions = (
        direction_alignment.get("alignment_actions_by_feature")
        if isinstance(direction_alignment.get("alignment_actions_by_feature"), dict)
        else {}
    )
    for row in evaluated:
        feature_name = str(row.get("feature") or "")
        expected_static = str(row.get("expected_direction") or "unknown")
        observed_direction = str(row.get("observed_direction") or "unknown")
        aligned_expected = str(aligned_overrides.get(feature_name) or expected_static)
        before_conflict = bool(
            expected_static in {"risk_up", "risk_down"}
            and observed_direction in {"risk_up", "risk_down"}
            and expected_static != observed_direction
        )
        after_conflict = bool(
            aligned_expected in {"risk_up", "risk_down"}
            and observed_direction in {"risk_up", "risk_down"}
            and aligned_expected != observed_direction
        )
        row["expected_direction_static"] = expected_static
        row["expected_direction_aligned"] = aligned_expected
        row["expected_direction"] = aligned_expected
        row["direction_conflict_before_alignment"] = before_conflict
        row["direction_conflict_after_alignment"] = after_conflict
        row["alignment_action"] = str(alignment_actions.get(feature_name) or "none")
        row["potentially_harmful_pre_alignment"] = before_conflict
        row["potentially_harmful"] = after_conflict
        row["potentially_harmful_reason"] = (
            "direction_conflict_after_alignment"
            if after_conflict
            else None
        )

    evaluated.sort(
        key=lambda row: (
            -float(row.get("signal_score") or 0.0),
            -float(row.get("coverage_fraction") or 0.0),
            str(row.get("feature") or ""),
        )
    )
    if len(evaluated) > int(max_features):
        evaluated = evaluated[: int(max_features)]

    top_predictive = [row for row in evaluated if int(row.get("rows_with_value") or 0) >= int(min_feature_rows)][:20]
    weak_noisy = sorted(
        evaluated,
        key=lambda row: (
            float(row.get("signal_score") or 0.0),
            -float(row.get("coverage_fraction") or 0.0),
            str(row.get("feature") or ""),
        ),
    )[:20]
    harmful = [
        row
        for row in evaluated
        if bool(row.get("potentially_harmful"))
    ][:20]

    family_summary: dict[str, Any] = {}
    for family in sorted({str(row.get("family") or "other") for row in evaluated}):
        bucket = [row for row in evaluated if str(row.get("family") or "other") == family]
        family_summary[family] = {
            "feature_count": len(bucket),
            "mean_signal_score": _mean([float(row.get("signal_score") or 0.0) for row in bucket]),
            "median_signal_score": _proxy_quantile(
                [float(row.get("signal_score") or 0.0) for row in bucket],
                0.5,
            ),
            "top_features": [
                {
                    "feature": row.get("feature"),
                    "signal_score": row.get("signal_score"),
                    "rank_correlation_with_outcome": row.get("rank_correlation_with_outcome"),
                    "coverage_fraction": row.get("coverage_fraction"),
                }
                for row in bucket[:5]
            ],
        }

    key_families = {
        "vegetation_metrics": family_summary.get("vegetation_metrics", {"feature_count": 0, "top_features": []}),
        "slope_terrain": family_summary.get("slope_terrain", {"feature_count": 0, "top_features": []}),
        "hazard_zone_context": family_summary.get("hazard_burn_context", {"feature_count": 0, "top_features": []}),
        "burn_probability": {
            "feature_count": len(
                [row for row in evaluated if "burn_probability" in str(row.get("feature") or "").lower()]
            ),
            "mean_signal_score": _mean(
                [
                    float(row.get("signal_score") or 0.0)
                    for row in evaluated
                    if "burn_probability" in str(row.get("feature") or "").lower()
                ]
            ),
            "median_signal_score": _proxy_quantile(
                [
                    float(row.get("signal_score") or 0.0)
                    for row in evaluated
                    if "burn_probability" in str(row.get("feature") or "").lower()
                ],
                0.5,
            ),
            "top_features": [
                {
                    "feature": row.get("feature"),
                    "signal_score": row.get("signal_score"),
                    "rank_correlation_with_outcome": row.get("rank_correlation_with_outcome"),
                    "coverage_fraction": row.get("coverage_fraction"),
                }
                for row in evaluated
                if "burn_probability" in str(row.get("feature") or "").lower()
            ][:5],
        },
        "structural_features": family_summary.get("structural_features", {"feature_count": 0, "top_features": []}),
    }

    return {
        "available": True,
        "row_count_used": total_rows,
        "feature_count_evaluated": len(evaluated),
        "near_zero_variance_feature_count": sum(
            1
            for row in evaluated
            if (_safe_float(row.get("feature_stddev")) is None) or (abs(float(_safe_float(row.get("feature_stddev")) or 0.0)) <= 1e-9)
        ),
        "features_with_variance_count": sum(
            1
            for row in evaluated
            if (_safe_float(row.get("feature_stddev")) is not None) and (abs(float(_safe_float(row.get("feature_stddev")) or 0.0)) > 1e-9)
        ),
        "method": {
            "signal_formula": (
                "signal_score = coverage * (0.45*max(|pearson|,|spearman|) + "
                "0.35*auc_signal + 0.20*standardized_delta_signal)"
            ),
            "auc_signal_definition": "abs(best_auc - 0.5) * 2",
            "plot_curve_definition": "feature-quantile bins with observed adverse outcome rate",
            "caveat": (
                "These diagnostics identify directional feature signal/noise in this dataset; "
                "they are not causal inference and not claims-grade validation."
            ),
        },
        "top_predictive_features": top_predictive,
        "weak_or_noisy_features": weak_noisy,
        "potentially_harmful_features": harmful,
        "direction_alignment": direction_alignment,
        "feature_vs_outcome_curves": [
            {
                "feature": row.get("feature"),
                "family": row.get("family"),
                "expected_direction": row.get("expected_direction"),
                "observed_direction": row.get("observed_direction"),
                "rows_with_value": row.get("rows_with_value"),
                "coverage_fraction": row.get("coverage_fraction"),
                "signal_score": row.get("signal_score"),
                "feature_vs_outcome_curve": row.get("feature_vs_outcome_curve"),
            }
            for row in top_predictive[:12]
        ],
        "key_feature_family_summary": key_families,
    }


def _build_data_sufficiency_flags(
    *,
    prepared_rows: list[dict[str, Any]],
    positive_rate: float | None,
    low_join_fraction: float,
    fallback_heavy_fraction: float,
    high_confidence_count: int,
    high_evidence_count: int,
) -> dict[str, Any]:
    n = len(prepared_rows)
    positives = sum(int(row.get("structure_loss_or_major_damage") or 0) for row in prepared_rows)
    flags = {
        "small_sample_size": n < 25,
        "very_small_sample_size": n < 10,
        "class_imbalance": bool(positive_rate is not None and (positive_rate < 0.1 or positive_rate > 0.9)),
        "low_join_confidence_prevalent": low_join_fraction >= 0.4,
        "fallback_heavy_prevalent": fallback_heavy_fraction >= 0.5,
        "no_high_confidence_rows": high_confidence_count == 0,
        "no_high_evidence_rows": high_evidence_count == 0,
    }
    return {
        "flags": flags,
        "sample_size": n,
        "positive_count": positives,
        "negative_count": max(0, n - positives),
        "positive_rate": positive_rate,
        "low_join_confidence_fraction": low_join_fraction,
        "fallback_heavy_fraction": fallback_heavy_fraction,
        "high_confidence_count": high_confidence_count,
        "high_evidence_count": high_evidence_count,
    }


def _build_narrative_summary(
    *,
    raw_auc: float | None,
    rank_hit_rate: float | None,
    data_sufficiency: dict[str, Any],
    by_evidence_group: dict[str, Any],
    metric_stability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flags = (data_sufficiency.get("flags") or {}) if isinstance(data_sufficiency, dict) else {}
    stability = metric_stability if isinstance(metric_stability, dict) else {}
    lines: list[str] = []
    if not bool(stability.get("auc_stable", True)):
        lines.append("Insufficient data for stable AUC/PR-AUC; treat discrimination values as unstable.")
    if isinstance(raw_auc, float):
        if raw_auc >= 0.7:
            lines.append("Model shows directional discrimination in this sample.")
        elif raw_auc >= 0.6:
            lines.append("Model shows weak-to-moderate directional discrimination in this sample.")
        else:
            lines.append("Directional discrimination appears weak in this sample.")
    elif isinstance(rank_hit_rate, float):
        if rank_hit_rate >= 0.6:
            lines.append("Directional separation observed in rank ordering despite limited sample size.")
        else:
            lines.append("Pairwise rank ordering is weak; directional signal is limited in this sample.")
    else:
        lines.append("Directional discrimination cannot be established from this sample.")

    if bool(flags.get("small_sample_size")):
        lines.append("Dataset too small for calibration trust; treat calibration/error estimates as provisional.")
    if bool(flags.get("class_imbalance")):
        lines.append("Class imbalance is present and may distort threshold-based metrics.")
    if bool(flags.get("low_join_confidence_prevalent")):
        lines.append("A substantial share of rows rely on low-confidence joins.")
    if bool(flags.get("fallback_heavy_prevalent")):
        lines.append("Fallback-heavy evidence is common and reduces reliability.")

    evidence_text = None
    if isinstance(by_evidence_group, dict):
        high = by_evidence_group.get("high_evidence") if isinstance(by_evidence_group.get("high_evidence"), dict) else {}
        low = by_evidence_group.get("fallback_heavy") if isinstance(by_evidence_group.get("fallback_heavy"), dict) else {}
        high_auc = high.get("wildfire_risk_score_auc")
        low_auc = low.get("wildfire_risk_score_auc")
        if isinstance(high_auc, (int, float)) and isinstance(low_auc, (int, float)):
            if float(high_auc) > float(low_auc):
                evidence_text = "Performance is stronger on higher-evidence rows than fallback-heavy rows."
            elif float(high_auc) < float(low_auc):
                evidence_text = "Fallback-heavy rows currently score better than high-evidence rows; verify sample composition."
    if evidence_text:
        lines.append(evidence_text)

    headline = lines[0] if lines else "Directional validation summary unavailable."
    return {"headline": headline, "bullets": lines}


def _proxy_feature_value(row: dict[str, Any], key: str) -> float | None:
    raw = row.get("raw_feature_vector") if isinstance(row.get("raw_feature_vector"), dict) else {}
    transformed = row.get("transformed_feature_vector") if isinstance(row.get("transformed_feature_vector"), dict) else {}
    if key in raw:
        value = _safe_float(raw.get(key))
        if value is not None:
            return value
    if key in transformed:
        value = _safe_float(transformed.get(key))
        if value is not None:
            return value
    return _safe_float(row.get(key))


def _proxy_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    clean = sorted(float(v) for v in values)
    if len(clean) == 1:
        return clean[0]
    pct = max(0.0, min(1.0, float(q)))
    idx = pct * (len(clean) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return clean[lo]
    frac = idx - lo
    return clean[lo] + (clean[hi] - clean[lo]) * frac


def _pairwise_hit_rate_from_scores(positive_scores: list[float], negative_scores: list[float]) -> dict[str, Any]:
    pair_count = len(positive_scores) * len(negative_scores)
    if pair_count <= 0:
        return {"available": False, "pair_count": pair_count, "hit_rate": None}
    wins = 0.0
    ties = 0
    for pos in positive_scores:
        for neg in negative_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
                ties += 1
    hit_rate = wins / float(pair_count)
    return {
        "available": True,
        "pair_count": pair_count,
        "hit_rate": hit_rate,
        "ties_fraction": ties / float(pair_count),
        "confidence_interval_95": _wilson_interval(int(round(wins)), pair_count),
    }


def _compute_proxy_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_ranges: dict[str, tuple[float, float, str]] = {}
    for key in PROXY_RISK_UP_FEATURE_KEYS:
        values = [
            _proxy_feature_value(row, key)
            for row in rows
            if _proxy_feature_value(row, key) is not None
        ]
        clean = [float(v) for v in values if v is not None]
        if len(clean) >= 2 and max(clean) > min(clean):
            feature_ranges[key] = (min(clean), max(clean), "risk_up")
    for key in PROXY_RISK_DOWN_FEATURE_KEYS:
        values = [
            _proxy_feature_value(row, key)
            for row in rows
            if _proxy_feature_value(row, key) is not None
        ]
        clean = [float(v) for v in values if v is not None]
        if len(clean) >= 2 and max(clean) > min(clean):
            feature_ranges[key] = (min(clean), max(clean), "risk_down")

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        components: list[float] = []
        for key, (lo, hi, direction) in feature_ranges.items():
            value = _proxy_feature_value(row, key)
            if value is None:
                continue
            norm = (float(value) - lo) / float(hi - lo)
            norm = max(0.0, min(1.0, norm))
            if direction == "risk_down":
                norm = 1.0 - norm
            components.append(norm)
        if len(components) < 2:
            continue
        scored_rows.append(
            {
                "record_id": row.get("record_id"),
                "event_id": row.get("event_id"),
                "wildfire_risk_score": float(row.get("wildfire_risk_score") or 0.0),
                "proxy_risk_index": sum(components) / float(len(components)),
            }
        )

    if len(scored_rows) < 3:
        return {
            "available": False,
            "caveat": "Proxy validation uses weak labels from environmental proxies and is not ground-truth validation.",
            "reason": "insufficient_proxy_feature_coverage",
            "feature_count_used": len(feature_ranges),
            "rows_with_proxy_index": len(scored_rows),
        }

    proxy_values = [float(row["proxy_risk_index"]) for row in scored_rows]
    low_cut = _proxy_quantile(proxy_values, 0.33)
    high_cut = _proxy_quantile(proxy_values, 0.67)
    if low_cut is None or high_cut is None or high_cut <= low_cut:
        return {
            "available": False,
            "caveat": "Proxy validation uses weak labels from environmental proxies and is not ground-truth validation.",
            "reason": "insufficient_proxy_separation",
            "feature_count_used": len(feature_ranges),
            "rows_with_proxy_index": len(scored_rows),
        }

    weak_labeled_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        idx = float(row["proxy_risk_index"])
        if idx >= high_cut:
            weak_label = 1
            weak_class = "high_proxy_risk"
        elif idx <= low_cut:
            weak_label = 0
            weak_class = "low_proxy_risk"
        else:
            weak_label = None
            weak_class = "mid_proxy_risk"
        weak_labeled_rows.append({**row, "weak_proxy_label": weak_label, "weak_proxy_class": weak_class})

    high_rows = [row for row in weak_labeled_rows if row.get("weak_proxy_label") == 1]
    low_rows = [row for row in weak_labeled_rows if row.get("weak_proxy_label") == 0]
    usable = high_rows + low_rows

    if len(usable) < 4:
        return {
            "available": False,
            "caveat": "Proxy validation uses weak labels from environmental proxies and is not ground-truth validation.",
            "reason": "insufficient_high_low_proxy_labels",
            "feature_count_used": len(feature_ranges),
            "rows_with_proxy_index": len(scored_rows),
            "weak_label_counts": {
                "high_proxy_risk": len(high_rows),
                "low_proxy_risk": len(low_rows),
                "mid_proxy_risk": len(weak_labeled_rows) - len(usable),
            },
        }

    y_true = [int(row["weak_proxy_label"]) for row in usable]
    y_score = [max(0.0, min(1.0, float(row["wildfire_risk_score"]) / 100.0)) for row in usable]
    raw_scores = [float(row["wildfire_risk_score"]) for row in usable]
    proxy_idx = [float(row["proxy_risk_index"]) for row in usable]
    threshold = 70.0
    pred = [1 if score >= threshold else 0 for score in raw_scores]
    conf = _confusion(y_true, pred)
    top_bucket_n = max(1, int(math.ceil(0.2 * len(scored_rows))))
    top_bucket = sorted(scored_rows, key=lambda row: float(row["wildfire_risk_score"]), reverse=True)[:top_bucket_n]
    top_bucket_high = sum(1 for row in top_bucket if float(row["proxy_risk_index"]) >= high_cut)
    overall_high = sum(1 for row in scored_rows if float(row["proxy_risk_index"]) >= high_cut)
    overall_high_rate = overall_high / float(len(scored_rows))
    top_bucket_high_rate = top_bucket_high / float(top_bucket_n)
    lift = (top_bucket_high_rate / overall_high_rate) if overall_high_rate > 0 else None

    return {
        "available": True,
        "evaluation_basis": "weak_proxy_labels",
        "caveat": "Proxy validation uses weak labels from environmental proxies and is not ground-truth validation.",
        "feature_count_used": len(feature_ranges),
        "features_used": sorted(feature_ranges.keys()),
        "rows_with_proxy_index": len(scored_rows),
        "weak_label_counts": {
            "high_proxy_risk": len(high_rows),
            "low_proxy_risk": len(low_rows),
            "mid_proxy_risk": len(weak_labeled_rows) - len(usable),
            "usable_high_low_rows": len(usable),
        },
        "proxy_thresholds": {"low_cut": low_cut, "high_cut": high_cut},
        "alignment_metrics": {
            "spearman_model_vs_proxy_index": _spearman(
                [float(row["wildfire_risk_score"]) for row in scored_rows],
                [float(row["proxy_risk_index"]) for row in scored_rows],
            ),
            "auc_model_vs_proxy_high_low": _roc_auc(y_true, y_score),
            "pr_auc_model_vs_proxy_high_low": _pr_auc(y_true, y_score),
            "rank_order_hit_rate_high_vs_low_proxy": _pairwise_hit_rate_from_scores(
                [float(row["wildfire_risk_score"]) for row in high_rows],
                [float(row["wildfire_risk_score"]) for row in low_rows],
            ),
            "accuracy_at_threshold_70_for_proxy_high": {
                "threshold": threshold,
                "confusion_matrix": conf,
                **_precision_recall(conf),
            },
            "top_risk_bucket_proxy_high_rate": {
                "bucket_definition": "top_20_percent_by_model_score",
                "bucket_count": top_bucket_n,
                "proxy_high_rate": top_bucket_high_rate,
                "overall_proxy_high_rate": overall_high_rate,
                "lift_vs_overall": lift,
            },
        },
    }


def _run_synthetic_stress_validation() -> dict[str, Any]:
    try:
        from backend.model_tuning import run_monotonic_guardrails
        from backend.model_tuning import _build_guardrail_context  # type: ignore
        from backend.models import PropertyAttributes
        from backend.risk_engine import RiskEngine
        from backend.scoring_config import load_scoring_config

        config = load_scoring_config()
        guardrails = run_monotonic_guardrails(config)
        checks = guardrails.get("checks") if isinstance(guardrails.get("checks"), list) else []
        passed = [row for row in checks if isinstance(row, dict) and bool(row.get("passed"))]
        failed = [row for row in checks if isinstance(row, dict) and not bool(row.get("passed"))]

        engine = RiskEngine(config)
        low_context = _build_guardrail_context(
            burn_probability=15.0,
            hazard=18.0,
            fuel=20.0,
            canopy=18.0,
            slope=8.0,
            wildland_distance=88.0,
            historic_fire=12.0,
            ring_0_5=10.0,
            ring_5_30=14.0,
            ring_30_100=18.0,
        )
        high_context = _build_guardrail_context(
            burn_probability=92.0,
            hazard=94.0,
            fuel=88.0,
            canopy=86.0,
            slope=52.0,
            wildland_distance=18.0,
            historic_fire=84.0,
            ring_0_5=90.0,
            ring_5_30=86.0,
            ring_30_100=82.0,
        )
        low_attrs = PropertyAttributes(
            roof_type="class a",
            vent_type="ember-resistant",
            defensible_space_ft=70.0,
            construction_year=2018,
        )
        high_attrs = PropertyAttributes(
            roof_type="wood",
            vent_type="standard",
            defensible_space_ft=3.0,
            construction_year=1975,
        )
        low_risk = engine.score(low_attrs, 46.0, -114.0, low_context)
        high_risk = engine.score(high_attrs, 46.0, -114.0, high_context)
        low_total = engine.compute_blended_wildfire_score(
            engine.compute_site_hazard_score(low_risk),
            engine.compute_home_ignition_vulnerability_score(low_risk),
        )
        high_total = engine.compute_blended_wildfire_score(
            engine.compute_site_hazard_score(high_risk),
            engine.compute_home_ignition_vulnerability_score(high_risk),
        )
        minimum_expected_delta = 10.0
        extreme_ranking_passed = bool(high_total >= (low_total + minimum_expected_delta))

        return {
            "available": True,
            "evaluation_basis": "synthetic_stress_scenarios",
            "caveat": "Synthetic stress validation checks directional behavior and is not real-outcome ground truth.",
            "passed": bool(guardrails.get("passed")) and extreme_ranking_passed,
            "check_count": len(checks),
            "pass_count": len(passed),
            "fail_count": len(failed),
            "checks": checks,
            "extreme_scenario_ranking": {
                "passed": extreme_ranking_passed,
                "minimum_expected_delta": minimum_expected_delta,
                "high_risk_score": round(float(high_total), 4),
                "low_risk_score": round(float(low_total), 4),
                "delta": round(float(high_total - low_total), 4),
                "scenarios": {
                    "high_risk": {
                        "context": "high_burn_probability_high_hazard_low_wildland_distance_high_ring_vegetation",
                        "attributes": "wood_roof_standard_vents_minimal_defensible_space",
                    },
                    "low_risk": {
                        "context": "low_burn_probability_low_hazard_high_wildland_distance_low_ring_vegetation",
                        "attributes": "class_a_roof_ember_resistant_vents_large_defensible_space",
                    },
                },
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "evaluation_basis": "synthetic_stress_scenarios",
            "caveat": "Synthetic stress validation checks directional behavior and is not real-outcome ground truth.",
            "error": str(exc),
        }


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
    *,
    allow_label_derived_target: bool = True,
    allow_surrogate_wildfire_score: bool = True,
    use_high_signal_simplified_model: bool = False,
    min_join_confidence_score_for_metrics: float | None = None,
    retain_unusable_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    missing_required = Counter()
    invalid_examples: list[dict[str, Any]] = []
    exclusion_reason_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for row in rows:
        label = _normalize_label(row.get("outcome_label"))
        target_int = _safe_int(row.get("structure_loss_or_major_damage"))
        if target_int not in {0, 1} and allow_label_derived_target:
            target_int = _derive_target_from_label(label)

        wildfire_score = _extract_score(row, "wildfire_risk_score")
        wildfire_score_source = "direct"
        if wildfire_score is None and allow_surrogate_wildfire_score:
            wildfire_score = _derive_surrogate_wildfire_score(row)
            if wildfire_score is not None:
                wildfire_score_source = "surrogate_from_site_and_vulnerability"
        simplified_score, simplified_meta = _compute_high_signal_simplified_score(row)
        if use_high_signal_simplified_model and simplified_score is not None:
            wildfire_score = simplified_score
            wildfire_score_source = "high_signal_simplified_model"

        fallback_flags = _extract_fallback_flags(row)
        evidence_tier = _extract_evidence_tier(row)
        confidence_tier = _extract_confidence_tier(row)
        join_tier = _extract_join_confidence_tier(row)
        join_score = _extract_join_confidence_score(row)
        evidence_group = _derive_evidence_group(
            evidence_tier=evidence_tier,
            fallback_flags=fallback_flags,
        )
        fallback_usage_summary = _derive_fallback_usage_summary(
            evidence_tier=evidence_tier,
            fallback_flags=fallback_flags,
        )

        exclusion_reasons: list[str] = []
        if target_int not in {0, 1}:
            missing_required["structure_loss_or_major_damage"] += 1
            exclusion_reasons.append("missing_or_invalid_structure_loss_or_major_damage")
        if wildfire_score is None:
            missing_required["scores.wildfire_risk_score"] += 1
            exclusion_reasons.append("missing_scores.wildfire_risk_score")
        if (
            min_join_confidence_score_for_metrics is not None
            and join_score is not None
            and float(join_score) < float(min_join_confidence_score_for_metrics)
        ):
            missing_required["join_confidence_below_min"] += 1
            exclusion_reasons.append("join_confidence_below_min")
        if (
            min_join_confidence_score_for_metrics is not None
            and join_score is None
        ):
            missing_required["join_confidence_missing"] += 1
            exclusion_reasons.append("join_confidence_missing")

        if exclusion_reasons:
            for reason in exclusion_reasons:
                exclusion_reason_counts[reason] += 1
            if len(invalid_examples) < 24:
                invalid_examples.append(
                    {
                        "record_id": row.get("record_id"),
                        "reasons": exclusion_reasons,
                    }
                )

        outcome_rank = _safe_int(row.get("outcome_rank"))
        if outcome_rank is None:
            outcome_rank = OUTCOME_RANKS.get(label, 0)

        low_confidence_join = bool(
            join_tier == "low"
            or "low_confidence_join" in (row.get("caveat_flags") or [])
            or (
                join_score is not None
                and min_join_confidence_score_for_metrics is not None
                and float(join_score) < float(min_join_confidence_score_for_metrics)
            )
        )
        missing_features = bool(
            fallback_flags["missing_factor_count"] > 0
            or fallback_flags["coverage_failed_count"] > 0
            or fallback_flags["coverage_fallback_count"] >= FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD
            or float(fallback_flags.get("fallback_weight_fraction") or 0.0) >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
            or wildfire_score_source != "direct"
        )
        fallback_heavy = evidence_group == "fallback_heavy"

        prepared_row = dict(row)
        prepared_row["outcome_label"] = label
        prepared_row["outcome_rank"] = int(outcome_rank or 0)
        prepared_row["structure_loss_or_major_damage"] = (int(target_int) if target_int in {0, 1} else None)
        prepared_row["wildfire_risk_score"] = (float(wildfire_score) if wildfire_score is not None else None)
        prepared_row["wildfire_probability_proxy"] = (
            max(0.0, min(1.0, float(wildfire_score) / 100.0))
            if wildfire_score is not None
            else None
        )
        prepared_row["calibrated_damage_likelihood"] = _extract_score(row, "calibrated_damage_likelihood")
        prepared_row["confidence_tier"] = confidence_tier
        prepared_row["evidence_quality_tier"] = evidence_tier
        prepared_row["evidence_group"] = evidence_group
        prepared_row["region_id"] = _extract_region_id(row)
        prepared_row["fallback_default_flags"] = fallback_flags
        prepared_row["fallback_usage"] = fallback_usage_summary
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
        prepared_row["hazard_level_segment"] = _derive_hazard_level_segment(prepared_row)
        prepared_row["vegetation_density_segment"] = _derive_vegetation_density_segment(prepared_row)
        prepared_row["low_confidence_join"] = low_confidence_join
        prepared_row["missing_features"] = missing_features
        prepared_row["fallback_heavy"] = fallback_heavy
        prepared_row["wildfire_score_source"] = wildfire_score_source
        prepared_row["high_signal_simplified_score"] = simplified_score
        prepared_row["high_signal_model_components"] = simplified_meta
        prepared_row["row_usable_for_metrics"] = len(exclusion_reasons) == 0
        prepared_row["exclusion_reasons"] = exclusion_reasons

        if low_confidence_join:
            flag_counts["low_confidence_join"] += 1
        if missing_features:
            flag_counts["missing_features"] += 1
        if fallback_heavy:
            flag_counts["fallback_heavy"] += 1
        if not prepared_row["row_usable_for_metrics"]:
            flag_counts["row_unusable_for_metrics"] += 1

        if prepared_row["row_usable_for_metrics"] or retain_unusable_rows:
            prepared.append(prepared_row)

    return prepared, {
        "missing_required_fields": dict(missing_required),
        "invalid_row_examples": invalid_examples,
        "exclusion_reason_counts": dict(exclusion_reason_counts),
        "row_flag_counts": dict(flag_counts),
        "retained_row_count": len(prepared),
        "usable_row_count": sum(1 for row in prepared if bool(row.get("row_usable_for_metrics"))),
        "unusable_row_count": sum(1 for row in prepared if not bool(row.get("row_usable_for_metrics"))),
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
    allow_label_derived_target: bool = True,
    allow_surrogate_wildfire_score: bool = True,
    use_high_signal_simplified_model: bool = False,
    min_join_confidence_score_for_metrics: float | None = None,
    retain_unusable_rows: bool = True,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_rows, validation = _prepare_rows(
        rows,
        allow_label_derived_target=bool(allow_label_derived_target),
        allow_surrogate_wildfire_score=bool(allow_surrogate_wildfire_score),
        use_high_signal_simplified_model=bool(use_high_signal_simplified_model),
        min_join_confidence_score_for_metrics=(
            float(min_join_confidence_score_for_metrics)
            if min_join_confidence_score_for_metrics is not None
            else None
        ),
        retain_unusable_rows=bool(retain_unusable_rows),
    )
    usable_rows = [row for row in prepared_rows if bool(row.get("row_usable_for_metrics"))]
    if len(usable_rows) < max(1, int(min_labeled_rows)):
        missing = validation.get("missing_required_fields") or {}
        raise ValueError(
            "Not enough usable labeled rows for evaluation. "
            f"usable_rows={len(usable_rows)} min_labeled_rows={max(1, int(min_labeled_rows))} "
            f"missing_required_counts={missing}"
        )

    y_true = [int(row["structure_loss_or_major_damage"]) for row in usable_rows]
    raw_probs = [float(row["wildfire_probability_proxy"]) for row in usable_rows]
    raw_scores = [float(row["wildfire_risk_score"]) for row in usable_rows]
    outcome_ranks = [float(row["outcome_rank"]) for row in usable_rows]
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

    for row, pred, truth in zip(usable_rows, default_preds, y_true):
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
        for row in usable_rows
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
    for row in usable_rows:
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
    low_join_count = sum(
        len(bucket)
        for key, bucket in join_tier_groups.items()
        if str(key).strip().lower() == "low"
    )
    low_join_fraction = (low_join_count / float(len(usable_rows))) if usable_rows else 0.0

    fallback_heavy_count = sum(1 for row in usable_rows if str(row.get("evidence_group")) == "fallback_heavy")
    fallback_heavy_fraction = (fallback_heavy_count / float(len(usable_rows))) if usable_rows else 0.0
    retained_fallback_heavy_count = sum(
        1 for row in prepared_rows if str(row.get("evidence_group")) == "fallback_heavy"
    )
    retained_fallback_heavy_fraction = (
        retained_fallback_heavy_count / float(len(prepared_rows))
        if prepared_rows
        else 0.0
    )
    rows_with_elevated_fallback_weight = sum(
        1
        for row in usable_rows
        if float((row.get("fallback_default_flags") or {}).get("fallback_weight_fraction") or 0.0)
        >= FALLBACK_HEAVY_ELEVATED_WEIGHT_THRESHOLD
    )
    rows_with_multi_layer_fallback = sum(
        1
        for row in usable_rows
        if int((row.get("fallback_default_flags") or {}).get("coverage_fallback_count") or 0)
        >= FALLBACK_HEAVY_COVERAGE_FALLBACK_COUNT_THRESHOLD
    )
    fallback_stats = {
        "rows_with_any_fallback_flag": sum(
            1
            for row in usable_rows
            if int((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0) > 0
            or int((row.get("fallback_default_flags") or {}).get("coverage_fallback_count") or 0) > 0
        ),
        "rows_with_elevated_fallback_weight": rows_with_elevated_fallback_weight,
        "rows_with_multi_layer_fallback": rows_with_multi_layer_fallback,
        "mean_fallback_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("fallback_factor_count") or 0) for row in usable_rows]
        ),
        "mean_missing_factor_count": _mean(
            [float((row.get("fallback_default_flags") or {}).get("missing_factor_count") or 0) for row in usable_rows]
        ),
        "fallback_heavy_count": fallback_heavy_count,
        "fallback_heavy_fraction": fallback_heavy_fraction,
        "retained_fallback_heavy_count": retained_fallback_heavy_count,
        "retained_fallback_heavy_fraction": round(retained_fallback_heavy_fraction, 4),
    }

    raw_auc = _roc_auc(y_true, raw_probs)
    raw_pr_auc = _pr_auc(y_true, raw_probs)
    baseline_model_comparison = _compute_simple_baseline_metrics(
        y_true=y_true,
        usable_rows=usable_rows,
        full_model_auc=raw_auc,
    )
    auc_ci = _bootstrap_metric_ci(y_true=y_true, y_score=raw_probs, metric="auc")
    pr_auc_ci = _bootstrap_metric_ci(y_true=y_true, y_score=raw_probs, metric="pr_auc")
    brier_ci = _bootstrap_metric_ci(y_true=y_true, y_score=raw_probs, metric="brier")
    metric_stability = _build_metric_stability(y_true=y_true, raw_auc=raw_auc)
    leak_risks = _detect_data_leakage_risks(usable_rows, raw_auc)
    guardrail_warnings = _build_guardrail_warnings(
        row_count=len(usable_rows),
        positive_rate=positive_rate,
        fallback_heavy_fraction=fallback_heavy_fraction,
        leak_warnings=list(leak_risks.get("warnings") or []),
    )
    review_sets = _false_review_sets(
        rows=usable_rows,
        false_low_max_score=float(false_low_max_score),
        false_high_min_score=float(false_high_min_score),
    )
    high_confidence_rows = [row for row in usable_rows if str(row.get("validation_confidence_tier")) == "high-confidence"]
    medium_confidence_rows = [row for row in usable_rows if str(row.get("validation_confidence_tier")) == "medium-confidence"]
    high_evidence_rows = [row for row in usable_rows if str(row.get("evidence_group")) == "high_evidence"]
    confidence_tier_performance = _build_confidence_tier_performance(
        all_rows=usable_rows,
        high_confidence_rows=high_confidence_rows,
        medium_confidence_rows=medium_confidence_rows,
        min_slice_size=min_slice_size,
    )
    feature_signal_diagnostics = _build_feature_signal_diagnostics(rows=usable_rows)
    modeling_viability = _build_modeling_viability(
        prepared_rows=usable_rows,
        feature_signal_diagnostics=feature_signal_diagnostics,
        baseline_model_comparison=baseline_model_comparison,
    )
    if not high_confidence_rows:
        guardrail_warnings.append("No high-confidence rows in this run; prioritize stronger join/feature coverage before calibration decisions.")
    if not high_evidence_rows:
        guardrail_warnings.append("No high-evidence rows in this run; results are dominated by inferred/fallback-heavy evidence.")
    if not bool(modeling_viability.get("dataset_viable_for_predictive_modeling")):
        guardrail_warnings.append(
            "Dataset not viable for predictive modeling in this run: "
            + str(modeling_viability.get("reason") or "viability guardrail check failed.")
        )
    guardrail_warnings.extend(list(confidence_tier_performance.get("warnings") or []))
    guardrail_warnings.extend(list(metric_stability.get("warnings") or []))
    guardrail_warnings = sorted(set(str(item) for item in guardrail_warnings if str(item).strip()))
    minimum_viable_metrics = _compute_minimum_viable_metrics(
        prepared_rows=usable_rows,
        default_threshold=default_threshold,
        default_confusion=default_conf,
    )
    data_sufficiency_flags = _build_data_sufficiency_flags(
        prepared_rows=usable_rows,
        positive_rate=positive_rate,
        low_join_fraction=low_join_fraction,
        fallback_heavy_fraction=fallback_heavy_fraction,
        high_confidence_count=len(high_confidence_rows),
        high_evidence_count=len(high_evidence_rows),
    )
    data_sufficiency_indicator = {
        "thresholds": {
            "insufficient_max_exclusive": DATA_SUFFICIENCY_LIMITED_MIN,
            "limited_max_exclusive": DATA_SUFFICIENCY_MODERATE_MIN,
            "moderate_max_inclusive": DATA_SUFFICIENCY_STRONG_MIN,
            "strong_min_exclusive": DATA_SUFFICIENCY_STRONG_MIN,
        },
        "total_dataset": _data_sufficiency_indicator(len(usable_rows)),
        "high_confidence_subset": _data_sufficiency_indicator(len(high_confidence_rows)),
    }

    directional_predictive_value = bool(
        raw_auc is not None
        and raw_auc >= 0.6
        and bool(modeling_viability.get("dataset_viable_for_predictive_modeling"))
    )
    calibration_recommendation = (
        "candidate"
        if directional_predictive_value
        and raw_ece is not None
        and raw_ece <= 0.12
        and fallback_heavy_fraction < 0.5
        and len(usable_rows) >= 100
        else "not_recommended_yet"
    )
    slice_metrics_data = {
        "by_confidence_tier": _slice_metrics(
            usable_rows,
            slice_key="confidence_tier",
            min_slice_size=min_slice_size,
        ),
        "by_evidence_quality_tier": _slice_metrics(
            usable_rows,
            slice_key="evidence_quality_tier",
            min_slice_size=min_slice_size,
        ),
        "by_evidence_group": _slice_metrics(
            usable_rows,
            slice_key="evidence_group",
            min_slice_size=min_slice_size,
        ),
        "by_join_confidence_tier": _slice_metrics(
            usable_rows,
            slice_key="join_confidence_tier",
            min_slice_size=min_slice_size,
        ),
        "by_validation_confidence_tier": _slice_metrics(
            usable_rows,
            slice_key="validation_confidence_tier",
            min_slice_size=min_slice_size,
        ),
        "by_fallback_status": _slice_metrics(
            usable_rows,
            slice_key="fallback_status",
            min_slice_size=min_slice_size,
        ),
        "by_hazard_level": _slice_metrics(
            usable_rows,
            slice_key="hazard_level_segment",
            min_slice_size=min_slice_size,
        ),
        "by_vegetation_density": _slice_metrics(
            usable_rows,
            slice_key="vegetation_density_segment",
            min_slice_size=min_slice_size,
        ),
        "by_region": _slice_metrics(
            usable_rows,
            slice_key="region_id",
            min_slice_size=min_slice_size,
        ),
    }
    segment_performance_summary = _build_segment_performance_summary(
        slice_metrics=slice_metrics_data,
        min_slice_size=min_slice_size,
    )

    report = {
        "schema_version": "1.1.0",
        "generated_at": generated_at or _utc_now_iso(),
        "row_count_labeled": len(usable_rows),
        "sample_counts": {
            "row_count_total": len(rows),
            "row_count_retained": len(prepared_rows),
            "row_count_usable": len(usable_rows),
            "row_count_unusable": max(0, len(prepared_rows) - len(usable_rows)),
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
            "filter_config": {
                "allow_label_derived_target": bool(allow_label_derived_target),
                "allow_surrogate_wildfire_score": bool(allow_surrogate_wildfire_score),
                "use_high_signal_simplified_model": bool(use_high_signal_simplified_model),
                "min_join_confidence_score_for_metrics": (
                    float(min_join_confidence_score_for_metrics)
                    if min_join_confidence_score_for_metrics is not None
                    else None
                ),
                "retain_unusable_rows": bool(retain_unusable_rows),
            },
        },
        "score_model": {
            "active_model": (
                "high_signal_simplified_model"
                if bool(use_high_signal_simplified_model)
                else "raw_wildfire_risk_score"
            ),
            "high_signal_model_weights": (
                {
                    key: (float(weight) / float(sum(HIGH_SIGNAL_MODEL_SIGNAL_WEIGHTS.values())))
                    for key, weight in HIGH_SIGNAL_MODEL_SIGNAL_WEIGHTS.items()
                }
                if bool(use_high_signal_simplified_model)
                else {}
            ),
            "high_signal_model_directionality": (
                {
                    "nearest_high_fuel_patch_distance_ft": "risk_up_via_inverse_distance_transform",
                    "canopy_adjacency_proxy_pct": "risk_down_observed_signal_alignment",
                    "vegetation_continuity_proxy_pct": "risk_up",
                    "slope_index": "risk_up",
                }
                if bool(use_high_signal_simplified_model)
                else {}
            ),
        },
        "discrimination_metrics": {
            "wildfire_risk_score_auc": raw_auc,
            "wildfire_risk_score_pr_auc": raw_pr_auc,
            "wildfire_risk_score_auc_confidence_interval_95": auc_ci,
            "wildfire_risk_score_pr_auc_confidence_interval_95": pr_auc_ci,
            "wildfire_discrimination_stability": (
                "stable" if bool(metric_stability.get("auc_stable")) else "unstable_small_sample"
            ),
            "site_hazard_score_auc": _roc_auc(
                y_true,
                [max(0.0, min(1.0, (_extract_score(row, "site_hazard_score") or 0.0) / 100.0)) for row in usable_rows],
            ),
            "home_ignition_vulnerability_score_auc": _roc_auc(
                y_true,
                [
                    max(
                        0.0,
                        min(1.0, (_extract_score(row, "home_ignition_vulnerability_score") or 0.0) / 100.0),
                    )
                    for row in usable_rows
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
            "wildfire_probability_proxy_confidence_interval_95": brier_ci,
            "calibrated_damage_likelihood": (
                _brier(calibrated_y, calibrated_probs) if calibrated_y and calibrated_probs else None
            ),
        },
        "score_distributions_by_outcome": {
            "wildfire_risk_score": _distribution_by_outcome(usable_rows, "wildfire_risk_score"),
            "site_hazard_score": _distribution_by_outcome(usable_rows, "site_hazard_score"),
            "home_ignition_vulnerability_score": _distribution_by_outcome(
                usable_rows, "home_ignition_vulnerability_score"
            ),
            "insurance_readiness_score": _distribution_by_outcome(usable_rows, "insurance_readiness_score"),
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
        "slice_metrics": slice_metrics_data,
        "segment_performance_summary": segment_performance_summary,
        "subset_metrics": {
            "full_dataset": _compute_subset_metrics(usable_rows),
            "high_confidence_subset": _compute_subset_metrics(high_confidence_rows),
            "medium_confidence_subset": _compute_subset_metrics(medium_confidence_rows),
            "high_evidence_subset": _compute_subset_metrics(high_evidence_rows),
        },
        "confidence_tier_performance": confidence_tier_performance,
        "minimum_viable_metrics": minimum_viable_metrics,
        "data_sufficiency_flags": data_sufficiency_flags,
        "data_sufficiency_indicator": data_sufficiency_indicator,
        "fallback_diagnostics": fallback_stats,
        "factor_contribution_summary_by_confusion_class": _factor_summary_by_confusion_class(usable_rows),
        "false_review_sets": review_sets,
        "data_leakage_risks": leak_risks,
        "guardrails": {
            "warnings": guardrail_warnings,
            "small_sample_warning": any("sample" in w.lower() for w in guardrail_warnings),
            "fallback_heavy_warning": fallback_heavy_fraction >= 0.5,
            "leakage_warning": bool(leak_risks.get("warnings")),
        },
        "metric_stability": metric_stability,
        "directional_predictive_value": directional_predictive_value,
        "calibration_artifact_recommendation": calibration_recommendation,
        "feature_signal_diagnostics": feature_signal_diagnostics,
        "direction_alignment": (
            feature_signal_diagnostics.get("direction_alignment")
            if isinstance(feature_signal_diagnostics.get("direction_alignment"), dict)
            else {"available": False, "reason": "not_available"}
        ),
        "modeling_viability": modeling_viability,
        "baseline_model_comparison": baseline_model_comparison,
    }
    report["narrative_summary"] = _build_narrative_summary(
        raw_auc=raw_auc,
        rank_hit_rate=((minimum_viable_metrics.get("rank_ordering") or {}).get("hit_rate") if isinstance(minimum_viable_metrics, dict) else None),
        data_sufficiency=data_sufficiency_flags,
        by_evidence_group=(report.get("slice_metrics") or {}).get("by_evidence_group") if isinstance(report.get("slice_metrics"), dict) else {},
        metric_stability=metric_stability,
    )
    proxy_validation = _compute_proxy_validation(usable_rows)
    synthetic_validation = _run_synthetic_stress_validation()
    report["proxy_validation"] = proxy_validation
    report["synthetic_validation"] = synthetic_validation
    report["validation_streams"] = {
        "real_outcome_validation": {
            "available": True,
            "row_count_labeled": len(usable_rows),
            "caveat": "Real-outcome validation uses public observed outcomes and remains directional rather than claims-grade truth.",
        },
        "proxy_validation": {
            "available": bool(proxy_validation.get("available")),
            "caveat": str(proxy_validation.get("caveat") or ""),
        },
        "synthetic_validation": {
            "available": bool(synthetic_validation.get("available")),
            "caveat": str(synthetic_validation.get("caveat") or ""),
        },
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
    allow_label_derived_target: bool = True,
    allow_surrogate_wildfire_score: bool = True,
    use_high_signal_simplified_model: bool = False,
    min_join_confidence_score_for_metrics: float | None = None,
    retain_unusable_rows: bool = True,
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
        allow_label_derived_target=allow_label_derived_target,
        allow_surrogate_wildfire_score=allow_surrogate_wildfire_score,
        use_high_signal_simplified_model=use_high_signal_simplified_model,
        min_join_confidence_score_for_metrics=min_join_confidence_score_for_metrics,
        retain_unusable_rows=retain_unusable_rows,
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
    allow_label_derived_target: bool = True,
    allow_surrogate_wildfire_score: bool = True,
    use_high_signal_simplified_model: bool = False,
    min_join_confidence_score_for_metrics: float | None = None,
    retain_unusable_rows: bool = True,
) -> dict[str, Any]:
    rows, dataset_format = _load_rows_from_dataset_file(dataset_path)
    prepared_rows, validation = _prepare_rows(
        rows,
        allow_label_derived_target=allow_label_derived_target,
        allow_surrogate_wildfire_score=allow_surrogate_wildfire_score,
        use_high_signal_simplified_model=use_high_signal_simplified_model,
        min_join_confidence_score_for_metrics=min_join_confidence_score_for_metrics,
        retain_unusable_rows=retain_unusable_rows,
    )
    usable_rows = [row for row in prepared_rows if bool(row.get("row_usable_for_metrics"))]
    missing = validation.get("missing_required_fields") if isinstance(validation, dict) else {}
    invalid = validation.get("invalid_row_examples") if isinstance(validation, dict) else []
    return {
        "dataset_path": str(dataset_path),
        "dataset_format": dataset_format,
        "loaded_rows": len(rows),
        "prepared_rows": len(usable_rows),
        "retained_rows": len(prepared_rows),
        "dropped_rows": max(0, len(rows) - len(prepared_rows)),
        "unusable_rows": max(0, len(prepared_rows) - len(usable_rows)),
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
                "hazard_level_segment",
                "vegetation_density_segment",
                "join_method",
                "row_usable_for_metrics",
                "low_confidence_join",
                "missing_features",
                "fallback_heavy",
                "wildfire_score_source",
                "exclusion_reasons",
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
                    "hazard_level_segment": row.get("hazard_level_segment"),
                    "vegetation_density_segment": row.get("vegetation_density_segment"),
                    "join_method": row.get("join_method"),
                    "row_usable_for_metrics": row.get("row_usable_for_metrics"),
                    "low_confidence_join": row.get("low_confidence_join"),
                    "missing_features": row.get("missing_features"),
                    "fallback_heavy": row.get("fallback_heavy"),
                    "wildfire_score_source": row.get("wildfire_score_source"),
                    "exclusion_reasons": ";".join(str(token) for token in (row.get("exclusion_reasons") or [])),
                    "confusion_class_default_threshold_70": row.get("_confusion_class"),
                    "fallback_factor_count": flags.get("fallback_factor_count"),
                    "missing_factor_count": flags.get("missing_factor_count"),
                    "inferred_factor_count": flags.get("inferred_factor_count"),
                    "coverage_failed_count": flags.get("coverage_failed_count"),
                    "coverage_fallback_count": flags.get("coverage_fallback_count"),
                    "fallback_weight_fraction": flags.get("fallback_weight_fraction"),
                }
            )
