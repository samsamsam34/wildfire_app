from __future__ import annotations

import json
import logging
import math
import random
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.benchmarking import (
    _resolve_ruleset,
    _scenario_snapshot,
    build_wildfire_context,
    patched_runtime_inputs,
)
from backend.models import AddressRequest
from backend.no_ground_truth_paths import (
    DEFAULT_NO_GROUND_TRUTH_ARTIFACT_ROOT,
    DEFAULT_NO_GROUND_TRUTH_FIXTURE_PATH,
)
from backend.scoring_config import load_scoring_config
from backend.version import (
    API_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    SCORING_MODEL_VERSION,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_FIXTURE_PATH = DEFAULT_NO_GROUND_TRUTH_FIXTURE_PATH
DEFAULT_OUTPUT_ROOT = DEFAULT_NO_GROUND_TRUTH_ARTIFACT_ROOT
DEFAULT_EVAL_VERSION = "1.0.0"
_RISK_BUCKET_THRESHOLDS = load_scoring_config().risk_bucket_thresholds or {}


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _deterministic_generated_at(run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def _path_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in str(path).split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


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


def _risk_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    try:
        low_max = float(_RISK_BUCKET_THRESHOLDS.get("low_max", 40.0))
    except (TypeError, ValueError):
        low_max = 40.0
    try:
        medium_max = float(_RISK_BUCKET_THRESHOLDS.get("medium_max", 60.0))
    except (TypeError, ValueError):
        medium_max = 60.0
    if score < low_max:
        return "low"
    if score < medium_max:
        return "medium"
    return "high"


def _fallback_group(snapshot: dict[str, Any]) -> str:
    metrics = snapshot.get("evidence_metrics") if isinstance(snapshot.get("evidence_metrics"), dict) else {}
    fallback_weight = _fallback_metric_value(metrics)
    observed_count = int((_safe_float(metrics.get("observed_feature_count")) or 0.0))
    if fallback_weight >= 0.45 or observed_count <= 2:
        return "fallback_heavy"
    if fallback_weight <= 0.2 and observed_count >= 6:
        return "high_evidence"
    return "mixed_evidence"


def _fallback_metric_value(metrics: dict[str, Any]) -> float:
    # Prefer feature-evidence fallback share when available. Contribution-weighted
    # fallback shares can drift with scenario severity and are not pure evidence quality.
    feature_fallback = _safe_float(metrics.get("fallback_evidence_fraction"))
    if feature_fallback is not None:
        return max(0.0, min(1.0, feature_fallback))
    weighted_fallback = _safe_float(metrics.get("fallback_weight_fraction"))
    if weighted_fallback is not None:
        return max(0.0, min(1.0, weighted_fallback))
    return 0.0


def load_no_ground_truth_fixture(path: str | Path | None = None) -> dict[str, Any]:
    fixture_path = Path(path or DEFAULT_FIXTURE_PATH).expanduser()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("No-ground-truth fixture must be a JSON object.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Fixture must include non-empty 'scenarios'.")
    seen: set[str] = set()
    for row in scenarios:
        if not isinstance(row, dict):
            raise ValueError("Each scenario must be an object.")
        scenario_id = str(row.get("scenario_id") or "").strip()
        if not scenario_id:
            raise ValueError("Each scenario must define scenario_id.")
        if scenario_id in seen:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        seen.add(scenario_id)
        payload_block = row.get("input_payload")
        if not isinstance(payload_block, dict) or not str(payload_block.get("address") or "").strip():
            raise ValueError(f"Scenario {scenario_id} is missing input_payload.address.")
    return payload


def _run_assessment_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import backend.main as app_main  # lazy import to avoid module cycle

    snapshots: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or "").strip()
        if not scenario_id:
            continue
        location = scenario.get("location") if isinstance(scenario.get("location"), dict) else {}
        lat = float(location.get("lat", 39.7392))
        lon = float(location.get("lon", -104.9903))
        geocode_source = str(location.get("geocode_source") or "no-ground-truth-fixture")
        context_overrides = scenario.get("context") if isinstance(scenario.get("context"), dict) else {}
        context = build_wildfire_context(context_overrides)
        payload = AddressRequest.model_validate(scenario.get("input_payload", {}))
        ruleset = _resolve_ruleset(payload.ruleset_id)
        org_id = str(scenario.get("organization_id") or "default_org")
        with patched_runtime_inputs(
            latitude=lat,
            longitude=lon,
            geocode_source=geocode_source,
            context=context,
        ):
            result, debug_payload = app_main._run_assessment(
                payload,
                organization_id=org_id,
                ruleset=ruleset,
            )
        snapshots[scenario_id] = _scenario_snapshot(scenario, result, debug_payload)
    return snapshots


def evaluate_monotonicity_rules(
    *,
    rules: list[dict[str, Any]],
    snapshots_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    violated: list[str] = []
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        baseline_id = str(rule.get("baseline") or "")
        variant_id = str(rule.get("variant") or "")
        metric = str(rule.get("metric") or "scores.wildfire_risk_score")
        expected = str(rule.get("expected") or "non_decrease")
        tolerance = abs(float(rule.get("tolerance") or 0.0))
        min_magnitude = abs(float(rule.get("min_magnitude") or 0.0))
        base = snapshots_by_id.get(baseline_id)
        variant = snapshots_by_id.get(variant_id)
        passed = False
        baseline_value = _safe_float(_path_get(base or {}, metric))
        variant_value = _safe_float(_path_get(variant or {}, metric))
        delta = (variant_value - baseline_value) if (baseline_value is not None and variant_value is not None) else None
        if base is None or variant is None or delta is None:
            detail = "missing scenario(s) or metric value"
        else:
            if expected == "increase":
                passed = delta > tolerance
            elif expected == "non_decrease":
                passed = delta >= (-1.0 * tolerance)
            elif expected == "decrease":
                passed = delta < (-1.0 * tolerance)
            elif expected == "non_increase":
                passed = delta <= tolerance
            else:
                detail = f"unsupported expectation: {expected}"
                rows.append(
                    {
                        "rule_id": rule_id,
                        "baseline": baseline_id,
                        "variant": variant_id,
                        "metric": metric,
                        "expected": expected,
                        "passed": False,
                        "warning": detail,
                        "baseline_value": baseline_value,
                        "variant_value": variant_value,
                        "delta": delta,
                    }
                )
                violated.append(rule_id)
                continue
            if passed and abs(delta) < min_magnitude:
                passed = False
                detail = f"delta={delta:.3f} smaller than min_magnitude={min_magnitude:.3f}"
            else:
                detail = "ok" if passed else f"expected={expected} tolerance={tolerance}"
        if not passed:
            violated.append(rule_id)
        rows.append(
            {
                "rule_id": rule_id,
                "description": rule.get("description"),
                "baseline": baseline_id,
                "variant": variant_id,
                "metric": metric,
                "expected": expected,
                "tolerance": tolerance,
                "min_magnitude": min_magnitude,
                "baseline_value": baseline_value,
                "variant_value": variant_value,
                "delta": delta,
                "passed": passed,
                "detail": detail,
            }
        )
    passed_count = sum(1 for row in rows if row.get("passed"))
    return {
        "rule_count": len(rows),
        "passed_count": passed_count,
        "failed_count": len(rows) - passed_count,
        "violated_rules": sorted(violated),
        "rows": rows,
        "status": "ok" if not violated else "warn",
    }


def evaluate_counterfactual_groups(
    *,
    groups: list[dict[str, Any]],
    snapshots_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    impact_by_intervention: dict[str, list[float]] = {}
    flagged: list[str] = []
    for group in groups:
        group_id = str(group.get("group_id") or "")
        baseline_id = str(group.get("baseline") or "")
        baseline = snapshots_by_id.get(baseline_id)
        baseline_risk = _safe_float(_path_get(baseline or {}, "scores.wildfire_risk_score"))
        baseline_readiness = _safe_float(_path_get(baseline or {}, "scores.insurance_readiness_score"))
        baseline_hardening = _safe_float(_path_get(baseline or {}, "scores.home_ignition_vulnerability_score"))
        interventions: list[dict[str, Any]] = []
        for variant_block in group.get("variants") or []:
            if not isinstance(variant_block, dict):
                continue
            scenario_id = str(variant_block.get("scenario_id") or "")
            intervention = str(variant_block.get("intervention") or scenario_id)
            expected = str(variant_block.get("expected") or "risk_down")
            snapshot = snapshots_by_id.get(scenario_id)
            risk = _safe_float(_path_get(snapshot or {}, "scores.wildfire_risk_score"))
            readiness = _safe_float(_path_get(snapshot or {}, "scores.insurance_readiness_score"))
            hardening = _safe_float(_path_get(snapshot or {}, "scores.home_ignition_vulnerability_score"))
            risk_delta = (risk - baseline_risk) if (risk is not None and baseline_risk is not None) else None
            readiness_delta = (
                readiness - baseline_readiness
                if (readiness is not None and baseline_readiness is not None)
                else None
            )
            hardening_delta = (
                hardening - baseline_hardening
                if (hardening is not None and baseline_hardening is not None)
                else None
            )
            direction_ok = True
            reason = "ok"
            if risk_delta is None:
                direction_ok = False
                reason = "missing_score"
            elif expected == "risk_down":
                direction_ok = risk_delta <= 0.0
                if not direction_ok:
                    reason = "risk_increased"
            elif expected == "risk_up":
                direction_ok = risk_delta >= 0.0
                if not direction_ok:
                    reason = "risk_decreased"
            elif expected == "readiness_up":
                direction_ok = readiness_delta is not None and readiness_delta >= 0.0
                if not direction_ok:
                    reason = "readiness_not_improved"
            elif expected == "readiness_down":
                direction_ok = readiness_delta is not None and readiness_delta <= 0.0
                if not direction_ok:
                    reason = "readiness_not_reduced"
            if risk_delta is not None:
                impact_by_intervention.setdefault(intervention, []).append(risk_delta)
            if not direction_ok:
                flagged.append(f"{group_id}:{intervention}")
            interventions.append(
                {
                    "scenario_id": scenario_id,
                    "intervention": intervention,
                    "expected": expected,
                    "risk_delta": risk_delta,
                    "insurance_readiness_delta": readiness_delta,
                    "home_ignition_vulnerability_delta": hardening_delta,
                    "directionally_consistent": direction_ok,
                    "detail": reason,
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "description": group.get("description"),
                "baseline": baseline_id,
                "baseline_scores": {
                    "wildfire_risk_score": baseline_risk,
                    "insurance_readiness_score": baseline_readiness,
                    "home_ignition_vulnerability_score": baseline_hardening,
                },
                "interventions": interventions,
            }
        )
    impact_table: list[dict[str, Any]] = []
    for intervention, deltas in impact_by_intervention.items():
        impact_table.append(
            {
                "intervention": intervention,
                "count": len(deltas),
                "median_risk_delta": _median(deltas),
                "mean_risk_delta": _mean(deltas),
            }
        )
    impact_table.sort(key=lambda row: (row.get("median_risk_delta") is None, row.get("median_risk_delta")))
    return {
        "group_count": len(group_rows),
        "groups": group_rows,
        "top_interventions_by_median_impact": impact_table,
        "flagged_interventions": sorted(flagged),
        "status": "ok" if not flagged else "warn",
    }


def _jitter_point(lat: float, lon: float, meters: float, angle_rad: float) -> tuple[float, float]:
    delta_lat = (meters / 111_320.0) * math.cos(angle_rad)
    meters_per_degree_lon = max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    delta_lon = (meters / meters_per_degree_lon) * math.sin(angle_rad)
    return lat + delta_lat, lon + delta_lon


def _coerce_context_numeric(context: dict[str, Any], field: str, delta: float) -> dict[str, Any]:
    next_context = deepcopy(context)
    current = _safe_float(next_context.get(field))
    if current is None:
        return next_context
    next_context[field] = max(0.0, min(100.0, current + delta))
    if field == "burn_probability_index":
        next_context["burn_probability"] = max(0.0, min(1.0, (next_context[field] / 100.0)))
    return next_context


def _apply_toggle_attribute(
    scenario: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    clone = deepcopy(scenario)
    payload = clone.setdefault("input_payload", {})
    attrs = payload.setdefault("attributes", {})
    attrs[field] = None
    confirmed = payload.get("confirmed_fields")
    if isinstance(confirmed, list):
        payload["confirmed_fields"] = [token for token in confirmed if str(token) != field]
    return clone


def _build_stability_variants(
    *,
    scenarios_by_id: dict[str, dict[str, Any]],
    specs: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for spec in specs:
        scenario_id = str(spec.get("scenario_id") or "")
        base = deepcopy(scenarios_by_id.get(scenario_id) or {})
        if not base:
            continue
        test_id = str(spec.get("test_id") or f"{scenario_id}_stability")
        location = base.get("location") if isinstance(base.get("location"), dict) else {}
        lat = float(location.get("lat", 39.7392))
        lon = float(location.get("lon", -104.9903))
        rng = random.Random(f"{seed}:{test_id}:{scenario_id}")

        for radius in spec.get("jitter_meters", []) or []:
            meters = abs(float(radius))
            for idx in range(max(1, int(spec.get("jitter_samples_per_radius") or 2))):
                angle = rng.uniform(0.0, 2.0 * math.pi)
                j_lat, j_lon = _jitter_point(lat, lon, meters, angle)
                scenario = deepcopy(base)
                scenario["scenario_id"] = f"{scenario_id}__{test_id}__jitter_{int(meters)}m_{idx}"
                scenario.setdefault("location", {})
                scenario["location"]["lat"] = j_lat
                scenario["location"]["lon"] = j_lon
                variants.append(scenario)
                descriptors.append(
                    {
                        "test_id": test_id,
                        "base_scenario_id": scenario_id,
                        "variant_scenario_id": scenario["scenario_id"],
                        "variant_type": "geocode_jitter",
                        "detail": {"radius_meters": meters, "sample_index": idx},
                    }
                )

        for perturb in spec.get("continuous_perturbations", []) or []:
            if not isinstance(perturb, dict):
                continue
            field = str(perturb.get("field") or "").strip()
            if not field:
                continue
            delta = abs(float(perturb.get("delta") or 0.0))
            if delta <= 0.0:
                continue
            for direction, signed in (("plus", delta), ("minus", -delta)):
                scenario = deepcopy(base)
                scenario["scenario_id"] = f"{scenario_id}__{test_id}__{field}_{direction}"
                context = scenario.get("context") if isinstance(scenario.get("context"), dict) else {}
                scenario["context"] = _coerce_context_numeric(context, field, signed)
                variants.append(scenario)
                descriptors.append(
                    {
                        "test_id": test_id,
                        "base_scenario_id": scenario_id,
                        "variant_scenario_id": scenario["scenario_id"],
                        "variant_type": "continuous_perturbation",
                        "detail": {"field": field, "delta": signed},
                    }
                )

        for field in spec.get("toggle_optional_attributes", []) or []:
            field_name = str(field).strip()
            if not field_name:
                continue
            scenario = _apply_toggle_attribute(base, field_name)
            scenario["scenario_id"] = f"{scenario_id}__{test_id}__toggle_attr_{field_name}"
            variants.append(scenario)
            descriptors.append(
                {
                    "test_id": test_id,
                    "base_scenario_id": scenario_id,
                    "variant_scenario_id": scenario["scenario_id"],
                    "variant_type": "optional_attribute_toggle",
                    "detail": {"field": field_name},
                }
            )

        for fallback_variant in spec.get("fallback_variants", []) or []:
            if not isinstance(fallback_variant, dict):
                continue
            name = str(fallback_variant.get("name") or "fallback_variant")
            scenario = deepcopy(base)
            scenario["scenario_id"] = f"{scenario_id}__{test_id}__{name}"
            context = scenario.get("context") if isinstance(scenario.get("context"), dict) else {}
            context = deepcopy(context)
            for field in fallback_variant.get("null_context_fields", []) or []:
                context[str(field)] = None
            layer_status_overrides = (
                fallback_variant.get("environmental_layer_status")
                if isinstance(fallback_variant.get("environmental_layer_status"), dict)
                else {}
            )
            status_block = (
                context.get("environmental_layer_status")
                if isinstance(context.get("environmental_layer_status"), dict)
                else {}
            )
            status_block = dict(status_block)
            for key, value in layer_status_overrides.items():
                status_block[str(key)] = str(value)
            if status_block:
                context["environmental_layer_status"] = status_block
            scenario["context"] = context
            variants.append(scenario)
            descriptors.append(
                {
                    "test_id": test_id,
                    "base_scenario_id": scenario_id,
                    "variant_scenario_id": scenario["scenario_id"],
                    "variant_type": "fallback_assumption_variant",
                    "detail": {"name": name},
                }
            )
    return variants, descriptors


def evaluate_stability(
    *,
    base_snapshots: dict[str, dict[str, Any]],
    variant_snapshots: dict[str, dict[str, Any]],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in descriptors:
        test_id = str(row.get("test_id") or "")
        grouped.setdefault(test_id, []).append(row)
    test_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for test_id in sorted(grouped.keys()):
        rows = grouped[test_id]
        base_id = str(rows[0].get("base_scenario_id") or "")
        base = base_snapshots.get(base_id) or {}
        baseline_score = _safe_float(_path_get(base, "scores.wildfire_risk_score"))
        baseline_conf = _safe_float(_path_get(base, "confidence.confidence_score"))
        baseline_tier = str(_path_get(base, "confidence.confidence_tier") or "unknown")
        baseline_band = str(_path_get(base, "scores.risk_band") or "unknown")
        deltas: list[float] = []
        tier_changes = 0
        band_changes = 0
        detail_rows: list[dict[str, Any]] = []
        for row in rows:
            variant_id = str(row.get("variant_scenario_id") or "")
            variant = variant_snapshots.get(variant_id) or {}
            score = _safe_float(_path_get(variant, "scores.wildfire_risk_score"))
            conf = _safe_float(_path_get(variant, "confidence.confidence_score"))
            tier = str(_path_get(variant, "confidence.confidence_tier") or "unknown")
            band = str(_path_get(variant, "scores.risk_band") or "unknown")
            score_delta = (score - baseline_score) if (score is not None and baseline_score is not None) else None
            conf_delta = (conf - baseline_conf) if (conf is not None and baseline_conf is not None) else None
            if score_delta is not None:
                deltas.append(abs(score_delta))
            if tier != baseline_tier:
                tier_changes += 1
            if band != baseline_band:
                band_changes += 1
            detail_rows.append(
                {
                    "variant_scenario_id": variant_id,
                    "variant_type": row.get("variant_type"),
                    "detail": row.get("detail"),
                    "wildfire_risk_score_delta": score_delta,
                    "confidence_score_delta": conf_delta,
                    "tier_changed": tier != baseline_tier,
                    "risk_band_changed": band != baseline_band,
                }
            )
        variant_count = len(rows)
        max_swing = max(deltas) if deltas else None
        p95_swing = _percentile(deltas, 0.95)
        change_rate = (tier_changes / float(variant_count)) if variant_count else 0.0
        if max_swing is not None and max_swing >= 15.0:
            warnings.append(f"{test_id}: max score swing {max_swing:.2f} from small perturbations.")
        if change_rate > 0.35:
            warnings.append(f"{test_id}: confidence tier changed too often ({change_rate:.2%}).")
        test_rows.append(
            {
                "test_id": test_id,
                "base_scenario_id": base_id,
                "variant_count": variant_count,
                "mean_abs_score_swing": _mean(deltas),
                "stddev_abs_score_swing": _stddev(deltas),
                "max_abs_score_swing": max_swing,
                "p95_abs_score_swing": p95_swing,
                "confidence_tier_change_rate": change_rate,
                "risk_band_change_rate": (band_changes / float(variant_count)) if variant_count else 0.0,
                "rows": detail_rows,
            }
        )
    return {
        "test_count": len(test_rows),
        "tests": test_rows,
        "warnings": warnings,
        "status": "ok" if not warnings else "warn",
    }


def evaluate_distribution(
    *,
    scenarios_by_id: dict[str, dict[str, Any]],
    snapshots_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for scenario_id in sorted(scenarios_by_id.keys()):
        scenario = scenarios_by_id[scenario_id]
        snap = snapshots_by_id.get(scenario_id) or {}
        risk = _safe_float(_path_get(snap, "scores.wildfire_risk_score"))
        confidence = _safe_float(_path_get(snap, "confidence.confidence_score"))
        fallback_group = _fallback_group(snap)
        segments = scenario.get("segments") if isinstance(scenario.get("segments"), list) else []
        records.append(
            {
                "scenario_id": scenario_id,
                "region": str(scenario.get("region") or "unknown"),
                "segments": [str(token) for token in segments if str(token).strip()],
                "risk_score": risk,
                "risk_bucket": _risk_bucket(risk),
                "confidence_score": confidence,
                "confidence_tier": str(_path_get(snap, "confidence.confidence_tier") or "unknown"),
                "fallback_group": fallback_group,
            }
        )

    risk_values = [float(row["risk_score"]) for row in records if row.get("risk_score") is not None]
    confidence_values = [float(row["confidence_score"]) for row in records if row.get("confidence_score") is not None]
    bucket_counts: dict[str, int] = {}
    confidence_tier_counts: dict[str, int] = {}
    fallback_group_counts: dict[str, int] = {}
    for row in records:
        bucket = str(row.get("risk_bucket") or "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        tier = str(row.get("confidence_tier") or "unknown")
        confidence_tier_counts[tier] = confidence_tier_counts.get(tier, 0) + 1
        fallback_group = str(row.get("fallback_group") or "unknown")
        fallback_group_counts[fallback_group] = fallback_group_counts.get(fallback_group, 0) + 1

    by_segment: dict[str, dict[str, Any]] = {}
    for row in records:
        for segment in row.get("segments") or []:
            block = by_segment.setdefault(segment, {"count": 0, "risk_scores": []})
            block["count"] += 1
            if row.get("risk_score") is not None:
                block["risk_scores"].append(float(row["risk_score"]))
    for segment, block in by_segment.items():
        values = block.pop("risk_scores")
        block["risk_mean"] = _mean(values)
        block["risk_median"] = _median(values)
        block["risk_stddev"] = _stddev(values)
        block["risk_min"] = min(values) if values else None
        block["risk_max"] = max(values) if values else None

    warnings: list[str] = []
    if risk_values:
        dynamic_range = max(risk_values) - min(risk_values)
        if dynamic_range < 12.0:
            warnings.append("Wildfire risk dynamic range is narrow; possible score compression.")
        if (_stddev(risk_values) or 0.0) < 5.0:
            warnings.append("Wildfire risk score spread is low (stddev < 5).")
    else:
        dynamic_range = None
    largest_bucket = max(bucket_counts.values()) if bucket_counts else 0
    largest_bucket_fraction = (largest_bucket / float(len(records))) if records else 0.0
    occupied_bucket_count = sum(1 for count in bucket_counts.values() if int(count) > 0)
    if records and largest_bucket_fraction >= 0.75:
        warnings.append("Most scenarios collapse into one risk bucket.")

    return {
        "record_count": len(records),
        "overall": {
            "wildfire_risk_score": {
                "min": min(risk_values) if risk_values else None,
                "max": max(risk_values) if risk_values else None,
                "mean": _mean(risk_values),
                "median": _median(risk_values),
                "stddev": _stddev(risk_values),
                "dynamic_range": dynamic_range,
            },
            "confidence_score": {
                "min": min(confidence_values) if confidence_values else None,
                "max": max(confidence_values) if confidence_values else None,
                "mean": _mean(confidence_values),
                "median": _median(confidence_values),
                "stddev": _stddev(confidence_values),
            },
        },
        "risk_bucket_counts": bucket_counts,
        "largest_bucket_fraction": round(largest_bucket_fraction, 4),
        "occupied_risk_bucket_count": occupied_bucket_count,
        "confidence_tier_counts": confidence_tier_counts,
        "fallback_group_counts": fallback_group_counts,
        "segment_stats": by_segment,
        "rows": records,
        "warnings": warnings,
        "status": "ok" if not warnings else "warn",
    }


def evaluate_external_alignment(
    *,
    scenarios_by_id: dict[str, dict[str, Any]],
    snapshots_by_id: dict[str, dict[str, Any]],
    alignment_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rule in alignment_rules:
        metric_path = str(rule.get("metric_path") or "scores.wildfire_risk_score")
        signal_key = str(rule.get("signal_key") or "")
        if not signal_key:
            continue
        mapping = rule.get("signal_mapping") if isinstance(rule.get("signal_mapping"), dict) else {}
        x_values: list[float] = []
        y_values: list[float] = []
        detail_rows: list[dict[str, Any]] = []
        for scenario_id in sorted(scenarios_by_id.keys()):
            scenario = scenarios_by_id[scenario_id]
            snap = snapshots_by_id.get(scenario_id) or {}
            signal_block = scenario.get("external_signals") if isinstance(scenario.get("external_signals"), dict) else {}
            signal_raw = signal_block.get(signal_key)
            signal_value = _safe_float(signal_raw)
            if signal_value is None and mapping:
                mapped = mapping.get(str(signal_raw).lower())
                signal_value = _safe_float(mapped)
            metric_value = _safe_float(_path_get(snap, metric_path))
            if metric_value is None or signal_value is None:
                continue
            x_values.append(metric_value)
            y_values.append(signal_value)
            detail_rows.append(
                {
                    "scenario_id": scenario_id,
                    "metric_value": metric_value,
                    "external_signal_value": signal_value,
                    "model_bucket": _risk_bucket(metric_value),
                    "signal_bucket": _risk_bucket(signal_value * 25.0 if signal_value <= 4.5 else signal_value),
                }
            )
        agreement_count = sum(
            1
            for row in detail_rows
            if str(row.get("model_bucket")) == str(row.get("signal_bucket"))
        )
        disagreement_cases = [
            row
            for row in detail_rows
            if str(row.get("model_bucket")) != str(row.get("signal_bucket"))
        ]
        disagreement_cases = sorted(
            disagreement_cases,
            key=lambda row: abs(float(row.get("metric_value") or 0.0) - float(row.get("external_signal_value") or 0.0)),
            reverse=True,
        )[:10]
        rows.append(
            {
                "metric_path": metric_path,
                "signal_key": signal_key,
                "sample_count": len(x_values),
                "spearman_rank_correlation": _spearman(x_values, y_values) if len(x_values) >= 3 else None,
                "bucket_agreement_ratio": (
                    agreement_count / float(len(detail_rows)) if detail_rows else None
                ),
                "disagreement_cases": disagreement_cases,
                "rows": detail_rows,
            }
        )
    warnings: list[str] = []
    if not rows:
        warnings.append("No external alignment rules or usable external signals were available.")
    for row in rows:
        count = int(row.get("sample_count") or 0)
        corr = row.get("spearman_rank_correlation")
        if count < 5:
            warnings.append(f"{row.get('signal_key')}: sample size is very small for stable alignment checks.")
        if isinstance(corr, (int, float)) and corr < 0.1:
            warnings.append(f"{row.get('signal_key')}: weak rank alignment with model metric.")
    return {
        "rule_count": len(rows),
        "rows": rows,
        "warnings": warnings,
        "status": "ok" if not warnings else "warn",
        "caveat": (
            "External alignment is a sanity check only. It is not ground-truth accuracy and "
            "must not be interpreted as validated loss prediction."
        ),
    }


def evaluate_confidence_diagnostics(
    *,
    snapshots_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tier_counts: dict[str, int] = {}
    fallback_group_counts: dict[str, int] = {}
    overconfidence_flags: list[dict[str, Any]] = []
    confidence_scores: list[float] = []
    fallback_weights: list[float] = []
    observed_counts: list[float] = []
    missing_critical_counts: list[float] = []
    inferred_counts: list[float] = []
    confidence_by_evidence_tier: dict[str, list[float]] = {}
    for scenario_id, snap in sorted(snapshots_by_id.items()):
        tier = str(_path_get(snap, "confidence.confidence_tier") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        group = _fallback_group(snap)
        fallback_group_counts[group] = fallback_group_counts.get(group, 0) + 1
        metrics = snap.get("evidence_metrics") if isinstance(snap.get("evidence_metrics"), dict) else {}
        fallback_weight = _fallback_metric_value(metrics)
        observed_features = _safe_float(metrics.get("observed_feature_count")) or 0.0
        missing_critical = _safe_float(_path_get(snap, "confidence.missing_critical_fields_count")) or 0.0
        inferred_fields = _safe_float(_path_get(snap, "confidence.inferred_fields_count")) or 0.0
        conf = _safe_float(_path_get(snap, "confidence.confidence_score"))
        if conf is not None:
            confidence_scores.append(conf)
            fallback_weights.append(fallback_weight)
            observed_counts.append(observed_features)
            missing_critical_counts.append(missing_critical)
            inferred_counts.append(inferred_fields)
            confidence_by_evidence_tier.setdefault(group, []).append(conf)
        if tier in {"high", "moderate"} and (fallback_weight >= 0.5 or observed_features <= 2.0):
            overconfidence_flags.append(
                {
                    "scenario_id": scenario_id,
                    "confidence_tier": tier,
                    "confidence_score": conf,
                    "fallback_weight_fraction": fallback_weight,
                    "observed_feature_count": observed_features,
                    "missing_critical_fields_count": missing_critical,
                    "inferred_fields_count": inferred_fields,
                    "reason": "confidence_tier_high_relative_to_evidence",
                }
            )
    warnings: list[str] = []
    if overconfidence_flags:
        warnings.append("Potential overconfidence detected in fallback-heavy or sparse-evidence scenarios.")
    corr_fallback = _spearman(confidence_scores, fallback_weights)
    corr_observed = _spearman(confidence_scores, observed_counts)
    corr_missing_critical = _spearman(confidence_scores, missing_critical_counts)
    corr_inferred = _spearman(confidence_scores, inferred_counts)
    if isinstance(corr_fallback, (int, float)) and corr_fallback > 0.0:
        warnings.append("Confidence score increases with fallback weight in this sample; review gating.")
    if isinstance(corr_observed, (int, float)) and corr_observed < 0.0:
        warnings.append("Confidence score decreases with observed feature count in this sample; review gating.")
    if isinstance(corr_missing_critical, (int, float)) and corr_missing_critical > 0.0:
        warnings.append("Confidence score increases with missing critical field count; review confidence penalties.")
    if isinstance(corr_inferred, (int, float)) and corr_inferred > 0.0:
        warnings.append("Confidence score increases with inferred-field count; review confidence penalties.")
    confidence_by_tier_summary = {
        key: {
            "count": len(values),
            "mean_confidence_score": _mean(values),
            "median_confidence_score": _median(values),
        }
        for key, values in confidence_by_evidence_tier.items()
    }
    missing_count_distribution: dict[str, int] = {}
    for count in missing_critical_counts:
        key = str(int(count))
        missing_count_distribution[key] = missing_count_distribution.get(key, 0) + 1
    return {
        "record_count": len(snapshots_by_id),
        "confidence_tier_distribution": tier_counts,
        "fallback_group_distribution": fallback_group_counts,
        "confidence_by_evidence_tier": confidence_by_tier_summary,
        "missing_critical_field_count_distribution": missing_count_distribution,
        "average_missing_critical_fields": _mean(missing_critical_counts),
        "average_inferred_fields": _mean(inferred_counts),
        "overconfidence_flags": overconfidence_flags,
        "confidence_vs_fallback_weight_spearman": corr_fallback,
        "confidence_vs_observed_feature_count_spearman": corr_observed,
        "confidence_vs_missing_critical_count_spearman": corr_missing_critical,
        "confidence_vs_inferred_field_count_spearman": corr_inferred,
        "warnings": warnings,
        "status": "ok" if not warnings else "warn",
    }


def build_no_ground_truth_summary_markdown(
    *,
    run_id: str,
    generated_at: str,
    fixture_path: Path,
    monotonicity: dict[str, Any],
    counterfactual: dict[str, Any],
    stability: dict[str, Any],
    distribution: dict[str, Any],
    benchmark_alignment: dict[str, Any],
    confidence_diagnostics: dict[str, Any],
) -> str:
    lines = [
        "# No-Ground-Truth Evaluation",
        "",
        "> This is not ground-truth accuracy validation. It is a coherence, stability, sensitivity, and trustworthiness check.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Fixture pack: `{fixture_path}`",
        "",
        "## What Was Tested",
        "- Monotonicity/directional expectations for paired scenarios.",
        "- Counterfactual mitigation sensitivity from controlled interventions.",
        "- Stability under small perturbations and fallback-assumption toggles.",
        "- Score distribution and segmentation health checks.",
        "- External benchmark alignment as a sanity check (not truth).",
        "- Confidence behavior relative to evidence quality and fallback pressure.",
        "",
        "## Pass/Fail/Warn Summary",
        (
            f"- Monotonicity: `{monotonicity.get('status')}` "
            f"({monotonicity.get('passed_count')}/{monotonicity.get('rule_count')} passed)"
        ),
        f"- Counterfactual sensitivity: `{counterfactual.get('status')}`",
        f"- Stability: `{stability.get('status')}`",
        f"- Distribution health: `{distribution.get('status')}`",
        f"- External alignment: `{benchmark_alignment.get('status')}`",
        f"- Confidence diagnostics: `{confidence_diagnostics.get('status')}`",
        "",
        "## Key Findings",
        f"- Monotonicity violations: `{len(monotonicity.get('violated_rules') or [])}`",
        f"- Flagged counterfactual interventions: `{len(counterfactual.get('flagged_interventions') or [])}`",
        f"- Stability warnings: `{len(stability.get('warnings') or [])}`",
        f"- Distribution warnings: `{len(distribution.get('warnings') or [])}`",
        f"- Risk bucket counts: `{distribution.get('risk_bucket_counts')}`",
        f"- Largest bucket fraction: `{distribution.get('largest_bucket_fraction')}`",
        f"- Confidence warnings: `{len(confidence_diagnostics.get('warnings') or [])}`",
        "",
        "## External Alignment Caveat",
        "- Agreement with external benchmark signals is directional sanity checking only.",
        "- It does not establish predictive accuracy for insured losses or structure damage outcomes.",
        "",
        "## Recommendation",
        "- Use this report to identify logic/coherence issues and data-evidence gaps.",
        "- Next step is labeled public-outcome validation and calibration once sufficient outcome data is available.",
    ]
    return "\n".join(lines) + "\n"


def _load_section_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _list_run_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    dirs = [row for row in root.iterdir() if row.is_dir() and (row / "evaluation_manifest.json").exists()]
    dirs.sort(key=lambda row: row.name)
    return dirs


def _resolve_baseline_run_dir(
    *,
    root: Path,
    current_run_id: str,
    compare_to_run_id: str | None,
) -> Path | None:
    if compare_to_run_id:
        candidate = root / str(compare_to_run_id)
        return candidate if candidate.exists() and candidate.is_dir() else None
    dirs = [row for row in _list_run_dirs(root) if row.name != current_run_id]
    return dirs[-1] if dirs else None


def _stability_rollup(payload: dict[str, Any]) -> dict[str, float | int | None]:
    tests = payload.get("tests") if isinstance(payload.get("tests"), list) else []
    rows = [row for row in tests if isinstance(row, dict)]
    mean_swings = [float(row.get("mean_abs_score_swing")) for row in rows if row.get("mean_abs_score_swing") is not None]
    max_swings = [float(row.get("max_abs_score_swing")) for row in rows if row.get("max_abs_score_swing") is not None]
    unstable_count = 0
    for row in rows:
        max_swing = float(row.get("max_abs_score_swing") or 0.0)
        tier_change = float(row.get("confidence_tier_change_rate") or 0.0)
        if max_swing >= 12.0 or tier_change >= 0.35:
            unstable_count += 1
    return {
        "avg_mean_abs_score_swing": _mean(mean_swings),
        "max_abs_score_swing": max(max_swings) if max_swings else None,
        "unstable_count": unstable_count,
    }


def _alignment_rollup(payload: dict[str, Any]) -> dict[str, float | int | None]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    valid_rows = [row for row in rows if isinstance(row, dict)]
    spearman = [
        float(row.get("spearman_rank_correlation"))
        for row in valid_rows
        if row.get("spearman_rank_correlation") is not None
    ]
    agreement = [
        float(row.get("bucket_agreement_ratio"))
        for row in valid_rows
        if row.get("bucket_agreement_ratio") is not None
    ]
    disagreements = 0
    for row in valid_rows:
        cases = row.get("disagreement_cases")
        if isinstance(cases, list):
            disagreements += len(cases)
    return {
        "rule_count": len(valid_rows),
        "avg_spearman_rank_correlation": _mean(spearman),
        "avg_bucket_agreement_ratio": _mean(agreement),
        "disagreement_case_count": disagreements,
    }


def _counterfactual_intervention_deltas(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> list[dict[str, Any]]:
    cur_table = (
        current.get("top_interventions_by_median_impact")
        if isinstance(current.get("top_interventions_by_median_impact"), list)
        else []
    )
    prev_table = (
        previous.get("top_interventions_by_median_impact")
        if isinstance(previous.get("top_interventions_by_median_impact"), list)
        else []
    )
    cur_map = {
        str(row.get("intervention")): row
        for row in cur_table
        if isinstance(row, dict) and str(row.get("intervention") or "").strip()
    }
    prev_map = {
        str(row.get("intervention")): row
        for row in prev_table
        if isinstance(row, dict) and str(row.get("intervention") or "").strip()
    }
    names = sorted(set(cur_map.keys()) | set(prev_map.keys()))
    deltas: list[dict[str, Any]] = []
    for name in names:
        cur = cur_map.get(name, {})
        prev = prev_map.get(name, {})
        cur_median = _safe_float(cur.get("median_risk_delta"))
        prev_median = _safe_float(prev.get("median_risk_delta"))
        deltas.append(
            {
                "intervention": name,
                "current_median_risk_delta": cur_median,
                "previous_median_risk_delta": prev_median,
                "median_delta_change": (
                    (cur_median - prev_median)
                    if (cur_median is not None and prev_median is not None)
                    else None
                ),
                "current_count": int(cur.get("count") or 0),
                "previous_count": int(prev.get("count") or 0),
            }
        )
    deltas.sort(
        key=lambda row: (
            row.get("median_delta_change") is None,
            -abs(float(row.get("median_delta_change") or 0.0)),
        )
    )
    return deltas[:12]


def build_no_ground_truth_run_comparison(
    *,
    current_run_id: str,
    current_manifest: dict[str, Any],
    current_sections: dict[str, dict[str, Any]],
    baseline_run_id: str | None,
    baseline_manifest: dict[str, Any] | None,
    baseline_sections: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not baseline_run_id or not baseline_manifest or not baseline_sections:
        return {
            "available": False,
            "run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "no_previous_run_available",
            "message": (
                "No previous run was found for before/after comparison. "
                "Run the evaluation again to compare latest vs previous."
            ),
        }

    current_mono = current_sections.get("monotonicity", {})
    baseline_mono = baseline_sections.get("monotonicity", {})
    current_counter = current_sections.get("counterfactual", {})
    baseline_counter = baseline_sections.get("counterfactual", {})
    current_stability = current_sections.get("stability", {})
    baseline_stability = baseline_sections.get("stability", {})
    current_dist = current_sections.get("distribution", {})
    baseline_dist = baseline_sections.get("distribution", {})
    current_conf = current_sections.get("confidence_diagnostics", {})
    baseline_conf = baseline_sections.get("confidence_diagnostics", {})
    current_align = current_sections.get("benchmark_alignment", {})
    baseline_align = baseline_sections.get("benchmark_alignment", {})

    cur_violations = {
        str(row)
        for row in (current_mono.get("violated_rules") if isinstance(current_mono.get("violated_rules"), list) else [])
    }
    prev_violations = {
        str(row)
        for row in (baseline_mono.get("violated_rules") if isinstance(baseline_mono.get("violated_rules"), list) else [])
    }

    current_stability_rollup = _stability_rollup(current_stability)
    baseline_stability_rollup = _stability_rollup(baseline_stability)
    current_alignment_rollup = _alignment_rollup(current_align)
    baseline_alignment_rollup = _alignment_rollup(baseline_align)

    current_bucket_counts = (
        current_dist.get("risk_bucket_counts")
        if isinstance(current_dist.get("risk_bucket_counts"), dict)
        else {}
    )
    baseline_bucket_counts = (
        baseline_dist.get("risk_bucket_counts")
        if isinstance(baseline_dist.get("risk_bucket_counts"), dict)
        else {}
    )
    current_tier_counts = (
        current_conf.get("confidence_tier_distribution")
        if isinstance(current_conf.get("confidence_tier_distribution"), dict)
        else {}
    )
    baseline_tier_counts = (
        baseline_conf.get("confidence_tier_distribution")
        if isinstance(baseline_conf.get("confidence_tier_distribution"), dict)
        else {}
    )

    same_fixture = str(current_manifest.get("fixture_path") or "") == str(baseline_manifest.get("fixture_path") or "")
    same_seed = str(current_manifest.get("seed") or "") == str(baseline_manifest.get("seed") or "")
    cur_versions = current_manifest.get("versions") if isinstance(current_manifest.get("versions"), dict) else {}
    prev_versions = baseline_manifest.get("versions") if isinstance(baseline_manifest.get("versions"), dict) else {}
    same_scoring_version = str(cur_versions.get("scoring_model_version") or "") == str(prev_versions.get("scoring_model_version") or "")
    same_rules_logic = str(cur_versions.get("rules_logic_version") or "") == str(prev_versions.get("rules_logic_version") or "")

    likely_change_drivers: list[str] = []
    if not same_scoring_version:
        likely_change_drivers.append("scoring_model_version_changed")
    if not same_rules_logic:
        likely_change_drivers.append("rules_logic_version_changed")
    if not same_fixture:
        likely_change_drivers.append("fixture_or_scenario_pack_changed")
    if not same_seed:
        likely_change_drivers.append("seed_changed")
    if not likely_change_drivers:
        likely_change_drivers.append("likely_logic_or_weight_tuning_with_constant_fixture")

    comparison = {
        "available": True,
        "run_id": current_run_id,
        "baseline_run_id": baseline_run_id,
        "comparability": {
            "same_fixture_path": same_fixture,
            "same_seed": same_seed,
            "same_scoring_model_version": same_scoring_version,
            "same_rules_logic_version": same_rules_logic,
        },
        "likely_change_drivers": likely_change_drivers,
        "monotonicity": {
            "current_status": current_mono.get("status"),
            "previous_status": baseline_mono.get("status"),
            "current_rule_count": int(current_mono.get("rule_count") or 0),
            "previous_rule_count": int(baseline_mono.get("rule_count") or 0),
            "current_failed_count": int(current_mono.get("failed_count") or 0),
            "previous_failed_count": int(baseline_mono.get("failed_count") or 0),
            "failed_count_delta": int(current_mono.get("failed_count") or 0) - int(baseline_mono.get("failed_count") or 0),
            "violations_added": sorted(cur_violations - prev_violations),
            "violations_resolved": sorted(prev_violations - cur_violations),
        },
        "counterfactual": {
            "current_status": current_counter.get("status"),
            "previous_status": baseline_counter.get("status"),
            "intervention_delta_table": _counterfactual_intervention_deltas(current_counter, baseline_counter),
            "current_flagged_count": len(current_counter.get("flagged_interventions") or []),
            "previous_flagged_count": len(baseline_counter.get("flagged_interventions") or []),
        },
        "stability": {
            "current_status": current_stability.get("status"),
            "previous_status": baseline_stability.get("status"),
            "current": current_stability_rollup,
            "previous": baseline_stability_rollup,
            "delta": {
                "avg_mean_abs_score_swing": (
                    (_safe_float(current_stability_rollup.get("avg_mean_abs_score_swing")) or 0.0)
                    - (_safe_float(baseline_stability_rollup.get("avg_mean_abs_score_swing")) or 0.0)
                ),
                "max_abs_score_swing": (
                    (_safe_float(current_stability_rollup.get("max_abs_score_swing")) or 0.0)
                    - (_safe_float(baseline_stability_rollup.get("max_abs_score_swing")) or 0.0)
                ),
                "unstable_count": int(current_stability_rollup.get("unstable_count") or 0)
                - int(baseline_stability_rollup.get("unstable_count") or 0),
            },
        },
        "confidence_diagnostics": {
            "current_status": current_conf.get("status"),
            "previous_status": baseline_conf.get("status"),
            "current_warning_count": len(current_conf.get("warnings") or []),
            "previous_warning_count": len(baseline_conf.get("warnings") or []),
            "warning_count_delta": len(current_conf.get("warnings") or []) - len(baseline_conf.get("warnings") or []),
            "current_confidence_tier_counts": current_tier_counts,
            "previous_confidence_tier_counts": baseline_tier_counts,
        },
        "distribution": {
            "current_status": current_dist.get("status"),
            "previous_status": baseline_dist.get("status"),
            "current_dynamic_range": _safe_float(((current_dist.get("overall") or {}).get("wildfire_risk_score") or {}).get("dynamic_range")),
            "previous_dynamic_range": _safe_float(((baseline_dist.get("overall") or {}).get("wildfire_risk_score") or {}).get("dynamic_range")),
            "dynamic_range_delta": (
                (_safe_float(((current_dist.get("overall") or {}).get("wildfire_risk_score") or {}).get("dynamic_range")) or 0.0)
                - (_safe_float(((baseline_dist.get("overall") or {}).get("wildfire_risk_score") or {}).get("dynamic_range")) or 0.0)
            ),
            "current_bucket_counts": current_bucket_counts,
            "previous_bucket_counts": baseline_bucket_counts,
            "current_largest_bucket_fraction": _safe_float(current_dist.get("largest_bucket_fraction")),
            "previous_largest_bucket_fraction": _safe_float(baseline_dist.get("largest_bucket_fraction")),
            "largest_bucket_fraction_delta": (
                (_safe_float(current_dist.get("largest_bucket_fraction")) or 0.0)
                - (_safe_float(baseline_dist.get("largest_bucket_fraction")) or 0.0)
            ),
            "current_occupied_bucket_count": int(current_dist.get("occupied_risk_bucket_count") or 0),
            "previous_occupied_bucket_count": int(baseline_dist.get("occupied_risk_bucket_count") or 0),
        },
        "benchmark_alignment": {
            "current_status": current_align.get("status"),
            "previous_status": baseline_align.get("status"),
            "current": current_alignment_rollup,
            "previous": baseline_alignment_rollup,
            "delta": {
                "avg_spearman_rank_correlation": (
                    (_safe_float(current_alignment_rollup.get("avg_spearman_rank_correlation")) or 0.0)
                    - (_safe_float(baseline_alignment_rollup.get("avg_spearman_rank_correlation")) or 0.0)
                ),
                "avg_bucket_agreement_ratio": (
                    (_safe_float(current_alignment_rollup.get("avg_bucket_agreement_ratio")) or 0.0)
                    - (_safe_float(baseline_alignment_rollup.get("avg_bucket_agreement_ratio")) or 0.0)
                ),
                "disagreement_case_count": int(current_alignment_rollup.get("disagreement_case_count") or 0)
                - int(baseline_alignment_rollup.get("disagreement_case_count") or 0),
            },
            "note": "Benchmark alignment is a sanity check only and not ground-truth validation.",
        },
    }

    overall_signals: list[str] = []
    mono_delta = int(comparison["monotonicity"]["failed_count_delta"])
    if mono_delta < 0:
        overall_signals.append("monotonicity_improved")
    elif mono_delta > 0:
        overall_signals.append("monotonicity_worsened")

    bucket_delta = _safe_float(comparison["distribution"]["largest_bucket_fraction_delta"]) or 0.0
    if bucket_delta < -0.01:
        overall_signals.append("bucket_collapse_reduced")
    elif bucket_delta > 0.01:
        overall_signals.append("bucket_collapse_worsened")

    warning_delta = int(comparison["confidence_diagnostics"]["warning_count_delta"])
    if warning_delta < 0:
        overall_signals.append("confidence_warnings_reduced")
    elif warning_delta > 0:
        overall_signals.append("confidence_warnings_increased")

    unstable_delta = int((comparison["stability"]["delta"] or {}).get("unstable_count") or 0)
    if unstable_delta < 0:
        overall_signals.append("stability_improved")
    elif unstable_delta > 0:
        overall_signals.append("stability_worsened")

    comparison["overall_direction_signals"] = overall_signals
    comparison["summary"] = (
        "Comparison generated between the latest run and prior baseline. "
        "Review comparability flags to distinguish logic changes from fixture/data differences."
    )
    return comparison


def build_no_ground_truth_comparison_markdown(payload: dict[str, Any]) -> str:
    if not bool(payload.get("available")):
        return (
            "# No-Ground-Truth Run Comparison\n\n"
            f"- Status: unavailable\n"
            f"- Message: {payload.get('message') or payload.get('reason') or 'No baseline run available.'}\n"
        )

    monotonicity = payload.get("monotonicity") if isinstance(payload.get("monotonicity"), dict) else {}
    counter = payload.get("counterfactual") if isinstance(payload.get("counterfactual"), dict) else {}
    stability = payload.get("stability") if isinstance(payload.get("stability"), dict) else {}
    distribution = payload.get("distribution") if isinstance(payload.get("distribution"), dict) else {}
    confidence = payload.get("confidence_diagnostics") if isinstance(payload.get("confidence_diagnostics"), dict) else {}
    comparability = payload.get("comparability") if isinstance(payload.get("comparability"), dict) else {}
    lines = [
        "# No-Ground-Truth Run Comparison",
        "",
        f"- Current run: `{payload.get('run_id')}`",
        f"- Baseline run: `{payload.get('baseline_run_id')}`",
        f"- Likely change drivers: `{payload.get('likely_change_drivers')}`",
        f"- Comparability flags: `{comparability}`",
        "",
        "## Monotonicity",
        f"- Current failed count: `{monotonicity.get('current_failed_count')}`",
        f"- Previous failed count: `{monotonicity.get('previous_failed_count')}`",
        f"- Failed-count delta: `{monotonicity.get('failed_count_delta')}`",
        f"- Violations resolved: `{monotonicity.get('violations_resolved')}`",
        f"- Violations added: `{monotonicity.get('violations_added')}`",
        "",
        "## Counterfactual Mitigation",
        f"- Current status: `{counter.get('current_status')}`",
        f"- Previous status: `{counter.get('previous_status')}`",
        f"- Intervention deltas (top): `{(counter.get('intervention_delta_table') or [])[:5]}`",
        "",
        "## Stability",
        f"- Delta avg mean swing: `{((stability.get('delta') or {}).get('avg_mean_abs_score_swing'))}`",
        f"- Delta max swing: `{((stability.get('delta') or {}).get('max_abs_score_swing'))}`",
        f"- Delta unstable count: `{((stability.get('delta') or {}).get('unstable_count'))}`",
        "",
        "## Confidence",
        f"- Warning-count delta: `{confidence.get('warning_count_delta')}`",
        f"- Current tier counts: `{confidence.get('current_confidence_tier_counts')}`",
        f"- Previous tier counts: `{confidence.get('previous_confidence_tier_counts')}`",
        "",
        "## Distribution",
        f"- Dynamic-range delta: `{distribution.get('dynamic_range_delta')}`",
        f"- Largest-bucket-fraction delta: `{distribution.get('largest_bucket_fraction_delta')}`",
        f"- Current buckets: `{distribution.get('current_bucket_counts')}`",
        f"- Previous buckets: `{distribution.get('previous_bucket_counts')}`",
        "",
        "## Caveat",
        "- This is directional diagnostics drift tracking, not ground-truth predictive validation.",
    ]
    return "\n".join(lines) + "\n"


def run_no_ground_truth_evaluation(
    *,
    fixture_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    compare_to_run_id: str | None = None,
) -> dict[str, Any]:
    fixture = load_no_ground_truth_fixture(fixture_path)
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_generated_at(run_id)
    fixture_seed = int(seed if seed is not None else int(fixture.get("seed") or 17))
    root = Path(output_root).expanduser()
    run_dir = root / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [row for row in fixture.get("scenarios", []) if isinstance(row, dict)]
    scenarios_by_id = {str(row.get("scenario_id")): row for row in scenarios}
    LOGGER.info("running no-ground-truth evaluation scenarios=%s run_id=%s", len(scenarios), run_token)
    base_snapshots = _run_assessment_scenarios(scenarios)

    monotonicity = evaluate_monotonicity_rules(
        rules=[row for row in fixture.get("monotonicity_rules", []) if isinstance(row, dict)],
        snapshots_by_id=base_snapshots,
    )
    counterfactual = evaluate_counterfactual_groups(
        groups=[row for row in fixture.get("counterfactual_groups", []) if isinstance(row, dict)],
        snapshots_by_id=base_snapshots,
    )

    stability_specs = [row for row in fixture.get("stability_tests", []) if isinstance(row, dict)]
    stability_variants, variant_descriptors = _build_stability_variants(
        scenarios_by_id=scenarios_by_id,
        specs=stability_specs,
        seed=fixture_seed,
    )
    variant_snapshots = _run_assessment_scenarios(stability_variants) if stability_variants else {}
    stability = evaluate_stability(
        base_snapshots=base_snapshots,
        variant_snapshots=variant_snapshots,
        descriptors=variant_descriptors,
    )
    distribution = evaluate_distribution(
        scenarios_by_id=scenarios_by_id,
        snapshots_by_id=base_snapshots,
    )
    benchmark_alignment = evaluate_external_alignment(
        scenarios_by_id=scenarios_by_id,
        snapshots_by_id=base_snapshots,
        alignment_rules=[row for row in fixture.get("benchmark_alignment_rules", []) if isinstance(row, dict)],
    )
    confidence_diagnostics = evaluate_confidence_diagnostics(snapshots_by_id=base_snapshots)

    artifacts = {
        "monotonicity_results.json": monotonicity,
        "counterfactual_results.json": counterfactual,
        "stability_results.json": stability,
        "distribution_results.json": distribution,
        "benchmark_alignment_results.json": benchmark_alignment,
        "confidence_diagnostics.json": confidence_diagnostics,
    }
    for filename, payload in artifacts.items():
        (run_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_md = build_no_ground_truth_summary_markdown(
        run_id=run_token,
        generated_at=generated_at,
        fixture_path=Path(fixture_path or DEFAULT_FIXTURE_PATH).expanduser(),
        monotonicity=monotonicity,
        counterfactual=counterfactual,
        stability=stability,
        distribution=distribution,
        benchmark_alignment=benchmark_alignment,
        confidence_diagnostics=confidence_diagnostics,
    )
    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    current_sections = {
        "monotonicity": monotonicity,
        "counterfactual": counterfactual,
        "stability": stability,
        "distribution": distribution,
        "benchmark_alignment": benchmark_alignment,
        "confidence_diagnostics": confidence_diagnostics,
    }

    current_versions = {
        "product_version": PRODUCT_VERSION,
        "api_version": API_VERSION,
        "scoring_model_version": SCORING_MODEL_VERSION,
        "rules_logic_version": RULESET_LOGIC_VERSION,
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "no_ground_truth_evaluation_version": DEFAULT_EVAL_VERSION,
    }
    current_manifest_seed = {
        "run_id": run_token,
        "generated_at": generated_at,
        "seed": fixture_seed,
        "fixture_path": str(Path(fixture_path or DEFAULT_FIXTURE_PATH).expanduser()),
        "versions": current_versions,
    }
    baseline_run_dir = _resolve_baseline_run_dir(
        root=root,
        current_run_id=run_token,
        compare_to_run_id=compare_to_run_id,
    )
    baseline_run_id = baseline_run_dir.name if baseline_run_dir is not None else None
    baseline_manifest = (
        _load_section_payload(baseline_run_dir / "evaluation_manifest.json")
        if baseline_run_dir is not None
        else None
    )
    baseline_sections = (
        {
            "monotonicity": _load_section_payload(baseline_run_dir / "monotonicity_results.json"),
            "counterfactual": _load_section_payload(baseline_run_dir / "counterfactual_results.json"),
            "stability": _load_section_payload(baseline_run_dir / "stability_results.json"),
            "distribution": _load_section_payload(baseline_run_dir / "distribution_results.json"),
            "benchmark_alignment": _load_section_payload(baseline_run_dir / "benchmark_alignment_results.json"),
            "confidence_diagnostics": _load_section_payload(baseline_run_dir / "confidence_diagnostics.json"),
        }
        if baseline_run_dir is not None
        else None
    )
    comparison_payload = build_no_ground_truth_run_comparison(
        current_run_id=run_token,
        current_manifest=current_manifest_seed,
        current_sections=current_sections,
        baseline_run_id=baseline_run_id,
        baseline_manifest=baseline_manifest,
        baseline_sections=baseline_sections,
    )
    comparison_json_path = run_dir / "comparison_to_previous.json"
    comparison_md_path = run_dir / "comparison_to_previous.md"
    comparison_json_path.write_text(
        json.dumps(comparison_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison_md_path.write_text(
        build_no_ground_truth_comparison_markdown(comparison_payload),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": DEFAULT_EVAL_VERSION,
        "run_id": run_token,
        "generated_at": generated_at,
        "seed": fixture_seed,
        "fixture_path": str(Path(fixture_path or DEFAULT_FIXTURE_PATH).expanduser()),
        "artifact_directory": str(run_dir),
        "artifacts": {
            "evaluation_manifest_json": str(run_dir / "evaluation_manifest.json"),
            "monotonicity_results_json": str(run_dir / "monotonicity_results.json"),
            "counterfactual_results_json": str(run_dir / "counterfactual_results.json"),
            "stability_results_json": str(run_dir / "stability_results.json"),
            "distribution_results_json": str(run_dir / "distribution_results.json"),
            "benchmark_alignment_results_json": str(run_dir / "benchmark_alignment_results.json"),
            "confidence_diagnostics_json": str(run_dir / "confidence_diagnostics.json"),
            "comparison_to_previous_json": str(comparison_json_path),
            "comparison_to_previous_markdown": str(comparison_md_path),
            "summary_markdown": str(summary_path),
        },
        "versions": current_versions,
        "status_summary": {
            "monotonicity": monotonicity.get("status"),
            "counterfactual": counterfactual.get("status"),
            "stability": stability.get("status"),
            "distribution": distribution.get("status"),
            "benchmark_alignment": benchmark_alignment.get("status"),
            "confidence_diagnostics": confidence_diagnostics.get("status"),
            "comparison_to_previous": comparison_payload.get("available"),
        },
        "comparison_to_previous": {
            "available": bool(comparison_payload.get("available")),
            "baseline_run_id": comparison_payload.get("baseline_run_id"),
            "compare_to_run_id_requested": compare_to_run_id,
            "likely_change_drivers": comparison_payload.get("likely_change_drivers"),
        },
    }
    manifest_path = run_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_id": run_token,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "comparison_json_path": str(comparison_json_path),
        "comparison_markdown_path": str(comparison_md_path),
    }
