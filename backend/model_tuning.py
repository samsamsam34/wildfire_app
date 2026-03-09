from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from backend.event_backtesting import run_event_backtest, spearman_rank_correlation
from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import ScoringConfig, load_scoring_config
from backend.version import (
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    MODEL_VERSION,
    build_model_governance,
)
from backend.wildfire_data import WildfireContext

DEFAULT_TUNING_RESULTS_DIR = Path("benchmark") / "tuning_results"
DEFAULT_SCORING_PARAMETERS_PATH = Path("config") / "scoring_parameters.yaml"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_yaml_like(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:  # pragma: no cover - exercised only when PyYAML is installed
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_scoring_parameters(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_SCORING_PARAMETERS_PATH).expanduser()
    payload = _load_yaml_like(target)
    if payload:
        return payload

    # Fall back to runtime defaults when no file is available.
    cfg = load_scoring_config()
    return {
        "submodel_weights": dict(cfg.submodel_weights),
        "risk_blending_weights": dict(cfg.risk_blending_weights),
        "vulnerability_ring_penalties": copy.deepcopy(cfg.vulnerability_ring_penalties),
        "readiness_penalties": dict(cfg.readiness_penalties),
        "readiness_bonuses": dict(cfg.readiness_bonuses),
        "readiness_thresholds": dict(cfg.readiness_thresholds),
        "risk_bucket_thresholds": dict(cfg.risk_bucket_thresholds),
        "benchmark_risk_band_thresholds": dict(cfg.benchmark_risk_band_thresholds),
        "error_analysis_thresholds": dict(cfg.error_analysis_thresholds),
    }


def write_scoring_parameters(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _risk_bucket(score: float | None, thresholds: dict[str, Any]) -> str:
    if score is None:
        return "unscored"
    low_max = float(thresholds.get("low_max", 33.0))
    medium_max = float(thresholds.get("medium_max", 66.0))
    if score < low_max:
        return "low"
    if score < medium_max:
        return "medium"
    return "high"


def _is_adverse(rank: int, thresholds: dict[str, Any]) -> bool:
    return rank >= _to_int(thresholds.get("adverse_outcome_min_rank"), 3)


def _is_non_adverse(rank: int, thresholds: dict[str, Any]) -> bool:
    return rank <= _to_int(thresholds.get("non_adverse_outcome_max_rank"), 1)


def _record_score(record: dict[str, Any], key: str) -> float | None:
    return _to_float((record.get("scores") or {}).get(key))


def _record_confidence(record: dict[str, Any]) -> float | None:
    return _to_float((record.get("confidence") or {}).get("confidence_score"))


def _keyword_bag(record: dict[str, Any]) -> list[str]:
    words: list[str] = []
    for note in record.get("scoring_notes") or []:
        text = str(note).lower()
        if "missing" in text:
            words.append("missing_input")
        if "fallback" in text:
            words.append("fallback_used")
        if "footprint" in text:
            words.append("footprint_gap")
        if "environment" in text or "hazard" in text:
            words.append("environment_gap")
        if "road" in text or "access" in text:
            words.append("access_gap")
    evidence = record.get("evidence_quality_summary") or {}
    if _to_int(evidence.get("fallback_factor_count"), 0) > 0:
        words.append("fallback_factors")
    if _to_int(evidence.get("missing_factor_count"), 0) > 0:
        words.append("missing_factors")
    return words


def _top_factor_keys(record: dict[str, Any], family_key: str = "wildfire_risk_score") -> list[str]:
    ledger = (record.get("score_evidence_ledger_summary") or {}).get(family_key) or {}
    if not isinstance(ledger, dict):
        return []
    scored: list[tuple[str, float]] = []
    for key, value in ledger.items():
        numeric = _to_float(value)
        if numeric is None:
            continue
        scored.append((str(key), numeric))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    return [k for k, _ in scored[:3]]


def _false_low_false_high(
    records: list[dict[str, Any]],
    error_thresholds: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_low_max = float(error_thresholds.get("false_low_max_score", 40.0))
    false_high_min = float(error_thresholds.get("false_high_min_score", 70.0))
    adverse_min = _to_int(error_thresholds.get("adverse_outcome_min_rank"), 3)
    non_adverse_max = _to_int(error_thresholds.get("non_adverse_outcome_max_rank"), 1)

    false_low: list[dict[str, Any]] = []
    false_high: list[dict[str, Any]] = []

    for record in records:
        score = _record_score(record, "wildfire_risk_score")
        if score is None:
            continue
        rank = _to_int(record.get("outcome_rank"), 0)
        if rank >= adverse_min and score < false_low_max:
            false_low.append(record)
        if rank <= non_adverse_max and score >= false_high_min:
            false_high.append(record)

    false_low.sort(key=lambda row: _record_score(row, "wildfire_risk_score") or 0.0)
    false_high.sort(key=lambda row: _record_score(row, "wildfire_risk_score") or 0.0, reverse=True)
    return false_low, false_high


def _score_distribution_by_outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        label = str(record.get("outcome_label") or "unknown")
        score = _record_score(record, "wildfire_risk_score")
        if score is None:
            continue
        grouped.setdefault(label, []).append(score)

    summary: dict[str, Any] = {}
    for label, scores in grouped.items():
        summary[label] = {
            "count": len(scores),
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
    return summary


def _bucket_analysis(records: list[dict[str, Any]], thresholds: dict[str, Any], error_thresholds: dict[str, Any]) -> dict[str, Any]:
    buckets = {"low": [], "medium": [], "high": [], "unscored": []}
    for record in records:
        bucket = _risk_bucket(_record_score(record, "wildfire_risk_score"), thresholds)
        buckets[bucket].append(record)

    out: dict[str, Any] = {}
    for bucket, rows in buckets.items():
        adverse = sum(1 for row in rows if _is_adverse(_to_int(row.get("outcome_rank"), 0), error_thresholds))
        out[bucket] = {
            "count": len(rows),
            "adverse_count": adverse,
            "adverse_rate": round(adverse / len(rows), 4) if rows else None,
        }
    return out


def _confidence_accuracy_correlation(records: list[dict[str, Any]], thresholds: dict[str, Any], error_thresholds: dict[str, Any]) -> float | None:
    pairs: list[tuple[float | None, float | None]] = []
    for record in records:
        confidence = _record_confidence(record)
        score = _record_score(record, "wildfire_risk_score")
        if confidence is None or score is None:
            continue
        predicted_adverse = _risk_bucket(score, thresholds) == "high"
        actual_adverse = _is_adverse(_to_int(record.get("outcome_rank"), 0), error_thresholds)
        accuracy = 1.0 if predicted_adverse == actual_adverse else 0.0
        pairs.append((confidence, accuracy))
    return spearman_rank_correlation(pairs)


def _evidence_group(record: dict[str, Any]) -> str:
    evidence = record.get("evidence_quality_summary") or {}
    observed = _to_int(evidence.get("observed_factor_count"), 0)
    missing = _to_int(evidence.get("missing_factor_count"), 0)
    fallback = _to_int(evidence.get("fallback_factor_count"), 0)
    if fallback > observed or missing + fallback >= max(2, observed):
        return "fallback_heavy"
    if fallback == 0 and missing <= 1 and observed >= 4:
        return "high_evidence"
    return "mixed_evidence"


def _confidence_stratified_metrics(
    records: list[dict[str, Any]],
    thresholds: dict[str, Any],
    error_thresholds: dict[str, Any],
) -> dict[str, Any]:
    grouped = {"high_evidence": [], "mixed_evidence": [], "fallback_heavy": []}
    for record in records:
        grouped[_evidence_group(record)].append(record)

    out: dict[str, Any] = {}
    for name, rows in grouped.items():
        pairs = [(_record_score(row, "wildfire_risk_score"), _to_float(row.get("outcome_rank"))) for row in rows]
        out[name] = {
            "count": len(rows),
            "wildfire_vs_outcome_spearman": spearman_rank_correlation(pairs),
            "confidence_accuracy_spearman": _confidence_accuracy_correlation(rows, thresholds, error_thresholds),
        }
    return out


def evaluate_backtest_records(
    records: list[dict[str, Any]],
    *,
    scoring_parameters: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(scoring_parameters.get("risk_bucket_thresholds") or {})
    error_thresholds = dict(scoring_parameters.get("error_analysis_thresholds") or {})

    false_low, false_high = _false_low_false_high(records, error_thresholds)

    rank_pairs = [
        (_record_score(record, "wildfire_risk_score"), _to_float(record.get("outcome_rank")))
        for record in records
    ]
    site_pairs = [
        (_record_score(record, "site_hazard_score"), _to_float(record.get("outcome_rank")))
        for record in records
    ]

    bucket = _bucket_analysis(records, thresholds, error_thresholds)
    high_rate = _to_float((bucket.get("high") or {}).get("adverse_rate")) or 0.0
    low_rate = _to_float((bucket.get("low") or {}).get("adverse_rate")) or 0.0

    total = max(1, len(records))
    false_low_rate = len(false_low) / total
    false_high_rate = len(false_high) / total

    confidence_stratified = _confidence_stratified_metrics(records, thresholds, error_thresholds)
    high_evidence_corr = _to_float((confidence_stratified.get("high_evidence") or {}).get("wildfire_vs_outcome_spearman")) or 0.0
    fallback_corr = _to_float((confidence_stratified.get("fallback_heavy") or {}).get("wildfire_vs_outcome_spearman")) or 0.0

    score = 0.0
    score += (_to_float(spearman_rank_correlation(rank_pairs)) or 0.0) * 3.0
    score += (_to_float(spearman_rank_correlation(site_pairs)) or 0.0) * 1.5
    score += max(0.0, high_rate - low_rate)
    score += max(0.0, high_evidence_corr - fallback_corr) * 0.5
    score -= false_low_rate * 2.5
    score -= false_high_rate * 1.2

    return {
        "record_count": len(records),
        "rank_correlation": {
            "wildfire_vs_outcome": spearman_rank_correlation(rank_pairs),
            "site_hazard_vs_outcome": spearman_rank_correlation(site_pairs),
        },
        "bucket_analysis": bucket,
        "false_low_count": len(false_low),
        "false_high_count": len(false_high),
        "false_low_rate": round(false_low_rate, 4),
        "false_high_rate": round(false_high_rate, 4),
        "confidence_stratified": confidence_stratified,
        "objective_score": round(score, 6),
    }


def analyze_backtest_errors(
    records: list[dict[str, Any]],
    *,
    scoring_parameters: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(scoring_parameters.get("error_analysis_thresholds") or {})
    false_low, false_high = _false_low_false_high(records, thresholds)

    missing_false_low = Counter()
    top_factor_false_high = Counter()

    false_low_cases: list[dict[str, Any]] = []
    for row in false_low:
        missing_tokens = _keyword_bag(row)
        missing_false_low.update(missing_tokens)
        false_low_cases.append(
            {
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "event_date": row.get("event_date"),
                "record_id": row.get("record_id"),
                "outcome_label": row.get("outcome_label"),
                "outcome_rank": row.get("outcome_rank"),
                "scores": row.get("scores"),
                "confidence": row.get("confidence"),
                "evidence_quality_summary": row.get("evidence_quality_summary"),
                "coverage_summary": row.get("coverage_summary"),
                "factor_contribution_ledger": row.get("score_evidence_ledger_summary"),
                "missing_or_fallback_layers": missing_tokens,
            }
        )

    false_high_cases: list[dict[str, Any]] = []
    for row in false_high:
        factor_keys = _top_factor_keys(row)
        top_factor_false_high.update(factor_keys)
        false_high_cases.append(
            {
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "event_date": row.get("event_date"),
                "record_id": row.get("record_id"),
                "outcome_label": row.get("outcome_label"),
                "outcome_rank": row.get("outcome_rank"),
                "scores": row.get("scores"),
                "confidence": row.get("confidence"),
                "evidence_quality_summary": row.get("evidence_quality_summary"),
                "coverage_summary": row.get("coverage_summary"),
                "factor_contribution_ledger": row.get("score_evidence_ledger_summary"),
                "top_contributing_factors": factor_keys,
            }
        )

    metrics = evaluate_backtest_records(records, scoring_parameters=scoring_parameters)
    candidate_tuning_signals: list[str] = []
    if metrics.get("false_low_rate", 0.0) > 0.12:
        candidate_tuning_signals.append("Increase vulnerability/defensible-space pressure for adverse low-score cases.")
    if metrics.get("false_high_rate", 0.0) > 0.12:
        candidate_tuning_signals.append("Reduce over-penalization in extreme vegetation/hazard-only contexts.")
    corr = _to_float((metrics.get("rank_correlation") or {}).get("wildfire_vs_outcome"))
    if corr is not None and corr < 0.25:
        candidate_tuning_signals.append("Wildfire ordering is weak; review submodel blend and hazard weighting.")

    return {
        "summary_statistics": {
            "record_count": len(records),
            "false_low_count": len(false_low_cases),
            "false_high_count": len(false_high_cases),
            "score_distribution_by_outcome": _score_distribution_by_outcome(records),
            "confidence_accuracy_correlation": _confidence_accuracy_correlation(
                records,
                dict(scoring_parameters.get("risk_bucket_thresholds") or {}),
                dict(scoring_parameters.get("error_analysis_thresholds") or {}),
            ),
        },
        "cluster_indicators": {
            "common_missing_evidence_false_low": missing_false_low.most_common(10),
            "common_top_factors_false_high": top_factor_false_high.most_common(10),
        },
        "candidate_tuning_signals": candidate_tuning_signals,
        "false_low_cases": false_low_cases[:100],
        "false_high_cases": false_high_cases[:100],
    }


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return weights
    return {k: round(max(0.0, float(v)) / total, 6) for k, v in weights.items()}


def generate_parameter_candidates(
    base_parameters: dict[str, Any],
    *,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def _add(candidate_id: str, description: str, mutate: callable) -> None:
        payload = copy.deepcopy(base_parameters)
        mutate(payload)
        candidates.append(
            {
                "parameter_set_id": candidate_id,
                "description": description,
                "parameters": payload,
            }
        )

    _add("baseline", "Current scoring parameter baseline.", lambda payload: None)

    _add(
        "blend_more_environment",
        "Increase environmental influence in final wildfire blend by 5 percentage points.",
        lambda payload: payload.setdefault("risk_blending_weights", {}).update({"environmental": 0.80, "structural": 0.20}),
    )
    _add(
        "blend_more_structure",
        "Increase structural influence in final wildfire blend by 5 percentage points.",
        lambda payload: payload.setdefault("risk_blending_weights", {}).update({"environmental": 0.70, "structural": 0.30}),
    )

    def _boost_structure(payload: dict[str, Any]) -> None:
        weights = dict(payload.get("submodel_weights") or {})
        for key in ("structure_vulnerability_risk", "defensible_space_risk"):
            if key in weights:
                weights[key] = float(weights[key]) * 1.12
        payload["submodel_weights"] = _renormalize_weights(weights)

    _add(
        "boost_structure_vulnerability",
        "Boost structure and defensible-space submodel weights within bounded range.",
        _boost_structure,
    )

    def _boost_vegetation(payload: dict[str, Any]) -> None:
        weights = dict(payload.get("submodel_weights") or {})
        for key in ("vegetation_intensity_risk", "fuel_proximity_risk", "flame_contact_risk"):
            if key in weights:
                weights[key] = float(weights[key]) * 1.08
        payload["submodel_weights"] = _renormalize_weights(weights)

    _add(
        "boost_vegetation_hazard",
        "Boost near-structure vegetation/fuel hazard submodel weights within bounded range.",
        _boost_vegetation,
    )

    def _tighten_buckets(payload: dict[str, Any]) -> None:
        buckets = dict(payload.get("risk_bucket_thresholds") or {})
        buckets["low_max"] = max(20.0, float(buckets.get("low_max", 33.0)) - 2.0)
        buckets["medium_max"] = max(45.0, float(buckets.get("medium_max", 66.0)) - 3.0)
        payload["risk_bucket_thresholds"] = buckets

    _add(
        "tighten_risk_buckets",
        "Tighten low/medium risk bucket thresholds for stronger adverse separation.",
        _tighten_buckets,
    )

    def _increase_false_low_penalty(payload: dict[str, Any]) -> None:
        thresholds = dict(payload.get("error_analysis_thresholds") or {})
        thresholds["false_low_max_score"] = min(55.0, float(thresholds.get("false_low_max_score", 40.0)) + 5.0)
        payload["error_analysis_thresholds"] = thresholds

    _add(
        "wider_false_low_detection",
        "Widen false-low detection threshold to bias review toward missed adverse outcomes.",
        _increase_false_low_penalty,
    )

    return candidates[:max(1, int(max_candidates))]


def _build_guardrail_context(
    *,
    burn_probability: float,
    hazard: float,
    fuel: float,
    canopy: float,
    slope: float,
    wildland_distance: float,
    historic_fire: float,
    ring_0_5: float,
    ring_5_30: float,
    ring_30_100: float,
) -> WildfireContext:
    return WildfireContext(
        environmental_index=50.0,
        slope_index=slope,
        aspect_index=50.0,
        fuel_index=fuel,
        moisture_index=50.0,
        canopy_index=canopy,
        wildland_distance_index=wildland_distance,
        historic_fire_index=historic_fire,
        burn_probability_index=burn_probability,
        hazard_severity_index=hazard,
        burn_probability=burn_probability / 100.0,
        wildfire_hazard=hazard / 20.0,
        slope=slope,
        fuel_model=fuel,
        canopy_cover=canopy,
        historic_fire_distance=max(0.0, 5.0 - historic_fire / 20.0),
        wildland_distance=max(10.0, 500.0 - wildland_distance * 4.0),
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["guardrail-fixture"],
        assumptions=[],
        structure_ring_metrics={
            "ring_0_5_ft": {"vegetation_density": ring_0_5},
            "ring_5_30_ft": {"vegetation_density": ring_5_30},
            "ring_30_100_ft": {"vegetation_density": ring_30_100},
        },
        property_level_context={"footprint_used": True, "fallback_mode": "footprint"},
    )


def run_monotonic_guardrails(config: ScoringConfig) -> dict[str, Any]:
    engine = RiskEngine(config)

    base_attrs = PropertyAttributes(
        roof_type="wood",
        vent_type="standard",
        defensible_space_ft=10,
        construction_year=1995,
    )

    low_veg_context = _build_guardrail_context(
        burn_probability=45.0,
        hazard=45.0,
        fuel=45.0,
        canopy=42.0,
        slope=30.0,
        wildland_distance=45.0,
        historic_fire=35.0,
        ring_0_5=30.0,
        ring_5_30=35.0,
        ring_30_100=40.0,
    )
    high_veg_context = _build_guardrail_context(
        burn_probability=45.0,
        hazard=45.0,
        fuel=80.0,
        canopy=82.0,
        slope=30.0,
        wildland_distance=45.0,
        historic_fire=35.0,
        ring_0_5=85.0,
        ring_5_30=82.0,
        ring_30_100=78.0,
    )

    low_whp_context = _build_guardrail_context(
        burn_probability=30.0,
        hazard=30.0,
        fuel=55.0,
        canopy=55.0,
        slope=30.0,
        wildland_distance=50.0,
        historic_fire=35.0,
        ring_0_5=40.0,
        ring_5_30=45.0,
        ring_30_100=50.0,
    )
    high_whp_context = _build_guardrail_context(
        burn_probability=80.0,
        hazard=85.0,
        fuel=55.0,
        canopy=55.0,
        slope=30.0,
        wildland_distance=50.0,
        historic_fire=35.0,
        ring_0_5=40.0,
        ring_5_30=45.0,
        ring_30_100=50.0,
    )

    weak_defensible_attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=5,
        construction_year=2012,
    )
    strong_defensible_attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=40,
        construction_year=2012,
    )

    weak_structure_attrs = PropertyAttributes(
        roof_type="wood",
        vent_type="standard",
        defensible_space_ft=20,
        construction_year=2000,
    )
    strong_structure_attrs = PropertyAttributes(
        roof_type="class a",
        vent_type="ember-resistant",
        defensible_space_ft=20,
        construction_year=2000,
    )

    shared_context = _build_guardrail_context(
        burn_probability=55.0,
        hazard=55.0,
        fuel=60.0,
        canopy=58.0,
        slope=40.0,
        wildland_distance=65.0,
        historic_fire=50.0,
        ring_0_5=65.0,
        ring_5_30=62.0,
        ring_30_100=58.0,
    )

    low_veg_risk = engine.score(base_attrs, 46.0, -114.0, low_veg_context)
    high_veg_risk = engine.score(base_attrs, 46.0, -114.0, high_veg_context)
    low_veg_site = engine.compute_site_hazard_score(low_veg_risk)
    high_veg_site = engine.compute_site_hazard_score(high_veg_risk)

    low_whp_risk = engine.score(base_attrs, 46.0, -114.0, low_whp_context)
    high_whp_risk = engine.score(base_attrs, 46.0, -114.0, high_whp_context)
    low_whp_total = engine.compute_blended_wildfire_score(
        engine.compute_site_hazard_score(low_whp_risk),
        engine.compute_home_ignition_vulnerability_score(low_whp_risk),
    )
    high_whp_total = engine.compute_blended_wildfire_score(
        engine.compute_site_hazard_score(high_whp_risk),
        engine.compute_home_ignition_vulnerability_score(high_whp_risk),
    )

    weak_defensible_risk = engine.score(weak_defensible_attrs, 46.0, -114.0, shared_context)
    strong_defensible_risk = engine.score(strong_defensible_attrs, 46.0, -114.0, shared_context)
    weak_defensible_total = engine.compute_blended_wildfire_score(
        engine.compute_site_hazard_score(weak_defensible_risk),
        engine.compute_home_ignition_vulnerability_score(weak_defensible_risk),
    )
    strong_defensible_total = engine.compute_blended_wildfire_score(
        engine.compute_site_hazard_score(strong_defensible_risk),
        engine.compute_home_ignition_vulnerability_score(strong_defensible_risk),
    )

    weak_structure_risk = engine.score(weak_structure_attrs, 46.0, -114.0, shared_context)
    strong_structure_risk = engine.score(strong_structure_attrs, 46.0, -114.0, shared_context)
    weak_structure_home = engine.compute_home_ignition_vulnerability_score(weak_structure_risk)
    strong_structure_home = engine.compute_home_ignition_vulnerability_score(strong_structure_risk)

    checks = [
        {
            "check": "increasing_vegetation_hazard_never_lowers_site_hazard",
            "passed": high_veg_site >= low_veg_site,
            "before": low_veg_site,
            "after": high_veg_site,
        },
        {
            "check": "increasing_whp_never_lowers_total_risk",
            "passed": high_whp_total >= low_whp_total,
            "before": low_whp_total,
            "after": high_whp_total,
        },
        {
            "check": "increasing_defensible_space_never_increases_total_risk",
            "passed": strong_defensible_total <= weak_defensible_total,
            "before": weak_defensible_total,
            "after": strong_defensible_total,
        },
        {
            "check": "improving_roof_vents_never_increases_vulnerability",
            "passed": strong_structure_home <= weak_structure_home,
            "before": weak_structure_home,
            "after": strong_structure_home,
        },
    ]
    return {
        "checks": checks,
        "passed": all(bool(check.get("passed")) for check in checks),
    }


def compare_before_after(
    baseline_artifact: dict[str, Any],
    candidate_artifact: dict[str, Any],
    *,
    scoring_parameters: dict[str, Any],
) -> dict[str, Any]:
    baseline_records = {str(row.get("record_id")): row for row in baseline_artifact.get("records", [])}
    candidate_records = {str(row.get("record_id")): row for row in candidate_artifact.get("records", [])}
    shared_ids = sorted(set(baseline_records.keys()).intersection(candidate_records.keys()))

    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []

    def _target(rank: int) -> float:
        return max(0.0, min(100.0, float(rank) * 25.0))

    for record_id in shared_ids:
        base = baseline_records[record_id]
        cand = candidate_records[record_id]
        rank = _to_int(base.get("outcome_rank"), 0)
        target = _target(rank)

        base_score = _record_score(base, "wildfire_risk_score")
        cand_score = _record_score(cand, "wildfire_risk_score")
        if base_score is None or cand_score is None:
            continue

        base_err = abs(base_score - target)
        cand_err = abs(cand_score - target)
        delta = cand_err - base_err
        row = {
            "record_id": record_id,
            "event_id": base.get("event_id"),
            "outcome_label": base.get("outcome_label"),
            "baseline_risk": base_score,
            "candidate_risk": cand_score,
            "error_delta": round(delta, 4),
        }
        if delta < 0:
            improved.append(row)
        elif delta > 0:
            worsened.append(row)

    base_metrics = evaluate_backtest_records(
        list(baseline_records.values()),
        scoring_parameters=scoring_parameters,
    )
    cand_metrics = evaluate_backtest_records(
        list(candidate_records.values()),
        scoring_parameters=scoring_parameters,
    )

    return {
        "shared_record_count": len(shared_ids),
        "metric_improvements": {
            "wildfire_vs_outcome_spearman_delta": round(
                (_to_float((cand_metrics.get("rank_correlation") or {}).get("wildfire_vs_outcome")) or 0.0)
                - (_to_float((base_metrics.get("rank_correlation") or {}).get("wildfire_vs_outcome")) or 0.0),
                6,
            ),
            "false_low_rate_delta": round(
                float(cand_metrics.get("false_low_rate") or 0.0)
                - float(base_metrics.get("false_low_rate") or 0.0),
                6,
            ),
            "false_high_rate_delta": round(
                float(cand_metrics.get("false_high_rate") or 0.0)
                - float(base_metrics.get("false_high_rate") or 0.0),
                6,
            ),
            "objective_score_delta": round(
                float(cand_metrics.get("objective_score") or 0.0)
                - float(base_metrics.get("objective_score") or 0.0),
                6,
            ),
        },
        "risk_distribution_shift": {
            "baseline": base_metrics.get("bucket_analysis"),
            "candidate": cand_metrics.get("bucket_analysis"),
        },
        "improved_case_count": len(improved),
        "worsened_case_count": len(worsened),
        "improved_cases": improved[:25],
        "worsened_cases": worsened[:25],
    }


@contextmanager
def _patched_runtime_scoring_config(config_path: Path):
    import backend.main as app_main

    previous_path = os.environ.get("WILDFIRE_SCORING_PARAMETERS_PATH")
    previous_config = app_main.scoring_config
    previous_engine = app_main.risk_engine

    os.environ["WILDFIRE_SCORING_PARAMETERS_PATH"] = str(config_path)
    app_main.scoring_config = load_scoring_config()
    app_main.risk_engine = RiskEngine(app_main.scoring_config)
    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop("WILDFIRE_SCORING_PARAMETERS_PATH", None)
        else:
            os.environ["WILDFIRE_SCORING_PARAMETERS_PATH"] = previous_path
        app_main.scoring_config = previous_config
        app_main.risk_engine = previous_engine


def _run_backtest_for_parameters(
    *,
    parameters: dict[str, Any],
    dataset_paths: list[str | Path],
    output_dir: Path,
    ruleset_id: str | None = None,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="wf_tuning_params_") as tmp:
        params_path = Path(tmp) / "scoring_parameters.yaml"
        write_scoring_parameters(params_path, parameters)
        with _patched_runtime_scoring_config(params_path):
            return run_event_backtest(
                dataset_paths=dataset_paths,
                output_dir=output_dir,
                ruleset_id=ruleset_id,
                reuse_existing_assessments=False,
            )


def _recommend_parameter_changes(
    baseline_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    best_candidate: dict[str, Any],
) -> list[str]:
    recs: list[str] = []
    if float(best_metrics.get("objective_score") or 0.0) > float(baseline_metrics.get("objective_score") or 0.0):
        recs.append(
            f"Candidate `{best_candidate.get('parameter_set_id')}` improved objective score "
            f"from {baseline_metrics.get('objective_score')} to {best_metrics.get('objective_score')}."
        )

    base_false_low = float(baseline_metrics.get("false_low_rate") or 0.0)
    best_false_low = float(best_metrics.get("false_low_rate") or 0.0)
    if best_false_low < base_false_low:
        recs.append("False-low rate decreased; this candidate better captures adverse outcomes.")

    base_corr = _to_float((baseline_metrics.get("rank_correlation") or {}).get("wildfire_vs_outcome")) or 0.0
    best_corr = _to_float((best_metrics.get("rank_correlation") or {}).get("wildfire_vs_outcome")) or 0.0
    if best_corr > base_corr:
        recs.append("Wildfire score rank-correlation versus outcome severity improved.")

    if not recs:
        recs.append("No candidate materially outperformed baseline under current thresholds and guardrails.")
    return recs


def run_model_tuning(
    *,
    dataset_paths: list[str | Path],
    scoring_parameters_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ruleset_id: str | None = None,
    max_candidates: int = 8,
) -> dict[str, Any]:
    previous_scoring_path = os.environ.get("WILDFIRE_SCORING_PARAMETERS_PATH")
    output = Path(output_dir or DEFAULT_TUNING_RESULTS_DIR).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    base_params = load_scoring_parameters(scoring_parameters_path)
    candidates = generate_parameter_candidates(base_params, max_candidates=max_candidates)

    experiments: list[dict[str, Any]] = []

    baseline_candidate = next((candidate for candidate in candidates if candidate.get("parameter_set_id") == "baseline"), None)
    if baseline_candidate is None:
        baseline_candidate = {
            "parameter_set_id": "baseline",
            "description": "Current scoring parameter baseline.",
            "parameters": copy.deepcopy(base_params),
        }
        candidates.insert(0, baseline_candidate)

    baseline_artifact = _run_backtest_for_parameters(
        parameters=baseline_candidate["parameters"],
        dataset_paths=dataset_paths,
        output_dir=output,
        ruleset_id=ruleset_id,
    )
    baseline_metrics = evaluate_backtest_records(
        baseline_artifact.get("records", []),
        scoring_parameters=baseline_candidate["parameters"],
    )
    baseline_error_analysis = analyze_backtest_errors(
        baseline_artifact.get("records", []),
        scoring_parameters=baseline_candidate["parameters"],
    )

    for candidate in candidates:
        candidate_id = str(candidate.get("parameter_set_id") or f"candidate_{len(experiments)+1}")
        params = copy.deepcopy(candidate.get("parameters") or {})

        run_artifact = baseline_artifact if candidate_id == "baseline" else _run_backtest_for_parameters(
            parameters=params,
            dataset_paths=dataset_paths,
            output_dir=output,
            ruleset_id=ruleset_id,
        )

        cfg_path = output / f"params_{candidate_id}.yaml"
        write_scoring_parameters(cfg_path, params)
        os.environ["WILDFIRE_SCORING_PARAMETERS_PATH"] = str(cfg_path)
        cfg = load_scoring_config()
        guardrails = run_monotonic_guardrails(cfg)

        metrics = evaluate_backtest_records(run_artifact.get("records", []), scoring_parameters=params)
        experiments.append(
            {
                "tuning_run_id": str(uuid4()),
                "timestamp": _now_iso(),
                "parameter_set_id": candidate_id,
                "description": candidate.get("description"),
                "parameter_path": str(cfg_path),
                "parameters": params,
                "metrics": metrics,
                "guardrails": guardrails,
                "artifact_path": run_artifact.get("artifact_path"),
                "markdown_summary_path": run_artifact.get("markdown_summary_path"),
                "model_governance": run_artifact.get("model_governance"),
            }
        )

    # Restore default env to avoid leaking candidate override.
    if previous_scoring_path is None:
        os.environ.pop("WILDFIRE_SCORING_PARAMETERS_PATH", None)
    else:
        os.environ["WILDFIRE_SCORING_PARAMETERS_PATH"] = previous_scoring_path

    passing = [row for row in experiments if bool((row.get("guardrails") or {}).get("passed"))]
    ranked = sorted(passing or experiments, key=lambda row: float((row.get("metrics") or {}).get("objective_score") or -math.inf), reverse=True)
    best = ranked[0]

    best_artifact_path = Path(str(best.get("artifact_path"))) if best.get("artifact_path") else None
    best_artifact = json.loads(best_artifact_path.read_text(encoding="utf-8")) if best_artifact_path and best_artifact_path.exists() else {}

    before_after = compare_before_after(
        baseline_artifact,
        best_artifact if best_artifact else baseline_artifact,
        scoring_parameters=best.get("parameters") or baseline_candidate["parameters"],
    )

    baseline_experiment = next(row for row in experiments if row.get("parameter_set_id") == "baseline")
    recommended_changes = _recommend_parameter_changes(
        baseline_experiment.get("metrics") or baseline_metrics,
        best.get("metrics") or {},
        best,
    )

    artifact = {
        "generated_at": _now_iso(),
        "tuning_framework_version": "1.0.0",
        "dataset_paths": [str(Path(path).expanduser()) for path in dataset_paths],
        "baseline_parameter_set_id": "baseline",
        "experiments": experiments,
        "baseline_metrics": baseline_experiment.get("metrics") or baseline_metrics,
        "baseline_error_analysis": baseline_error_analysis,
        "best_experiment": {
            "parameter_set_id": best.get("parameter_set_id"),
            "description": best.get("description"),
            "metrics": best.get("metrics"),
            "guardrails": best.get("guardrails"),
            "artifact_path": best.get("artifact_path"),
        },
        "before_after_comparison": before_after,
        "recommended_parameter_changes": recommended_changes,
        "model_governance": build_model_governance(
            scoring_model_version=MODEL_VERSION,
            calibration_version=CALIBRATION_VERSION,
            benchmark_pack_version=BENCHMARK_PACK_VERSION,
        ),
        "summary": {
            "candidate_count": len(experiments),
            "passing_guardrail_count": len(passing),
            "best_objective_score": (best.get("metrics") or {}).get("objective_score"),
            "baseline_objective_score": (baseline_experiment.get("metrics") or {}).get("objective_score"),
            "baseline_artifact_path": baseline_artifact.get("artifact_path"),
        },
    }

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output / f"model_tuning_{stamp}.json"
    markdown_path = output / f"model_tuning_{stamp}.md"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        "# Model Tuning Summary",
        "",
        f"- Generated at: `{artifact['generated_at']}`",
        f"- Dataset count: `{len(dataset_paths)}`",
        f"- Candidate count: `{artifact['summary']['candidate_count']}`",
        f"- Guardrail pass count: `{artifact['summary']['passing_guardrail_count']}`",
        f"- Baseline objective: `{artifact['summary']['baseline_objective_score']}`",
        f"- Best objective: `{artifact['summary']['best_objective_score']}`",
        "",
        "## Best Parameter Set",
        f"- ID: `{artifact['best_experiment']['parameter_set_id']}`",
        f"- Description: {artifact['best_experiment']['description']}",
        f"- Artifact: `{artifact['best_experiment']['artifact_path']}`",
        "",
        "## Recommended Changes",
    ]
    for rec in artifact["recommended_parameter_changes"]:
        markdown_lines.append(f"- {rec}")

    markdown_lines.extend(
        [
            "",
            "## Notes",
            "- Tuning is bounded and deterministic; no opaque ML fitting is applied.",
            "- Candidate sets that fail monotonic guardrails should not be promoted.",
            "- Review fallback-heavy records separately from high-evidence records before adopting changes.",
        ]
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    artifact["artifact_path"] = str(artifact_path)
    artifact["markdown_summary_path"] = str(markdown_path)
    return artifact
