from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


DEFAULT_SCORING_PARAMETERS_PATH = Path("config") / "scoring_parameters.yaml"


@dataclass
class ScoringConfig:
    # Deterministic heuristic configuration (not underwriting-approved).
    submodel_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "vegetation_intensity_risk": 0.16,
            "fuel_proximity_risk": 0.14,
            "slope_topography_risk": 0.08,
            "ember_exposure_risk": 0.11,
            "flame_contact_risk": 0.18,
            "historic_fire_risk": 0.04,
            "structure_vulnerability_risk": 0.12,
            "defensible_space_risk": 0.17,
        }
    )
    risk_blending_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "environmental": 0.45,
            "structural": 0.45,
            "readiness": 0.10,
        }
    )
    vulnerability_ring_penalties: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "zone_0_5_ft": {"threshold": 42.0, "slope": 0.46, "nonlinear_boost": 0.60},
            "zone_5_30_ft": {"threshold": 52.0, "slope": 0.27, "nonlinear_boost": 0.35},
            "zone_30_100_ft": {"threshold": 63.0, "slope": 0.10},
            "nearest_vegetation_distance_ft": {
                "ultra_critical_max_ft": 2.0,
                "ultra_critical_penalty": 13.0,
                "critical_max_ft": 5.0,
                "watch_max_ft": 15.0,
                "critical_penalty": 8.0,
                "watch_penalty": 4.0,
            },
        }
    )

    readiness_penalties: Dict[str, float] = field(
        default_factory=lambda: {
            "roof_fail": 26.0,
            "roof_watch": 8.0,
            "vent_fail": 12.0,
            "vent_watch": 7.0,
            "defensible_fail": 25.0,
            "defensible_watch": 14.0,
            "fuel_fail": 15.0,
            "fuel_watch": 8.0,
            "vegetation_fail": 14.0,
            "vegetation_watch": 7.0,
            "immediate_zone_0_5_fail": 8.5,
            "immediate_zone_0_5_watch": 4.0,
            "intermediate_zone_5_30_fail": 5.0,
            "intermediate_zone_5_30_watch": 2.5,
            "severe_env_fail": 12.0,
            "severe_env_watch": 6.0,
        }
    )

    readiness_bonuses: Dict[str, float] = field(
        default_factory=lambda: {
            "roof_pass": 4.0,
            "vent_pass": 3.0,
            "defensible_pass": 5.0,
            "fuel_pass": 2.0,
            "vegetation_pass": 2.0,
            "severe_env_pass": 1.0,
        }
    )
    readiness_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "defensible_space_pass_ft": 30.0,
            "defensible_space_watch_min_ft": 5.0,
            "structure_vulnerability_fail_score": 75.0,
            "structure_vulnerability_watch_score": 60.0,
            "adjacent_fuel_fail_score": 75.0,
            "adjacent_fuel_watch_score": 55.0,
            "vegetation_intensity_fail_score": 75.0,
            "vegetation_intensity_watch_score": 55.0,
            "ember_exposure_fail_score": 80.0,
            "ember_exposure_watch_score": 65.0,
            "severe_environment_fail_score": 85.0,
            "severe_environment_watch_score": 70.0,
            "zone_0_5_fail_density": 55.0,
            "zone_0_5_watch_density": 40.0,
            "zone_5_30_fail_density": 68.0,
            "zone_5_30_watch_density": 52.0,
        }
    )
    risk_bucket_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low_max": 40.0,
            "medium_max": 60.0,
        }
    )
    benchmark_risk_band_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low_max": 30.0,
            "moderate_max": 55.0,
            "high_max": 75.0,
        }
    )
    error_analysis_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "false_low_max_score": 40.0,
            "false_high_min_score": 70.0,
            "adverse_outcome_min_rank": 3.0,
            "non_adverse_outcome_max_rank": 1.0,
        }
    )


def _merge_float_map(defaults: Dict[str, float], raw: str | None) -> Dict[str, float]:
    if not raw:
        return defaults
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return defaults
        merged = dict(defaults)
        for k, v in payload.items():
            try:
                merged[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return merged
    except json.JSONDecodeError:
        return defaults


def _merge_float_map_from_payload(defaults: Dict[str, float], payload: object) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return defaults
    merged = dict(defaults)
    for k, v in payload.items():
        try:
            merged[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return merged


def _merge_nested_float_map_from_payload(defaults: Dict[str, Dict[str, float]], payload: object) -> Dict[str, Dict[str, float]]:
    if not isinstance(payload, dict):
        return defaults
    merged: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in defaults.items()}
    for outer_k, outer_v in payload.items():
        if not isinstance(outer_v, dict):
            continue
        bucket = dict(merged.get(str(outer_k), {}))
        for inner_k, inner_v in outer_v.items():
            try:
                bucket[str(inner_k)] = float(inner_v)
            except (TypeError, ValueError):
                continue
        merged[str(outer_k)] = bucket
    return merged


def _parse_yaml_like(text: str) -> dict:
    # Prefer PyYAML when installed, but keep JSON-compatible fallback so the
    # repo can run without adding a hard dependency for basic parameter loading.
    try:  # pragma: no cover - exercised when PyYAML is installed
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


def _load_file_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _parse_yaml_like(text)


def load_scoring_config() -> ScoringConfig:
    config = ScoringConfig()
    config_path = Path(os.getenv("WILDFIRE_SCORING_PARAMETERS_PATH", str(DEFAULT_SCORING_PARAMETERS_PATH))).expanduser()
    payload = _load_file_payload(config_path)

    config.submodel_weights = _merge_float_map_from_payload(
        config.submodel_weights,
        payload.get("submodel_weights"),
    )
    config.risk_blending_weights = _merge_float_map_from_payload(
        config.risk_blending_weights,
        payload.get("risk_blending_weights"),
    )
    config.vulnerability_ring_penalties = _merge_nested_float_map_from_payload(
        config.vulnerability_ring_penalties,
        payload.get("vulnerability_ring_penalties"),
    )
    config.readiness_penalties = _merge_float_map_from_payload(
        config.readiness_penalties,
        payload.get("readiness_penalties"),
    )
    config.readiness_bonuses = _merge_float_map_from_payload(
        config.readiness_bonuses,
        payload.get("readiness_bonuses"),
    )
    config.readiness_thresholds = _merge_float_map_from_payload(
        config.readiness_thresholds,
        payload.get("readiness_thresholds"),
    )
    config.risk_bucket_thresholds = _merge_float_map_from_payload(
        config.risk_bucket_thresholds,
        payload.get("risk_bucket_thresholds"),
    )
    config.benchmark_risk_band_thresholds = _merge_float_map_from_payload(
        config.benchmark_risk_band_thresholds,
        payload.get("benchmark_risk_band_thresholds"),
    )
    config.error_analysis_thresholds = _merge_float_map_from_payload(
        config.error_analysis_thresholds,
        payload.get("error_analysis_thresholds"),
    )

    config.submodel_weights = _merge_float_map(config.submodel_weights, os.getenv("WILDFIRE_SUBMODEL_WEIGHTS_JSON"))
    config.readiness_penalties = _merge_float_map(config.readiness_penalties, os.getenv("WILDFIRE_READINESS_PENALTIES_JSON"))
    config.readiness_bonuses = _merge_float_map(config.readiness_bonuses, os.getenv("WILDFIRE_READINESS_BONUSES_JSON"))
    return config
