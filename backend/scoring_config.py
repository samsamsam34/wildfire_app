from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScoringConfig:
    # MVP insurer-oriented heuristic weights (not underwriting-approved).
    submodel_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "vegetation_intensity_risk": 0.12,
            "fuel_proximity_risk": 0.14,
            "slope_topography_risk": 0.12,
            "ember_exposure_risk": 0.15,
            "flame_contact_risk": 0.14,
            "historic_fire_risk": 0.11,
            "structure_vulnerability_risk": 0.12,
            "defensible_space_risk": 0.10,
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


def load_scoring_config() -> ScoringConfig:
    config = ScoringConfig()
    config.submodel_weights = _merge_float_map(config.submodel_weights, os.getenv("WILDFIRE_SUBMODEL_WEIGHTS_JSON"))
    config.readiness_penalties = _merge_float_map(config.readiness_penalties, os.getenv("WILDFIRE_READINESS_PENALTIES_JSON"))
    config.readiness_bonuses = _merge_float_map(config.readiness_bonuses, os.getenv("WILDFIRE_READINESS_BONUSES_JSON"))
    return config
