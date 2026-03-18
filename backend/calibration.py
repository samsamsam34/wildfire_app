from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=8)
def load_calibration_artifact(path: str) -> dict[str, Any]:
    return _load_json(Path(path))


def _apply_logistic(score: float, artifact: dict[str, Any]) -> float | None:
    params = artifact.get("parameters")
    if not isinstance(params, dict):
        return None
    intercept = _safe_float(params.get("intercept"))
    slope = _safe_float(params.get("slope"))
    x_scale = _safe_float(params.get("x_scale")) or 100.0
    if intercept is None or slope is None:
        return None
    score_scaled = score / x_scale if x_scale > 0 else score
    z = intercept + (slope * score_scaled)
    # Clamp z for numerical stability.
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _apply_piecewise(score: float, artifact: dict[str, Any]) -> float | None:
    points = artifact.get("points")
    if not isinstance(points, list):
        params = artifact.get("parameters")
        if isinstance(params, dict):
            candidate = params.get("points")
            if isinstance(candidate, list):
                points = candidate
    if not isinstance(points, list) or len(points) < 2:
        return None
    parsed: list[tuple[float, float]] = []
    for row in points:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        x = _safe_float(row[0])
        y = _safe_float(row[1])
        if x is None or y is None:
            continue
        parsed.append((x, max(0.0, min(1.0, y))))
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda item: item[0])

    if score <= parsed[0][0]:
        return parsed[0][1]
    if score >= parsed[-1][0]:
        return parsed[-1][1]

    for idx in range(1, len(parsed)):
        x0, y0 = parsed[idx - 1]
        x1, y1 = parsed[idx]
        if x0 <= score <= x1:
            if x1 == x0:
                return (y0 + y1) / 2.0
            ratio = (score - x0) / (x1 - x0)
            return y0 + (ratio * (y1 - y0))
    return None


def _scope_status(artifact: dict[str, Any], resolved_region_id: str | None) -> tuple[bool, str | None]:
    scope = artifact.get("scope")
    if not isinstance(scope, dict):
        return True, None
    region_ok = True
    region_warning = None
    regions = scope.get("region_ids")
    if isinstance(regions, list) and regions:
        resolved = str(resolved_region_id or "").strip()
        if not resolved:
            return False, "Calibration artifact is region-scoped but no resolved_region_id was provided."
        allowed = {str(r).strip() for r in regions if str(r).strip()}
        if resolved not in allowed:
            region_ok = False
            region_warning = f"Calibration artifact scope excludes region '{resolved}'."

    year_now = datetime.now(tz=timezone.utc).year
    start_year = _safe_float(scope.get("year_start"))
    end_year = _safe_float(scope.get("year_end"))
    if start_year is not None and year_now < int(start_year):
        return False, (
            f"Calibration artifact temporal scope starts at {int(start_year)}; "
            f"current year {year_now} is earlier."
        )
    if end_year is not None and year_now > int(end_year):
        return False, (
            f"Calibration artifact temporal scope ends at {int(end_year)}; "
            f"current year {year_now} is later."
        )
    if not region_ok:
        return False, region_warning
    return True, None


def resolve_public_calibration(
    *,
    raw_wildfire_score: float | None,
    artifact_path: str | None = None,
    resolved_region_id: str | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "calibration_enabled": False,
        "calibration_applied": False,
        "calibration_status": "disabled",
        "calibration_method": None,
        "artifact_path": None,
        "artifact_version": None,
        "artifact_generated_at": None,
        "outcome_dataset": None,
        "calibration_limitations": [],
        "calibrated_damage_likelihood": None,
        "empirical_damage_likelihood_proxy": None,
        "empirical_loss_likelihood_proxy": None,
        "raw_wildfire_risk_score": _safe_float(raw_wildfire_score),
        "scope_included": None,
        "scope_warning": None,
    }
    score = _safe_float(raw_wildfire_score)
    configured_path = str(artifact_path or os.getenv("WF_PUBLIC_CALIBRATION_ARTIFACT", "")).strip()
    if not configured_path:
        base["calibration_status"] = "disabled_no_artifact"
        return base
    artifact = load_calibration_artifact(configured_path)
    base["calibration_enabled"] = True
    base["artifact_path"] = configured_path
    base["artifact_version"] = artifact.get("artifact_version")
    base["artifact_generated_at"] = artifact.get("generated_at")
    base["outcome_dataset"] = artifact.get("dataset")
    base["calibration_limitations"] = list(artifact.get("limitations") or artifact.get("notes") or [])
    if not artifact:
        base["calibration_status"] = "invalid_artifact"
        return base
    if score is None:
        base["calibration_status"] = "score_unavailable"
        return base

    in_scope, scope_warning = _scope_status(artifact, resolved_region_id=resolved_region_id)
    base["scope_included"] = bool(in_scope)
    base["scope_warning"] = scope_warning
    if not in_scope:
        base["calibration_status"] = "out_of_scope"
        return base

    method = str(artifact.get("method") or "").strip().lower()
    base["calibration_method"] = method or None
    if method in {"logistic", "platt_logistic"}:
        calibrated = _apply_logistic(score, artifact)
    elif method in {"piecewise_linear", "piecewise", "isotonic", "isotonic_piecewise"}:
        calibrated = _apply_piecewise(score, artifact)
    else:
        calibrated = None
    if calibrated is None:
        base["calibration_status"] = "invalid_method_or_parameters"
        return base
    calibrated = max(0.0, min(1.0, float(calibrated)))
    base["calibration_applied"] = True
    base["calibration_status"] = "applied"
    base["calibrated_damage_likelihood"] = round(calibrated, 4)
    base["empirical_damage_likelihood_proxy"] = round(calibrated, 4)
    base["empirical_loss_likelihood_proxy"] = round(calibrated, 4)
    return base


def apply_public_calibration(
    *,
    raw_wildfire_score: float | None,
    artifact_path: str | None = None,
    resolved_region_id: str | None = None,
) -> dict[str, Any] | None:
    payload = resolve_public_calibration(
        raw_wildfire_score=raw_wildfire_score,
        artifact_path=artifact_path,
        resolved_region_id=resolved_region_id,
    )
    if not payload.get("calibration_applied"):
        return None
    return payload
