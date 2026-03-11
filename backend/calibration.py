from __future__ import annotations

import json
import math
import os
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


def apply_public_calibration(
    *,
    raw_wildfire_score: float | None,
    artifact_path: str | None = None,
) -> dict[str, Any] | None:
    if raw_wildfire_score is None:
        return None
    score = _safe_float(raw_wildfire_score)
    if score is None:
        return None
    configured_path = str(artifact_path or os.getenv("WF_PUBLIC_CALIBRATION_ARTIFACT", "")).strip()
    if not configured_path:
        return None
    artifact = load_calibration_artifact(configured_path)
    if not artifact:
        return None

    method = str(artifact.get("method") or "").strip().lower()
    if method == "logistic":
        calibrated = _apply_logistic(score, artifact)
    elif method in {"piecewise_linear", "piecewise"}:
        calibrated = _apply_piecewise(score, artifact)
    else:
        calibrated = None
    if calibrated is None:
        return None
    calibrated = max(0.0, min(1.0, float(calibrated)))
    return {
        "calibration_applied": True,
        "calibration_method": method,
        "artifact_path": configured_path,
        "calibrated_damage_likelihood": round(calibrated, 4),
        "raw_wildfire_risk_score": round(score, 2),
        "artifact_version": artifact.get("artifact_version"),
        "outcome_dataset": artifact.get("dataset"),
    }
