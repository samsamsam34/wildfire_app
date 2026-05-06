from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.version import SCORING_MODEL_VERSION

LOGGER = logging.getLogger("wildfire_app.calibration")

# Default artifact path — the only artifact currently in the repository.
_DEFAULT_ARTIFACT_PATH = Path("config") / "public_outcome_calibration.json"

# Minimum quality thresholds an artifact must pass before it may be activated.
_QUALITY_THRESHOLDS: dict[str, tuple[str, float, str]] = {
    # (metric_description, required_value, comparison)
    "roc_auc":      ("ROC-AUC",                     0.65,  ">="),
    "spearman_r":   ("Spearman r",                  0.25,  ">="),
    "dataset_size": ("dataset row_count",          1000.0, ">="),
    "ece":          ("ECE post-calibration",         0.10,  "<="),
}

# Module-level flag set once at import time by _run_quality_gate().
# False means the artifact failed at least one threshold and must not be activated.
CALIBRATION_ARTIFACT_SAFE: bool = False


def _extract_quality_metrics(artifact: dict[str, Any]) -> dict[str, float | None]:
    post = artifact.get("metrics", {}).get("post", {})
    calib = post.get("calibration", {})
    dataset = artifact.get("dataset", {})
    return {
        "roc_auc":      _safe_float(post.get("roc_auc_probability")),
        "spearman_r":   _safe_float(post.get("spearman_score_vs_label")),
        "dataset_size": _safe_float(dataset.get("row_count")),
        "ece":          _safe_float(calib.get("expected_calibration_error")),
    }


def _check_quality_thresholds(metrics: dict[str, float | None]) -> list[str]:
    """Return a list of human-readable failure reasons; empty list means all pass."""
    failures: list[str] = []
    thresholds = [
        ("roc_auc",      0.65,   ">=", "ROC-AUC"),
        ("spearman_r",   0.25,   ">=", "Spearman r"),
        ("dataset_size", 1000.0, ">=", "dataset row_count"),
        ("ece",          0.10,   "<=", "ECE post-calibration"),
    ]
    for key, required, cmp, label in thresholds:
        actual = metrics.get(key)
        if actual is None:
            failures.append(f"{label}: value missing from artifact")
            continue
        if cmp == ">=" and actual < required:
            failures.append(f"{label}: {actual:.4f} < required {required}")
        elif cmp == "<=" and actual > required:
            failures.append(f"{label}: {actual:.4f} > required {required}")
    return failures


def _run_quality_gate() -> bool:
    """
    Load the default artifact, check quality thresholds, set CALIBRATION_ARTIFACT_SAFE.
    Called once at module import time.
    """
    global CALIBRATION_ARTIFACT_SAFE  # noqa: PLW0603
    artifact = _load_json(_DEFAULT_ARTIFACT_PATH)
    if not artifact:
        LOGGER.warning(
            "calibration: default artifact not found at %s — CALIBRATION_ARTIFACT_SAFE=False",
            _DEFAULT_ARTIFACT_PATH,
        )
        CALIBRATION_ARTIFACT_SAFE = False
        return False
    metrics = _extract_quality_metrics(artifact)
    failures = _check_quality_thresholds(metrics)
    if failures:
        LOGGER.critical(
            "calibration: artifact at %s fails minimum quality thresholds — "
            "CALIBRATION_ARTIFACT_SAFE=False. Failures: %s",
            _DEFAULT_ARTIFACT_PATH,
            "; ".join(failures),
        )
        CALIBRATION_ARTIFACT_SAFE = False
        return False
    LOGGER.info(
        "calibration: artifact at %s passed all quality thresholds — CALIBRATION_ARTIFACT_SAFE=True",
        _DEFAULT_ARTIFACT_PATH,
    )
    CALIBRATION_ARTIFACT_SAFE = True
    return True


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


# Run the quality gate at import time so CALIBRATION_ARTIFACT_SAFE is always set.
_run_quality_gate()


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


def _apply_bin_rate_table(score: float, artifact: dict[str, Any]) -> float | None:
    params = artifact.get("parameters")
    table = None
    if isinstance(params, dict) and isinstance(params.get("bin_table"), list):
        table = params.get("bin_table")
    elif isinstance(artifact.get("bin_table"), list):
        table = artifact.get("bin_table")
    if not isinstance(table, list) or not table:
        return None
    parsed: list[tuple[float, float, float]] = []
    for row in table:
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
        return None
    parsed.sort(key=lambda item: (item[0], item[1]))
    for lo, hi, prob in parsed:
        if lo <= float(score) <= hi:
            return prob
    if float(score) < parsed[0][0]:
        return parsed[0][2]
    return parsed[-1][2]


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


def _version_compatibility_status(artifact: dict[str, Any]) -> tuple[bool, str | None]:
    model_compat = artifact.get("model_compatibility")
    if isinstance(model_compat, dict):
        allowed = model_compat.get("scoring_model_versions")
        if isinstance(allowed, list) and allowed:
            allowed_set = {str(v).strip() for v in allowed if str(v).strip()}
            if SCORING_MODEL_VERSION not in allowed_set:
                return (
                    False,
                    "Calibration artifact model-compatibility excludes this scoring_model_version.",
                )
    allowed_direct = artifact.get("compatible_scoring_model_versions")
    if isinstance(allowed_direct, list) and allowed_direct:
        allowed_set = {str(v).strip() for v in allowed_direct if str(v).strip()}
        if SCORING_MODEL_VERSION not in allowed_set:
            return (
                False,
                "Calibration artifact compatible_scoring_model_versions excludes this scoring_model_version.",
            )
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

    # Quality gate: if the artifact fails minimum thresholds, never apply it.
    if not CALIBRATION_ARTIFACT_SAFE:
        LOGGER.critical(
            "calibration: WF_PUBLIC_CALIBRATION_ARTIFACT is set but artifact fails minimum "
            "quality thresholds — calibration will not be applied. Raw score returned unchanged."
        )
        base["calibration_status"] = "disabled_quality_gate"
        base["calibration_limitations"] = [
            "Artifact blocked by quality gate (ROC-AUC, Spearman r, dataset size, or ECE threshold not met)."
        ]
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
    version_ok, version_warning = _version_compatibility_status(artifact)
    if not version_ok:
        base["calibration_status"] = "incompatible_version"
        if version_warning:
            base["calibration_limitations"] = list(base["calibration_limitations"] or []) + [version_warning]
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
    elif method in {"bin_rate_table", "binned", "histogram_binned"}:
        calibrated = _apply_bin_rate_table(score, artifact)
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
