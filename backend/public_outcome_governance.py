from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.no_ground_truth_paths import repository_root
from backend.version import compare_model_governance

DEFAULT_PUBLIC_OUTCOME_VALIDATION_ROOT = (
    repository_root() / "benchmark" / "public_outcomes" / "validation"
)
DEFAULT_PUBLIC_OUTCOME_CALIBRATION_ROOT = (
    repository_root() / "benchmark" / "public_outcomes" / "calibration"
)


def resolve_public_outcome_validation_root(path_hint: str | Path | None = None) -> Path:
    hint = str(path_hint or os.getenv("WF_PUBLIC_OUTCOME_VALIDATION_DIR") or "").strip()
    if hint:
        return Path(hint).expanduser()
    return DEFAULT_PUBLIC_OUTCOME_VALIDATION_ROOT


def resolve_public_outcome_calibration_root(path_hint: str | Path | None = None) -> Path:
    hint = str(path_hint or os.getenv("WF_PUBLIC_OUTCOME_CALIBRATION_DIR") or "").strip()
    if hint:
        return Path(hint).expanduser()
    return DEFAULT_PUBLIC_OUTCOME_CALIBRATION_ROOT


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _parse_run_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    # ISO timestamps (with optional Z suffix).
    try:
        iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    # Compact run-id timestamp format, e.g. 20260325T180150Z.
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def list_public_outcome_runs(
    *,
    artifact_root: Path,
    manifest_filename: str = "manifest.json",
) -> dict[str, Any]:
    root = artifact_root.expanduser()
    root_exists = root.exists()
    is_dir = root.is_dir()
    if not root_exists or not is_dir:
        return {
            "available": False,
            "artifact_root": str(root),
            "artifact_root_exists": bool(root_exists),
            "artifact_root_is_dir": bool(is_dir),
            "run_directory_count": 0,
            "runs": [],
            "latest_run_id": None,
            "message": (
                "No artifact runs found. "
                f"Checked artifact root: {root} (exists={root_exists}, is_dir={is_dir})."
            ),
        }

    runs: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / manifest_filename
        manifest = _safe_load_json(manifest_path) if manifest_path.exists() else None
        generated_at = (
            str(manifest.get("generated_at"))
            if isinstance(manifest, dict) and manifest.get("generated_at")
            else _iso_mtime(entry)
        )
        sort_dt = _parse_run_datetime(generated_at)
        if sort_dt is None:
            sort_dt = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        runs.append(
            {
                "run_id": entry.name,
                "path": str(entry),
                "generated_at": generated_at,
                "sort_timestamp": sort_dt.isoformat(),
                "has_manifest": bool(manifest),
            }
        )
    runs.sort(key=lambda row: str(row.get("sort_timestamp") or ""), reverse=True)
    for row in runs:
        if isinstance(row, dict):
            row.pop("sort_timestamp", None)
    return {
        "available": bool(runs),
        "artifact_root": str(root),
        "artifact_root_exists": True,
        "artifact_root_is_dir": True,
        "run_directory_count": len(runs),
        "runs": runs,
        "latest_run_id": (runs[0]["run_id"] if runs else None),
        "message": (
            None
            if runs
            else (
                f"No run directories found in artifact root: {root}. "
                "Run the public-outcome workflows first."
            )
        ),
    }


def resolve_baseline_run_id(
    *,
    ordered_run_ids: list[str],
    current_run_id: str,
    baseline_run_id: str | None,
) -> str | None:
    if baseline_run_id:
        requested = str(baseline_run_id).strip()
        if requested in ordered_run_ids and requested != current_run_id:
            return requested
        return None
    for run_id in ordered_run_ids:
        if run_id != current_run_id:
            return run_id
    return None


def _metric_delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def _extract_validation_confidence_slices(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    slice_metrics = report.get("slice_metrics") if isinstance(report.get("slice_metrics"), dict) else {}
    by_tier = (
        slice_metrics.get("by_confidence_tier")
        if isinstance(slice_metrics.get("by_confidence_tier"), dict)
        else {}
    )
    out: dict[str, dict[str, Any]] = {}
    for tier, payload in sorted(by_tier.items()):
        if not isinstance(payload, dict):
            continue
        out[str(tier)] = {
            "count": int(payload.get("count") or 0),
            "wildfire_risk_score_auc": _safe_float(payload.get("wildfire_risk_score_auc")),
            "wildfire_risk_score_pr_auc": _safe_float(payload.get("wildfire_risk_score_pr_auc")),
            "wildfire_risk_score_brier": _safe_float(payload.get("wildfire_risk_score_brier")),
        }
    return out


def _validation_run_summary(report: dict[str, Any]) -> dict[str, Any]:
    sample_counts = report.get("sample_counts") if isinstance(report.get("sample_counts"), dict) else {}
    discrimination = report.get("discrimination_metrics") if isinstance(report.get("discrimination_metrics"), dict) else {}
    brier = report.get("brier_scores") if isinstance(report.get("brier_scores"), dict) else {}
    reported_metrics = report.get("reported_metrics") if isinstance(report.get("reported_metrics"), dict) else {}
    reported_primary = (
        reported_metrics.get("primary")
        if isinstance(reported_metrics.get("primary"), dict)
        else {}
    )
    cal = report.get("calibration_metrics") if isinstance(report.get("calibration_metrics"), dict) else {}
    wildfire_cal = cal.get("wildfire_risk_score") if isinstance(cal.get("wildfire_risk_score"), dict) else {}
    review = report.get("false_review_sets") if isinstance(report.get("false_review_sets"), dict) else {}
    calibrated_cal = (
        cal.get("calibrated_damage_likelihood")
        if isinstance(cal.get("calibrated_damage_likelihood"), dict)
        else {}
    )
    return {
        "sample_count": int(
            reported_primary.get("count")
            if reported_primary.get("count") is not None
            else (sample_counts.get("row_count_usable") or 0)
        ),
        "prevalence": (
            _safe_float(reported_primary.get("positive_rate"))
            if reported_primary.get("positive_rate") is not None
            else _safe_float(sample_counts.get("positive_rate"))
        ),
        "roc_auc": (
            _safe_float(reported_primary.get("wildfire_risk_score_auc"))
            if reported_primary.get("wildfire_risk_score_auc") is not None
            else _safe_float(discrimination.get("wildfire_risk_score_auc"))
        ),
        "pr_auc": (
            _safe_float(reported_primary.get("wildfire_risk_score_pr_auc"))
            if reported_primary.get("wildfire_risk_score_pr_auc") is not None
            else _safe_float(discrimination.get("wildfire_risk_score_pr_auc"))
        ),
        "brier_score": (
            _safe_float(reported_primary.get("wildfire_risk_score_brier"))
            if reported_primary.get("wildfire_risk_score_brier") is not None
            else _safe_float(brier.get("wildfire_probability_proxy"))
        ),
        "metrics_source": (reported_metrics.get("primary_source") or "in_sample_fallback"),
        "calibration_error": _safe_float(wildfire_cal.get("expected_calibration_error")),
        "false_low_count": int(review.get("false_low_count") or 0),
        "false_high_count": int(review.get("false_high_count") or 0),
        "confidence_tier_metrics": _extract_validation_confidence_slices(report),
        "raw_vs_calibrated_within_run": {
            "calibrated_metrics_available": (
                discrimination.get("calibrated_damage_likelihood_auc") is not None
                or brier.get("calibrated_damage_likelihood") is not None
            ),
            "raw_roc_auc": _safe_float(discrimination.get("wildfire_risk_score_auc")),
            "calibrated_roc_auc": _safe_float(discrimination.get("calibrated_damage_likelihood_auc")),
            "roc_auc_delta_calibrated_minus_raw": _metric_delta(
                _safe_float(discrimination.get("calibrated_damage_likelihood_auc")),
                _safe_float(discrimination.get("wildfire_risk_score_auc")),
            ),
            "raw_brier": _safe_float(brier.get("wildfire_probability_proxy")),
            "calibrated_brier": _safe_float(brier.get("calibrated_damage_likelihood")),
            "brier_delta_calibrated_minus_raw": _metric_delta(
                _safe_float(brier.get("calibrated_damage_likelihood")),
                _safe_float(brier.get("wildfire_probability_proxy")),
            ),
            "raw_ece": _safe_float(wildfire_cal.get("expected_calibration_error")),
            "calibrated_ece": _safe_float(calibrated_cal.get("expected_calibration_error")),
            "ece_delta_calibrated_minus_raw": _metric_delta(
                _safe_float(calibrated_cal.get("expected_calibration_error")),
                _safe_float(wildfire_cal.get("expected_calibration_error")),
            ),
        },
    }


def build_validation_run_comparison(
    *,
    current_run_id: str,
    current_manifest: dict[str, Any],
    current_report: dict[str, Any],
    baseline_run_id: str | None,
    baseline_manifest: dict[str, Any] | None,
    baseline_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline_run_id or not baseline_manifest or not baseline_report:
        return {
            "available": False,
            "run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "no_previous_run_available",
            "message": (
                "No previous validation run was found for before/after comparison. "
                "Run the validation workflow again to compare latest vs previous."
            ),
        }

    current = _validation_run_summary(current_report)
    previous = _validation_run_summary(baseline_report)
    confidence_table: list[dict[str, Any]] = []
    tier_names = sorted(
        set(current["confidence_tier_metrics"].keys()) | set(previous["confidence_tier_metrics"].keys())
    )
    for tier in tier_names:
        cur = current["confidence_tier_metrics"].get(tier) or {}
        prev = previous["confidence_tier_metrics"].get(tier) or {}
        confidence_table.append(
            {
                "confidence_tier": tier,
                "current_count": int(cur.get("count") or 0),
                "previous_count": int(prev.get("count") or 0),
                "count_delta": int(cur.get("count") or 0) - int(prev.get("count") or 0),
                "current_auc": _safe_float(cur.get("wildfire_risk_score_auc")),
                "previous_auc": _safe_float(prev.get("wildfire_risk_score_auc")),
                "auc_delta": _metric_delta(
                    _safe_float(cur.get("wildfire_risk_score_auc")),
                    _safe_float(prev.get("wildfire_risk_score_auc")),
                ),
                "current_brier": _safe_float(cur.get("wildfire_risk_score_brier")),
                "previous_brier": _safe_float(prev.get("wildfire_risk_score_brier")),
                "brier_delta": _metric_delta(
                    _safe_float(cur.get("wildfire_risk_score_brier")),
                    _safe_float(prev.get("wildfire_risk_score_brier")),
                ),
            }
        )

    comparability = compare_model_governance(
        current_manifest.get("versions") if isinstance(current_manifest.get("versions"), dict) else {},
        baseline_manifest.get("versions") if isinstance(baseline_manifest.get("versions"), dict) else {},
    )
    dataset_changed = (
        str(((current_manifest.get("inputs") or {}).get("evaluation_dataset_path")) or "").strip()
        != str(((baseline_manifest.get("inputs") or {}).get("evaluation_dataset_path")) or "").strip()
    )
    likely_change_drivers: list[str] = []
    if dataset_changed:
        likely_change_drivers.append("evaluation_dataset_changed")
    for key in (comparability.get("critical_differences") or []):
        likely_change_drivers.append(f"{key}_changed")
    for key in (comparability.get("review_differences") or []):
        if f"{key}_changed" not in likely_change_drivers:
            likely_change_drivers.append(f"{key}_changed")
    if not likely_change_drivers:
        likely_change_drivers.append("likely_logic_or_parameter_change_with_constant_versions")

    payload = {
        "available": True,
        "run_id": current_run_id,
        "baseline_run_id": baseline_run_id,
        "comparability": comparability,
        "likely_change_drivers": likely_change_drivers,
        "current": current,
        "previous": previous,
        "delta": {
            "sample_count": int(current["sample_count"]) - int(previous["sample_count"]),
            "prevalence": _metric_delta(current["prevalence"], previous["prevalence"]),
            "roc_auc": _metric_delta(current["roc_auc"], previous["roc_auc"]),
            "pr_auc": _metric_delta(current["pr_auc"], previous["pr_auc"]),
            "brier_score": _metric_delta(current["brier_score"], previous["brier_score"]),
            "calibration_error": _metric_delta(
                current["calibration_error"], previous["calibration_error"]
            ),
            "false_low_count": int(current["false_low_count"]) - int(previous["false_low_count"]),
            "false_high_count": int(current["false_high_count"]) - int(previous["false_high_count"]),
        },
        "confidence_tier_delta_table": confidence_table,
    }

    signals: list[str] = []
    if (payload["delta"].get("roc_auc") or 0.0) > 0.005:
        signals.append("discrimination_improved")
    elif (payload["delta"].get("roc_auc") or 0.0) < -0.005:
        signals.append("discrimination_worsened")
    if (payload["delta"].get("brier_score") or 0.0) < -0.002:
        signals.append("calibration_improved")
    elif (payload["delta"].get("brier_score") or 0.0) > 0.002:
        signals.append("calibration_worsened")
    if int(payload["delta"].get("false_low_count") or 0) < 0:
        signals.append("false_lows_reduced")
    elif int(payload["delta"].get("false_low_count") or 0) > 0:
        signals.append("false_lows_increased")
    if int(payload["delta"].get("false_high_count") or 0) < 0:
        signals.append("false_highs_reduced")
    elif int(payload["delta"].get("false_high_count") or 0) > 0:
        signals.append("false_highs_increased")
    payload["overall_direction_signals"] = signals
    return payload


def build_validation_comparison_markdown(payload: dict[str, Any]) -> str:
    if not bool(payload.get("available")):
        return (
            "# Public Outcome Validation Comparison\n\n"
            f"- Status: unavailable\n"
            f"- Message: {payload.get('message') or payload.get('reason') or 'No baseline run available.'}\n"
        )
    delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
    return "\n".join(
        [
            "# Public Outcome Validation Comparison",
            "",
            f"- Current run: `{payload.get('run_id')}`",
            f"- Baseline run: `{payload.get('baseline_run_id')}`",
            f"- Likely change drivers: `{payload.get('likely_change_drivers')}`",
            "",
            "## Core Metrics Delta (current - previous)",
            f"- Sample count: `{delta.get('sample_count')}`",
            f"- Prevalence: `{delta.get('prevalence')}`",
            f"- ROC AUC: `{delta.get('roc_auc')}`",
            f"- PR AUC: `{delta.get('pr_auc')}`",
            f"- Brier: `{delta.get('brier_score')}`",
            f"- Calibration error (ECE): `{delta.get('calibration_error')}`",
            f"- False-low count: `{delta.get('false_low_count')}`",
            f"- False-high count: `{delta.get('false_high_count')}`",
            "",
            "## Confidence Tier Delta Table",
            f"- `{(payload.get('confidence_tier_delta_table') or [])}`",
            "",
            "## Direction Signals",
            f"- `{payload.get('overall_direction_signals') or []}`",
            "",
            "## Caveat",
            "- This is directional comparison on public observed outcomes, not carrier claims validation.",
            "",
        ]
    )


def _calibration_run_summary(pre_post: dict[str, Any]) -> dict[str, Any]:
    pre = pre_post.get("pre") if isinstance(pre_post.get("pre"), dict) else {}
    post = pre_post.get("post") if isinstance(pre_post.get("post"), dict) else {}
    delta = pre_post.get("delta") if isinstance(pre_post.get("delta"), dict) else {}
    slices = pre_post.get("slices") if isinstance(pre_post.get("slices"), dict) else {}
    return {
        "sample_count": int(pre.get("row_count") or post.get("row_count") or 0),
        "prevalence": _safe_float(pre.get("positive_rate") or post.get("positive_rate")),
        "pre_roc_auc_probability": _safe_float(pre.get("roc_auc_probability")),
        "post_roc_auc_probability": _safe_float(post.get("roc_auc_probability")),
        "pre_pr_auc_probability": _safe_float(pre.get("pr_auc_probability")),
        "post_pr_auc_probability": _safe_float(post.get("pr_auc_probability")),
        "pre_brier": _safe_float(pre.get("brier_probability")),
        "post_brier": _safe_float(post.get("brier_probability")),
        "pre_calibration_error": _safe_float(
            ((pre.get("calibration") or {}).get("expected_calibration_error"))
            if isinstance(pre.get("calibration"), dict)
            else None
        ),
        "post_calibration_error": _safe_float(
            ((post.get("calibration") or {}).get("expected_calibration_error"))
            if isinstance(post.get("calibration"), dict)
            else None
        ),
        "brier_improvement": _safe_float(delta.get("brier_improvement")),
        "log_loss_improvement": _safe_float(delta.get("log_loss_improvement")),
        "slice_metrics": slices,
    }


def build_calibration_run_comparison(
    *,
    current_run_id: str,
    current_manifest: dict[str, Any],
    current_pre_post: dict[str, Any],
    baseline_run_id: str | None,
    baseline_manifest: dict[str, Any] | None,
    baseline_pre_post: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline_run_id or not baseline_manifest or not baseline_pre_post:
        return {
            "available": False,
            "run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "no_previous_run_available",
            "message": (
                "No previous calibration run was found for before/after comparison. "
                "Run calibration again to compare latest vs previous."
            ),
        }

    current = _calibration_run_summary(current_pre_post)
    previous = _calibration_run_summary(baseline_pre_post)
    comparability = compare_model_governance(
        current_manifest.get("versions") if isinstance(current_manifest.get("versions"), dict) else {},
        baseline_manifest.get("versions") if isinstance(baseline_manifest.get("versions"), dict) else {},
    )
    dataset_changed = (
        str(((current_manifest.get("inputs") or {}).get("dataset_path")) or "").strip()
        != str(((baseline_manifest.get("inputs") or {}).get("dataset_path")) or "").strip()
    )
    likely_change_drivers: list[str] = []
    if dataset_changed:
        likely_change_drivers.append("evaluation_dataset_changed")
    for key in (comparability.get("critical_differences") or []):
        likely_change_drivers.append(f"{key}_changed")
    for key in (comparability.get("review_differences") or []):
        if f"{key}_changed" not in likely_change_drivers:
            likely_change_drivers.append(f"{key}_changed")
    if not likely_change_drivers:
        likely_change_drivers.append("likely_logic_or_parameter_change_with_constant_versions")

    payload = {
        "available": True,
        "run_id": current_run_id,
        "baseline_run_id": baseline_run_id,
        "comparability": comparability,
        "likely_change_drivers": likely_change_drivers,
        "current": current,
        "previous": previous,
        "delta": {
            "sample_count": int(current["sample_count"]) - int(previous["sample_count"]),
            "prevalence": _metric_delta(current["prevalence"], previous["prevalence"]),
            "pre_brier": _metric_delta(current["pre_brier"], previous["pre_brier"]),
            "post_brier": _metric_delta(current["post_brier"], previous["post_brier"]),
            "brier_improvement": _metric_delta(
                current["brier_improvement"], previous["brier_improvement"]
            ),
            "pre_calibration_error": _metric_delta(
                current["pre_calibration_error"], previous["pre_calibration_error"]
            ),
            "post_calibration_error": _metric_delta(
                current["post_calibration_error"], previous["post_calibration_error"]
            ),
            "log_loss_improvement": _metric_delta(
                current["log_loss_improvement"], previous["log_loss_improvement"]
            ),
            "post_roc_auc_probability": _metric_delta(
                current["post_roc_auc_probability"], previous["post_roc_auc_probability"]
            ),
            "post_pr_auc_probability": _metric_delta(
                current["post_pr_auc_probability"], previous["post_pr_auc_probability"]
            ),
        },
    }
    signals: list[str] = []
    if (payload["delta"].get("brier_improvement") or 0.0) > 0.001:
        signals.append("calibration_gain_improved")
    elif (payload["delta"].get("brier_improvement") or 0.0) < -0.001:
        signals.append("calibration_gain_worsened")
    if (payload["delta"].get("post_brier") or 0.0) < -0.001:
        signals.append("post_calibration_brier_improved")
    elif (payload["delta"].get("post_brier") or 0.0) > 0.001:
        signals.append("post_calibration_brier_worsened")
    payload["overall_direction_signals"] = signals
    return payload


def build_calibration_comparison_markdown(payload: dict[str, Any]) -> str:
    if not bool(payload.get("available")):
        return (
            "# Public Outcome Calibration Comparison\n\n"
            f"- Status: unavailable\n"
            f"- Message: {payload.get('message') or payload.get('reason') or 'No baseline run available.'}\n"
        )
    delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
    return "\n".join(
        [
            "# Public Outcome Calibration Comparison",
            "",
            f"- Current run: `{payload.get('run_id')}`",
            f"- Baseline run: `{payload.get('baseline_run_id')}`",
            f"- Likely change drivers: `{payload.get('likely_change_drivers')}`",
            "",
            "## Core Metrics Delta (current - previous)",
            f"- Sample count: `{delta.get('sample_count')}`",
            f"- Prevalence: `{delta.get('prevalence')}`",
            f"- Pre Brier: `{delta.get('pre_brier')}`",
            f"- Post Brier: `{delta.get('post_brier')}`",
            f"- Brier improvement delta: `{delta.get('brier_improvement')}`",
            f"- Pre calibration error delta: `{delta.get('pre_calibration_error')}`",
            f"- Post calibration error delta: `{delta.get('post_calibration_error')}`",
            f"- Log-loss improvement delta: `{delta.get('log_loss_improvement')}`",
            f"- Post ROC AUC delta: `{delta.get('post_roc_auc_probability')}`",
            f"- Post PR AUC delta: `{delta.get('post_pr_auc_probability')}`",
            "",
            "## Direction Signals",
            f"- `{payload.get('overall_direction_signals') or []}`",
            "",
            "## Caveat",
            "- This compares calibration fit quality against public observed outcomes only.",
            "",
        ]
    )
