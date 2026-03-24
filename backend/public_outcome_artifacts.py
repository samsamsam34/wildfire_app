from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.public_outcome_governance import (
    build_calibration_run_comparison,
    build_validation_run_comparison,
    list_public_outcome_runs,
    resolve_baseline_run_id,
    resolve_public_outcome_calibration_root,
    resolve_public_outcome_validation_root,
)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sufficiency_from_count(sample_size: int) -> dict[str, Any]:
    n = max(0, int(sample_size))
    if n < 20:
        return {
            "sample_size": n,
            "tier": "insufficient",
            "explanation": f"Sample size {n} is below 20; reliability is low.",
        }
    if n < 100:
        return {
            "sample_size": n,
            "tier": "limited",
            "explanation": f"Sample size {n} is between 20 and 99; interpret with caution.",
        }
    if n <= 500:
        return {
            "sample_size": n,
            "tier": "moderate",
            "explanation": f"Sample size {n} is between 100 and 500; reliability is moderate.",
        }
    return {
        "sample_size": n,
        "tier": "strong",
        "explanation": f"Sample size {n} exceeds 500; reliability is comparatively strong.",
    }


def _load_run_payload(
    *,
    root: Path,
    run_id: str,
    files: dict[str, str],
) -> dict[str, Any]:
    run_dir = root / run_id
    out = {"run_id": run_id, "run_dir": str(run_dir)}
    for key, filename in files.items():
        payload = _safe_load_json(run_dir / filename)
        out[key] = payload
    summary_path = run_dir / "summary.md"
    out["summary_markdown"] = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
    return out


def _validation_summary(manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    sample = report.get("sample_counts") if isinstance(report.get("sample_counts"), dict) else {}
    discr = report.get("discrimination_metrics") if isinstance(report.get("discrimination_metrics"), dict) else {}
    brier = report.get("brier_scores") if isinstance(report.get("brier_scores"), dict) else {}
    cal = report.get("calibration_metrics") if isinstance(report.get("calibration_metrics"), dict) else {}
    wildfire_cal = cal.get("wildfire_risk_score") if isinstance(cal.get("wildfire_risk_score"), dict) else {}
    review = report.get("false_review_sets") if isinstance(report.get("false_review_sets"), dict) else {}
    sufficiency = (
        report.get("data_sufficiency_indicator")
        if isinstance(report.get("data_sufficiency_indicator"), dict)
        else {}
    )
    subset = report.get("subset_metrics") if isinstance(report.get("subset_metrics"), dict) else {}
    slice_metrics = report.get("slice_metrics") if isinstance(report.get("slice_metrics"), dict) else {}
    by_confidence = (
        slice_metrics.get("by_confidence_tier")
        if isinstance(slice_metrics.get("by_confidence_tier"), dict)
        else {}
    )
    high_conf_subset = (
        subset.get("high_confidence_subset")
        if isinstance(subset.get("high_confidence_subset"), dict)
        else {}
    )
    total_count = int(sample.get("row_count_usable") or 0)
    high_conf_count = int(high_conf_subset.get("count") or 0)
    if high_conf_count <= 0:
        high_conf_bucket = (
            by_confidence.get("high")
            if isinstance(by_confidence.get("high"), dict)
            else {}
        )
        high_conf_count = int(high_conf_bucket.get("count") or 0)
    total_sufficiency = (
        sufficiency.get("total_dataset")
        if isinstance(sufficiency.get("total_dataset"), dict)
        else _sufficiency_from_count(total_count)
    )
    high_conf_sufficiency = (
        sufficiency.get("high_confidence_subset")
        if isinstance(sufficiency.get("high_confidence_subset"), dict)
        else _sufficiency_from_count(high_conf_count)
    )
    return {
        "available": True,
        "run_id": manifest.get("run_id"),
        "generated_at": manifest.get("generated_at"),
        "sample_count": total_count,
        "prevalence": _safe_float(sample.get("positive_rate")),
        "roc_auc": _safe_float(discr.get("wildfire_risk_score_auc")),
        "pr_auc": _safe_float(discr.get("wildfire_risk_score_pr_auc")),
        "brier_score": _safe_float(brier.get("wildfire_probability_proxy")),
        "calibration_error": _safe_float(wildfire_cal.get("expected_calibration_error")),
        "false_low_count": int(review.get("false_low_count") or 0),
        "false_high_count": int(review.get("false_high_count") or 0),
        "data_sufficiency": {
            "total_dataset": total_sufficiency,
            "high_confidence_subset": high_conf_sufficiency,
        },
        "confidence_tier_slice_count": len(
            ((report.get("slice_metrics") or {}).get("by_confidence_tier") or {})
            if isinstance(report.get("slice_metrics"), dict)
            else {}
        ),
        "raw_vs_calibrated_within_run": (
            {
                "available": (
                    discr.get("calibrated_damage_likelihood_auc") is not None
                    or brier.get("calibrated_damage_likelihood") is not None
                ),
                "raw_roc_auc": _safe_float(discr.get("wildfire_risk_score_auc")),
                "calibrated_roc_auc": _safe_float(discr.get("calibrated_damage_likelihood_auc")),
                "raw_brier": _safe_float(brier.get("wildfire_probability_proxy")),
                "calibrated_brier": _safe_float(brier.get("calibrated_damage_likelihood")),
            }
        ),
    }


def _calibration_summary(
    manifest: dict[str, Any],
    pre_post: dict[str, Any],
    calibration_model: dict[str, Any] | None,
) -> dict[str, Any]:
    pre = pre_post.get("pre") if isinstance(pre_post.get("pre"), dict) else {}
    post = pre_post.get("post") if isinstance(pre_post.get("post"), dict) else {}
    delta = pre_post.get("delta") if isinstance(pre_post.get("delta"), dict) else {}
    return {
        "available": True,
        "run_id": manifest.get("run_id"),
        "generated_at": manifest.get("generated_at"),
        "fitted": bool(manifest.get("fitted")),
        "method": (calibration_model or {}).get("method"),
        "sample_count": int(pre.get("row_count") or post.get("row_count") or 0),
        "prevalence": _safe_float(pre.get("positive_rate") or post.get("positive_rate")),
        "pre_brier": _safe_float(pre.get("brier_probability")),
        "post_brier": _safe_float(post.get("brier_probability")),
        "brier_improvement": _safe_float(delta.get("brier_improvement")),
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
        "post_roc_auc_probability": _safe_float(post.get("roc_auc_probability")),
        "post_pr_auc_probability": _safe_float(post.get("pr_auc_probability")),
    }


def _compare_validation_runs(root: Path, run_id: str, baseline_run_id: str | None) -> dict[str, Any]:
    listing = list_public_outcome_runs(artifact_root=root)
    ordered_ids = [
        str(item.get("run_id"))
        for item in (listing.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    ]
    if run_id not in ordered_ids:
        return {
            "available": False,
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "run_not_found",
            "message": f"Validation run '{run_id}' was not found.",
            "available_run_ids": ordered_ids,
        }
    if baseline_run_id and str(baseline_run_id).strip() not in ordered_ids:
        return {
            "available": False,
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "baseline_run_not_found",
            "message": f"Validation baseline run '{baseline_run_id}' was not found.",
            "available_run_ids": ordered_ids,
        }
    baseline = resolve_baseline_run_id(
        ordered_run_ids=ordered_ids,
        current_run_id=run_id,
        baseline_run_id=baseline_run_id,
    )
    current = _load_run_payload(
        root=root,
        run_id=run_id,
        files={"manifest": "manifest.json", "validation_metrics": "validation_metrics.json"},
    )
    baseline_payload = (
        _load_run_payload(
            root=root,
            run_id=baseline,
            files={"manifest": "manifest.json", "validation_metrics": "validation_metrics.json"},
        )
        if baseline
        else None
    )
    return build_validation_run_comparison(
        current_run_id=run_id,
        current_manifest=current.get("manifest") if isinstance(current.get("manifest"), dict) else {},
        current_report=current.get("validation_metrics")
        if isinstance(current.get("validation_metrics"), dict)
        else {},
        baseline_run_id=baseline,
        baseline_manifest=(
            baseline_payload.get("manifest")
            if baseline_payload is not None and isinstance(baseline_payload.get("manifest"), dict)
            else None
        ),
        baseline_report=(
            baseline_payload.get("validation_metrics")
            if baseline_payload is not None and isinstance(baseline_payload.get("validation_metrics"), dict)
            else None
        ),
    )


def _compare_calibration_runs(root: Path, run_id: str, baseline_run_id: str | None) -> dict[str, Any]:
    listing = list_public_outcome_runs(artifact_root=root)
    ordered_ids = [
        str(item.get("run_id"))
        for item in (listing.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    ]
    if run_id not in ordered_ids:
        return {
            "available": False,
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "run_not_found",
            "message": f"Calibration run '{run_id}' was not found.",
            "available_run_ids": ordered_ids,
        }
    if baseline_run_id and str(baseline_run_id).strip() not in ordered_ids:
        return {
            "available": False,
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "reason": "baseline_run_not_found",
            "message": f"Calibration baseline run '{baseline_run_id}' was not found.",
            "available_run_ids": ordered_ids,
        }
    baseline = resolve_baseline_run_id(
        ordered_run_ids=ordered_ids,
        current_run_id=run_id,
        baseline_run_id=baseline_run_id,
    )
    current = _load_run_payload(
        root=root,
        run_id=run_id,
        files={"manifest": "manifest.json", "pre_vs_post_metrics": "pre_vs_post_metrics.json"},
    )
    baseline_payload = (
        _load_run_payload(
            root=root,
            run_id=baseline,
            files={"manifest": "manifest.json", "pre_vs_post_metrics": "pre_vs_post_metrics.json"},
        )
        if baseline
        else None
    )
    return build_calibration_run_comparison(
        current_run_id=run_id,
        current_manifest=current.get("manifest") if isinstance(current.get("manifest"), dict) else {},
        current_pre_post=current.get("pre_vs_post_metrics")
        if isinstance(current.get("pre_vs_post_metrics"), dict)
        else {},
        baseline_run_id=baseline,
        baseline_manifest=(
            baseline_payload.get("manifest")
            if baseline_payload is not None and isinstance(baseline_payload.get("manifest"), dict)
            else None
        ),
        baseline_pre_post=(
            baseline_payload.get("pre_vs_post_metrics")
            if baseline_payload is not None and isinstance(baseline_payload.get("pre_vs_post_metrics"), dict)
            else None
        ),
    )


def _select_run_id(listing: dict[str, Any], requested_run_id: str | None) -> tuple[str | None, str | None]:
    ordered_ids = [
        str(item.get("run_id"))
        for item in (listing.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    ]
    if not ordered_ids:
        return None, "No runs are available in artifact listing."
    if requested_run_id:
        requested = str(requested_run_id).strip()
        if requested in ordered_ids:
            return requested, None
        return None, f"Run '{requested}' was not found."
    latest = str(listing.get("latest_run_id") or ordered_ids[0])
    if latest in ordered_ids:
        return latest, None
    return ordered_ids[0], None


def load_public_outcome_governance_snapshot(
    *,
    validation_run_id: str | None = None,
    validation_baseline_run_id: str | None = None,
    calibration_run_id: str | None = None,
    calibration_baseline_run_id: str | None = None,
) -> dict[str, Any]:
    validation_root = resolve_public_outcome_validation_root()
    calibration_root = resolve_public_outcome_calibration_root()
    validation_listing = list_public_outcome_runs(artifact_root=validation_root)
    calibration_listing = list_public_outcome_runs(artifact_root=calibration_root)

    validation_summary: dict[str, Any]
    validation_comparison: dict[str, Any]
    if not bool(validation_listing.get("available")):
        validation_summary = {
            "available": False,
            "message": validation_listing.get("message"),
        }
        validation_comparison = {
            "available": False,
            "message": validation_listing.get("message"),
        }
        selected_validation_run_id = None
    else:
        selected_validation_run_id, validation_select_error = _select_run_id(
            validation_listing,
            validation_run_id,
        )
        if validation_select_error:
            validation_summary = {
                "available": False,
                "message": validation_select_error,
                "available_run_ids": [
                    str(item.get("run_id"))
                    for item in (validation_listing.get("runs") or [])
                    if isinstance(item, dict) and item.get("run_id")
                ],
            }
            validation_comparison = {
                "available": False,
                "message": validation_select_error,
            }
        else:
            assert selected_validation_run_id is not None
            validation_bundle = _load_run_payload(
                root=validation_root,
                run_id=selected_validation_run_id,
                files={"manifest": "manifest.json", "validation_metrics": "validation_metrics.json"},
            )
            manifest = (
                validation_bundle.get("manifest")
                if isinstance(validation_bundle.get("manifest"), dict)
                else {}
            )
            metrics = (
                validation_bundle.get("validation_metrics")
                if isinstance(validation_bundle.get("validation_metrics"), dict)
                else {}
            )
            validation_summary = _validation_summary(manifest, metrics)
            validation_comparison = _compare_validation_runs(
                validation_root,
                selected_validation_run_id,
                baseline_run_id=validation_baseline_run_id,
            )

    calibration_summary: dict[str, Any]
    calibration_comparison: dict[str, Any]
    if not bool(calibration_listing.get("available")):
        calibration_summary = {
            "available": False,
            "message": calibration_listing.get("message"),
        }
        calibration_comparison = {
            "available": False,
            "message": calibration_listing.get("message"),
        }
        selected_calibration_run_id = None
    else:
        selected_calibration_run_id, calibration_select_error = _select_run_id(
            calibration_listing,
            calibration_run_id,
        )
        if calibration_select_error:
            calibration_summary = {
                "available": False,
                "message": calibration_select_error,
                "available_run_ids": [
                    str(item.get("run_id"))
                    for item in (calibration_listing.get("runs") or [])
                    if isinstance(item, dict) and item.get("run_id")
                ],
            }
            calibration_comparison = {
                "available": False,
                "message": calibration_select_error,
            }
        else:
            assert selected_calibration_run_id is not None
            calibration_bundle = _load_run_payload(
                root=calibration_root,
                run_id=selected_calibration_run_id,
                files={
                    "manifest": "manifest.json",
                    "pre_vs_post_metrics": "pre_vs_post_metrics.json",
                    "calibration_model": "calibration_model.json",
                },
            )
            manifest = (
                calibration_bundle.get("manifest")
                if isinstance(calibration_bundle.get("manifest"), dict)
                else {}
            )
            pre_post = (
                calibration_bundle.get("pre_vs_post_metrics")
                if isinstance(calibration_bundle.get("pre_vs_post_metrics"), dict)
                else {}
            )
            calibration_model = (
                calibration_bundle.get("calibration_model")
                if isinstance(calibration_bundle.get("calibration_model"), dict)
                else {}
            )
            calibration_summary = _calibration_summary(manifest, pre_post, calibration_model)
            calibration_comparison = _compare_calibration_runs(
                calibration_root,
                selected_calibration_run_id,
                baseline_run_id=calibration_baseline_run_id,
            )

    return {
        "available": bool(validation_listing.get("available") or calibration_listing.get("available")),
        "caveat": (
            "Public-outcome reports are directional validation artifacts based on public observed outcomes. "
            "They are not carrier-claims truth or underwriting-performance guarantees."
        ),
        "selected_run_ids": {
            "validation_run_id": selected_validation_run_id,
            "validation_baseline_run_id": validation_baseline_run_id,
            "calibration_run_id": selected_calibration_run_id,
            "calibration_baseline_run_id": calibration_baseline_run_id,
        },
        "validation": {
            "artifact_root": str(validation_root),
            "listing": validation_listing,
            "latest_summary": validation_summary,
            "comparison_to_previous": validation_comparison,
        },
        "calibration": {
            "artifact_root": str(calibration_root),
            "listing": calibration_listing,
            "latest_summary": calibration_summary,
            "comparison_to_previous": calibration_comparison,
        },
    }
