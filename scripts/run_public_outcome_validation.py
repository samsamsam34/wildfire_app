#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.public_outcome_validation import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    evaluate_public_outcome_dataset_file,
    trace_public_outcome_dataset_flow,
    write_evaluation_rows_csv,
)
from backend.public_outcome_governance import (  # noqa: E402
    build_validation_comparison_markdown,
    build_validation_run_comparison,
    list_public_outcome_runs,
    resolve_baseline_run_id,
)
from backend.version import (  # noqa: E402
    API_VERSION,
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    SCORING_MODEL_VERSION,
)

DEFAULT_EVALUATION_DATASET_ROOT = Path("benchmark/public_outcomes/evaluation_dataset")
DEFAULT_VALIDATION_OUTPUT_ROOT = Path("benchmark/public_outcomes/validation")


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
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


def _load_and_filter_high_signal_weights_from_feature_signal_report(
    path: Path,
    *,
    min_feature_auc: float = 0.55,
    min_feature_stddev: float = 1e-6,
) -> tuple[dict[str, float], dict[str, Any]]:
    payload = _safe_load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid feature signal report JSON: {path}")
    candidates: list[dict[str, Any]] = []
    for key in ("feature_ranking_strongest_to_weakest", "top_predictive_features", "weak_or_noisy_features"):
        rows = payload.get(key)
        if isinstance(rows, list):
            candidates.extend([row for row in rows if isinstance(row, dict)])
    signal_by_feature: dict[str, float] = {}
    auc_by_feature: dict[str, float] = {}
    for row in candidates:
        feature_name = str(row.get("feature") or "").strip()
        if not feature_name:
            continue
        signal = _safe_float(row.get("signal_score"))
        if signal is None:
            signal = 0.0
        if feature_name not in signal_by_feature or float(signal) > float(signal_by_feature[feature_name]):
            signal_by_feature[feature_name] = float(signal)
        best_auc = _safe_float(row.get("best_auc"))
        if best_auc is not None:
            auc_strength = max(0.0, min(0.5, float(best_auc) - 0.5))
            if feature_name not in auc_by_feature or auc_strength > float(auc_by_feature[feature_name]):
                auc_by_feature[feature_name] = float(auc_strength)

    required = [
        "nearest_high_fuel_patch_distance_ft",
        "canopy_adjacency_proxy_pct",
        "vegetation_continuity_proxy_pct",
        "slope_index",
    ]
    out: dict[str, float] = {}
    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for feature_name in required:
        feature_row = next(
            (
                row
                for row in candidates
                if str(row.get("feature") or "").strip() == feature_name
            ),
            {},
        )
        best_auc = _safe_float(feature_row.get("best_auc"))
        feature_stddev = _safe_float(feature_row.get("feature_stddev"))
        auc_strength = auc_by_feature.get(feature_name)
        signal = signal_by_feature.get(feature_name)

        exclude_reasons: list[str] = []
        if best_auc is None:
            exclude_reasons.append("missing_best_auc")
        elif float(best_auc) < float(min_feature_auc):
            exclude_reasons.append(
                f"best_auc_below_threshold({best_auc:.4f}<{float(min_feature_auc):.4f})"
            )
        if feature_stddev is not None and abs(float(feature_stddev)) < float(min_feature_stddev):
            exclude_reasons.append(
                f"feature_stddev_below_threshold({feature_stddev:.6f}<{float(min_feature_stddev):.6f})"
            )
        if signal is None and auc_strength is None:
            exclude_reasons.append("missing_signal_and_auc_strength")

        if exclude_reasons:
            excluded_rows.append(
                {
                    "feature": feature_name,
                    "best_auc": best_auc,
                    "feature_stddev": feature_stddev,
                    "signal_score": signal,
                    "reasons": exclude_reasons,
                }
            )
            continue

        if auc_strength is None:
            # Fall back to signal-score scaling when best_auc is present but parsing
            # failed for auc-strength extraction.
            out[feature_name] = max(0.01, float(signal or 0.01))
        else:
            # Exponential emphasis on stronger univariate AUC separation so weak
            # features (for example low-AUC slope variants) do not dominate.
            out[feature_name] = max(1e-6, float(auc_strength) ** 4)
        included_rows.append(
            {
                "feature": feature_name,
                "best_auc": best_auc,
                "feature_stddev": feature_stddev,
                "signal_score": signal,
                "raw_weight": out[feature_name],
            }
        )
    if not out:
        raise ValueError(
            "No high-signal features met thresholds. "
            f"min_feature_auc={float(min_feature_auc):.4f}, min_feature_stddev={float(min_feature_stddev):.6f}. "
            f"excluded={excluded_rows}"
        )
    summary = {
        "feature_signal_report_path": str(path),
        "thresholds": {
            "min_feature_auc": float(min_feature_auc),
            "min_feature_stddev": float(min_feature_stddev),
        },
        "included_feature_count": len(included_rows),
        "excluded_feature_count": len(excluded_rows),
        "included_features": included_rows,
        "excluded_features": excluded_rows,
    }
    return out, summary


def _load_high_signal_weights_from_feature_signal_report(path: Path) -> dict[str, float]:
    weights, _summary = _load_and_filter_high_signal_weights_from_feature_signal_report(path)
    return weights


def _extract_feature_input_versions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    observed: dict[str, set[str]] = {
        "scoring_model_versions": set(),
        "factor_schema_versions": set(),
        "rules_logic_versions": set(),
        "region_data_versions": set(),
        "data_bundle_versions": set(),
    }
    for row in rows:
        governance = row.get("model_governance") if isinstance(row.get("model_governance"), dict) else {}
        for key, field in (
            ("scoring_model_versions", "scoring_model_version"),
            ("factor_schema_versions", "factor_schema_version"),
            ("rules_logic_versions", "rules_logic_version"),
            ("region_data_versions", "region_data_version"),
            ("data_bundle_versions", "data_bundle_version"),
        ):
            text = str(governance.get(field) or "").strip()
            if text:
                observed[key].add(text)
    return {key: sorted(values) for key, values in observed.items() if values}


def _dataset_governance(dataset_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_manifest_path = dataset_path.parent / "manifest.json"
    dataset_manifest = _safe_load_json(dataset_manifest_path) if dataset_manifest_path.exists() else None
    dataset_inputs = dataset_manifest.get("inputs") if isinstance(dataset_manifest, dict) else {}
    outcomes_path: Path | None = None
    if isinstance(dataset_inputs, dict):
        single = dataset_inputs.get("normalized_outcomes_path")
        multiple = dataset_inputs.get("normalized_outcomes_paths")
        if isinstance(single, str) and single.strip():
            outcomes_path = Path(single).expanduser()
        elif isinstance(multiple, list) and multiple:
            first = next((str(item) for item in multiple if str(item).strip()), "")
            if first:
                outcomes_path = Path(first).expanduser()
    outcomes_manifest_path = outcomes_path.parent / "manifest.json" if outcomes_path else None
    outcomes_manifest = (
        _safe_load_json(outcomes_manifest_path)
        if outcomes_manifest_path is not None and outcomes_manifest_path.exists()
        else None
    )
    return {
        "evaluation_dataset_run_id": dataset_path.parent.name,
        "evaluation_dataset_schema_version": (
            dataset_manifest.get("schema_version")
            if isinstance(dataset_manifest, dict)
            else None
        ),
        "evaluation_dataset_manifest_path": (
            str(dataset_manifest_path) if dataset_manifest_path.exists() else None
        ),
        "outcome_dataset_run_id": (
            outcomes_manifest.get("run_id")
            if isinstance(outcomes_manifest, dict)
            else None
        ),
        "outcome_dataset_schema_version": (
            outcomes_manifest.get("schema_version")
            if isinstance(outcomes_manifest, dict)
            else None
        ),
        "outcome_dataset_manifest_path": (
            str(outcomes_manifest_path)
            if outcomes_manifest_path is not None and outcomes_manifest_path.exists()
            else None
        ),
        "feature_input_versions": _extract_feature_input_versions(rows),
    }


def _dataset_join_stage_counts(dataset_path: Path) -> dict[str, Any]:
    join_path = dataset_path.parent / "join_quality_report.json"
    if not join_path.exists():
        return {}
    payload = _safe_load_json(join_path)
    if not isinstance(payload, dict):
        return {}
    return {
        "outcomes_loaded": int(payload.get("total_outcomes_loaded") or 0),
        "feature_rows_loaded": int(payload.get("total_feature_rows_loaded") or 0),
        "joined_rows": int(payload.get("total_joined_records") or 0),
        "join_rate": payload.get("join_rate"),
        "excluded_rows": int(payload.get("excluded_row_count") or 0),
        "join_confidence_tier_counts": (
            payload.get("join_confidence_tier_counts")
            if isinstance(payload.get("join_confidence_tier_counts"), dict)
            else {}
        ),
        "join_confidence_non_high_reason_counts": (
            payload.get("join_confidence_non_high_reason_counts")
            if isinstance(payload.get("join_confidence_non_high_reason_counts"), dict)
            else {}
        ),
        "high_confidence_threshold_diagnostics": (
            payload.get("high_confidence_threshold_diagnostics")
            if isinstance(payload.get("high_confidence_threshold_diagnostics"), dict)
            else {}
        ),
        "score_backfill": payload.get("score_backfill") if isinstance(payload.get("score_backfill"), dict) else {},
        "retention_fallback": payload.get("retention_fallback") if isinstance(payload.get("retention_fallback"), dict) else {},
        "retention_fallback_triggered": bool(payload.get("retention_fallback_triggered")),
        "retention_fallback_used": bool(payload.get("retention_fallback_used")),
    }


def _count_rows_in_dataset(dataset_path: Path) -> int:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as fh:
            return sum(1 for _ in fh) - 1
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return len(payload["rows"])
        if isinstance(payload.get("records"), list):
            return len(payload["records"])
    if isinstance(payload, list):
        return len(payload)
    return 0


def _resolve_latest_dataset_jsonl(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise ValueError(
            f"Evaluation dataset root does not exist: {dataset_root}. "
            "Run scripts/build_public_outcome_evaluation_dataset.py first."
        )
    run_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidate = run_dir / "evaluation_dataset.jsonl"
        if candidate.exists():
            return candidate
    raise ValueError(
        f"No evaluation_dataset.jsonl files found under {dataset_root}. "
        "Run scripts/build_public_outcome_evaluation_dataset.py first."
    )


def _resolve_dataset_path(
    *,
    dataset_path: Path | None,
    dataset_root: Path,
    dataset_run_id: str | None,
) -> Path:
    if dataset_path is not None:
        resolved = dataset_path.expanduser()
        if not resolved.exists():
            raise ValueError(f"Evaluation dataset not found: {resolved}")
        return resolved
    if dataset_run_id:
        run_dir = dataset_root / str(dataset_run_id)
        candidate_jsonl = run_dir / "evaluation_dataset.jsonl"
        candidate_json = run_dir / "evaluation_dataset.json"
        if candidate_jsonl.exists():
            return candidate_jsonl
        if candidate_json.exists():
            return candidate_json
        raise ValueError(
            f"Run '{dataset_run_id}' not found under {dataset_root} or missing evaluation_dataset.jsonl."
        )
    return _resolve_latest_dataset_jsonl(dataset_root)


def _format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.4f}"


def _format_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return "n/a"
    low = value.get("low")
    high = value.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "n/a"
    return f"[{float(low):.4f}, {float(high):.4f}]"


def _build_summary_markdown(
    *,
    run_id: str,
    generated_at: str,
    dataset_path: Path,
    report: dict[str, Any],
) -> str:
    sample_counts = report.get("sample_counts") if isinstance(report.get("sample_counts"), dict) else {}
    discrimination = report.get("discrimination_metrics") if isinstance(report.get("discrimination_metrics"), dict) else {}
    brier = report.get("brier_scores") if isinstance(report.get("brier_scores"), dict) else {}
    default_threshold = report.get("default_threshold_70") if isinstance(report.get("default_threshold_70"), dict) else {}
    guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
    metric_stability = report.get("metric_stability") if isinstance(report.get("metric_stability"), dict) else {}
    narrative = report.get("narrative_summary") if isinstance(report.get("narrative_summary"), dict) else {}
    minimum_viable = report.get("minimum_viable_metrics") if isinstance(report.get("minimum_viable_metrics"), dict) else {}
    data_suff = report.get("data_sufficiency_flags") if isinstance(report.get("data_sufficiency_flags"), dict) else {}
    data_suff_indicator = (
        report.get("data_sufficiency_indicator")
        if isinstance(report.get("data_sufficiency_indicator"), dict)
        else {}
    )
    slice_metrics = report.get("slice_metrics") if isinstance(report.get("slice_metrics"), dict) else {}
    proxy_validation = report.get("proxy_validation") if isinstance(report.get("proxy_validation"), dict) else {}
    synthetic_validation = report.get("synthetic_validation") if isinstance(report.get("synthetic_validation"), dict) else {}
    subset_metrics = report.get("subset_metrics") if isinstance(report.get("subset_metrics"), dict) else {}
    confidence_tier_performance = (
        report.get("confidence_tier_performance")
        if isinstance(report.get("confidence_tier_performance"), dict)
        else {}
    )
    segment_summary = (
        report.get("segment_performance_summary")
        if isinstance(report.get("segment_performance_summary"), dict)
        else {}
    )
    baseline_comparison = (
        report.get("baseline_model_comparison")
        if isinstance(report.get("baseline_model_comparison"), dict)
        else {}
    )
    modeling_viability = (
        report.get("modeling_viability")
        if isinstance(report.get("modeling_viability"), dict)
        else {}
    )
    reported_metrics = (
        report.get("reported_metrics")
        if isinstance(report.get("reported_metrics"), dict)
        else {}
    )
    reported_primary = (
        reported_metrics.get("primary")
        if isinstance(reported_metrics.get("primary"), dict)
        else {}
    )
    reported_in_sample = (
        reported_metrics.get("in_sample")
        if isinstance(reported_metrics.get("in_sample"), dict)
        else {}
    )
    reported_holdout = (
        reported_metrics.get("holdout_event_level_out_of_sample")
        if isinstance(reported_metrics.get("holdout_event_level_out_of_sample"), dict)
        else {}
    )
    reporting_claim_guardrails = (
        report.get("reporting_claim_guardrails")
        if isinstance(report.get("reporting_claim_guardrails"), dict)
        else {}
    )
    feature_diag = (
        report.get("feature_signal_diagnostics")
        if isinstance(report.get("feature_signal_diagnostics"), dict)
        else {}
    )
    review_sets = report.get("false_review_sets") if isinstance(report.get("false_review_sets"), dict) else {}
    calibration_metrics = report.get("calibration_metrics") if isinstance(report.get("calibration_metrics"), dict) else {}
    wildfire_calibration = (
        calibration_metrics.get("wildfire_risk_score")
        if isinstance(calibration_metrics.get("wildfire_risk_score"), dict)
        else {}
    )

    lines = [
        "# Public Outcome Validation",
        "",
        "Validation against public observed wildfire outcomes.",
        "This is directional validation and not carrier claims truth or underwriting-performance truth.",
        "Outcome target: `structure_loss_or_major_damage` (major damage or destroyed = 1).",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Input labeled dataset: `{dataset_path}`",
        f"- Retained rows: `{sample_counts.get('row_count_retained')}`",
        f"- Usable labeled rows: `{sample_counts.get('row_count_usable')}`",
        f"- Unusable retained rows: `{sample_counts.get('row_count_unusable')}`",
        f"- Outcome prevalence (adverse): `{_format_float(sample_counts.get('positive_rate'))}`",
        f"- Reported metrics source: `{reported_metrics.get('primary_source') or 'in_sample_fallback'}`",
        f"- Narrative headline: `{str(narrative.get('headline') or 'n/a')}`",
        "",
        "## Headline Metrics (Strict Default)",
        f"- ROC AUC: `{_format_float(reported_primary.get('wildfire_risk_score_auc'))}`",
        f"- PR AUC: `{_format_float(reported_primary.get('wildfire_risk_score_pr_auc'))}`",
        f"- Brier: `{_format_float(reported_primary.get('wildfire_risk_score_brier'))}`",
        f"- Row count / positives / negatives: "
        f"`{reported_primary.get('count')} / {reported_primary.get('positive_count')} / {reported_primary.get('negative_count')}`",
        "",
        "## In-Sample Discrimination",
        f"- ROC AUC (wildfire risk): `{_format_float(discrimination.get('wildfire_risk_score_auc'))}`",
        f"- ROC AUC 95% CI: `{_format_ci(discrimination.get('wildfire_risk_score_auc_confidence_interval_95'))}`",
        f"- PR AUC (wildfire risk): `{_format_float(discrimination.get('wildfire_risk_score_pr_auc'))}`",
        f"- PR AUC 95% CI: `{_format_ci(discrimination.get('wildfire_risk_score_pr_auc_confidence_interval_95'))}`",
        f"- Spearman rank correlation: `{_format_float(discrimination.get('wildfire_vs_outcome_rank_spearman'))}`",
        f"- Discrimination stability: `{discrimination.get('wildfire_discrimination_stability')}`",
        f"- In-sample row count / positives / negatives: "
        f"`{reported_in_sample.get('count')} / {reported_in_sample.get('positive_count')} / {reported_in_sample.get('negative_count')}`",
        "",
        "## Event-Level Holdout (Out-of-Sample)",
        f"- ROC AUC: `{_format_float(reported_holdout.get('wildfire_risk_score_auc'))}`",
        f"- PR AUC: `{_format_float(reported_holdout.get('wildfire_risk_score_pr_auc'))}`",
        f"- Brier: `{_format_float(reported_holdout.get('wildfire_risk_score_brier'))}`",
        f"- Holdout row count / positives / negatives: "
        f"`{reported_holdout.get('count')} / {reported_holdout.get('positive_count')} / {reported_holdout.get('negative_count')}`",
        "",
        "## Reporting Guardrails",
        f"- Strong performance claims allowed: `{reporting_claim_guardrails.get('strong_performance_claims_allowed')}`",
        f"- Calibration claims allowed: `{reporting_claim_guardrails.get('calibration_claims_allowed')}`",
        f"- Failed checks: `{reporting_claim_guardrails.get('failed_checks')}`",
        "",
        "## Baseline Comparison",
    ]

    if baseline_comparison:
        baseline_rows = (
            baseline_comparison.get("baselines")
            if isinstance(baseline_comparison.get("baselines"), dict)
            else {}
        )
        comparison = (
            baseline_comparison.get("comparison")
            if isinstance(baseline_comparison.get("comparison"), dict)
            else {}
        )
        lines.extend(
            [
                f"- Full model AUC: `{_format_float(baseline_comparison.get('full_model_auc'))}`",
                f"- Beats all simple baselines by AUC: `{comparison.get('beats_all_baselines_by_auc')}`",
                f"- Best baseline: `{comparison.get('best_baseline_name')}`",
                f"- Best baseline AUC: `{_format_float(comparison.get('best_baseline_auc'))}`",
                f"- AUC margin vs best baseline: `{_format_float(comparison.get('auc_margin_vs_best_baseline'))}`",
                f"- Complexity justified signal: `{comparison.get('complexity_justified_signal')}`",
            ]
        )
        if baseline_rows:
            lines.append("- Baseline AUCs:")
            for name in ("random", "hazard_only", "vegetation_only"):
                payload = baseline_rows.get(name) if isinstance(baseline_rows.get(name), dict) else {}
                if not payload:
                    continue
                lines.append(
                    f"  - {name}: auc={_format_float(payload.get('auc'))}, "
                    f"pr_auc={_format_float(payload.get('pr_auc'))}, "
                    f"brier={_format_float(payload.get('brier'))}, "
                    f"missing_signal_count={payload.get('missing_signal_count')}"
                )
        caveat = baseline_comparison.get("caveat")
        if isinstance(caveat, str) and caveat.strip():
            lines.append(f"- Caveat: {caveat}")
        lines.append("")
    else:
        lines.extend(
            [
                "- Baseline comparison unavailable.",
                "",
            ]
        )

    if modeling_viability:
        checks = (
            modeling_viability.get("checks")
            if isinstance(modeling_viability.get("checks"), dict)
            else {}
        )
        lines.extend(
            [
                "## Dataset Viability Guardrail",
                f"- Dataset viable for predictive modeling: `{modeling_viability.get('dataset_viable_for_predictive_modeling')}`",
                f"- Classification: `{modeling_viability.get('classification')}`",
                f"- Reason: `{modeling_viability.get('reason')}`",
                (
                    f"- Independent samples: `{checks.get('independent_sample_count')}` "
                    f"(labeled rows={checks.get('labeled_sample_count')}, duplication_factor={_format_float(checks.get('duplication_factor'))})"
                ),
                (
                    f"- Feature variance: `{checks.get('features_with_variance_count')}` / "
                    f"`{checks.get('feature_count_evaluated')}` varying "
                    f"(ratio={_format_float(checks.get('feature_variation_ratio'))})"
                ),
                (
                    f"- Model vs random AUC margin: `{_format_float(checks.get('auc_margin_vs_random_baseline'))}` "
                    f"(full_auc={_format_float(checks.get('full_model_auc'))}, random_auc={_format_float(checks.get('random_baseline_auc'))})"
                ),
            ]
        )
        lines.append("")

    lines.extend(
        [
        "## Calibration",
        f"- Brier score (raw wildfire probability proxy): `{_format_float(brier.get('wildfire_probability_proxy'))}`",
        f"- Brier score 95% CI: `{_format_ci(brier.get('wildfire_probability_proxy_confidence_interval_95'))}`",
        f"- ECE (raw wildfire risk): `{_format_float(wildfire_calibration.get('expected_calibration_error'))}`",
        "",
        "## Default Threshold (70)",
        f"- Precision: `{_format_float(default_threshold.get('precision'))}`",
        f"- Recall: `{_format_float(default_threshold.get('recall'))}`",
        f"- F1: `{_format_float(default_threshold.get('f1'))}`",
        "",
        "## Sliced Analysis Highlights",
    ]
    )

    narrative_bullets = narrative.get("bullets") if isinstance(narrative.get("bullets"), list) else []
    if narrative_bullets:
        lines.append("### Narrative Summary")
        for item in narrative_bullets[:8]:
            lines.append(f"- {item}")
        lines.append("")

    if minimum_viable:
        rank_order = minimum_viable.get("rank_ordering") if isinstance(minimum_viable.get("rank_ordering"), dict) else {}
        acc = minimum_viable.get("simple_accuracy_at_threshold") if isinstance(minimum_viable.get("simple_accuracy_at_threshold"), dict) else {}
        top_bucket = minimum_viable.get("top_risk_bucket_hit_rate") if isinstance(minimum_viable.get("top_risk_bucket_hit_rate"), dict) else {}
        deciles = minimum_viable.get("adverse_rate_by_score_decile") if isinstance(minimum_viable.get("adverse_rate_by_score_decile"), dict) else {}
        lines.append("### Minimum Viable Metrics")
        lines.append(f"- Rank-order hit rate: `{_format_float(rank_order.get('hit_rate'))}`")
        lines.append(f"- Accuracy@{acc.get('threshold')}: `{_format_float(acc.get('accuracy'))}`")
        lines.append(f"- Top-risk bucket adverse rate: `{_format_float(top_bucket.get('adverse_rate'))}`")
        lines.append(f"- Top-risk lift vs baseline: `{_format_float(top_bucket.get('lift_vs_baseline'))}`")
        lines.append(f"- Score-decile bins available: `{deciles.get('bin_count')}`")
        lines.append("")

    if data_suff:
        flags = data_suff.get("flags") if isinstance(data_suff.get("flags"), dict) else {}
        lines.append("### Data Sufficiency Flags")
        lines.append(f"- Small sample size: `{flags.get('small_sample_size')}`")
        lines.append(f"- Very small sample size: `{flags.get('very_small_sample_size')}`")
        lines.append(f"- Class imbalance: `{flags.get('class_imbalance')}`")
        lines.append(f"- Low join-confidence prevalent: `{flags.get('low_join_confidence_prevalent')}`")
        lines.append(f"- Fallback-heavy prevalent: `{flags.get('fallback_heavy_prevalent')}`")
        lines.append("")
    if data_suff_indicator:
        total = (
            data_suff_indicator.get("total_dataset")
            if isinstance(data_suff_indicator.get("total_dataset"), dict)
            else {}
        )
        high_conf = (
            data_suff_indicator.get("high_confidence_subset")
            if isinstance(data_suff_indicator.get("high_confidence_subset"), dict)
            else {}
        )
        lines.append("### Data Sufficiency Indicator")
        lines.append(
            f"- Total dataset sufficiency: `{total.get('tier')}` "
            f"(n={total.get('sample_size')})"
        )
        lines.append(
            f"- High-confidence subset sufficiency: `{high_conf.get('tier')}` "
            f"(n={high_conf.get('sample_size')})"
        )
        if total.get("explanation"):
            lines.append(f"- Total explanation: {total.get('explanation')}")
        if high_conf.get("explanation"):
            lines.append(f"- High-confidence explanation: {high_conf.get('explanation')}")
        lines.append("")
    if metric_stability:
        lines.append("### Metric Stability")
        lines.append(f"- AUC stable for interpretation: `{metric_stability.get('auc_stable')}`")
        lines.append(
            f"- Sample size / positives / negatives: "
            f"`{metric_stability.get('sample_size')} / {metric_stability.get('positive_count')} / {metric_stability.get('negative_count')}`"
        )
        instability_warnings = (
            metric_stability.get("warnings")
            if isinstance(metric_stability.get("warnings"), list)
            else []
        )
        if instability_warnings:
            lines.append("- Instability warnings:")
            for warning in instability_warnings:
                lines.append(f"  - {warning}")
        lines.append("")

    lines.append("## Supplemental Validation (Non-Ground-Truth)")
    if synthetic_validation:
        extreme = synthetic_validation.get("extreme_scenario_ranking") if isinstance(synthetic_validation.get("extreme_scenario_ranking"), dict) else {}
        lines.append("### Synthetic Stress Validation")
        lines.append(f"- Available: `{synthetic_validation.get('available')}`")
        lines.append(f"- Passed: `{synthetic_validation.get('passed')}`")
        lines.append(
            f"- Checks: `{synthetic_validation.get('check_count')}` "
            f"(pass={synthetic_validation.get('pass_count')}, fail={synthetic_validation.get('fail_count')})"
        )
        if extreme:
            lines.append(
                f"- Extreme high-vs-low ranking: `passed={extreme.get('passed')}` "
                f"(delta={_format_float(extreme.get('delta'))}, min_expected={_format_float(extreme.get('minimum_expected_delta'))})"
            )
        lines.append(
            "- Caveat: `Synthetic stress scenarios test directional behavior only; they are not real-outcome truth.`"
        )
    if proxy_validation:
        align = proxy_validation.get("alignment_metrics") if isinstance(proxy_validation.get("alignment_metrics"), dict) else {}
        lines.append("")
        lines.append("### Proxy Validation")
        lines.append(f"- Available: `{proxy_validation.get('available')}`")
        lines.append(f"- Rows with proxy index: `{proxy_validation.get('rows_with_proxy_index')}`")
        weak_counts = proxy_validation.get("weak_label_counts")
        lines.append(f"- Weak-label counts: `{weak_counts}`")
        lines.append(
            f"- Spearman(model, proxy index): `{_format_float(align.get('spearman_model_vs_proxy_index'))}`"
        )
        lines.append(
            f"- AUC(model vs high/low proxy labels): `{_format_float(align.get('auc_model_vs_proxy_high_low'))}`"
        )
        rank_align = align.get("rank_order_hit_rate_high_vs_low_proxy") if isinstance(align.get("rank_order_hit_rate_high_vs_low_proxy"), dict) else {}
        lines.append(
            f"- Rank-hit(high proxy vs low proxy): `{_format_float(rank_align.get('hit_rate'))}`"
        )
        lines.append(
            "- Caveat: `Proxy validation uses weak labels from perimeter/burn-probability-style signals and is not ground truth.`"
        )
    lines.append("")

    by_evidence = slice_metrics.get("by_evidence_group") if isinstance(slice_metrics.get("by_evidence_group"), dict) else {}
    by_confidence = slice_metrics.get("by_confidence_tier") if isinstance(slice_metrics.get("by_confidence_tier"), dict) else {}
    by_join = slice_metrics.get("by_join_confidence_tier") if isinstance(slice_metrics.get("by_join_confidence_tier"), dict) else {}
    by_hazard = slice_metrics.get("by_hazard_level") if isinstance(slice_metrics.get("by_hazard_level"), dict) else {}
    by_vegetation = slice_metrics.get("by_vegetation_density") if isinstance(slice_metrics.get("by_vegetation_density"), dict) else {}
    by_region = slice_metrics.get("by_region") if isinstance(slice_metrics.get("by_region"), dict) else {}
    if by_evidence:
        lines.append("### By Evidence Group")
        for name in sorted(by_evidence):
            detail = by_evidence[name] if isinstance(by_evidence[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"pr_auc={_format_float(detail.get('wildfire_risk_score_pr_auc'))}"
            )
    if by_confidence:
        lines.append("")
        lines.append("### By Confidence Tier")
        for name in sorted(by_confidence):
            detail = by_confidence[name] if isinstance(by_confidence[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if by_join:
        lines.append("")
        lines.append("### By Join-Confidence Tier")
        for name in sorted(by_join):
            detail = by_join[name] if isinstance(by_join[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"pr_auc={_format_float(detail.get('wildfire_risk_score_pr_auc'))}"
            )
    by_validation_tier = slice_metrics.get("by_validation_confidence_tier") if isinstance(slice_metrics.get("by_validation_confidence_tier"), dict) else {}
    if by_validation_tier:
        lines.append("")
        lines.append("### By Validation Confidence Tier")
        for name in sorted(by_validation_tier):
            detail = by_validation_tier[name] if isinstance(by_validation_tier[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if by_hazard:
        lines.append("")
        lines.append("### By Hazard Level Segment")
        for name in sorted(by_hazard):
            detail = by_hazard[name] if isinstance(by_hazard[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if by_vegetation:
        lines.append("")
        lines.append("### By Vegetation Density Segment")
        for name in sorted(by_vegetation):
            detail = by_vegetation[name] if isinstance(by_vegetation[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if by_region:
        lines.append("")
        lines.append("### By Region Segment")
        for name in sorted(by_region):
            detail = by_region[name] if isinstance(by_region[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if subset_metrics:
        lines.append("")
        lines.append("### Subset Evaluation")
        for name in ("full_dataset", "high_confidence_subset", "medium_confidence_subset", "high_evidence_subset"):
            detail = subset_metrics.get(name) if isinstance(subset_metrics.get(name), dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"pr_auc={_format_float(detail.get('wildfire_risk_score_pr_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}"
            )
    if confidence_tier_performance:
        tier_rows = confidence_tier_performance.get("tiers") if isinstance(confidence_tier_performance.get("tiers"), dict) else {}
        deltas = confidence_tier_performance.get("deltas_vs_all_data") if isinstance(confidence_tier_performance.get("deltas_vs_all_data"), dict) else {}
        warnings_by_tier = (
            confidence_tier_performance.get("warnings")
            if isinstance(confidence_tier_performance.get("warnings"), list)
            else []
        )
        lines.append("")
        lines.append("### Confidence-Tier Performance (All vs High vs Medium)")
        for name in ("all_data", "high_confidence", "medium_confidence"):
            detail = tier_rows.get(name) if isinstance(tier_rows.get(name), dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}, "
                f"small_sample_warning={detail.get('small_sample_warning')}"
            )
        lines.append(
            "- Delta vs all_data: "
            f"high_auc={_format_float(deltas.get('high_confidence_auc_delta'))}, "
            f"high_brier={_format_float(deltas.get('high_confidence_brier_delta'))}, "
            f"medium_auc={_format_float(deltas.get('medium_confidence_auc_delta'))}, "
            f"medium_brier={_format_float(deltas.get('medium_confidence_brier_delta'))}"
        )
        if warnings_by_tier:
            for warning in warnings_by_tier:
                lines.append(f"- Tier warning: {warning}")
    if segment_summary:
        highlights = (
            segment_summary.get("highlights")
            if isinstance(segment_summary.get("highlights"), list)
            else []
        )
        strongest = (
            segment_summary.get("strongest_segments")
            if isinstance(segment_summary.get("strongest_segments"), list)
            else []
        )
        weakest = (
            segment_summary.get("weakest_segments")
            if isinstance(segment_summary.get("weakest_segments"), list)
            else []
        )
        lines.append("")
        lines.append("### Segment Strengths and Weaknesses")
        lines.append(
            f"- Eligible segments (n >= {segment_summary.get('min_slice_size')}): "
            f"`{segment_summary.get('eligible_segment_count')}`"
        )
        for item in highlights[:4]:
            lines.append(f"- {item}")
        if strongest:
            top = strongest[0] if isinstance(strongest[0], dict) else {}
            lines.append(
                f"- Best segment: `{top.get('segment_family')}={top.get('segment_name')}` "
                f"(auc={_format_float(top.get('auc'))}, brier={_format_float(top.get('brier'))}, n={top.get('count')})"
            )
        if weakest:
            tail = weakest[0] if isinstance(weakest[0], dict) else {}
            lines.append(
                f"- Weakest segment: `{tail.get('segment_family')}={tail.get('segment_name')}` "
                f"(auc={_format_float(tail.get('auc'))}, brier={_format_float(tail.get('brier'))}, n={tail.get('count')})"
            )

    if feature_diag:
        top_features = (
            feature_diag.get("top_predictive_features")
            if isinstance(feature_diag.get("top_predictive_features"), list)
            else []
        )
        weak_features = (
            feature_diag.get("weak_or_noisy_features")
            if isinstance(feature_diag.get("weak_or_noisy_features"), list)
            else []
        )
        harmful_features = (
            feature_diag.get("potentially_harmful_features")
            if isinstance(feature_diag.get("potentially_harmful_features"), list)
            else []
        )
        family_summary = (
            feature_diag.get("key_feature_family_summary")
            if isinstance(feature_diag.get("key_feature_family_summary"), dict)
            else {}
        )
        direction_alignment = (
            feature_diag.get("direction_alignment")
            if isinstance(feature_diag.get("direction_alignment"), dict)
            else {}
        )
        lines.append("")
        lines.append("## Feature Signal Diagnostics")
        lines.append(
            f"- Features evaluated: `{feature_diag.get('feature_count_evaluated')}` "
            f"(rows used={feature_diag.get('row_count_used')})"
        )
        lines.append(
            "- Caveat: `Feature diagnostics reflect directional signal/noise in this labeled sample; "
            "they are not causal inference or insurer-claims truth.`"
        )
        if top_features:
            lines.append("### Top Predictive Features")
            for row in top_features[:8]:
                lines.append(
                    f"- `{row.get('feature')}`: signal={_format_float(row.get('signal_score'))}, "
                    f"spearman={_format_float(row.get('rank_correlation_with_outcome'))}, "
                    f"coverage={_format_float(row.get('coverage_fraction'))}, "
                    f"family={row.get('family')}"
                )
        if weak_features:
            lines.append("")
            lines.append("### Weak / Noisy Features")
            for row in weak_features[:6]:
                lines.append(
                    f"- `{row.get('feature')}`: signal={_format_float(row.get('signal_score'))}, "
                    f"coverage={_format_float(row.get('coverage_fraction'))}, "
                    f"rows={row.get('rows_with_value')}"
                )
        if harmful_features:
            lines.append("")
            lines.append("### Potentially Harmful Features")
            for row in harmful_features[:6]:
                lines.append(
                    f"- `{row.get('feature')}`: expected={row.get('expected_direction')}, "
                    f"observed={row.get('observed_direction')}, "
                    f"signal={_format_float(row.get('signal_score'))}"
                )
        if direction_alignment:
            lines.append("")
            lines.append("### Direction Alignment")
            lines.append(
                f"- Conflicts detected before alignment: `{direction_alignment.get('conflicts_detected_pre_alignment')}`"
            )
            lines.append(
                f"- Conflicts remaining after alignment: `{direction_alignment.get('conflicts_remaining_post_alignment')}`"
            )
            lines.append(
                f"- Conflicts resolved by alignment: `{direction_alignment.get('conflicts_resolved_count')}`"
            )
        if family_summary:
            lines.append("")
            lines.append("### Key Feature Families")
            for family_name in (
                "vegetation_metrics",
                "slope_terrain",
                "hazard_zone_context",
                "burn_probability",
                "structural_features",
            ):
                detail = family_summary.get(family_name)
                if not isinstance(detail, dict):
                    continue
                lines.append(
                    f"- `{family_name}`: count={detail.get('feature_count')}, "
                    f"mean_signal={_format_float(detail.get('mean_signal_score'))}"
                )

    lines.extend(
        [
            "",
            "## Error Review Sets",
            f"- False-low count: `{review_sets.get('false_low_count')}`",
            f"- False-high count: `{review_sets.get('false_high_count')}`",
            f"- Unstable but outcome-positive count: `{review_sets.get('unstable_positive_count')}`",
            f"- Low-confidence but outcome-positive count: `{review_sets.get('low_confidence_positive_count')}`",
            "",
            "## Guardrails",
        ]
    )
    warnings = guardrails.get("warnings") if isinstance(guardrails.get("warnings"), list) else []
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No guardrail warnings.")

    lines.extend(
        [
            "",
            "## Caveats",
            "- Public observed outcomes are imperfect and incomplete.",
            "- This report preserves and evaluates raw, uncalibrated model scores before any calibration overlay.",
            "- Treat results as directional model validation, not insurer claims validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_segment_report_markdown(
    *,
    run_id: str,
    generated_at: str,
    report: dict[str, Any],
) -> str:
    slice_metrics = (
        report.get("slice_metrics")
        if isinstance(report.get("slice_metrics"), dict)
        else {}
    )
    segment_summary = (
        report.get("segment_performance_summary")
        if isinstance(report.get("segment_performance_summary"), dict)
        else {}
    )
    by_hazard = (
        slice_metrics.get("by_hazard_level")
        if isinstance(slice_metrics.get("by_hazard_level"), dict)
        else {}
    )
    by_vegetation = (
        slice_metrics.get("by_vegetation_density")
        if isinstance(slice_metrics.get("by_vegetation_density"), dict)
        else {}
    )
    by_confidence = (
        slice_metrics.get("by_confidence_tier")
        if isinstance(slice_metrics.get("by_confidence_tier"), dict)
        else {}
    )
    by_region = (
        slice_metrics.get("by_region")
        if isinstance(slice_metrics.get("by_region"), dict)
        else {}
    )

    lines = [
        "# Segment Performance Report",
        "",
        "Segment-level validation against public observed outcomes.",
        "This is directional performance analysis and does not establish carrier-claims truth.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{generated_at}`",
        f"- Min segment size for stable interpretation: `{segment_summary.get('min_slice_size')}`",
        "",
    ]

    def _append_segment_block(title: str, payload: dict[str, Any]) -> None:
        if not payload:
            return
        lines.append(f"## {title}")
        for name in sorted(payload.keys()):
            detail = payload[name] if isinstance(payload[name], dict) else {}
            lines.append(
                f"- `{name}`: n={detail.get('count')}, "
                f"auc={_format_float(detail.get('wildfire_risk_score_auc'))}, "
                f"brier={_format_float(detail.get('wildfire_risk_score_brier'))}, "
                f"small_sample_warning={detail.get('small_sample_warning')}"
            )
        lines.append("")

    _append_segment_block("Hazard Level Segments", by_hazard)
    _append_segment_block("Vegetation Density Segments", by_vegetation)
    _append_segment_block("Confidence Tier Segments", by_confidence)
    _append_segment_block("Region Segments", by_region)

    strongest = (
        segment_summary.get("strongest_segments")
        if isinstance(segment_summary.get("strongest_segments"), list)
        else []
    )
    weakest = (
        segment_summary.get("weakest_segments")
        if isinstance(segment_summary.get("weakest_segments"), list)
        else []
    )
    highlights = (
        segment_summary.get("highlights")
        if isinstance(segment_summary.get("highlights"), list)
        else []
    )
    strength_map = (
        segment_summary.get("segment_strength_map")
        if isinstance(segment_summary.get("segment_strength_map"), dict)
        else {}
    )
    lines.append("## Strengths and Weaknesses")
    lines.append(
        f"- Eligible segments: `{segment_summary.get('eligible_segment_count')}` "
        f"of `{segment_summary.get('segment_count')}`"
    )
    for item in highlights[:6]:
        lines.append(f"- {item}")

    family_statuses: list[dict[str, Any]] = []
    for key in ("hazard_level", "vegetation_density", "confidence_tier", "region"):
        payload = strength_map.get(key) if isinstance(strength_map.get(key), dict) else {}
        if payload:
            family_statuses.append(payload)
    has_comparative_family = any(
        int(payload.get("eligible_segment_count") or 0) >= 2
        and str(payload.get("status") or "") not in {"single_segment_only", "insufficient_data"}
        for payload in family_statuses
    )

    strong_rows = [row for row in strongest if isinstance(row, dict)]
    strong_ids = {
        (str(row.get("segment_family") or ""), str(row.get("segment_name") or ""))
        for row in strong_rows
    }
    weak_rows = [
        row
        for row in weakest
        if isinstance(row, dict)
        and (str(row.get("segment_family") or ""), str(row.get("segment_name") or "")) not in strong_ids
    ]

    if not has_comparative_family:
        lines.append(
            "- No segment family currently has multiple eligible class-balanced slices; "
            "strength/weakness ranking is provisional until segment coverage improves."
        )
    elif strong_rows:
        lines.append("- Strong segments:")
        for row in strong_rows[:5]:
            lines.append(
                f"  - {row.get('segment_family')}={row.get('segment_name')} "
                f"(auc={_format_float(row.get('auc'))}, brier={_format_float(row.get('brier'))}, n={row.get('count')})"
            )
    if has_comparative_family and weak_rows:
        lines.append("- Weak segments:")
        for row in weak_rows[:5]:
            lines.append(
                f"  - {row.get('segment_family')}={row.get('segment_name')} "
                f"(auc={_format_float(row.get('auc'))}, brier={_format_float(row.get('brier'))}, n={row.get('count')})"
            )
    if strength_map:
        lines.append("")
        lines.append("## Segment Strength Map")
        family_order = ["hazard_level", "vegetation_density", "confidence_tier", "region"]
        for family in family_order:
            payload = strength_map.get(family) if isinstance(strength_map.get(family), dict) else {}
            if not payload:
                continue
            lines.append(
                f"- `{family}`: status=`{payload.get('status')}`, "
                f"eligible_segments=`{payload.get('eligible_segment_count')}`/{payload.get('segment_count')}, "
                f"auc_spread=`{_format_float(payload.get('auc_spread'))}`"
            )
            best = payload.get("best_segment") if isinstance(payload.get("best_segment"), dict) else {}
            if best:
                lines.append(
                    f"  - best: `{best.get('segment_name')}` "
                    f"(auc={_format_float(best.get('auc'))}, brier={_format_float(best.get('brier'))}, n={best.get('count')})"
                )
            worst = payload.get("worst_segment") if isinstance(payload.get("worst_segment"), dict) else {}
            if worst:
                lines.append(
                    f"  - worst: `{worst.get('segment_name')}` "
                    f"(auc={_format_float(worst.get('auc'))}, brier={_format_float(worst.get('brier'))}, n={worst.get('count')})"
                )
            notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
            for note in notes[:2]:
                lines.append(f"  - note: {note}")
    lines.append("")
    lines.append("## Caveat")
    lines.append(
        "- Segment diagnostics help identify where the model appears stronger or weaker, "
        "but they remain limited by sample size, join quality, and public-outcome coverage."
    )
    return "\n".join(lines) + "\n"


def _insufficient_data_report(
    *,
    generated_at: str,
    dataset_path: Path,
    error: str,
    dataset_flow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    insuff_threshold = 20
    limited_threshold = 100
    strong_threshold = 500
    total_rows = _count_rows_in_dataset(dataset_path)
    flow = dataset_flow if isinstance(dataset_flow, dict) else {}
    missing_required = flow.get("missing_required_fields") if isinstance(flow, dict) else {}
    invalid_examples = flow.get("invalid_row_examples") if isinstance(flow, dict) else []
    return {
        "schema_version": "1.1.0",
        "generated_at": generated_at,
        "status": "insufficient_data",
        "error": error,
        "row_count_labeled": 0,
        "sample_counts": {
            "row_count_total": total_rows,
            "row_count_retained": int(flow.get("retained_rows") or 0),
            "row_count_usable": 0,
            "row_count_unusable": int(flow.get("unusable_rows") or 0),
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": None,
            "validation_exclusions": {
                "missing_required_fields": (
                    missing_required if isinstance(missing_required, dict) else {}
                ),
                "invalid_row_examples": (
                    invalid_examples if isinstance(invalid_examples, list) else []
                ),
            },
        },
        "discrimination_metrics": {
            "wildfire_risk_score_auc": None,
            "wildfire_risk_score_pr_auc": None,
            "wildfire_vs_outcome_rank_spearman": None,
        },
        "threshold_metrics_wildfire_risk_score": {},
        "default_threshold_70": {"confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}, "precision": None, "recall": None, "f1": None},
        "brier_scores": {"wildfire_probability_proxy": None},
        "calibration_metrics": {"wildfire_risk_score": {"bins": [], "expected_calibration_error": None}},
        "slice_metrics": {
            "by_confidence_tier": {},
            "by_evidence_group": {},
            "by_join_confidence_tier": {},
            "by_validation_confidence_tier": {},
            "by_hazard_level": {},
            "by_vegetation_density": {},
            "by_region": {},
        },
        "segment_performance_summary": {
            "min_slice_size": 20,
            "segment_count": 0,
            "eligible_segment_count": 0,
            "insufficient_segment_count": 0,
            "strongest_segments": [],
            "weakest_segments": [],
            "insufficient_or_unstable_segments": [],
            "segment_strength_map": {
                "hazard_level": {
                    "segment_count": 0,
                    "eligible_segment_count": 0,
                    "status": "insufficient_data",
                    "auc_spread": None,
                    "best_segment": None,
                    "worst_segment": None,
                    "strong_segments": [],
                    "weak_segments": [],
                    "notes": ["No usable rows available for hazard-level segmentation."],
                },
                "vegetation_density": {
                    "segment_count": 0,
                    "eligible_segment_count": 0,
                    "status": "insufficient_data",
                    "auc_spread": None,
                    "best_segment": None,
                    "worst_segment": None,
                    "strong_segments": [],
                    "weak_segments": [],
                    "notes": ["No usable rows available for vegetation-density segmentation."],
                },
                "confidence_tier": {
                    "segment_count": 0,
                    "eligible_segment_count": 0,
                    "status": "insufficient_data",
                    "auc_spread": None,
                    "best_segment": None,
                    "worst_segment": None,
                    "strong_segments": [],
                    "weak_segments": [],
                    "notes": ["No usable rows available for confidence-tier segmentation."],
                },
                "region": {
                    "segment_count": 0,
                    "eligible_segment_count": 0,
                    "status": "insufficient_data",
                    "auc_spread": None,
                    "best_segment": None,
                    "worst_segment": None,
                    "strong_segments": [],
                    "weak_segments": [],
                    "notes": ["No usable rows available for region segmentation."],
                },
            },
            "highlights": [
                "No segment-level evaluation is available because there are no usable labeled rows."
            ],
        },
        "subset_metrics": {
            "full_dataset": {
                "count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "high_confidence_subset": {
                "count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "medium_confidence_subset": {
                "count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "high_evidence_subset": {
                "count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
        },
        "event_level_out_of_sample": {
            "available": False,
            "method": "event_level_holdout_aggregation",
            "reason": "no_usable_rows",
            "thresholds": {
                "min_event_rows": 4,
                "min_event_positive_count": 1,
                "min_event_negative_count": 1,
            },
            "event_count_total": 0,
            "event_count_eligible": 0,
            "event_count_ineligible": 0,
            "aggregate": {
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "per_event": [],
        },
        "reported_metrics": {
            "default_reporting_path": "event_level_holdout_out_of_sample",
            "primary_source": "in_sample_fallback",
            "primary": {
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "in_sample": {
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
            "holdout_event_level_out_of_sample": {
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": None,
                "wildfire_risk_score_auc": None,
                "wildfire_risk_score_pr_auc": None,
                "wildfire_risk_score_brier": None,
            },
        },
        "confidence_tier_performance": {
            "min_slice_size": 20,
            "tiers": {
                "all_data": {
                    "count": 0,
                    "positive_rate": None,
                    "wildfire_risk_score_auc": None,
                    "wildfire_risk_score_pr_auc": None,
                    "wildfire_risk_score_brier": None,
                    "small_sample_warning": True,
                },
                "high_confidence": {
                    "count": 0,
                    "positive_rate": None,
                    "wildfire_risk_score_auc": None,
                    "wildfire_risk_score_pr_auc": None,
                    "wildfire_risk_score_brier": None,
                    "small_sample_warning": True,
                },
                "medium_confidence": {
                    "count": 0,
                    "positive_rate": None,
                    "wildfire_risk_score_auc": None,
                    "wildfire_risk_score_pr_auc": None,
                    "wildfire_risk_score_brier": None,
                    "small_sample_warning": True,
                },
            },
            "deltas_vs_all_data": {
                "high_confidence_auc_delta": None,
                "high_confidence_brier_delta": None,
                "medium_confidence_auc_delta": None,
                "medium_confidence_brier_delta": None,
            },
            "warnings": [
                "High-confidence slice is too small for stable interpretation (n=0 < 20).",
                "Medium-confidence slice is too small for stable interpretation (n=0 < 20).",
            ],
        },
        "minimum_viable_metrics": {
            "available": False,
            "rank_ordering": {"available": False},
            "simple_accuracy_at_threshold": {"available": False},
            "top_risk_bucket_hit_rate": {"available": False},
            "adverse_rate_by_score_decile": {"available": False, "bins": []},
        },
        "data_sufficiency_flags": {
            "flags": {
                "small_sample_size": True,
                "very_small_sample_size": True,
                "class_imbalance": False,
                "low_join_confidence_prevalent": False,
                "fallback_heavy_prevalent": False,
                "no_high_confidence_rows": True,
                "no_high_evidence_rows": True,
            },
            "sample_size": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": None,
            "low_join_confidence_fraction": None,
            "fallback_heavy_fraction": None,
            "high_confidence_count": 0,
            "high_evidence_count": 0,
        },
        "data_sufficiency_indicator": {
            "thresholds": {
                "insufficient_max_exclusive": insuff_threshold,
                "limited_max_exclusive": limited_threshold,
                "moderate_max_inclusive": strong_threshold,
                "strong_min_exclusive": strong_threshold,
            },
            "total_dataset": {
                "sample_size": 0,
                "tier": "insufficient",
                "explanation": (
                    f"Sample size 0 is below {insuff_threshold}; "
                    "discrimination/calibration metrics are highly unstable."
                ),
            },
            "high_confidence_subset": {
                "sample_size": 0,
                "tier": "insufficient",
                "explanation": (
                    f"Sample size 0 is below {insuff_threshold}; "
                    "high-confidence subset metrics are unavailable."
                ),
            },
        },
        "reporting_claim_guardrails": {
            "strong_performance_claims_allowed": False,
            "calibration_claims_allowed": False,
            "checks": {
                "sample_count_ok": False,
                "positive_class_count_ok": False,
                "negative_class_count_ok": False,
                "class_balance_ok": False,
                "event_level_holdout_available": False,
            },
            "failed_checks": [
                "sample_count_ok",
                "positive_class_count_ok",
                "negative_class_count_ok",
                "class_balance_ok",
                "event_level_holdout_available",
            ],
        },
        "narrative_summary": {
            "headline": "Directional validation is unavailable because no usable labeled rows were found.",
            "bullets": [
                "Directional validation is unavailable because no usable labeled rows were found.",
                "Insufficient data for stable calibration conclusions.",
            ],
        },
        "proxy_validation": {
            "available": False,
            "caveat": "Proxy validation uses weak proxy labels and is not ground-truth validation.",
            "reason": "insufficient_real_outcome_rows",
        },
        "synthetic_validation": {
            "available": False,
            "caveat": "Synthetic stress validation checks directional behavior and is not real-outcome ground truth.",
            "reason": "insufficient_real_outcome_rows",
        },
        "validation_streams": {
            "real_outcome_validation": {
                "available": False,
                "row_count_labeled": 0,
                "caveat": "No usable public-outcome rows were available in this run.",
            },
            "proxy_validation": {
                "available": False,
                "caveat": "Proxy validation uses weak labels and is not ground-truth validation.",
            },
            "synthetic_validation": {
                "available": False,
                "caveat": "Synthetic stress validation is not real-outcome ground truth.",
            },
        },
        "feature_signal_diagnostics": {
            "available": False,
            "reason": "insufficient_real_outcome_rows",
            "top_predictive_features": [],
            "weak_or_noisy_features": [],
            "potentially_harmful_features": [],
            "direction_alignment": {
                "available": False,
                "reason": "insufficient_real_outcome_rows",
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
        },
        "modeling_viability": {
            "dataset_viable_for_predictive_modeling": False,
            "classification": "dataset_not_viable_for_predictive_modeling",
            "reason": "No usable labeled rows were retained for independent-sample and feature-variation checks.",
            "thresholds": {
                "min_independent_samples": 30,
                "min_features_with_variance": 5,
                "min_feature_variation_ratio": 0.25,
                "min_auc_margin_vs_random_baseline": 0.05,
            },
            "checks": {
                "independent_sample_count": 0,
                "labeled_sample_count": 0,
                "duplication_factor": None,
                "feature_count_evaluated": 0,
                "near_zero_variance_feature_count": 0,
                "features_with_variance_count": 0,
                "feature_variation_ratio": 0.0,
                "full_model_auc": None,
                "random_baseline_auc": None,
                "auc_margin_vs_random_baseline": None,
                "independent_sample_size_ok": False,
                "feature_variation_ok": False,
                "model_vs_random_auc_ok": False,
            },
            "failed_checks": [
                "independent_sample_size_ok",
                "feature_variation_ok",
                "model_vs_random_auc_ok",
            ],
            "caveat": (
                "This viability check is a guardrail for directional public-outcome evaluation. "
                "It does not establish insurer-claims predictive validity."
            ),
        },
        "baseline_model_comparison": {
            "available": False,
            "reason": "insufficient_real_outcome_rows",
            "full_model_auc": None,
            "baselines": {},
            "comparison": {
                "beats_all_baselines_by_auc": None,
                "best_baseline_name": None,
                "best_baseline_auc": None,
                "auc_margin_vs_best_baseline": None,
                "baselines_compared_count": 0,
                "complexity_justified_signal": "no_or_inconclusive",
            },
            "caveat": (
                "Baseline comparison is unavailable because no usable labeled rows were retained. "
                "Any baseline comparisons are directional only."
            ),
        },
        "false_review_sets": {
            "false_low_count": 0,
            "false_high_count": 0,
            "unstable_positive_count": 0,
            "low_confidence_positive_count": 0,
            "false_low_examples": [],
            "false_high_examples": [],
        },
        "guardrails": {
            "warnings": [
                "Insufficient usable labeled rows for stable public-outcome validation metrics.",
                error,
            ],
            "small_sample_warning": True,
            "fallback_heavy_warning": False,
            "leakage_warning": False,
        },
        "directional_predictive_value": False,
        "calibration_artifact_recommendation": "not_recommended_yet",
        "pipeline_stage_counts": {
            "loaded_rows": int(flow.get("loaded_rows") or total_rows),
            "prepared_rows": int(flow.get("prepared_rows") or 0),
            "retained_rows": int(flow.get("retained_rows") or 0),
            "unusable_rows": int(flow.get("unusable_rows") or 0),
            "dropped_rows": int(flow.get("dropped_rows") or max(0, total_rows)),
        },
    }


def run_public_outcome_validation(
    *,
    evaluation_dataset: Path | None = None,
    evaluation_dataset_root: Path = DEFAULT_EVALUATION_DATASET_ROOT,
    evaluation_dataset_run_id: str | None = None,
    output_root: Path = DEFAULT_VALIDATION_OUTPUT_ROOT,
    run_id: str | None = None,
    thresholds: list[float] | None = None,
    bins: int = 10,
    min_slice_size: int = 20,
    false_low_max_score: float = 40.0,
    false_high_min_score: float = 70.0,
    min_labeled_rows: int = 1,
    allow_label_derived_target: bool = False,
    allow_surrogate_wildfire_score: bool = False,
    use_high_signal_simplified_model: bool = False,
    high_signal_feature_weights: dict[str, float] | None = None,
    high_signal_feature_filtering: dict[str, Any] | None = None,
    min_join_confidence_score_for_metrics: float | None = None,
    retain_unusable_rows: bool = True,
    baseline_run_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = str(run_id) if run_id else _iso_now()
    output_dir = output_root.expanduser() / run_token
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {output_dir}. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = _resolve_dataset_path(
        dataset_path=evaluation_dataset,
        dataset_root=evaluation_dataset_root.expanduser(),
        dataset_run_id=evaluation_dataset_run_id,
    )
    dataset_flow = trace_public_outcome_dataset_flow(
        dataset_path=dataset_path,
        allow_label_derived_target=allow_label_derived_target,
        allow_surrogate_wildfire_score=allow_surrogate_wildfire_score,
        use_high_signal_simplified_model=use_high_signal_simplified_model,
        high_signal_feature_weights=high_signal_feature_weights,
        min_join_confidence_score_for_metrics=min_join_confidence_score_for_metrics,
        retain_unusable_rows=retain_unusable_rows,
    )
    join_stage_counts = _dataset_join_stage_counts(dataset_path)
    if join_stage_counts:
        print(
            "[public-validation] Join stage: "
            f"outcomes={join_stage_counts.get('outcomes_loaded')} "
            f"feature_rows={join_stage_counts.get('feature_rows_loaded')} "
            f"joined={join_stage_counts.get('joined_rows')} "
            f"excluded={join_stage_counts.get('excluded_rows')} "
            f"join_rate={join_stage_counts.get('join_rate')}"
        )
        print(
            "[public-validation] Join confidence tiers: "
            f"{join_stage_counts.get('join_confidence_tier_counts')}"
        )
        print(
            "[public-validation] Join non-high reasons: "
            f"{join_stage_counts.get('join_confidence_non_high_reason_counts')}"
        )
        print(
            "[public-validation] Join threshold diagnostics: "
            f"{join_stage_counts.get('high_confidence_threshold_diagnostics')}"
        )
        score_backfill = join_stage_counts.get("score_backfill")
        if isinstance(score_backfill, dict) and score_backfill:
            print(
                "[public-validation] Score backfill stage: "
                f"before_missing={score_backfill.get('missing_score_record_count_before')} "
                f"backfilled={score_backfill.get('backfilled_record_count')} "
                f"remaining_missing={score_backfill.get('remaining_missing_score_record_count')}"
            )
        if bool(join_stage_counts.get("retention_fallback_triggered")):
            print(
                "[public-validation] WARNING: evaluation dataset minimum-retention fallback was triggered; "
                "metrics include lower-confidence joins (see retention_fallback metadata)."
            )
    print(
        f"[public-validation] Loaded {dataset_flow.get('loaded_rows')} dataset rows "
        f"from {dataset_flow.get('dataset_path')}"
    )
    print(
        f"[public-validation] Prepared usable={dataset_flow.get('prepared_rows')} "
        f"retained={dataset_flow.get('retained_rows')} "
        f"unusable={dataset_flow.get('unusable_rows')} "
        f"(dropped {dataset_flow.get('dropped_rows')})"
    )
    missing_map = dataset_flow.get("missing_required_fields")
    if isinstance(missing_map, dict) and missing_map:
        print(f"[public-validation] Missing required field counts: {missing_map}")
    invalid_examples = dataset_flow.get("invalid_row_examples")
    if isinstance(invalid_examples, list) and invalid_examples:
        print(
            "[public-validation] Invalid-row examples: "
            + str(invalid_examples[:5])
        )

    rows: list[dict[str, Any]]
    try:
        report, rows = evaluate_public_outcome_dataset_file(
            dataset_path=dataset_path,
            thresholds=(thresholds or list(DEFAULT_THRESHOLDS)),
            bins=max(2, int(bins)),
            min_slice_size=max(2, int(min_slice_size)),
            false_low_max_score=float(false_low_max_score),
            false_high_min_score=float(false_high_min_score),
            min_labeled_rows=max(1, int(min_labeled_rows)),
            allow_label_derived_target=bool(allow_label_derived_target),
            allow_surrogate_wildfire_score=bool(allow_surrogate_wildfire_score),
            use_high_signal_simplified_model=bool(use_high_signal_simplified_model),
            high_signal_feature_weights=high_signal_feature_weights,
            min_join_confidence_score_for_metrics=(
                float(min_join_confidence_score_for_metrics)
                if min_join_confidence_score_for_metrics is not None
                else None
            ),
            retain_unusable_rows=bool(retain_unusable_rows),
            generated_at=generated_at,
        )
        report["pipeline_stage_counts"] = {
            "outcomes_loaded": int(join_stage_counts.get("outcomes_loaded") or 0),
            "feature_rows_loaded": int(join_stage_counts.get("feature_rows_loaded") or 0),
            "joined_rows": int(join_stage_counts.get("joined_rows") or 0),
            "join_excluded_rows": int(join_stage_counts.get("excluded_rows") or 0),
            "join_confidence_tier_counts": (
                join_stage_counts.get("join_confidence_tier_counts")
                if isinstance(join_stage_counts.get("join_confidence_tier_counts"), dict)
                else {}
            ),
            "join_confidence_non_high_reason_counts": (
                join_stage_counts.get("join_confidence_non_high_reason_counts")
                if isinstance(join_stage_counts.get("join_confidence_non_high_reason_counts"), dict)
                else {}
            ),
            "high_confidence_threshold_diagnostics": (
                join_stage_counts.get("high_confidence_threshold_diagnostics")
                if isinstance(join_stage_counts.get("high_confidence_threshold_diagnostics"), dict)
                else {}
            ),
            "retention_fallback_triggered": bool(join_stage_counts.get("retention_fallback_triggered")),
            "retention_fallback_used": bool(join_stage_counts.get("retention_fallback_used")),
            "retention_fallback": (
                join_stage_counts.get("retention_fallback")
                if isinstance(join_stage_counts.get("retention_fallback"), dict)
                else {}
            ),
            "loaded_rows": int(dataset_flow.get("loaded_rows") or 0),
            "prepared_rows": int(dataset_flow.get("prepared_rows") or 0),
            "retained_rows": int(dataset_flow.get("retained_rows") or 0),
            "unusable_rows": int(dataset_flow.get("unusable_rows") or 0),
            "dropped_rows": int(dataset_flow.get("dropped_rows") or 0),
        }
        if bool(join_stage_counts.get("retention_fallback_triggered")):
            guardrails = report.get("guardrails") if isinstance(report.get("guardrails"), dict) else {}
            guardrail_warnings = guardrails.get("warnings") if isinstance(guardrails.get("warnings"), list) else []
            guardrail_warnings = list(guardrail_warnings)
            warning_text = (
                "Evaluation dataset minimum-retention fallback mode was triggered; "
                "lower-confidence joins were included to avoid near-zero sample collapse."
            )
            if warning_text not in guardrail_warnings:
                guardrail_warnings.append(warning_text)
            guardrails["warnings"] = guardrail_warnings
            report["guardrails"] = guardrails
        subset_metrics = report.get("subset_metrics") if isinstance(report.get("subset_metrics"), dict) else {}
        if subset_metrics:
            print(
                "[public-validation] Subsets: "
                f"full={((subset_metrics.get('full_dataset') or {}).get('count'))} "
                f"high_confidence={((subset_metrics.get('high_confidence_subset') or {}).get('count'))} "
                f"medium_confidence={((subset_metrics.get('medium_confidence_subset') or {}).get('count'))} "
                f"high_evidence={((subset_metrics.get('high_evidence_subset') or {}).get('count'))}"
            )
        if isinstance(high_signal_feature_filtering, dict):
            report["high_signal_feature_filtering"] = high_signal_feature_filtering
            print(
                "[public-validation] High-signal feature filtering: "
                f"included={high_signal_feature_filtering.get('included_feature_count')} "
                f"excluded={high_signal_feature_filtering.get('excluded_feature_count')} "
                f"thresholds={high_signal_feature_filtering.get('thresholds')}"
            )
            excluded_rows = (
                high_signal_feature_filtering.get("excluded_features")
                if isinstance(high_signal_feature_filtering.get("excluded_features"), list)
                else []
            )
            if excluded_rows:
                print(
                    "[public-validation] Excluded high-signal features: "
                    + str(
                        [
                            {
                                "feature": row.get("feature"),
                                "reasons": row.get("reasons"),
                            }
                            for row in excluded_rows
                            if isinstance(row, dict)
                        ]
                    )
                )
    except ValueError as exc:
        rows = []
        report = _insufficient_data_report(
            generated_at=generated_at,
            dataset_path=dataset_path,
            error=str(exc),
            dataset_flow=dataset_flow,
        )

    validation_metrics_path = output_dir / "validation_metrics.json"
    _write_json(validation_metrics_path, report)

    calibration_table_payload = {
        "wildfire_risk_score": (
            (report.get("calibration_metrics") or {}).get("wildfire_risk_score")
            if isinstance(report.get("calibration_metrics"), dict)
            else None
        ),
        "calibrated_damage_likelihood": (
            (report.get("calibration_metrics") or {}).get("calibrated_damage_likelihood")
            if isinstance(report.get("calibration_metrics"), dict)
            else None
        ),
        "generated_at": generated_at,
    }
    calibration_table_path = output_dir / "calibration_table.json"
    _write_json(calibration_table_path, calibration_table_payload)

    threshold_metrics_payload = {
        "threshold_metrics_wildfire_risk_score": report.get("threshold_metrics_wildfire_risk_score"),
        "default_threshold_70": report.get("default_threshold_70"),
        "generated_at": generated_at,
    }
    threshold_metrics_path = output_dir / "threshold_metrics.json"
    _write_json(threshold_metrics_path, threshold_metrics_payload)

    review_sets = report.get("false_review_sets") if isinstance(report.get("false_review_sets"), dict) else {}
    false_low_rows = (
        review_sets.get("false_low_examples")
        if isinstance(review_sets.get("false_low_examples"), list)
        else []
    )
    false_high_rows = (
        review_sets.get("false_high_examples")
        if isinstance(review_sets.get("false_high_examples"), list)
        else []
    )
    false_low_path = output_dir / "false_low_review_set.jsonl"
    false_high_path = output_dir / "false_high_review_set.jsonl"
    _write_jsonl(false_low_path, [row for row in false_low_rows if isinstance(row, dict)])
    _write_jsonl(false_high_path, [row for row in false_high_rows if isinstance(row, dict)])

    # Keep a compact row export for operator review.
    evaluated_rows_csv_path = output_dir / "evaluation_rows.csv"
    write_evaluation_rows_csv(rows=rows, output_csv=evaluated_rows_csv_path)

    feature_signal_diagnostics = (
        report.get("feature_signal_diagnostics")
        if isinstance(report.get("feature_signal_diagnostics"), dict)
        else {
            "available": False,
            "reason": "not_present_in_validation_report",
            "top_predictive_features": [],
            "weak_or_noisy_features": [],
            "potentially_harmful_features": [],
            "direction_alignment": {
                "available": False,
                "reason": "not_present_in_validation_report",
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
    )

    feature_diag_payload = {
        "run_id": run_token,
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "feature_signal_diagnostics": feature_signal_diagnostics,
        "caveat": (
            "Feature diagnostics describe directional signal/noise in this labeled sample. "
            "They are not causal inference and do not establish insurer-claims predictive truth."
        ),
    }
    feature_diagnostics_path = output_dir / "feature_diagnostics.json"
    _write_json(feature_diagnostics_path, feature_diag_payload)

    feature_signal_report_payload = {
        "run_id": run_token,
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "method": (
            feature_signal_diagnostics.get("method")
            if isinstance(feature_signal_diagnostics.get("method"), dict)
            else {}
        ),
        "feature_count_evaluated": feature_signal_diagnostics.get("feature_count_evaluated"),
        "row_count_used": feature_signal_diagnostics.get("row_count_used"),
        "top_predictive_features": (
            feature_signal_diagnostics.get("top_predictive_features")
            if isinstance(feature_signal_diagnostics.get("top_predictive_features"), list)
            else []
        ),
        "weak_or_noisy_features": (
            feature_signal_diagnostics.get("weak_or_noisy_features")
            if isinstance(feature_signal_diagnostics.get("weak_or_noisy_features"), list)
            else []
        ),
        "potentially_harmful_features": (
            feature_signal_diagnostics.get("potentially_harmful_features")
            if isinstance(feature_signal_diagnostics.get("potentially_harmful_features"), list)
            else []
        ),
        "feature_ranking_strongest_to_weakest": (
            feature_signal_diagnostics.get("top_predictive_features")
            if isinstance(feature_signal_diagnostics.get("top_predictive_features"), list)
            else []
        ),
        "scoring_caveat": (
            "Feature signal diagnostics are directional and sample-dependent. They do not establish "
            "causal effects, insurer claims truth, or underwriting-grade predictive validity."
        ),
    }
    feature_signal_report_path = output_dir / "feature_signal_report.json"
    _write_json(feature_signal_report_path, feature_signal_report_payload)

    direction_alignment_payload = {
        "run_id": run_token,
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "direction_alignment": (
            feature_signal_diagnostics.get("direction_alignment")
            if isinstance(feature_signal_diagnostics.get("direction_alignment"), dict)
            else {
                "available": False,
                "reason": "not_present_in_feature_signal_diagnostics",
                "conflicts_detected_pre_alignment": 0,
                "conflicts_remaining_post_alignment": 0,
                "conflicts_resolved_count": 0,
                "conflicts_detected": [],
                "conflicts_resolved": [],
                "unresolved_conflicts": [],
                "aligned_expected_direction_overrides": {},
            }
        ),
        "caveat": (
            "Direction alignment is a diagnostic harmonization against this labeled sample. "
            "It does not establish causal directionality or claims-grade truth."
        ),
    }
    direction_alignment_report_path = output_dir / "direction_alignment_report.json"
    _write_json(direction_alignment_report_path, direction_alignment_payload)

    baseline_comparison_payload = {
        "run_id": run_token,
        "generated_at": generated_at,
        "dataset_path": str(dataset_path),
        "baseline_model_comparison": (
            report.get("baseline_model_comparison")
            if isinstance(report.get("baseline_model_comparison"), dict)
            else {
                "available": False,
                "reason": "not_present_in_validation_report",
                "full_model_auc": None,
                "baselines": {},
                "comparison": {
                    "beats_all_baselines_by_auc": None,
                    "best_baseline_name": None,
                    "best_baseline_auc": None,
                    "auc_margin_vs_best_baseline": None,
                    "baselines_compared_count": 0,
                    "complexity_justified_signal": "no_or_inconclusive",
                },
            }
        ),
        "caveat": (
            "Baseline comparison is directional only and evaluates whether the full score "
            "beats simple baseline signals on this public-outcome sample."
        ),
    }
    baseline_comparison_path = output_dir / "baseline_model_comparison.json"
    _write_json(baseline_comparison_path, baseline_comparison_payload)

    segment_metrics_payload = {
        "run_id": run_token,
        "generated_at": generated_at,
        "segment_performance_summary": (
            report.get("segment_performance_summary")
            if isinstance(report.get("segment_performance_summary"), dict)
            else {}
        ),
        "segment_strength_map": (
            ((report.get("segment_performance_summary") or {}).get("segment_strength_map"))
            if isinstance(report.get("segment_performance_summary"), dict)
            else {}
        ),
        "segment_slice_metrics": {
            "by_hazard_level": (
                ((report.get("slice_metrics") or {}).get("by_hazard_level"))
                if isinstance(report.get("slice_metrics"), dict)
                else {}
            ),
            "by_vegetation_density": (
                ((report.get("slice_metrics") or {}).get("by_vegetation_density"))
                if isinstance(report.get("slice_metrics"), dict)
                else {}
            ),
            "by_confidence_tier": (
                ((report.get("slice_metrics") or {}).get("by_confidence_tier"))
                if isinstance(report.get("slice_metrics"), dict)
                else {}
            ),
            "by_region": (
                ((report.get("slice_metrics") or {}).get("by_region"))
                if isinstance(report.get("slice_metrics"), dict)
                else {}
            ),
        },
        "caveat": (
            "Segment metrics are directional diagnostics from public observed outcomes. "
            "They do not establish insurer-claims predictive truth."
        ),
    }
    segment_metrics_path = output_dir / "segment_metrics.json"
    _write_json(segment_metrics_path, segment_metrics_payload)
    segment_report_path = output_dir / "segment_report.md"
    segment_report_path.write_text(
        _build_segment_report_markdown(
            run_id=run_token,
            generated_at=generated_at,
            report=report,
        ),
        encoding="utf-8",
    )

    summary_text = _build_summary_markdown(
        run_id=run_token,
        generated_at=generated_at,
        dataset_path=dataset_path,
        report=report,
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    listing = list_public_outcome_runs(artifact_root=output_root)
    ordered_ids = [
        str(item.get("run_id"))
        for item in (listing.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    ]
    selected_baseline_run = resolve_baseline_run_id(
        ordered_run_ids=ordered_ids,
        current_run_id=run_token,
        baseline_run_id=baseline_run_id,
    )
    baseline_manifest: dict[str, Any] | None = None
    baseline_metrics: dict[str, Any] | None = None
    if selected_baseline_run:
        baseline_dir = output_root.expanduser() / selected_baseline_run
        baseline_manifest = _safe_load_json(baseline_dir / "manifest.json")
        baseline_metrics = _safe_load_json(baseline_dir / "validation_metrics.json")
    comparison_payload = build_validation_run_comparison(
        current_run_id=run_token,
        current_manifest={
            "versions": {
                "product_version": PRODUCT_VERSION,
                "api_version": API_VERSION,
                "scoring_model_version": SCORING_MODEL_VERSION,
                "rules_logic_version": RULESET_LOGIC_VERSION,
                "factor_schema_version": FACTOR_SCHEMA_VERSION,
                "benchmark_pack_version": BENCHMARK_PACK_VERSION,
                "calibration_version": CALIBRATION_VERSION,
            },
            "inputs": {"evaluation_dataset_path": str(dataset_path)},
        },
        current_report=report,
        baseline_run_id=selected_baseline_run,
        baseline_manifest=baseline_manifest,
        baseline_report=baseline_metrics,
    )
    comparison_json_path = output_dir / "comparison_to_previous.json"
    _write_json(comparison_json_path, comparison_payload)
    comparison_md_path = output_dir / "comparison_to_previous.md"
    comparison_md_path.write_text(
        build_validation_comparison_markdown(comparison_payload),
        encoding="utf-8",
    )

    dataset_governance = _dataset_governance(dataset_path, rows)
    reported_metrics = report.get("reported_metrics") if isinstance(report.get("reported_metrics"), dict) else {}
    reported_primary = (
        reported_metrics.get("primary")
        if isinstance(reported_metrics.get("primary"), dict)
        else {}
    )

    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_token,
        "generated_at": generated_at,
        "evaluation_basis": "public_observed_outcomes",
        "caveat": (
            "Validation uses public observed wildfire outcomes and is directional; "
            "it is not carrier claims truth or underwriting-performance truth."
        ),
        "inputs": {
            "evaluation_dataset_path": str(dataset_path),
            "evaluation_dataset_root": str(evaluation_dataset_root.expanduser()),
            "evaluation_dataset_run_id": evaluation_dataset_run_id,
            "thresholds": [float(v) for v in (thresholds or list(DEFAULT_THRESHOLDS))],
            "bins": int(max(2, int(bins))),
            "min_slice_size": int(max(2, int(min_slice_size))),
            "false_low_max_score": float(false_low_max_score),
            "false_high_min_score": float(false_high_min_score),
            "min_labeled_rows": int(max(1, int(min_labeled_rows))),
            "allow_label_derived_target": bool(allow_label_derived_target),
            "allow_surrogate_wildfire_score": bool(allow_surrogate_wildfire_score),
            "use_high_signal_simplified_model": bool(use_high_signal_simplified_model),
            "high_signal_feature_weights": (
                {k: float(v) for k, v in sorted((high_signal_feature_weights or {}).items())}
                if isinstance(high_signal_feature_weights, dict)
                else {}
            ),
            "high_signal_feature_filtering": (
                high_signal_feature_filtering
                if isinstance(high_signal_feature_filtering, dict)
                else {}
            ),
            "min_join_confidence_score_for_metrics": (
                float(min_join_confidence_score_for_metrics)
                if min_join_confidence_score_for_metrics is not None
                else None
            ),
            "retain_unusable_rows": bool(retain_unusable_rows),
            "baseline_run_id_requested": baseline_run_id,
        },
        "versions": {
            "product_version": PRODUCT_VERSION,
            "api_version": API_VERSION,
            "scoring_model_version": SCORING_MODEL_VERSION,
            "rules_logic_version": RULESET_LOGIC_VERSION,
            "factor_schema_version": FACTOR_SCHEMA_VERSION,
            "benchmark_pack_version": BENCHMARK_PACK_VERSION,
            "calibration_version": CALIBRATION_VERSION,
        },
        "governance": {
            "model_version": SCORING_MODEL_VERSION,
            "score_logic_version": RULESET_LOGIC_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            **dataset_governance,
            "command_config": {
                "script": "scripts/run_public_outcome_validation.py",
                "thresholds": [float(v) for v in (thresholds or list(DEFAULT_THRESHOLDS))],
                "bins": int(max(2, int(bins))),
                "min_slice_size": int(max(2, int(min_slice_size))),
                "false_low_max_score": float(false_low_max_score),
                "false_high_min_score": float(false_high_min_score),
                "min_labeled_rows": int(max(1, int(min_labeled_rows))),
                "allow_label_derived_target": bool(allow_label_derived_target),
                "allow_surrogate_wildfire_score": bool(allow_surrogate_wildfire_score),
                "use_high_signal_simplified_model": bool(use_high_signal_simplified_model),
                "high_signal_feature_weights": (
                    {k: float(v) for k, v in sorted((high_signal_feature_weights or {}).items())}
                    if isinstance(high_signal_feature_weights, dict)
                    else {}
                ),
                "high_signal_feature_filtering": (
                    high_signal_feature_filtering
                    if isinstance(high_signal_feature_filtering, dict)
                    else {}
                ),
                "min_join_confidence_score_for_metrics": (
                    float(min_join_confidence_score_for_metrics)
                    if min_join_confidence_score_for_metrics is not None
                    else None
                ),
                "retain_unusable_rows": bool(retain_unusable_rows),
                "baseline_run_id": selected_baseline_run,
            },
        },
        "raw_score_integrity": {
            "raw_wildfire_risk_score_preserved": True,
            "note": "Validation metrics are computed on raw deterministic model outputs before optional calibration overlays.",
        },
        "artifacts": {
            "validation_metrics_json": str(validation_metrics_path),
            "calibration_table_json": str(calibration_table_path),
            "threshold_metrics_json": str(threshold_metrics_path),
            "false_low_review_set_jsonl": str(false_low_path),
            "false_high_review_set_jsonl": str(false_high_path),
            "evaluation_rows_csv": str(evaluated_rows_csv_path),
            "feature_diagnostics_json": str(feature_diagnostics_path),
            "feature_signal_report_json": str(feature_signal_report_path),
            "direction_alignment_report_json": str(direction_alignment_report_path),
            "baseline_model_comparison_json": str(baseline_comparison_path),
            "segment_metrics_json": str(segment_metrics_path),
            "segment_report_markdown": str(segment_report_path),
            "comparison_to_previous_json": str(comparison_json_path),
            "comparison_to_previous_markdown": str(comparison_md_path),
            "summary_markdown": str(summary_path),
        },
        "headline": {
            "metrics_source": reported_metrics.get("primary_source") or "in_sample_fallback",
            "row_count_labeled": reported_primary.get("count"),
            "positive_rate": reported_primary.get("positive_rate"),
            "roc_auc": reported_primary.get("wildfire_risk_score_auc"),
            "pr_auc": reported_primary.get("wildfire_risk_score_pr_auc"),
            "brier": reported_primary.get("wildfire_risk_score_brier"),
            "in_sample_metrics": (
                reported_metrics.get("in_sample")
                if isinstance(reported_metrics.get("in_sample"), dict)
                else {}
            ),
            "holdout_metrics": (
                reported_metrics.get("holdout_event_level_out_of_sample")
                if isinstance(reported_metrics.get("holdout_event_level_out_of_sample"), dict)
                else {}
            ),
            "dataset_viable_for_predictive_modeling": (
                ((report.get("modeling_viability") or {}).get("dataset_viable_for_predictive_modeling"))
                if isinstance(report.get("modeling_viability"), dict)
                else None
            ),
        },
        "comparison_to_previous": {
            "available": bool(comparison_payload.get("available")),
            "baseline_run_id": comparison_payload.get("baseline_run_id"),
            "overall_direction_signals": comparison_payload.get("overall_direction_signals"),
            "likely_change_drivers": comparison_payload.get("likely_change_drivers"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "run_id": run_token,
        "run_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "validation_metrics_path": str(validation_metrics_path),
        "feature_diagnostics_path": str(feature_diagnostics_path),
        "feature_signal_report_path": str(feature_signal_report_path),
        "direction_alignment_report_path": str(direction_alignment_report_path),
        "baseline_model_comparison_path": str(baseline_comparison_path),
        "comparison_json_path": str(comparison_json_path),
        "comparison_markdown_path": str(comparison_md_path),
        "segment_metrics_path": str(segment_metrics_path),
        "segment_report_path": str(segment_report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run public-outcome model validation on a labeled evaluation dataset "
            "and write a reproducible metrics bundle."
        )
    )
    parser.add_argument(
        "--evaluation-dataset",
        default="",
        help=(
            "Path to labeled evaluation dataset (.json, .jsonl, .csv). "
            "If omitted, latest run under --evaluation-dataset-root is used."
        ),
    )
    parser.add_argument(
        "--evaluation-dataset-root",
        default=str(DEFAULT_EVALUATION_DATASET_ROOT),
        help="Root containing timestamped evaluation dataset runs.",
    )
    parser.add_argument(
        "--evaluation-dataset-run-id",
        default="",
        help="Optional specific evaluation dataset run id under --evaluation-dataset-root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_VALIDATION_OUTPUT_ROOT),
        help="Root output directory for validation bundles.",
    )
    parser.add_argument("--run-id", default="", help="Optional deterministic output run id.")
    parser.add_argument(
        "--thresholds",
        default="30,40,50,60,70,80",
        help="Comma-separated wildfire-risk thresholds for precision/recall/confusion summaries.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins for calibration table.")
    parser.add_argument("--min-slice-size", type=int, default=20, help="Minimum slice size before small-sample warning.")
    parser.add_argument(
        "--min-labeled-rows",
        type=int,
        default=1,
        help="Minimum usable labeled rows required before evaluation raises insufficient-data.",
    )
    parser.add_argument(
        "--allow-label-derived-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to deriving adverse-outcome target from outcome label when binary target is missing. "
            "Disabled by default for strict validation."
        ),
    )
    parser.add_argument(
        "--allow-surrogate-wildfire-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to surrogate wildfire-risk score from available hazard/vulnerability components when "
            "wildfire_risk_score is missing. Disabled by default for strict validation."
        ),
    )
    parser.add_argument(
        "--use-high-signal-simplified-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Override wildfire_risk_score with a simplified high-signal model using only: "
            "nearest_high_fuel_patch_distance_ft, canopy_adjacency_proxy_pct, "
            "vegetation_continuity_proxy_pct, and slope_index."
        ),
    )
    parser.add_argument(
        "--feature-signal-report",
        default="",
        help=(
            "Optional path to feature_signal_report.json. When provided with "
            "--use-high-signal-simplified-model, per-feature strength from report metrics "
            "(best_auc emphasis with threshold gating) is used as weights."
        ),
    )
    parser.add_argument(
        "--high-signal-min-feature-auc",
        type=float,
        default=0.55,
        help="Minimum per-feature univariate best_auc required for inclusion when loading feature_signal_report weights.",
    )
    parser.add_argument(
        "--high-signal-min-feature-stddev",
        type=float,
        default=1e-6,
        help="Minimum per-feature stddev required for inclusion when loading feature_signal_report weights.",
    )
    parser.add_argument(
        "--min-join-confidence-score-for-metrics",
        type=float,
        default=None,
        help=(
            "Optional minimum join-confidence score required for a row to be usable in metrics. "
            "Rows below threshold are retained and flagged, not dropped from artifacts."
        ),
    )
    parser.add_argument(
        "--retain-unusable-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain rows that are unusable for metrics in exported evaluation rows with exclusion flags.",
    )
    parser.add_argument("--false-low-max-score", type=float, default=40.0)
    parser.add_argument("--false-high-min-score", type=float, default=70.0)
    parser.add_argument(
        "--baseline-run-id",
        default="",
        help="Optional baseline validation run id to compare against. Defaults to previous run.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output run directory if present.")
    args = parser.parse_args()

    thresholds = [float(token) for token in str(args.thresholds).split(",") if token.strip()]
    high_signal_feature_weights: dict[str, float] | None = None
    high_signal_feature_filtering: dict[str, Any] | None = None
    if bool(args.use_high_signal_simplified_model) and str(args.feature_signal_report or "").strip():
        high_signal_feature_weights, high_signal_feature_filtering = _load_and_filter_high_signal_weights_from_feature_signal_report(
            Path(str(args.feature_signal_report)).expanduser(),
            min_feature_auc=float(args.high_signal_min_feature_auc),
            min_feature_stddev=float(args.high_signal_min_feature_stddev),
        )
    result = run_public_outcome_validation(
        evaluation_dataset=(Path(args.evaluation_dataset).expanduser() if args.evaluation_dataset else None),
        evaluation_dataset_root=Path(args.evaluation_dataset_root).expanduser(),
        evaluation_dataset_run_id=(args.evaluation_dataset_run_id or None),
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        thresholds=thresholds,
        bins=max(2, int(args.bins)),
        min_slice_size=max(2, int(args.min_slice_size)),
        false_low_max_score=float(args.false_low_max_score),
        false_high_min_score=float(args.false_high_min_score),
        min_labeled_rows=max(1, int(args.min_labeled_rows)),
        allow_label_derived_target=bool(args.allow_label_derived_target),
        allow_surrogate_wildfire_score=bool(args.allow_surrogate_wildfire_score),
        use_high_signal_simplified_model=bool(args.use_high_signal_simplified_model),
        high_signal_feature_weights=high_signal_feature_weights,
        high_signal_feature_filtering=high_signal_feature_filtering,
        min_join_confidence_score_for_metrics=(
            float(args.min_join_confidence_score_for_metrics)
            if args.min_join_confidence_score_for_metrics is not None
            else None
        ),
        retain_unusable_rows=bool(args.retain_unusable_rows),
        baseline_run_id=(args.baseline_run_id or None),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
