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

from backend.event_backtesting import DEFAULT_DATASET_PATH, run_event_backtest
from backend.public_outcome_validation import (
    DEFAULT_THRESHOLDS,
    evaluate_public_outcome_dataset_file,
    write_evaluation_rows_csv,
)
from backend.version import (
    API_VERSION,
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    SCORING_MODEL_VERSION,
)
from scripts.build_calibration_dataset import build_calibration_dataset
from scripts.fit_public_outcome_calibration import fit_calibration
from scripts.ingest_public_structure_damage import normalize_public_damage_rows


def _timestamp_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _deterministic_timestamp(run_id: str | None) -> str:
    if run_id:
        return str(run_id)
    return datetime.now(tz=timezone.utc).isoformat()


def _collect_governance(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    keys = (
        "product_version",
        "api_version",
        "scoring_model_version",
        "ruleset_version",
        "rules_logic_version",
        "factor_schema_version",
        "benchmark_pack_version",
        "calibration_version",
        "region_data_version",
        "data_bundle_version",
    )
    values: dict[str, set[str]] = {key: set() for key in keys}
    for row in rows:
        governance = row.get("model_governance")
        if not isinstance(governance, dict):
            continue
        for key in keys:
            token = str(governance.get(key) or "").strip()
            if token:
                values[key].add(token)
    return {key: sorted(tokens) for key, tokens in values.items() if tokens}


def _headline_assessment(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("discrimination_metrics") or {}
    brier = evaluation.get("brier_scores") or {}
    calibration = evaluation.get("calibration_metrics") or {}
    auc = metrics.get("wildfire_risk_score_auc")
    spearman = metrics.get("wildfire_vs_outcome_rank_spearman")
    ece = ((calibration.get("wildfire_risk_score") or {}).get("expected_calibration_error"))
    raw_brier = brier.get("wildfire_probability_proxy")
    directional = bool(isinstance(auc, (int, float)) and auc >= 0.6)

    if isinstance(ece, (int, float)):
        if ece <= 0.06:
            calibration_quality = "well_calibrated"
        elif ece <= 0.12:
            calibration_quality = "mixed_calibration"
        else:
            calibration_quality = "poorly_calibrated"
    else:
        calibration_quality = "insufficient_data"

    recommendation = str(evaluation.get("calibration_artifact_recommendation") or "not_recommended_yet")
    return {
        "directional_predictive_value": directional,
        "wildfire_risk_score_auc": auc,
        "wildfire_vs_outcome_rank_spearman": spearman,
        "wildfire_probability_proxy_brier": raw_brier,
        "wildfire_probability_proxy_ece": ece,
        "calibration_quality_assessment": calibration_quality,
        "calibration_artifact_recommendation": recommendation,
    }


def _format_slice_line(name: str, detail: dict[str, Any]) -> str:
    return (
        f"- `{name}`: count={detail.get('count')}, "
        f"auc={detail.get('wildfire_risk_score_auc')}, "
        f"brier={detail.get('wildfire_risk_score_brier')}, "
        f"positive_rate={detail.get('positive_rate')}"
    )


def _build_markdown_summary(
    *,
    run_id: str,
    evaluation: dict[str, Any],
    manifest: dict[str, Any],
    fitted_calibration_artifact_path: str | None,
) -> str:
    headline = _headline_assessment(evaluation)
    guardrails = evaluation.get("guardrails") or {}
    slice_metrics = evaluation.get("slice_metrics") or {}
    by_confidence = slice_metrics.get("by_confidence_tier") or {}
    by_evidence = slice_metrics.get("by_evidence_group") or {}

    lines = [
        "# Public Outcome Validation v1",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{manifest.get('generated_at')}`",
        f"- Dataset rows (usable): `{(evaluation.get('sample_counts') or {}).get('row_count_usable')}`",
        f"- Positive rate: `{(evaluation.get('sample_counts') or {}).get('positive_rate')}`",
        "",
        "## Directional Predictive Value",
        (
            "- **Directional predictive value detected**."
            if headline.get("directional_predictive_value")
            else "- **Directional predictive value not yet established**."
        ),
        f"- ROC AUC (wildfire risk): `{headline.get('wildfire_risk_score_auc')}`",
        f"- Spearman (risk vs outcome rank): `{headline.get('wildfire_vs_outcome_rank_spearman')}`",
        "",
        "## Calibration Quality",
        f"- Brier score (raw wildfire probability proxy): `{headline.get('wildfire_probability_proxy_brier')}`",
        f"- ECE (raw wildfire probability proxy): `{headline.get('wildfire_probability_proxy_ece')}`",
        f"- Assessment: `{headline.get('calibration_quality_assessment')}`",
        "",
        "## Confidence/Evidence Slices",
    ]
    if by_confidence:
        lines.append("### By Confidence Tier")
        for name in sorted(by_confidence.keys()):
            lines.append(_format_slice_line(name, by_confidence[name]))
    if by_evidence:
        lines.append("")
        lines.append("### By Evidence Group")
        for name in sorted(by_evidence.keys()):
            lines.append(_format_slice_line(name, by_evidence[name]))

    lines.extend(
        [
            "",
            "## Guardrails",
        ]
    )
    warnings = list(guardrails.get("warnings") or [])
    if not warnings:
        lines.append("- No additional guardrail warnings.")
    else:
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Calibration Artifact Decision",
            f"- Recommendation: `{headline.get('calibration_artifact_recommendation')}`",
            (
                f"- Fitted artifact: `{fitted_calibration_artifact_path}`"
                if fitted_calibration_artifact_path
                else "- Fitted artifact: not produced in this run."
            ),
            "",
            "## Caveats",
            "- This is directional validation against public outcomes, not carrier claims truth.",
            "- Deterministic raw model scores remain unchanged; calibration artifacts are optional overlays.",
            "- Fallback-heavy cohorts should not be used as primary calibration anchors.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_public_outcome_validation(
    *,
    outcomes_input: Path,
    feature_artifacts: list[Path] | None = None,
    backtest_datasets: list[Path] | None = None,
    output_root: Path = Path("benchmark/public_outcome_validation"),
    run_id: str | None = None,
    thresholds: list[float] | None = None,
    bins: int = 10,
    fit_calibration_artifact: bool = False,
    min_rows_for_fit: int = 50,
    fallback_heavy_fit_threshold: float = 0.50,
    allow_fallback_heavy_fit: bool = False,
    source_name: str | None = None,
    reuse_existing_assessments: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_token = str(run_id or _timestamp_id())
    generated_at = _deterministic_timestamp(run_id)
    run_dir = Path(output_root).expanduser() / run_token
    if run_dir.exists() and not overwrite:
        raise ValueError(f"Output run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Normalize public outcomes.
    normalized = normalize_public_damage_rows(
        input_path=Path(outcomes_input).expanduser(),
        source_name=source_name,
    )
    normalized["generated_at"] = generated_at
    normalized_path = run_dir / "public_outcomes_normalized.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    if int(normalized.get("record_count") or 0) == 0:
        raise ValueError("Outcome normalization produced zero usable records; cannot run validation.")

    # 2) Run or load backtest feature artifacts.
    artifact_paths: list[Path]
    backtest_info: dict[str, Any]
    if feature_artifacts:
        artifact_paths = [Path(path).expanduser() for path in feature_artifacts]
        for path in artifact_paths:
            if not path.exists():
                raise ValueError(f"Feature artifact not found: {path}")
        backtest_info = {
            "mode": "loaded_existing",
            "feature_artifacts": [str(path) for path in artifact_paths],
        }
    else:
        datasets = backtest_datasets or [DEFAULT_DATASET_PATH]
        backtest_dir = run_dir / "event_backtest_results"
        backtest_artifact = run_event_backtest(
            dataset_paths=[str(path) for path in datasets],
            output_dir=backtest_dir,
            reuse_existing_assessments=reuse_existing_assessments,
        )
        artifact_paths = [Path(str(backtest_artifact.get("artifact_path")))]
        backtest_info = {
            "mode": "executed",
            "datasets": [str(Path(path).expanduser()) for path in datasets],
            "artifact_path": str(backtest_artifact.get("artifact_path")),
            "markdown_summary_path": str(backtest_artifact.get("markdown_summary_path")),
        }

    # 3) Build joined calibration dataset.
    calibration_dataset_path = run_dir / "public_outcome_calibration_dataset.json"
    calibration_dataset_csv_path = run_dir / "public_outcome_calibration_dataset.csv"
    calibration_dataset = build_calibration_dataset(
        outcome_path=normalized_path,
        feature_artifacts=artifact_paths,
        output_path=calibration_dataset_path,
        output_csv=calibration_dataset_csv_path,
    )
    calibration_dataset["generated_at"] = generated_at
    calibration_dataset_path.write_text(
        json.dumps(calibration_dataset, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 4) Evaluate discrimination + calibration.
    evaluation_path = run_dir / "public_outcome_evaluation.json"
    evaluation_rows_path = run_dir / "public_outcome_evaluation_rows.csv"
    evaluation, evaluated_rows = evaluate_public_outcome_dataset_file(
        dataset_path=calibration_dataset_path,
        thresholds=thresholds or list(DEFAULT_THRESHOLDS),
        bins=max(2, int(bins)),
        generated_at=generated_at,
    )
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    write_evaluation_rows_csv(rows=evaluated_rows, output_csv=evaluation_rows_path)

    # 5) Optionally fit calibration artifact.
    fitted_calibration_artifact_path: Path | None = None
    fit_status = "not_requested"
    fit_warnings: list[str] = []
    fallback_heavy_fraction = float(
        ((evaluation.get("sample_counts") or {}).get("fallback_heavy_fraction") or 0.0)
    )
    usable_rows = int((evaluation.get("sample_counts") or {}).get("row_count_usable") or 0)
    if fit_calibration_artifact:
        if usable_rows < max(10, int(min_rows_for_fit)):
            fit_status = "skipped_small_sample"
            fit_warnings.append(
                f"Calibration fit skipped: usable_rows={usable_rows} is below min_rows_for_fit={min_rows_for_fit}."
            )
        elif fallback_heavy_fraction > float(fallback_heavy_fit_threshold) and not allow_fallback_heavy_fit:
            fit_status = "skipped_fallback_heavy"
            fit_warnings.append(
                "Calibration fit skipped: fallback-heavy fraction exceeds threshold. "
                "Use --allow-fallback-heavy-fit only for exploratory analysis."
            )
        else:
            fitted_calibration_artifact_path = run_dir / "public_outcome_calibration_artifact.json"
            artifact = fit_calibration(
                dataset_path=calibration_dataset_path,
                output_path=fitted_calibration_artifact_path,
            )
            artifact["generated_at"] = generated_at
            fitted_calibration_artifact_path.write_text(
                json.dumps(artifact, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            fit_status = "fitted"
            if fallback_heavy_fraction > float(fallback_heavy_fit_threshold):
                fit_warnings.append(
                    "Calibration was fit on fallback-heavy data. Treat artifact as exploratory and non-production."
                )

    # 6) Write manifest and markdown summary.
    governance_versions = _collect_governance(calibration_dataset.get("rows") or [])
    headline = _headline_assessment(evaluation)
    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_token,
        "generated_at": generated_at,
        "artifacts": {
            "normalized_outcomes_json": str(normalized_path),
            "calibration_dataset_json": str(calibration_dataset_path),
            "calibration_dataset_csv": str(calibration_dataset_csv_path),
            "evaluation_json": str(evaluation_path),
            "evaluation_rows_csv": str(evaluation_rows_path),
            "calibration_artifact_json": (
                str(fitted_calibration_artifact_path) if fitted_calibration_artifact_path else None
            ),
            "summary_markdown": str(run_dir / "public_outcome_validation_summary.md"),
        },
        "inputs": {
            "outcomes_input": str(Path(outcomes_input).expanduser()),
            "feature_artifacts": [str(path) for path in artifact_paths],
            "backtest": backtest_info,
        },
        "versions": {
            "product_version": PRODUCT_VERSION,
            "api_version": API_VERSION,
            "scoring_model_version": SCORING_MODEL_VERSION,
            "rules_logic_version": RULESET_LOGIC_VERSION,
            "factor_schema_version": FACTOR_SCHEMA_VERSION,
            "benchmark_pack_version": BENCHMARK_PACK_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "observed_model_governance_versions": governance_versions,
        },
        "evaluation_headline": headline,
        "guardrails": {
            "warnings": list((evaluation.get("guardrails") or {}).get("warnings") or []) + fit_warnings,
            "data_leakage_risks": evaluation.get("data_leakage_risks"),
        },
        "calibration_fit": {
            "requested": bool(fit_calibration_artifact),
            "status": fit_status,
            "min_rows_for_fit": int(min_rows_for_fit),
            "fallback_heavy_fraction": fallback_heavy_fraction,
            "fallback_heavy_fit_threshold": float(fallback_heavy_fit_threshold),
            "allow_fallback_heavy_fit": bool(allow_fallback_heavy_fit),
        },
        "raw_score_integrity": {
            "raw_wildfire_risk_score_preserved": True,
            "note": "Deterministic raw model outputs are unchanged; optional calibration artifacts are additive.",
        },
    }

    summary_markdown = _build_markdown_summary(
        run_id=run_token,
        evaluation=evaluation,
        manifest=manifest,
        fitted_calibration_artifact_path=(
            str(fitted_calibration_artifact_path) if fitted_calibration_artifact_path else None
        ),
    )
    summary_path = run_dir / "public_outcome_validation_summary.md"
    summary_path.write_text(summary_markdown, encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "evaluation_path": str(evaluation_path),
        "fit_status": fit_status,
        "calibration_artifact_path": (
            str(fitted_calibration_artifact_path) if fitted_calibration_artifact_path else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Public Outcome Validation v1 end-to-end and emit a reproducible artifact bundle."
    )
    parser.add_argument(
        "--outcomes-input",
        default=str(DEFAULT_DATASET_PATH),
        help="Public outcomes input path (CSV/JSON/GeoJSON) for normalization.",
    )
    parser.add_argument(
        "--feature-artifact",
        action="append",
        default=[],
        help="Existing event backtest artifact path(s). If omitted, backtest will be executed.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Event dataset path(s) for backtest execution when --feature-artifact is not provided.",
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/public_outcome_validation",
        help="Root output directory for timestamped validation runs.",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run id for deterministic output naming.")
    parser.add_argument(
        "--thresholds",
        default="30,40,50,60,70,80",
        help="Comma-separated wildfire-risk thresholds for PR/confusion summaries.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins for calibration tables.")
    parser.add_argument("--source-name", default="", help="Optional normalized outcome source-name override.")
    parser.add_argument("--fit-calibration", action="store_true", help="Fit and save optional calibration artifact.")
    parser.add_argument("--min-rows-for-fit", type=int, default=50)
    parser.add_argument("--fallback-heavy-fit-threshold", type=float, default=0.5)
    parser.add_argument("--allow-fallback-heavy-fit", action="store_true")
    parser.add_argument("--reuse-existing-assessments", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite run directory if it already exists.")
    args = parser.parse_args()

    thresholds = [float(token) for token in str(args.thresholds).split(",") if token.strip()]
    result = run_public_outcome_validation(
        outcomes_input=Path(args.outcomes_input).expanduser(),
        feature_artifacts=[Path(path).expanduser() for path in args.feature_artifact],
        backtest_datasets=[Path(path).expanduser() for path in args.dataset],
        output_root=Path(args.output_root).expanduser(),
        run_id=(args.run_id or None),
        thresholds=thresholds,
        bins=max(2, int(args.bins)),
        fit_calibration_artifact=bool(args.fit_calibration),
        min_rows_for_fit=max(10, int(args.min_rows_for_fit)),
        fallback_heavy_fit_threshold=float(args.fallback_heavy_fit_threshold),
        allow_fallback_heavy_fit=bool(args.allow_fallback_heavy_fit),
        source_name=(args.source_name or None),
        reuse_existing_assessments=bool(args.reuse_existing_assessments),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
