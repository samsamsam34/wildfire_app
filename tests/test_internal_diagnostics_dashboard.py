from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.internal_diagnostics_artifacts import (
    build_no_ground_truth_health_summary,
    list_no_ground_truth_runs,
    load_no_ground_truth_run_bundle,
)

client = TestClient(app_main.app)


def _write_run(root: Path, run_id: str, *, include_alignment: bool = True) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": run_id,
                "generated_at": "2026-03-18T00:00:00Z",
                "status_summary": {
                    "monotonicity": "warn",
                    "counterfactual": "ok",
                    "stability": "warn",
                    "distribution": "ok",
                    "benchmark_alignment": "ok",
                    "confidence_diagnostics": "warn",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(
        "# Sample Summary\n\n## Recommendation\n- prioritize larger fixture coverage\n- review instability outliers\n",
        encoding="utf-8",
    )
    (run_dir / "monotonicity_results.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "rule_count": 4,
                "passed_count": 3,
                "failed_count": 1,
                "rows": [{"rule_id": "r1", "passed": False, "detail": "test detail"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "counterfactual_results.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "top_interventions_by_median_impact": [
                    {"intervention": "clear_0_5ft_zone", "median_risk_delta": -4.0, "mean_risk_delta": -3.7, "count": 5}
                ],
                "flagged_interventions": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stability_results.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "test_count": 2,
                "tests": [
                    {
                        "test_id": "s1",
                        "max_abs_score_swing": 14.0,
                        "mean_abs_score_swing": 8.0,
                        "confidence_tier_change_rate": 0.4,
                        "rows": [
                            {"variant_type": "geocode_jitter", "wildfire_risk_score_delta": 4.0},
                            {"variant_type": "fallback_assumption", "wildfire_risk_score_delta": 9.0},
                        ],
                    }
                ],
                "warnings": ["large swing"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "distribution_results.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "overall": {"wildfire_risk_score": {"dynamic_range": 28.0}},
                "confidence_tier_counts": {"high": 2, "moderate": 3},
                "fallback_group_counts": {"high_evidence": 3, "fallback_heavy": 2},
                "rows": [{"risk_score": 45.0}, {"risk_score": 71.0}],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "confidence_diagnostics.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "record_count": 5,
                "confidence_tier_distribution": {"high": 2, "moderate": 2, "low": 1},
                "fallback_group_distribution": {"high_evidence": 3, "fallback_heavy": 2},
                "warnings": ["overconfidence warning"],
            }
        ),
        encoding="utf-8",
    )
    if include_alignment:
        (run_dir / "benchmark_alignment_results.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "rule_count": 2,
                    "rows": [
                        {
                            "signal_key": "fire_regime_index",
                            "spearman_rank_correlation": 0.52,
                            "bucket_agreement_ratio": 0.62,
                            "disagreement_cases": [{"scenario_id": "x"}],
                        }
                    ],
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def _write_public_validation_run(root: Path, run_id: str, *, roc_auc: float) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": run_id,
                "generated_at": "2026-03-18T00:00:00Z",
                "versions": {
                    "scoring_model_version": "1.10.0",
                    "rules_logic_version": "1.1.0",
                    "factor_schema_version": "1.3.0",
                    "calibration_version": "0.3.0",
                },
                "inputs": {"evaluation_dataset_path": "benchmark/public_outcomes/evaluation_dataset/demo/evaluation_dataset.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "sample_counts": {"row_count_usable": 20, "positive_rate": 0.25},
                "discrimination_metrics": {"wildfire_risk_score_auc": roc_auc, "wildfire_risk_score_pr_auc": 0.52},
                "brier_scores": {"wildfire_probability_proxy": 0.16},
                "calibration_metrics": {"wildfire_risk_score": {"expected_calibration_error": 0.08}},
                "false_review_sets": {"false_low_count": 3, "false_high_count": 2},
                "slice_metrics": {"by_confidence_tier": {"high": {"count": 10, "wildfire_risk_score_auc": 0.71, "wildfire_risk_score_brier": 0.12}}},
                "baseline_model_comparison": {
                    "available": True,
                    "full_model_auc": roc_auc,
                    "comparison": {
                        "beats_all_baselines_by_auc": True,
                        "best_baseline_name": "hazard_only",
                        "best_baseline_auc": max(0.0, float(roc_auc) - 0.02),
                        "auc_margin_vs_best_baseline": 0.02,
                        "complexity_justified_signal": "yes",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Validation Summary\n", encoding="utf-8")
    return run_dir


def _write_public_calibration_run(root: Path, run_id: str, *, brier_improvement: float) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": run_id,
                "generated_at": "2026-03-18T00:00:00Z",
                "fitted": True,
                "versions": {
                    "scoring_model_version": "1.10.0",
                    "rules_logic_version": "1.1.0",
                    "factor_schema_version": "1.3.0",
                    "calibration_version": "0.3.0",
                },
                "inputs": {"dataset_path": "benchmark/public_outcomes/evaluation_dataset/demo/evaluation_dataset.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pre_vs_post_metrics.json").write_text(
        json.dumps(
            {
                "pre": {"row_count": 20, "positive_rate": 0.25, "brier_probability": 0.18, "roc_auc_probability": 0.68, "pr_auc_probability": 0.44, "calibration": {"expected_calibration_error": 0.09}},
                "post": {"row_count": 20, "positive_rate": 0.25, "brier_probability": 0.16, "roc_auc_probability": 0.70, "pr_auc_probability": 0.47, "calibration": {"expected_calibration_error": 0.07}},
                "delta": {"brier_improvement": brier_improvement, "log_loss_improvement": 0.01},
                "slices": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "calibration_model.json").write_text(
        json.dumps({"method": "logistic"}),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# Calibration Summary\n", encoding="utf-8")
    return run_dir


def test_artifact_loader_handles_missing_root(monkeypatch, tmp_path: Path) -> None:
    empty_root = tmp_path / "missing"
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(empty_root))
    listing = list_no_ground_truth_runs()
    assert listing["available"] is False
    assert listing["artifact_root_exists"] is False
    assert listing["run_directory_count"] == 0
    assert "Checked artifact root" in str(listing.get("message") or "")
    bundle = load_no_ground_truth_run_bundle()
    assert bundle["available"] is False
    summary = build_no_ground_truth_health_summary(bundle)
    assert summary["available"] is False


def test_artifact_loader_and_summary_with_fixture(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_run(root, "20260318T000000Z")
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(root))
    listing = list_no_ground_truth_runs()
    assert listing["available"] is True
    assert listing["latest_run_id"] == "20260318T000000Z"
    bundle = load_no_ground_truth_run_bundle()
    assert bundle["available"] is True
    summary = build_no_ground_truth_health_summary(bundle)
    assert summary["available"] is True
    assert summary["monotonicity"]["failed_count"] == 1
    assert summary["stability"]["unstable_scenario_count"] == 1
    assert summary["stability"]["average_score_swing"] is not None
    assert summary["stability"]["top_unstable_factors"]
    assert summary["benchmark_alignment"]["average_spearman_rank_correlation"] is not None
    assert summary["benchmark_alignment"]["average_bucket_agreement_ratio"] is not None
    assert "comparison_to_previous" in summary
    assert summary["recommended_next_actions"]


def test_internal_dashboard_endpoints_with_and_without_artifacts(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    missing_root = tmp_path / "none"
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(missing_root))

    missing_res = client.get("/internal/diagnostics/api/latest")
    assert missing_res.status_code == 200
    missing_body = missing_res.json()
    assert missing_body["available"] is False

    root = tmp_path / "runs"
    _write_run(root, "20260318T000000Z", include_alignment=False)
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(root))

    runs_res = client.get("/internal/diagnostics/api/runs")
    assert runs_res.status_code == 200
    assert runs_res.json()["available"] is True

    latest_res = client.get("/internal/diagnostics/api/latest")
    assert latest_res.status_code == 200
    latest_body = latest_res.json()
    assert latest_body["available"] is True
    assert latest_body["summary"]["available"] is True
    assert "average_score_swing" in latest_body["summary"]["stability"]
    assert "average_spearman_rank_correlation" in latest_body["summary"]["benchmark_alignment"]

    section_missing = client.get("/internal/diagnostics/api/latest/benchmark_alignment")
    assert section_missing.status_code == 200
    assert section_missing.json()["available"] is False


def test_internal_dashboard_compare_endpoint(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    root = tmp_path / "runs"
    _write_run(root, "20260318T000000Z", include_alignment=True)
    _write_run(root, "20260319T000000Z", include_alignment=True)
    # Improve monotonicity in newer run so comparison delta is non-zero.
    (root / "20260319T000000Z" / "monotonicity_results.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "rule_count": 4,
                "passed_count": 4,
                "failed_count": 0,
                "rows": [],
                "violated_rules": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(root))

    explicit = client.get(
        "/internal/diagnostics/api/compare",
        params={"run_id": "20260319T000000Z", "baseline_run_id": "20260318T000000Z"},
    )
    assert explicit.status_code == 200
    body = explicit.json()
    assert body["available"] is True
    assert body["run_id"] == "20260319T000000Z"
    assert body["baseline_run_id"] == "20260318T000000Z"
    assert body["comparison_mode"] == "explicit_runs"
    assert body["monotonicity"]["failed_count_delta"] == -1

    auto = client.get(
        "/internal/diagnostics/api/compare",
        params={"run_id": "20260319T000000Z"},
    )
    assert auto.status_code == 200
    auto_body = auto.json()
    assert auto_body["available"] is True
    assert auto_body["comparison_mode"] == "latest_vs_previous"


def test_internal_dashboard_public_outcome_governance_endpoint(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    validation_root = tmp_path / "validation_runs"
    calibration_root = tmp_path / "calibration_runs"
    _write_public_validation_run(validation_root, "20260318T000000Z", roc_auc=0.66)
    _write_public_validation_run(validation_root, "20260319T000000Z", roc_auc=0.71)
    _write_public_calibration_run(calibration_root, "20260318T000000Z", brier_improvement=0.012)
    _write_public_calibration_run(calibration_root, "20260319T000000Z", brier_improvement=0.018)
    monkeypatch.setenv("WF_PUBLIC_OUTCOME_VALIDATION_DIR", str(validation_root))
    monkeypatch.setenv("WF_PUBLIC_OUTCOME_CALIBRATION_DIR", str(calibration_root))

    response = client.get("/internal/diagnostics/api/public-outcomes")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["validation"]["latest_summary"]["run_id"] == "20260319T000000Z"
    assert payload["calibration"]["latest_summary"]["run_id"] == "20260319T000000Z"
    baseline_cmp = ((payload["validation"]["latest_summary"] or {}).get("baseline_comparison") or {})
    assert baseline_cmp.get("available") is True
    assert baseline_cmp.get("best_baseline_name") == "hazard_only"
    assert baseline_cmp.get("complexity_justified_signal") == "yes"
    suff = (payload["validation"]["latest_summary"] or {}).get("data_sufficiency") or {}
    assert ((suff.get("total_dataset") or {}).get("tier")) == "limited"
    assert ((suff.get("high_confidence_subset") or {}).get("tier")) == "insufficient"
    viability = (payload["validation"]["latest_summary"] or {}).get("modeling_viability") or {}
    assert "dataset_viable_for_predictive_modeling" in viability
    assert "classification" in viability
    assert payload["validation"]["comparison_to_previous"]["available"] is True
    assert payload["calibration"]["comparison_to_previous"]["available"] is True

    explicit = client.get(
        "/internal/diagnostics/api/public-outcomes",
        params={
            "validation_run_id": "20260318T000000Z",
            "validation_baseline_run_id": "20260319T000000Z",
            "calibration_run_id": "20260318T000000Z",
            "calibration_baseline_run_id": "20260319T000000Z",
        },
    )
    assert explicit.status_code == 200
    explicit_body = explicit.json()
    selected = explicit_body.get("selected_run_ids") or {}
    assert selected.get("validation_run_id") == "20260318T000000Z"
    assert selected.get("calibration_run_id") == "20260318T000000Z"
    assert ((explicit_body.get("validation") or {}).get("latest_summary") or {}).get("run_id") == "20260318T000000Z"
    assert ((explicit_body.get("calibration") or {}).get("latest_summary") or {}).get("run_id") == "20260318T000000Z"


def test_internal_dashboard_page_contains_property_diagnostics_hooks() -> None:
    auth.API_KEYS = set()
    response = client.get("/internal/diagnostics")
    assert response.status_code == 200
    html = response.text
    assert "Model Diagnostics" in html
    assert "Model Health" in html
    assert "Property Diagnostics" in html
    assert "include_diagnostics=true" in html
    assert "runPropertyDiagnostics" in html
    assert "Property Diagnostics Query" in html
    assert "Run a property search to render assessment and diagnostics cards." in html
    assert "Public Outcome Governance (Internal)" in html
    assert 'id="propertyDiagMissingWarning"' in html
    assert 'id="inferredFields"' in html
    assert 'id="confidenceReasons"' in html
    assert 'id="vegetationSignal"' in html
    assert "Benchmark / Distribution" in html
    assert "Near-Structure Vegetation Signal" in html
