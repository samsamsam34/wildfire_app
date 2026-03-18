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


def test_artifact_loader_handles_missing_root(monkeypatch, tmp_path: Path) -> None:
    empty_root = tmp_path / "missing"
    monkeypatch.setenv("WF_NO_GROUND_TRUTH_EVAL_DIR", str(empty_root))
    listing = list_no_ground_truth_runs()
    assert listing["available"] is False
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


def test_internal_dashboard_page_contains_property_diagnostics_hooks() -> None:
    auth.API_KEYS = set()
    response = client.get("/internal/diagnostics")
    assert response.status_code == 200
    html = response.text
    assert "Model Diagnostics" in html
    assert "include_diagnostics=true" in html
    assert "runPropertyDiagnostics" in html
    assert "Property Diagnostics Query" in html
    assert "Benchmark / Distribution" in html
