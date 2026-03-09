from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.model_tuning import (
    analyze_backtest_errors,
    evaluate_backtest_records,
    load_scoring_parameters,
    run_model_tuning,
    run_monotonic_guardrails,
)
from backend.scoring_config import load_scoring_config


def _records_fixture() -> list[dict]:
    return [
        {
            "event_id": "evt_1",
            "event_name": "Event 1",
            "event_date": "2021-01-01",
            "record_id": "r1",
            "outcome_label": "major_damage",
            "outcome_rank": 3,
            "scores": {
                "wildfire_risk_score": 30.0,
                "site_hazard_score": 40.0,
                "home_ignition_vulnerability_score": 35.0,
                "insurance_readiness_score": 55.0,
            },
            "confidence": {"confidence_score": 45.0, "confidence_tier": "preliminary"},
            "evidence_quality_summary": {
                "observed_factor_count": 2,
                "inferred_factor_count": 2,
                "missing_factor_count": 3,
                "fallback_factor_count": 3,
            },
            "coverage_summary": {"failed_count": 2},
            "score_evidence_ledger_summary": {
                "wildfire_risk_score": {"site_hazard_component": 18.0, "home_ignition_component": 12.0}
            },
            "scoring_notes": [
                "Environmental fuel layer unavailable — score uses partial data",
                "Building footprint not found — vulnerability estimated using point context",
            ],
        },
        {
            "event_id": "evt_1",
            "event_name": "Event 1",
            "event_date": "2021-01-01",
            "record_id": "r2",
            "outcome_label": "no_known_damage",
            "outcome_rank": 1,
            "scores": {
                "wildfire_risk_score": 82.0,
                "site_hazard_score": 78.0,
                "home_ignition_vulnerability_score": 70.0,
                "insurance_readiness_score": 35.0,
            },
            "confidence": {"confidence_score": 70.0, "confidence_tier": "medium"},
            "evidence_quality_summary": {
                "observed_factor_count": 6,
                "inferred_factor_count": 1,
                "missing_factor_count": 0,
                "fallback_factor_count": 0,
            },
            "coverage_summary": {"failed_count": 0},
            "score_evidence_ledger_summary": {
                "wildfire_risk_score": {"site_hazard_component": 58.0, "home_ignition_component": 24.0}
            },
            "scoring_notes": [],
        },
        {
            "event_id": "evt_2",
            "event_name": "Event 2",
            "event_date": "2022-09-01",
            "record_id": "r3",
            "outcome_label": "destroyed",
            "outcome_rank": 4,
            "scores": {
                "wildfire_risk_score": 88.0,
                "site_hazard_score": 84.0,
                "home_ignition_vulnerability_score": 76.0,
                "insurance_readiness_score": 28.0,
            },
            "confidence": {"confidence_score": 78.0, "confidence_tier": "high"},
            "evidence_quality_summary": {
                "observed_factor_count": 8,
                "inferred_factor_count": 1,
                "missing_factor_count": 0,
                "fallback_factor_count": 0,
            },
            "coverage_summary": {"failed_count": 0},
            "score_evidence_ledger_summary": {
                "wildfire_risk_score": {"site_hazard_component": 60.0, "home_ignition_component": 28.0}
            },
            "scoring_notes": [],
        },
    ]


def test_load_scoring_parameters_from_yaml_json(tmp_path: Path):
    payload = {
        "risk_blending_weights": {"environmental": 0.8, "structural": 0.2},
        "risk_bucket_thresholds": {"low_max": 30.0, "medium_max": 60.0},
    }
    path = tmp_path / "params.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_scoring_parameters(path)
    assert loaded["risk_blending_weights"]["environmental"] == 0.8
    assert loaded["risk_bucket_thresholds"]["medium_max"] == 60.0


def test_evaluate_backtest_records_and_error_analysis_outputs():
    records = _records_fixture()
    params = load_scoring_parameters("config/scoring_parameters.yaml")

    metrics = evaluate_backtest_records(records, scoring_parameters=params)
    assert metrics["record_count"] == 3
    assert "wildfire_vs_outcome" in metrics["rank_correlation"]
    assert "bucket_analysis" in metrics
    assert "false_low_rate" in metrics

    errors = analyze_backtest_errors(records, scoring_parameters=params)
    assert errors["summary_statistics"]["false_low_count"] >= 1
    assert errors["summary_statistics"]["false_high_count"] >= 1
    assert isinstance(errors["cluster_indicators"]["common_missing_evidence_false_low"], list)
    assert isinstance(errors["candidate_tuning_signals"], list)


def test_run_monotonic_guardrails_passes_with_default_config():
    cfg = load_scoring_config()
    result = run_monotonic_guardrails(cfg)
    assert "checks" in result
    assert isinstance(result["checks"], list)
    assert result["passed"] is True


def test_run_model_tuning_generates_artifact_with_mocked_backtest(monkeypatch, tmp_path: Path):
    records_base = _records_fixture()

    def _fake_runner(*, parameters, dataset_paths, output_dir, ruleset_id=None):
        env = float((parameters.get("risk_blending_weights") or {}).get("environmental", 0.75))
        adjusted = []
        for row in records_base:
            clone = json.loads(json.dumps(row))
            score = float(clone["scores"]["wildfire_risk_score"])
            # Lower environmental blend produces slightly lower false-high behavior in this fixture.
            clone["scores"]["wildfire_risk_score"] = round(max(0.0, min(100.0, score - (env - 0.7) * 30.0)), 1)
            adjusted.append(clone)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"fake_{env:.2f}.json"
        artifact = {
            "records": adjusted,
            "model_governance": {
                "product_version": "0.10.0",
                "api_version": "1.0.0",
                "scoring_model_version": "1.5.0",
            },
            "artifact_path": str(artifact_path),
            "markdown_summary_path": str(artifact_path.with_suffix(".md")),
        }
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        artifact_path.with_suffix(".md").write_text("fake summary\n", encoding="utf-8")
        return artifact

    monkeypatch.setattr("backend.model_tuning._run_backtest_for_parameters", _fake_runner)

    artifact = run_model_tuning(
        dataset_paths=["benchmark/event_backtest_sample_v1.json"],
        scoring_parameters_path="config/scoring_parameters.yaml",
        output_dir=tmp_path / "tuning_out",
        max_candidates=3,
    )

    assert Path(artifact["artifact_path"]).exists()
    assert Path(artifact["markdown_summary_path"]).exists()
    assert artifact["summary"]["candidate_count"] >= 1
    assert "best_experiment" in artifact
    assert "before_after_comparison" in artifact
    assert isinstance(artifact["recommended_parameter_changes"], list)
    assert "model_governance" in artifact
    assert artifact["experiments"][0]["tuning_run_id"]
    assert artifact["experiments"][0]["parameter_set_id"]
    assert "metrics" in artifact["experiments"][0]


def test_run_model_tuning_script_exit_behavior(tmp_path: Path):
    script = Path("scripts") / "run_model_tuning.py"
    ok = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "benchmark/event_backtest_sample_v1.json",
            "--max-candidates",
            "1",
            "--output-dir",
            str(tmp_path / "out_ok"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
