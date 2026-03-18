from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.public_outcome_validation import evaluate_public_outcome_dataset_file
from scripts.build_calibration_dataset import build_calibration_dataset
from scripts.ingest_public_structure_damage import normalize_public_damage_rows
from scripts.run_public_outcome_validation import run_public_outcome_validation


def _write_outcomes_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "record_id,event_id,incident_name,inspection_date,address,city,state,zip,latitude,longitude,damage_state",
                "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Destroyed",
                "2,evt-a,Event A,2021-08-01,101 Main St,Town,CA,90001,39.1002,-120.1002,Major Damage",
                "3,evt-a,Event A,2021-08-01,102 Main St,Town,CA,90001,39.1003,-120.1003,Major Damage",
                "4,evt-a,Event A,2021-08-01,103 Main St,Town,CA,90001,39.1004,-120.1004,Minor Damage",
                "5,evt-a,Event A,2021-08-01,104 Main St,Town,CA,90001,39.1005,-120.1005,No Damage",
                "6,evt-a,Event A,2021-08-01,105 Main St,Town,CA,90001,39.1006,-120.1006,No Damage",
                "7,evt-a,Event A,2021-08-01,106 Main St,Town,CA,90001,39.1007,-120.1007,Destroyed",
                "8,evt-a,Event A,2021-08-01,107 Main St,Town,CA,90001,39.1008,-120.1008,No Damage",
            ]
        ),
        encoding="utf-8",
    )


def _write_feature_artifact(path: Path) -> None:
    rows = [
        ("1", 92.0, "high", 0, 0, 0),
        ("2", 84.0, "high", 0, 0, 0),
        ("3", 75.0, "moderate", 1, 0, 1),
        ("4", 58.0, "moderate", 1, 1, 0),
        ("5", 43.0, "high", 0, 0, 0),
        ("6", 30.0, "high", 0, 0, 0),
        ("7", 88.0, "low", 3, 2, 2),
        ("8", 72.0, "low", 2, 1, 1),
    ]
    records = []
    for source_record_id, score, tier, fallback_count, missing_count, coverage_fallback in rows:
        records.append(
            {
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "record_id": f"rec-{source_record_id}",
                "source_record_id": source_record_id,
                "latitude": 39.1 + (int(source_record_id) * 0.0001),
                "longitude": -120.1 - (int(source_record_id) * 0.0001),
                "scores": {
                    "wildfire_risk_score": score,
                    "site_hazard_score": max(0.0, min(100.0, score - 5.0)),
                    "home_ignition_vulnerability_score": max(0.0, min(100.0, score - 3.0)),
                    "insurance_readiness_score": max(0.0, min(100.0, 100.0 - score)),
                },
                "confidence": {
                    "confidence_tier": tier,
                    "confidence_score": 85.0 if tier == "high" else (62.0 if tier == "moderate" else 38.0),
                    "use_restriction": "none" if tier == "high" else "limited_review",
                },
                "assessment_status": "scored",
                "evidence_quality_summary": {
                    "evidence_tier": "high" if tier == "high" else ("moderate" if tier == "moderate" else "low"),
                    "fallback_factor_count": fallback_count,
                    "missing_factor_count": missing_count,
                    "inferred_factor_count": 1 if tier != "high" else 0,
                },
                "coverage_summary": {"failed_count": 0 if tier != "low" else 1, "fallback_count": coverage_fallback},
                "factor_contribution_breakdown": {
                    "fuel_proximity_risk": {"contribution": score / 5.0},
                    "defensible_space_risk": {"contribution": (fallback_count + 1.0)},
                },
                "raw_feature_vector": {"burn_probability": score / 100.0},
                "transformed_feature_vector": {"burn_probability_index": score},
                "property_level_context": {"resolved_region_id": "test_region"},
                "model_governance": {
                    "product_version": "0.17.1",
                    "api_version": "1.5.0",
                    "scoring_model_version": "1.10.0",
                    "ruleset_version": "1.0.0",
                    "rules_logic_version": "1.1.0",
                    "factor_schema_version": "1.3.0",
                    "benchmark_pack_version": "1.0.0",
                    "calibration_version": "0.3.0",
                    "region_data_version": "test_region",
                    "data_bundle_version": "test_bundle_v1",
                },
            }
        )
    path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")


def test_evaluate_public_outcome_dataset_reports_required_metrics(tmp_path: Path) -> None:
    outcomes_csv = tmp_path / "outcomes.csv"
    _write_outcomes_csv(outcomes_csv)
    normalized = normalize_public_damage_rows(input_path=outcomes_csv, source_name="fixture_outcomes")
    normalized_path = tmp_path / "normalized.json"
    normalized_path.write_text(json.dumps(normalized), encoding="utf-8")

    feature_artifact = tmp_path / "feature_artifact.json"
    _write_feature_artifact(feature_artifact)
    dataset_path = tmp_path / "joined.json"
    build_calibration_dataset(
        outcome_path=normalized_path,
        feature_artifacts=[feature_artifact],
        output_path=dataset_path,
    )

    report, rows = evaluate_public_outcome_dataset_file(dataset_path=dataset_path, thresholds=[40.0, 70.0], bins=5)
    assert report["row_count_labeled"] >= 8
    assert "sample_counts" in report
    assert "by_event" in (report["sample_counts"] or {})
    assert "by_region" in (report["sample_counts"] or {})
    assert report["discrimination_metrics"]["wildfire_risk_score_auc"] is not None
    assert report["brier_scores"]["wildfire_probability_proxy"] is not None
    assert "40" in report["threshold_metrics_wildfire_risk_score"]
    assert "70" in report["threshold_metrics_wildfire_risk_score"]
    assert "by_confidence_tier" in (report["slice_metrics"] or {})
    assert "by_evidence_group" in (report["slice_metrics"] or {})
    assert "false_review_sets" in report
    assert isinstance(rows, list) and rows


def test_public_outcome_validation_orchestration_is_deterministic_with_fixed_run_id(tmp_path: Path) -> None:
    outcomes_csv = tmp_path / "outcomes.csv"
    _write_outcomes_csv(outcomes_csv)
    feature_artifact = tmp_path / "feature_artifact.json"
    _write_feature_artifact(feature_artifact)
    output_root = tmp_path / "validation_runs"

    first = run_public_outcome_validation(
        outcomes_input=outcomes_csv,
        feature_artifacts=[feature_artifact],
        output_root=output_root,
        run_id="fixed_validation_run",
        thresholds=[40.0, 70.0],
        bins=5,
        fit_calibration_artifact=False,
        overwrite=True,
    )
    second = run_public_outcome_validation(
        outcomes_input=outcomes_csv,
        feature_artifacts=[feature_artifact],
        output_root=output_root,
        run_id="fixed_validation_run",
        thresholds=[40.0, 70.0],
        bins=5,
        fit_calibration_artifact=False,
        overwrite=True,
    )

    assert first["manifest_path"] == second["manifest_path"]
    manifest_text_1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    manifest_text_2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert manifest_text_1 == manifest_text_2

    evaluation_text_1 = Path(first["evaluation_path"]).read_text(encoding="utf-8")
    evaluation_text_2 = Path(second["evaluation_path"]).read_text(encoding="utf-8")
    assert evaluation_text_1 == evaluation_text_2
    assert Path(first["summary_path"]).exists()


def test_evaluate_public_outcome_dataset_fails_when_required_fields_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"record_id": "a", "structure_loss_or_major_damage": 1, "scores": {}},
                    {"record_id": "b", "structure_loss_or_major_damage": None, "scores": {"wildfire_risk_score": 60}},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        evaluate_public_outcome_dataset_file(dataset_path=dataset_path)
