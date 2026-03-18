from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.public_outcome_validation import evaluate_public_outcome_dataset_file
from scripts.build_calibration_dataset import build_calibration_dataset
from scripts.build_public_outcome_evaluation_dataset import build_public_outcome_evaluation_dataset
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
    assert report["discrimination_metrics"]["wildfire_risk_score_pr_auc"] is not None
    assert report["brier_scores"]["wildfire_probability_proxy"] is not None
    assert "40" in report["threshold_metrics_wildfire_risk_score"]
    assert "70" in report["threshold_metrics_wildfire_risk_score"]
    assert report["calibration_metrics"]["wildfire_risk_score"]["bins"]
    assert "by_confidence_tier" in (report["slice_metrics"] or {})
    assert "by_evidence_group" in (report["slice_metrics"] or {})
    assert "false_review_sets" in report
    assert isinstance(rows, list) and rows


def test_public_outcome_validation_orchestration_is_deterministic_with_fixed_run_id(tmp_path: Path) -> None:
    outcomes_csv = tmp_path / "outcomes.csv"
    _write_outcomes_csv(outcomes_csv)
    feature_artifact = tmp_path / "feature_artifact.json"
    _write_feature_artifact(feature_artifact)
    normalized = normalize_public_damage_rows(input_path=outcomes_csv, source_name="fixture_outcomes")
    normalized_path = tmp_path / "normalized.json"
    normalized_path.write_text(json.dumps(normalized), encoding="utf-8")

    evaluation_dataset_run = build_public_outcome_evaluation_dataset(
        outcomes_path=normalized_path,
        feature_artifacts=[feature_artifact],
        output_root=tmp_path / "eval_dataset_runs",
        run_id="joined_fixture",
        overwrite=True,
    )
    evaluation_dataset_path = Path(evaluation_dataset_run["run_dir"]) / "evaluation_dataset.jsonl"
    output_root = tmp_path / "validation_runs"

    first = run_public_outcome_validation(
        evaluation_dataset=evaluation_dataset_path,
        output_root=output_root,
        run_id="fixed_validation_run",
        thresholds=[40.0, 70.0],
        bins=5,
        overwrite=True,
    )
    second = run_public_outcome_validation(
        evaluation_dataset=evaluation_dataset_path,
        output_root=output_root,
        run_id="fixed_validation_run",
        thresholds=[40.0, 70.0],
        bins=5,
        overwrite=True,
    )

    assert first["manifest_path"] == second["manifest_path"]
    manifest_text_1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    manifest_text_2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert manifest_text_1 == manifest_text_2

    evaluation_text_1 = Path(first["validation_metrics_path"]).read_text(encoding="utf-8")
    evaluation_text_2 = Path(second["validation_metrics_path"]).read_text(encoding="utf-8")
    assert evaluation_text_1 == evaluation_text_2
    assert Path(first["summary_path"]).exists()
    assert (output_root / "fixed_validation_run" / "calibration_table.json").exists()
    assert (output_root / "fixed_validation_run" / "threshold_metrics.json").exists()
    assert (output_root / "fixed_validation_run" / "false_low_review_set.jsonl").exists()
    assert (output_root / "fixed_validation_run" / "false_high_review_set.jsonl").exists()
    assert (output_root / "fixed_validation_run" / "comparison_to_previous.json").exists()
    assert (output_root / "fixed_validation_run" / "comparison_to_previous.md").exists()


def test_evaluation_jsonl_dataset_supports_join_confidence_slices(tmp_path: Path) -> None:
    rows = [
        {
            "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
            "feature": {"record_id": "f1", "source_record_id": "1", "address_text": "100 Main St", "latitude": 39.1, "longitude": -120.1},
            "outcome": {"record_id": "o1", "damage_label": "destroyed", "damage_severity_class": "destroyed", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 90.0, "site_hazard_score": 84.0, "home_ignition_vulnerability_score": 86.0, "insurance_readiness_score": 32.0},
            "confidence": {"confidence_tier": "high", "confidence_score": 84.0},
            "evidence": {"evidence_quality_tier": "high", "evidence_quality_summary": {"fallback_factor_count": 0, "missing_factor_count": 0, "inferred_factor_count": 0}, "coverage_summary": {"failed_count": 0, "fallback_count": 0}},
            "feature_snapshot": {"raw_feature_vector": {"fuel": 0.9}, "transformed_feature_vector": {"fuel_idx": 90.0}, "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 18.0}}},
            "join_metadata": {"join_method": "exact_source_record_id", "join_confidence_tier": "high", "join_confidence_score": 0.98},
        },
        {
            "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
            "feature": {"record_id": "f2", "source_record_id": "2", "address_text": "101 Main St", "latitude": 39.101, "longitude": -120.101},
            "outcome": {"record_id": "o2", "damage_label": "major_damage", "damage_severity_class": "major", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 72.0, "site_hazard_score": 66.0, "home_ignition_vulnerability_score": 67.0, "insurance_readiness_score": 41.0},
            "confidence": {"confidence_tier": "moderate", "confidence_score": 63.0},
            "evidence": {"evidence_quality_tier": "moderate", "evidence_quality_summary": {"fallback_factor_count": 1, "missing_factor_count": 1, "inferred_factor_count": 1}, "coverage_summary": {"failed_count": 0, "fallback_count": 1}},
            "feature_snapshot": {"raw_feature_vector": {"fuel": 0.6}, "transformed_feature_vector": {"fuel_idx": 60.0}, "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 13.0}}},
            "join_metadata": {"join_method": "nearest_event_coordinates", "join_confidence_tier": "moderate", "join_confidence_score": 0.78},
        },
        {
            "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
            "feature": {"record_id": "f3", "source_record_id": "3", "address_text": "102 Main St", "latitude": 39.102, "longitude": -120.102},
            "outcome": {"record_id": "o3", "damage_label": "no_damage", "damage_severity_class": "none", "structure_loss_or_major_damage": 0},
            "scores": {"wildfire_risk_score": 28.0, "site_hazard_score": 25.0, "home_ignition_vulnerability_score": 30.0, "insurance_readiness_score": 78.0},
            "confidence": {"confidence_tier": "low", "confidence_score": 39.0},
            "evidence": {"evidence_quality_tier": "low", "evidence_quality_summary": {"fallback_factor_count": 2, "missing_factor_count": 2, "inferred_factor_count": 1}, "coverage_summary": {"failed_count": 1, "fallback_count": 1}},
            "feature_snapshot": {"raw_feature_vector": {"fuel": 0.2}, "transformed_feature_vector": {"fuel_idx": 20.0}, "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 4.0}}},
            "join_metadata": {"join_method": "nearest_global_coordinates", "join_confidence_tier": "low", "join_confidence_score": 0.42},
        },
        {
            "event": {"event_id": "evt-b", "event_name": "Event B", "event_date": "2022-09-05"},
            "feature": {"record_id": "f4", "source_record_id": "4", "address_text": "200 Main St", "latitude": 38.5, "longitude": -121.5},
            "outcome": {"record_id": "o4", "damage_label": "destroyed", "damage_severity_class": "destroyed", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 86.0, "site_hazard_score": 82.0, "home_ignition_vulnerability_score": 83.0, "insurance_readiness_score": 35.0},
            "confidence": {"confidence_tier": "high", "confidence_score": 82.0},
            "evidence": {"evidence_quality_tier": "high", "evidence_quality_summary": {"fallback_factor_count": 0, "missing_factor_count": 0, "inferred_factor_count": 0}, "coverage_summary": {"failed_count": 0, "fallback_count": 0}},
            "feature_snapshot": {"raw_feature_vector": {"fuel": 0.82}, "transformed_feature_vector": {"fuel_idx": 82.0}, "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 16.4}}},
            "join_metadata": {"join_method": "exact_event_record_id", "join_confidence_tier": "high", "join_confidence_score": 0.96},
        },
        {
            "event": {"event_id": "evt-b", "event_name": "Event B", "event_date": "2022-09-05"},
            "feature": {"record_id": "f5", "source_record_id": "5", "address_text": "201 Main St", "latitude": 38.501, "longitude": -121.501},
            "outcome": {"record_id": "o5", "damage_label": "no_damage", "damage_severity_class": "none", "structure_loss_or_major_damage": 0},
            "scores": {"wildfire_risk_score": 34.0, "site_hazard_score": 31.0, "home_ignition_vulnerability_score": 35.0, "insurance_readiness_score": 72.0},
            "confidence": {"confidence_tier": "moderate", "confidence_score": 61.0},
            "evidence": {"evidence_quality_tier": "moderate", "evidence_quality_summary": {"fallback_factor_count": 1, "missing_factor_count": 0, "inferred_factor_count": 1}, "coverage_summary": {"failed_count": 0, "fallback_count": 0}},
            "feature_snapshot": {"raw_feature_vector": {"fuel": 0.34}, "transformed_feature_vector": {"fuel_idx": 34.0}, "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 6.8}}},
            "join_metadata": {"join_method": "exact_event_address", "join_confidence_tier": "moderate", "join_confidence_score": 0.9},
        },
    ]
    dataset_path = tmp_path / "evaluation_dataset.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report, _ = evaluate_public_outcome_dataset_file(dataset_path=dataset_path, thresholds=[40.0, 70.0], bins=4)
    assert report["dataset_format"] == "jsonl"
    assert "by_join_confidence_tier" in (report["slice_metrics"] or {})
    assert "high" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert "moderate" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert "low" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert report["discrimination_metrics"]["wildfire_risk_score_pr_auc"] is not None


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


def test_orchestration_writes_bundle_for_insufficient_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "insufficient.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"event": {"event_id": "evt-a"}, "outcome": {"damage_label": "destroyed", "structure_loss_or_major_damage": 1}, "scores": {"wildfire_risk_score": None}}),
                json.dumps({"event": {"event_id": "evt-a"}, "outcome": {"damage_label": "no_damage", "structure_loss_or_major_damage": 0}, "scores": {"wildfire_risk_score": None}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_public_outcome_validation(
        evaluation_dataset=dataset_path,
        output_root=tmp_path / "validation_runs",
        run_id="insufficient_case",
        overwrite=True,
    )
    metrics = json.loads(Path(result["validation_metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["status"] == "insufficient_data"
    assert metrics["row_count_labeled"] == 0
    assert "Insufficient usable labeled rows" in " ".join((metrics.get("guardrails") or {}).get("warnings") or [])
