from __future__ import annotations

import json
from pathlib import Path

from scripts.build_public_outcome_evaluation_dataset import (
    build_public_outcome_evaluation_dataset,
)


def _write_outcomes(path: Path) -> None:
    payload = {
        "records": [
            {
                "record_id": "o1",
                "source_record_id": "SRC-001",
                "source_name": "public_source",
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "event_year": 2021,
                "latitude": 39.1001,
                "longitude": -120.1001,
                "address_text": "100 Main St, Town, CA 90001",
                "damage_label": "destroyed",
                "damage_severity_class": "destroyed",
                "structure_loss_or_major_damage": 1,
                "source_native_label": "Destroyed",
            },
            {
                "record_id": "o2",
                "source_record_id": "SRC-002",
                "source_name": "public_source",
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "event_year": 2021,
                "latitude": 39.1005,
                "longitude": -120.1005,
                "address_text": "101 Main St, Town, CA 90001",
                "damage_label": "no_damage",
                "damage_severity_class": "none",
                "structure_loss_or_major_damage": 0,
                "source_native_label": "No Damage",
            },
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_feature_artifact(path: Path) -> None:
    payload = {
        "records": [
            {
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "record_id": "f1",
                "source_record_id": "SRC-001",
                "latitude": 39.10011,
                "longitude": -120.10011,
                "address_text": "100 Main Street, Town, CA 90001",
                "scores": {
                    "wildfire_risk_score": 88.0,
                    "site_hazard_score": 82.0,
                    "home_ignition_vulnerability_score": 84.0,
                    "insurance_readiness_score": 32.0,
                },
                "confidence": {"confidence_tier": "high", "confidence_score": 84.0},
                "evidence_quality_summary": {"evidence_tier": "high"},
                "coverage_summary": {"failed_count": 0, "fallback_count": 0},
                "raw_feature_vector": {"burn_probability": 0.82},
                "transformed_feature_vector": {"burn_probability_index": 82.0},
                "factor_contribution_breakdown": {"fuel_proximity_risk": {"contribution": 12.3}},
            },
            {
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "record_id": "f2",
                "source_record_id": "NO_MATCH",
                "latitude": 39.10049,
                "longitude": -120.10049,
                "address_text": "101 Main St, Town, CA 90001",
                "scores": {"wildfire_risk_score": 26.0},
                "raw_feature_vector": {"damage_hint": 1.0},
                "transformed_feature_vector": {"fuel_index": 22.0},
                "factor_contribution_breakdown": {"defensible_space_risk": {"contribution": 2.1}},
            },
            {
                "event_id": "evt-z",
                "event_name": "Event Z",
                "event_date": "2022-09-01",
                "record_id": "f3",
                "source_record_id": "MISS-003",
                "latitude": 38.0,
                "longitude": -122.0,
                "address_text": "No Match Address",
                "scores": {"wildfire_risk_score": 44.0},
            },
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_builder_joins_rows_and_reports_join_quality(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="fixed_eval_ds",
        max_distance_m=150.0,
        overwrite=True,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["summary"]["joined_records"] == 2
    assert manifest["summary"]["excluded_rows"] == 1
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["total_outcomes_loaded"] == 2
    assert join_quality["total_feature_rows_loaded"] == 3
    assert join_quality["total_joined_records"] == 2
    assert join_quality["join_method_counts"]
    assert join_quality["join_confidence_tier_counts"]
    assert join_quality["by_event_join_counts"]["evt-a"] == 2
    assert join_quality["by_label_join_counts"]["destroyed"] == 1
    assert join_quality["by_label_join_counts"]["no_damage"] == 1


def test_join_confidence_and_leakage_flags_are_exposed(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="leakage_eval_ds",
        max_distance_m=150.0,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 2
    assert all("join_metadata" in row for row in rows)
    assert all("join_confidence_score" in row["join_metadata"] for row in rows)
    # second row includes leakage token in raw feature vector key.
    leaked = [row for row in rows if row["feature"]["record_id"] == "f2"][0]
    assert "potential_outcome_leakage_token_in_raw_feature_vector" in leaked["leakage_flags"]


def test_deterministic_with_fixed_run_id(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)
    root = tmp_path / "out"

    first = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=root,
        run_id="deterministic_eval_ds",
        overwrite=True,
    )
    second = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=root,
        run_id="deterministic_eval_ds",
        overwrite=True,
    )

    assert first["manifest_path"] == second["manifest_path"]
    m1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    m2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert m1 == m2
