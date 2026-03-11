from __future__ import annotations

import json
from pathlib import Path

from scripts.build_calibration_dataset import build_calibration_dataset


def test_build_calibration_dataset_joins_outcomes_and_features(tmp_path: Path):
    outcomes_path = tmp_path / "outcomes.json"
    outcomes_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "out_001",
                        "source_record_id": "r-1",
                        "event_id": "evt-1",
                        "event_name": "Test Event",
                        "event_date": "2021-08-01",
                        "latitude": 39.101,
                        "longitude": -120.101,
                        "damage_label": "destroyed",
                        "structure_loss_or_major_damage": 1,
                        "source_quality_flags": ["high_quality_source"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    feature_artifact = tmp_path / "artifact.json"
    feature_artifact.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-1",
                        "record_id": "abc",
                        "source_record_id": "r-1",
                        "latitude": 39.101,
                        "longitude": -120.101,
                        "outcome_label": "destroyed",
                        "scores": {
                            "wildfire_risk_score": 88.0,
                            "site_hazard_score": 83.0,
                            "home_ignition_vulnerability_score": 86.0,
                            "insurance_readiness_score": 34.0,
                        },
                        "raw_feature_vector": {"burn_probability": 0.74},
                        "transformed_feature_vector": {"burn_probability_index": 74.0},
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 16.5}
                        },
                        "evidence_quality_summary": {
                            "fallback_factor_count": 1,
                            "missing_factor_count": 0,
                            "inferred_factor_count": 2,
                        },
                        "coverage_summary": {"failed_count": 0, "fallback_count": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "dataset.json"
    payload = build_calibration_dataset(
        outcome_path=outcomes_path,
        feature_artifacts=[feature_artifact],
        output_path=output_path,
    )
    assert payload["row_count"] == 1
    assert payload["matched_outcome_count"] == 1
    row = payload["rows"][0]
    assert row["outcome_label"] == "destroyed"
    assert row["structure_loss_or_major_damage"] == 1
    assert row["scores"]["wildfire_risk_score"] == 88.0
    assert row["raw_feature_vector"]["burn_probability"] == 0.74
    assert row["factor_contribution_breakdown"]["fuel_proximity_risk"]["contribution"] == 16.5
    assert output_path.exists()
