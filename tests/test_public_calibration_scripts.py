from __future__ import annotations

import json
from pathlib import Path

from scripts.build_public_outcome_calibration_dataset import build_dataset
from scripts.fit_public_outcome_calibration import fit_calibration


def test_build_and_fit_public_calibration(tmp_path):
    backtest_path = tmp_path / "backtest.json"
    backtest_payload = {
        "records": [
            {"event_id": "e1", "record_id": "r1", "outcome_rank": 4, "outcome_label": "destroyed", "scores": {"wildfire_risk_score": 92}},
            {"event_id": "e1", "record_id": "r2", "outcome_rank": 3, "outcome_label": "major_damage", "scores": {"wildfire_risk_score": 81}},
            {"event_id": "e1", "record_id": "r3", "outcome_rank": 2, "outcome_label": "minor_damage", "scores": {"wildfire_risk_score": 63}},
            {"event_id": "e1", "record_id": "r4", "outcome_rank": 1, "outcome_label": "no_known_damage", "scores": {"wildfire_risk_score": 45}},
            {"event_id": "e1", "record_id": "r5", "outcome_rank": 1, "outcome_label": "no_known_damage", "scores": {"wildfire_risk_score": 37}},
            {"event_id": "e2", "record_id": "r6", "outcome_rank": 4, "outcome_label": "destroyed", "scores": {"wildfire_risk_score": 88}},
            {"event_id": "e2", "record_id": "r7", "outcome_rank": 3, "outcome_label": "major_damage", "scores": {"wildfire_risk_score": 76}},
            {"event_id": "e2", "record_id": "r8", "outcome_rank": 2, "outcome_label": "minor_damage", "scores": {"wildfire_risk_score": 59}},
            {"event_id": "e2", "record_id": "r9", "outcome_rank": 1, "outcome_label": "no_known_damage", "scores": {"wildfire_risk_score": 41}},
            {"event_id": "e2", "record_id": "r10", "outcome_rank": 1, "outcome_label": "no_known_damage", "scores": {"wildfire_risk_score": 32}},
        ]
    }
    backtest_path.write_text(json.dumps(backtest_payload), encoding="utf-8")

    dataset_path = tmp_path / "calibration_dataset.json"
    dataset = build_dataset(
        input_path=backtest_path,
        output_path=dataset_path,
        adverse_min_rank=3,
    )
    assert dataset["row_count"] == 10
    assert dataset_path.exists()

    artifact_path = tmp_path / "artifact.json"
    artifact = fit_calibration(
        dataset_path=dataset_path,
        output_path=artifact_path,
    )
    assert artifact["method"] == "logistic"
    assert artifact["dataset"]["row_count"] == 10
    assert "metrics" in artifact
    assert artifact_path.exists()

