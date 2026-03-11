from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_model_against_public_outcomes import evaluate_dataset


def test_evaluate_dataset_generates_metrics(tmp_path: Path):
    dataset_path = tmp_path / "cal_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "record_id": "r1",
                        "event_id": "e1",
                        "outcome_label": "destroyed",
                        "structure_loss_or_major_damage": 1,
                        "scores": {
                            "wildfire_risk_score": 91.0,
                            "site_hazard_score": 84.0,
                            "home_ignition_vulnerability_score": 89.0,
                            "insurance_readiness_score": 21.0,
                            "calibrated_damage_likelihood": 0.92,
                        },
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 20.0}
                        },
                        "fallback_default_flags": {"fallback_factor_count": 0, "missing_factor_count": 0},
                    },
                    {
                        "record_id": "r2",
                        "event_id": "e1",
                        "outcome_label": "major_damage",
                        "structure_loss_or_major_damage": 1,
                        "scores": {
                            "wildfire_risk_score": 78.0,
                            "site_hazard_score": 72.0,
                            "home_ignition_vulnerability_score": 74.0,
                            "insurance_readiness_score": 33.0,
                            "calibrated_damage_likelihood": 0.8,
                        },
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 16.0}
                        },
                        "fallback_default_flags": {"fallback_factor_count": 1, "missing_factor_count": 0},
                    },
                    {
                        "record_id": "r3",
                        "event_id": "e1",
                        "outcome_label": "minor_damage",
                        "structure_loss_or_major_damage": 0,
                        "scores": {
                            "wildfire_risk_score": 59.0,
                            "site_hazard_score": 55.0,
                            "home_ignition_vulnerability_score": 58.0,
                            "insurance_readiness_score": 44.0,
                            "calibrated_damage_likelihood": 0.45,
                        },
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 10.0}
                        },
                        "fallback_default_flags": {"fallback_factor_count": 1, "missing_factor_count": 1},
                    },
                    {
                        "record_id": "r4",
                        "event_id": "e1",
                        "outcome_label": "no_damage",
                        "structure_loss_or_major_damage": 0,
                        "scores": {
                            "wildfire_risk_score": 36.0,
                            "site_hazard_score": 30.0,
                            "home_ignition_vulnerability_score": 34.0,
                            "insurance_readiness_score": 70.0,
                            "calibrated_damage_likelihood": 0.12,
                        },
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 4.0}
                        },
                        "fallback_default_flags": {"fallback_factor_count": 0, "missing_factor_count": 0},
                    },
                    {
                        "record_id": "r5",
                        "event_id": "e1",
                        "outcome_label": "no_damage",
                        "structure_loss_or_major_damage": 0,
                        "scores": {
                            "wildfire_risk_score": 29.0,
                            "site_hazard_score": 26.0,
                            "home_ignition_vulnerability_score": 28.0,
                            "insurance_readiness_score": 76.0,
                            "calibrated_damage_likelihood": 0.09,
                        },
                        "factor_contribution_breakdown": {
                            "fuel_proximity_risk": {"contribution": 3.0}
                        },
                        "fallback_default_flags": {"fallback_factor_count": 0, "missing_factor_count": 0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out_json = tmp_path / "eval.json"
    out_csv = tmp_path / "eval.csv"
    report = evaluate_dataset(
        dataset_path=dataset_path,
        output_json=out_json,
        output_csv=out_csv,
        thresholds=[40.0, 70.0],
    )
    assert report["row_count_labeled"] == 5
    assert report["discrimination_metrics"]["wildfire_risk_score_auc"] is not None
    assert "70" in report["threshold_metrics_wildfire_risk_score"]
    assert report["default_threshold_70"]["confusion_matrix"]["tp"] >= 1
    assert out_json.exists()
    assert out_csv.exists()
