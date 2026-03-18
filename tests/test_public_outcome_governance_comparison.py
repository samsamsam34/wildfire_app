from __future__ import annotations

import json
from pathlib import Path

from scripts.fit_public_outcome_calibration import run_public_outcome_calibration
from scripts.run_public_outcome_validation import run_public_outcome_validation


def _write_validation_dataset(path: Path) -> None:
    rows = []
    samples = [
        ("a1", 91.0, 1, "high"),
        ("a2", 83.0, 1, "high"),
        ("a3", 74.0, 1, "moderate"),
        ("a4", 62.0, 0, "moderate"),
        ("a5", 48.0, 0, "low"),
        ("a6", 35.0, 0, "low"),
    ]
    for idx, (record_id, risk, outcome, tier) in enumerate(samples, start=1):
        rows.append(
            {
                "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
                "feature": {
                    "record_id": record_id,
                    "source_record_id": str(idx),
                    "address_text": f"{idx} Main St",
                    "latitude": 39.1 + (idx * 0.0001),
                    "longitude": -120.1 - (idx * 0.0001),
                },
                "outcome": {
                    "record_id": f"o{idx}",
                    "damage_label": "destroyed" if outcome == 1 else "no_damage",
                    "damage_severity_class": "destroyed" if outcome == 1 else "none",
                    "structure_loss_or_major_damage": outcome,
                },
                "scores": {
                    "wildfire_risk_score": risk,
                    "site_hazard_score": max(0.0, risk - 6.0),
                    "home_ignition_vulnerability_score": max(0.0, risk - 4.0),
                    "insurance_readiness_score": max(0.0, 100.0 - risk),
                },
                "confidence": {"confidence_tier": tier, "confidence_score": 80.0 if tier == "high" else 60.0},
                "evidence": {
                    "evidence_quality_tier": "high" if tier == "high" else ("moderate" if tier == "moderate" else "low"),
                    "evidence_quality_summary": {
                        "fallback_factor_count": 0 if tier == "high" else (1 if tier == "moderate" else 2),
                        "missing_factor_count": 0 if tier == "high" else (1 if tier == "moderate" else 2),
                        "inferred_factor_count": 0 if tier == "high" else 1,
                    },
                    "coverage_summary": {"failed_count": 0 if tier != "low" else 1, "fallback_count": 0 if tier == "high" else 1},
                },
                "feature_snapshot": {"raw_feature_vector": {}, "transformed_feature_vector": {}, "factor_contribution_breakdown": {}},
                "join_metadata": {"join_method": "exact_source_record_id", "join_confidence_tier": "high", "join_confidence_score": 0.95},
                "provenance": {
                    "model_governance": {
                        "scoring_model_version": "1.10.0",
                        "rules_logic_version": "1.1.0",
                        "factor_schema_version": "1.3.0",
                        "region_data_version": "test_region",
                        "data_bundle_version": "test_bundle",
                    }
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_calibration_dataset(path: Path) -> None:
    rows = []
    for idx, score in enumerate((95, 90, 82, 76, 68, 60, 52, 44, 36, 28), start=1):
        label = 1 if score >= 70 else 0
        rows.append(
            {
                "event_id": "evt-cal",
                "record_id": f"r{idx}",
                "outcome_label": "destroyed" if label else "no_damage",
                "structure_loss_or_major_damage": label,
                "scores": {"wildfire_risk_score": float(score)},
                "confidence_tier": "high" if score >= 80 else ("moderate" if score >= 50 else "low"),
                "evidence_quality_tier": "high" if score >= 80 else ("moderate" if score >= 50 else "low"),
                "join_confidence_tier": "high",
                "fallback_default_flags": {
                    "fallback_factor_count": 0 if score >= 80 else 1,
                    "missing_factor_count": 0 if score >= 80 else 1,
                    "inferred_factor_count": 0 if score >= 80 else 1,
                    "coverage_failed_count": 0,
                    "coverage_fallback_count": 0,
                    "fallback_weight_fraction": 0.05 if score >= 80 else 0.35,
                },
                "model_governance": {
                    "scoring_model_version": "1.10.0",
                    "rules_logic_version": "1.1.0",
                    "factor_schema_version": "1.3.0",
                    "region_data_version": "test_region",
                    "data_bundle_version": "test_bundle",
                },
            }
        )
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def test_validation_workflow_writes_comparison_to_previous(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation_dataset.jsonl"
    _write_validation_dataset(dataset)
    output_root = tmp_path / "validation_runs"
    run_public_outcome_validation(
        evaluation_dataset=dataset,
        output_root=output_root,
        run_id="20260318T000000Z",
        overwrite=True,
    )
    second = run_public_outcome_validation(
        evaluation_dataset=dataset,
        output_root=output_root,
        run_id="20260319T000000Z",
        overwrite=True,
    )
    comparison = json.loads(
        (Path(second["run_dir"]) / "comparison_to_previous.json").read_text(encoding="utf-8")
    )
    assert comparison["available"] is True
    assert comparison["baseline_run_id"] == "20260318T000000Z"
    assert "delta" in comparison
    assert "false_low_count" in (comparison["delta"] or {})


def test_calibration_workflow_writes_comparison_to_previous(tmp_path: Path) -> None:
    dataset = tmp_path / "calibration_dataset.json"
    _write_calibration_dataset(dataset)
    output_root = tmp_path / "calibration_runs"
    run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=output_root,
        run_id="20260318T000000Z",
        min_rows=8,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    second = run_public_outcome_calibration(
        dataset_path=dataset,
        output_root=output_root,
        run_id="20260319T000000Z",
        min_rows=8,
        min_positive=2,
        min_negative=2,
        overwrite=True,
    )
    comparison = json.loads(
        (Path(second["run_dir"]) / "comparison_to_previous.json").read_text(encoding="utf-8")
    )
    assert comparison["available"] is True
    assert comparison["baseline_run_id"] == "20260318T000000Z"
    assert "delta" in comparison
    assert "brier_improvement" in (comparison["delta"] or {})
