from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.event_backtesting import (
    load_event_backtest_dataset,
    run_event_backtest,
    spearman_rank_correlation,
)


def _dataset_payload() -> dict:
    return {
        "event_backtest_version": "1.0.0",
        "dataset_id": "test_event_pack",
        "dataset_name": "Test Event Pack",
        "source_name": "test-fixture",
        "records": [
            {
                "event_id": "event_a",
                "event_name": "Event A",
                "event_date": "2020-01-01",
                "record_id": "a_destroyed",
                "latitude": 39.75,
                "longitude": -105.0,
                "address_text": "1 Test Way",
                "outcome_label": "destroyed",
                "outcome_rank": 4,
                "label_confidence": 0.9,
                "input_payload": {
                    "attributes": {
                        "roof_type": "wood",
                        "vent_type": "standard",
                        "defensible_space_ft": 5
                    },
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
                    "audience": "insurer"
                },
                "context_overrides": {
                    "burn_probability_index": 90.0,
                    "hazard_severity_index": 92.0,
                    "slope_index": 72.0,
                    "fuel_index": 86.0,
                    "canopy_index": 84.0,
                    "historic_fire_index": 80.0,
                    "wildland_distance_index": 88.0
                }
            },
            {
                "event_id": "event_a",
                "event_name": "Event A",
                "event_date": "2020-01-01",
                "record_id": "a_no_damage",
                "latitude": 39.74,
                "longitude": -104.99,
                "address_text": "2 Test Way",
                "outcome_label": "no_known_damage",
                "outcome_rank": 1,
                "label_confidence": 0.8,
                "input_payload": {
                    "attributes": {
                        "roof_type": "class a",
                        "vent_type": "ember-resistant",
                        "defensible_space_ft": 40
                    },
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
                    "audience": "insurer"
                },
                "context_overrides": {
                    "burn_probability_index": 25.0,
                    "hazard_severity_index": 28.0,
                    "slope_index": 24.0,
                    "fuel_index": 30.0,
                    "canopy_index": 22.0,
                    "historic_fire_index": 20.0,
                    "wildland_distance_index": 30.0
                }
            },
            {
                "event_id": "event_a",
                "event_name": "Event A",
                "event_date": "2020-01-01",
                "record_id": "a_unknown",
                "latitude": 39.73,
                "longitude": -104.98,
                "address_text": "3 Test Way",
                "outcome_label": "unknown",
                "input_payload": {
                    "attributes": {},
                    "confirmed_fields": [],
                    "audience": "insurer"
                },
                "context_overrides": {
                    "environmental_layer_status": {
                        "burn_probability": "missing",
                        "hazard": "missing",
                        "slope": "ok",
                        "fuel": "missing",
                        "canopy": "missing",
                        "fire_history": "missing"
                    },
                    "property_level_context": {
                        "footprint_used": False,
                        "footprint_status": "not_found",
                        "fallback_mode": "point_based"
                    },
                    "structure_ring_metrics": {}
                }
            }
        ]
    }


def _write_dataset(path: Path) -> Path:
    path.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    return path


def test_event_dataset_normalization_from_json(tmp_path: Path):
    path = _write_dataset(tmp_path / "event_dataset.json")
    ds = load_event_backtest_dataset(path)
    assert ds.dataset_id == "test_event_pack"
    assert len(ds.records) == 3
    assert ds.records[0].outcome_label == "destroyed"
    assert ds.records[2].outcome_label == "unknown"
    assert ds.records[2].outcome_rank == 0


def test_event_dataset_normalization_from_csv(tmp_path: Path):
    csv_path = tmp_path / "event_dataset.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_id,event_name,event_date,source_name,record_id,latitude,longitude,outcome_label,outcome_rank,input_payload,context_overrides",
                'event_csv,Event CSV,2021-01-01,csv-fixture,row1,39.7,-105.0,major_damage,3,"{""attributes"": {""roof_type"": ""wood""}}","{""burn_probability_index"": 80.0}"',
            ]
        ),
        encoding="utf-8",
    )
    ds = load_event_backtest_dataset(csv_path)
    assert ds.dataset_id == "event_dataset"
    assert len(ds.records) == 1
    assert ds.records[0].outcome_label == "major_damage"
    assert ds.records[0].outcome_rank == 3


def test_rank_metric_helper():
    corr = spearman_rank_correlation([(10.0, 1.0), (20.0, 2.0), (30.0, 3.0), (40.0, 4.0)])
    assert isinstance(corr, float)
    assert corr > 0.95


def test_run_event_backtest_generates_artifacts_and_governance(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path / "event_dataset.json")
    out_dir = tmp_path / "artifacts"
    artifact = run_event_backtest(dataset_paths=[dataset_path], output_dir=out_dir)
    assert Path(artifact["artifact_path"]).exists()
    assert Path(artifact["markdown_summary_path"]).exists()
    assert artifact["summary"]["record_count"] == 3
    assert "model_governance" in artifact
    governance = artifact["model_governance"]
    for key in [
        "product_version",
        "api_version",
        "scoring_model_version",
        "ruleset_version",
        "factor_schema_version",
        "benchmark_pack_version",
        "calibration_version",
        "data_bundle_version",
    ]:
        assert key in governance
    assert "false_low_count" in artifact["analysis"]
    assert "false_high_count" in artifact["analysis"]
    assert "recommendations" in artifact["analysis"]


def test_false_low_false_high_extraction_present(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path / "event_dataset.json")
    artifact = run_event_backtest(dataset_paths=[dataset_path], output_dir=tmp_path / "artifacts")
    assert isinstance(artifact["false_low_examples"], list)
    assert isinstance(artifact["false_high_examples"], list)
    assert artifact["analysis"]["false_low_count"] == len(artifact["false_low_examples"])
    assert artifact["analysis"]["false_high_count"] == len(artifact["false_high_examples"])


def test_run_event_backtest_script_exit_behavior(tmp_path: Path):
    script = Path("scripts") / "run_event_backtest.py"
    dataset_path = _write_dataset(tmp_path / "event_dataset.json")
    ok = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "out_ok"),
            "--min-records",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    bad = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "out_bad"),
            "--min-records",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert bad.returncode == 2, bad.stdout + bad.stderr
