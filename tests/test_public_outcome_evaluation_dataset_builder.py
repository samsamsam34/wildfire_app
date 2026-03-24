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


def _write_outcomes_second_source(path: Path) -> None:
    payload = {
        "records": [
            {
                "record_id": "o3",
                "source_record_id": "SRC-003",
                "source_name": "public_source_b",
                "event_id": "evt-b",
                "event_name": "Event B",
                "event_date": "2022-09-04",
                "event_year": 2022,
                "latitude": 38.2001,
                "longitude": -121.2001,
                "address_text": "200 Oak St, Town, CA 90002",
                "damage_label": "major_damage",
                "damage_severity_class": "major",
                "structure_loss_or_major_damage": 1,
                "source_native_label": "Major Damage",
            }
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
                "event_id": "evt-b",
                "event_name": "Event B",
                "event_date": "2022-09-04",
                "record_id": "f-b-1",
                "source_record_id": "SRC-003",
                "latitude": 38.20009,
                "longitude": -121.20009,
                "address_text": "200 Oak Street, Town, CA 90002",
                "scores": {"wildfire_risk_score": 74.0},
                "confidence": {"confidence_tier": "moderate", "confidence_score": 61.0},
                "evidence_quality_summary": {"evidence_tier": "moderate"},
                "coverage_summary": {"failed_count": 0, "fallback_count": 0},
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


def _write_unscored_feature_artifact(path: Path) -> None:
    payload = {
        "records": [
            {
                "event_id": "evt-a",
                "event_name": "Event A",
                "event_date": "2021-08-01",
                "record_id": "f-unscored-1",
                "source_record_id": "SRC-001",
                "latitude": 39.10011,
                "longitude": -120.10011,
                "address_text": "100 Main Street, Town, CA 90001",
                "input_payload": {"attributes": {"roof_type": "wood"}},
                "context_overrides": {"fuel_index": 80.0},
            }
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_builder_joins_rows_and_reports_join_quality(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    outcomes_b = tmp_path / "normalized_outcomes_b.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_outcomes_second_source(outcomes_b)
    _write_feature_artifact(features)

    result = build_public_outcome_evaluation_dataset(
        outcomes_paths=[outcomes, outcomes_b],
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="fixed_eval_ds",
        max_distance_m=150.0,
        overwrite=True,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["summary"]["joined_records"] == 3
    assert manifest["summary"]["excluded_rows"] == 1
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["total_outcomes_loaded"] == 3
    assert join_quality["total_feature_rows_loaded"] == 4
    assert join_quality["total_joined_records"] == 3
    assert join_quality["join_method_counts"]
    assert join_quality["join_confidence_tier_counts"]
    assert join_quality["by_event_join_counts"]["evt-a"] == 2
    assert join_quality["by_event_join_counts"]["evt-b"] == 1
    assert join_quality["by_label_join_counts"]["destroyed"] == 1
    assert join_quality["by_label_join_counts"]["no_damage"] == 1
    assert join_quality["by_label_join_counts"]["major_damage"] == 1
    assert join_quality["row_confidence_tier_counts"]


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
    assert all((row.get("evaluation") or {}).get("row_confidence_tier") in {"high-confidence", "medium-confidence", "low-confidence"} for row in rows)
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


def test_builder_backfills_missing_scores_via_event_backtest(monkeypatch, tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest_unscored.json"
    _write_outcomes(outcomes)
    _write_unscored_feature_artifact(features)

    def _fake_run_event_backtest(*, dataset_paths, output_dir, ruleset_id=None, reuse_existing_assessments=False):
        return {
            "artifact_path": str(Path(output_dir) / "fake_event_backtest.json"),
            "records": [
                {
                    "event_id": "evt-a",
                    "record_id": "f-unscored-1",
                    "source_record_id": "SRC-001",
                    "scores": {
                        "wildfire_risk_score": 81.0,
                        "site_hazard_score": 73.0,
                        "home_ignition_vulnerability_score": 75.0,
                        "insurance_readiness_score": 36.0,
                    },
                    "confidence": {"confidence_tier": "moderate", "confidence_score": 63.0},
                    "evidence_quality_summary": {"evidence_tier": "moderate"},
                    "coverage_summary": {"failed_count": 0, "fallback_count": 0},
                    "model_governance": {"scoring_model_version": "1.10.0"},
                }
            ],
        }

    import backend.event_backtesting as event_backtesting

    monkeypatch.setattr(event_backtesting, "run_event_backtest", _fake_run_event_backtest)

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="backfill_eval_ds",
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert rows
    assert rows[0]["scores"]["wildfire_risk_score"] == 81.0
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    score_backfill = join_quality.get("score_backfill") or {}
    assert score_backfill.get("backfilled_record_count") == 1
    assert score_backfill.get("remaining_missing_score_record_count") == 0


def test_builder_respects_global_fallback_toggle(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_far.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-z",
                        "event_name": "Event Z",
                        "event_date": "2022-01-01",
                        "record_id": "far-1",
                        "source_record_id": "UNKNOWN",
                        "latitude": 39.11,
                        "longitude": -120.11,
                        "scores": {"wildfire_risk_score": 57.0},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    disabled = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="fallback_disabled",
        max_distance_m=50.0,
        global_max_distance_m=100000.0,
        enable_global_nearest_fallback=False,
        overwrite=True,
    )
    disabled_manifest = json.loads(Path(disabled["manifest_path"]).read_text(encoding="utf-8"))
    assert disabled_manifest["summary"]["joined_records"] == 0

    enabled = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="fallback_enabled",
        max_distance_m=50.0,
        global_max_distance_m=100000.0,
        enable_global_nearest_fallback=True,
        overwrite=True,
    )
    enabled_manifest = json.loads(Path(enabled["manifest_path"]).read_text(encoding="utf-8"))
    assert enabled_manifest["summary"]["joined_records"] == 1


def test_builder_assigns_extended_geospatial_match_tier(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_extended_radius.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "extended-1",
                        "source_record_id": "UNKNOWN",
                        "latitude": 39.10095,
                        "longitude": -120.10095,
                        "address_text": "Unmatched Address",
                        "scores": {"wildfire_risk_score": 63.0},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="extended_radius",
        near_match_distance_m=30.0,
        max_distance_m=200.0,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 1
    join_meta = rows[0]["join_metadata"]
    assert join_meta["join_method"] in {"extended_event_coordinates", "nearest_event_name_coordinates_tolerant_year"}
    assert join_meta.get("match_tier") in {"extended", "near"}

    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["match_tier_counts"]
    assert join_quality["match_rate_percent"] == 100.0


def test_builder_uses_global_address_overlap_fallback_when_coordinates_missing(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_address_overlap.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-z",
                        "event_name": "Event Z",
                        "event_date": "2021-08-01",
                        "record_id": "addr-overlap-1",
                        "source_record_id": "UNKNOWN",
                        "address_text": "100 Main Street Town CA 90001",
                        "scores": {"wildfire_risk_score": 49.0},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="address_overlap",
        enable_global_nearest_fallback=False,
        address_token_overlap_min=0.6,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 1
    join_meta = rows[0]["join_metadata"]
    assert join_meta["join_method"] == "approx_global_address_token_overlap"
    assert join_meta["match_tier"] == "fallback"
    assert join_meta["join_confidence_tier"] in {"low", "moderate"}
