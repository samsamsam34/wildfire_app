from __future__ import annotations

import json
from pathlib import Path

from scripts.build_public_outcome_evaluation_dataset import (
    _resolve_all_normalized_outcomes,
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
    assert Path(result["join_quality_metrics_path"]).exists()
    assert Path(result["join_quality_markdown_path"]).exists()
    assert Path(result["filter_summary_path"]).exists()
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["total_outcomes_loaded"] == 3
    assert join_quality["total_feature_rows_loaded"] == 4
    assert join_quality["total_joined_records"] == 3
    assert "min_join_distance_m" in join_quality
    assert "max_join_distance_m" in join_quality
    assert join_quality["outcomes_by_event_counts"]
    assert join_quality["feature_rows_by_event_counts"]
    assert join_quality["join_method_counts"]
    assert join_quality["join_confidence_tier_counts"]
    assert "join_quality_warnings" in join_quality
    assert join_quality["by_event_join_counts"]["evt-a"] == 2
    assert join_quality["by_event_join_counts"]["evt-b"] == 1
    assert join_quality["by_label_join_counts"]["destroyed"] == 1
    assert join_quality["by_label_join_counts"]["no_damage"] == 1
    assert join_quality["by_label_join_counts"]["major_damage"] == 1
    assert join_quality["row_confidence_tier_counts"]
    assert "excluded_reason_counts" in join_quality
    assert join_quality.get("no_silent_data_loss_guarantee") is True
    filter_summary = join_quality.get("filter_summary") or {}
    assert isinstance(filter_summary.get("filter_reason_counts"), dict)
    assert isinstance(filter_summary.get("soft_flag_counts"), dict)
    assert (filter_summary.get("accounting") or {}).get("matches_attempts_accounted") is True


def test_builder_join_quality_warnings_for_weak_spatial_matches(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_far_match.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "far-1",
                        "source_record_id": "UNKNOWN",
                        "latitude": 39.1016,
                        "longitude": -120.1001,
                        "address_text": "Far Address",
                        "scores": {"wildfire_risk_score": 58.0},
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
        run_id="weak_join_warning",
        near_match_distance_m=30.0,
        max_distance_m=300.0,
        medium_confidence_distance_m=60.0,
        overwrite=True,
    )
    join_quality = json.loads(Path(result["join_quality_metrics_path"]).read_text(encoding="utf-8"))
    warnings = join_quality.get("join_quality_warnings") if isinstance(join_quality.get("join_quality_warnings"), list) else []
    assert warnings
    assert any("No high-confidence matches were found" in warning for warning in warnings)
    assert any("Average join distance is high" in warning for warning in warnings)


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
    assert all(isinstance((row.get("evaluation") or {}).get("soft_filter_flags"), list) for row in rows)
    assert any("missing_features" in ((row.get("evaluation") or {}).get("soft_filter_flags") or []) for row in rows)
    # second row includes leakage token in raw feature vector key.
    leaked = [row for row in rows if row["feature"]["record_id"] == "f2"][0]
    assert "potential_outcome_leakage_token_in_raw_feature_vector" in leaked["leakage_flags"]
    assert (leaked.get("evaluation") or {}).get("row_confidence_tier") == "high-confidence"


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


def test_consecutive_runs_are_stable_with_different_run_ids(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)
    root = tmp_path / "out"

    run_a = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=root,
        run_id="deterministic_eval_ds_a",
        overwrite=True,
    )
    run_b = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=root,
        run_id="deterministic_eval_ds_b",
        overwrite=True,
    )

    dataset_a = Path(run_a["run_dir"]) / "evaluation_dataset.jsonl"
    dataset_b = Path(run_b["run_dir"]) / "evaluation_dataset.jsonl"
    assert dataset_a.read_text(encoding="utf-8") == dataset_b.read_text(encoding="utf-8")

    metrics_a = json.loads(Path(run_a["join_quality_metrics_path"]).read_text(encoding="utf-8"))
    metrics_b = json.loads(Path(run_b["join_quality_metrics_path"]).read_text(encoding="utf-8"))
    metrics_a.pop("generated_at", None)
    metrics_b.pop("generated_at", None)
    assert metrics_a == metrics_b
    assert metrics_a.get("candidate_match_attempt_count") == metrics_a.get("total_feature_rows_loaded")
    assert metrics_a.get("final_dataset_size") == metrics_a.get("total_joined_records")


def test_nearest_join_tie_break_is_deterministic(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes_tie.json"
    features = tmp_path / "feature_artifact_tie.json"
    outcomes.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "z-row",
                        "source_record_id": "SRC-Z",
                        "source_name": "public_source",
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "event_year": 2021,
                        "latitude": 39.1,
                        "longitude": -120.1001,
                        "damage_label": "destroyed",
                        "damage_severity_class": "destroyed",
                        "structure_loss_or_major_damage": 1,
                    },
                    {
                        "record_id": "a-row",
                        "source_record_id": "SRC-A",
                        "source_name": "public_source",
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "event_year": 2021,
                        "latitude": 39.1,
                        "longitude": -120.0999,
                        "damage_label": "no_damage",
                        "damage_severity_class": "none",
                        "structure_loss_or_major_damage": 0,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "f-tie",
                        "source_record_id": "UNMATCHED",
                        "latitude": 39.1,
                        "longitude": -120.1,
                        "address_text": "Unknown",
                        "scores": {"wildfire_risk_score": 51.0},
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
        run_id="tie_break_determinism",
        exact_match_distance_m=0.5,
        near_match_distance_m=30.0,
        max_distance_m=150.0,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 1
    assert ((rows[0].get("outcome") or {}).get("record_id")) == "a-row"


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
    assert join_meta["join_method"] in {
        "buffered_event_coordinates",
        "extended_event_coordinates",
        "nearest_event_name_coordinates_tolerant_year",
    }
    assert join_meta.get("match_tier") in {"extended", "near"}

    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["match_tier_counts"]
    assert join_quality["match_rate_percent"] == 100.0


def test_builder_applies_distance_based_join_confidence_tiers(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_distance_tiers.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "dist-high",
                        "source_record_id": "UNKNOWN-H",
                        "latitude": 39.10023,
                        "longitude": -120.1001,
                        "address_text": "Unknown A",
                        "scores": {"wildfire_risk_score": 70.0},
                    },
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "dist-moderate",
                        "source_record_id": "UNKNOWN-M",
                        "latitude": 39.10082,
                        "longitude": -120.1001,
                        "address_text": "Unknown B",
                        "scores": {"wildfire_risk_score": 55.0},
                    },
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "dist-low",
                        "source_record_id": "UNKNOWN-L",
                        "latitude": 39.1022,
                        "longitude": -120.1001,
                        "address_text": "Unknown C",
                        "scores": {"wildfire_risk_score": 42.0},
                    },
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
        run_id="distance_tiers",
        near_match_distance_m=30.0,
        max_distance_m=220.0,
        high_confidence_distance_m=20.0,
        medium_confidence_distance_m=100.0,
        allow_duplicate_outcome_matches=True,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 3
    by_record = {str((row.get("feature") or {}).get("record_id")): row for row in rows}
    assert ((by_record["dist-high"].get("join_metadata") or {}).get("join_confidence_tier")) == "high"
    assert ((by_record["dist-moderate"].get("join_metadata") or {}).get("join_confidence_tier")) == "moderate"
    assert ((by_record["dist-low"].get("join_metadata") or {}).get("join_confidence_tier")) == "low"

    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert "join_confidence_tier_distance_stats" in join_quality
    assert "join_confidence_tier_examples" in join_quality
    assert "match_distance_histogram_m" in join_quality
    assert "distance_outlier_examples" in join_quality
    assert (join_quality.get("join_confidence_tier_counts") or {}).get("high", 0) >= 1
    assert "join_confidence_non_high_reason_counts" in join_quality
    assert "high_confidence_threshold_diagnostics" in join_quality


def test_builder_prevents_duplicate_outcome_matches_by_default(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "feature_artifact_duplicates.json"
    _write_outcomes(outcomes)
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "dup-1",
                        "source_record_id": "SRC-001",
                        "latitude": 39.1001,
                        "longitude": -120.1001,
                        "scores": {"wildfire_risk_score": 75.0},
                    },
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "dup-2",
                        "source_record_id": "SRC-001",
                        "latitude": 39.1001,
                        "longitude": -120.1001,
                        "scores": {"wildfire_risk_score": 74.0},
                    },
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
        run_id="dedupe_default",
        overwrite=True,
    )
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert join_quality["total_joined_records"] == 1
    assert join_quality["duplicate_outcome_match_prevented_count"] >= 0
    assert join_quality["allow_duplicate_outcome_matches"] is False


def test_builder_converts_web_mercator_coordinates_to_wgs84(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes_mercator.json"
    features = tmp_path / "feature_artifact_wgs84.json"
    outcomes.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "m1",
                        "source_record_id": "M-001",
                        "source_name": "public_source",
                        "event_id": "evt-m",
                        "event_name": "Event Mercator",
                        "event_date": "2021-08-01",
                        "x": -13369492.572,
                        "y": 4747094.953,
                        "address_text": "Mercator Point",
                        "damage_label": "destroyed",
                        "damage_severity_class": "destroyed",
                        "structure_loss_or_major_damage": 1,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-m",
                        "event_name": "Event Mercator",
                        "event_date": "2021-08-01",
                        "record_id": "f-m-1",
                        "source_record_id": "M-001",
                        "latitude": 39.1001,
                        "longitude": -120.1001,
                        "scores": {"wildfire_risk_score": 88.0},
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
        run_id="mercator_conversion",
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == 1
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    coord_summary = join_quality.get("coordinate_normalization_summary") or {}
    outcome_modes = coord_summary.get("matched_outcomes_by_mode") or {}
    assert "web_mercator_xy" in outcome_modes


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


def test_high_confidence_matches_are_recognized_with_realistic_distance(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    _write_outcomes(outcomes)
    features = tmp_path / "feature_artifact_realistic_high_conf.json"
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "f-realistic-high",
                        "source_record_id": "UNKNOWN-R",
                        "latitude": 39.10033,
                        "longitude": -120.1001,
                        "address_text": "Unknown Realistic High",
                        "scores": {"wildfire_risk_score": 79.0},
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
        run_id="realistic_high_confidence",
        near_match_distance_m=40.0,
        max_distance_m=180.0,
        overwrite=True,
    )
    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert rows
    join_meta = rows[0].get("join_metadata") or {}
    assert join_meta.get("join_confidence_tier") == "high"
    debug = join_meta.get("join_confidence_debug") or {}
    assert debug.get("resolved_tier") == "high"
    assert isinstance(debug.get("distance_m"), (int, float))

    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    assert (join_quality.get("join_confidence_tier_counts") or {}).get("high", 0) >= 1
    threshold_diag = join_quality.get("high_confidence_threshold_diagnostics") or {}
    assert "just_above_high_distance_threshold_count" in threshold_diag
    assert "just_below_high_score_threshold_count" in threshold_diag
    debug_path = Path(result["join_confidence_debug_path"])
    assert debug_path.exists()


def test_resolve_all_normalized_outcomes_returns_all_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_c = tmp_path / "run_c"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    run_c.mkdir(parents=True, exist_ok=True)
    (run_a / "normalized_outcomes.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    (run_b / "normalized_outcomes.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    # run_c intentionally missing normalized_outcomes.json

    resolved = _resolve_all_normalized_outcomes(tmp_path)
    assert [path.parent.name for path in resolved] == ["run_a", "run_b"]


def test_min_retention_fallback_relaxes_matching_when_dataset_is_too_small(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    outcomes.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "o-ret-1",
                        "source_record_id": "RET-001",
                        "source_name": "public_source",
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "event_year": 2021,
                        "latitude": 39.1001,
                        "longitude": -120.1001,
                        "damage_label": "major_damage",
                        "damage_severity_class": "major",
                        "structure_loss_or_major_damage": 1,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    features = tmp_path / "feature_artifact_retention.json"
    features.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "event_id": "evt-a",
                        "event_name": "Event A",
                        "event_date": "2021-08-01",
                        "record_id": "ret-1",
                        "source_record_id": "UNKNOWN",
                        "latitude": 39.1020,
                        "longitude": -120.1001,
                        "scores": {"wildfire_risk_score": 62.0},
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
        run_id="retention_fallback",
        max_distance_m=50.0,
        near_match_distance_m=30.0,
        enable_global_nearest_fallback=False,
        min_retained_records=20,
        auto_relax_for_min_retention=True,
        overwrite=True,
    )
    join_quality = json.loads(Path(result["join_quality_report_path"]).read_text(encoding="utf-8"))
    retention = join_quality.get("retention_fallback") or {}
    assert retention.get("triggered") is True
    assert retention.get("used") is True
    assert int(join_quality.get("total_joined_records") or 0) >= 1

    rows = []
    with (Path(result["run_dir"]) / "evaluation_dataset.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert rows
    assert any(
        "retention_fallback_mode" in ((row.get("evaluation") or {}).get("soft_filter_flags") or [])
        for row in rows
    )


def test_audit_mode_writes_pipeline_audit_report_with_samples(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="audit_mode_run",
        audit_mode=True,
        audit_sample_rows=9,
        overwrite=True,
    )
    audit_path = result.get("pipeline_audit_report_path")
    assert isinstance(audit_path, str) and audit_path
    report_path = Path(audit_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "## Ingestion" in report
    assert "## Join Pipeline" in report
    assert "## Filtering" in report
    assert "## Joined Sample Rows" in report
    assert "## Filtered Sample Rows" in report
    assert "join_confidence_tier" in report
    assert "outcome_label" in report


def test_audit_mode_clamps_sample_rows_and_records_in_manifest(tmp_path: Path) -> None:
    outcomes = tmp_path / "normalized_outcomes.json"
    features = tmp_path / "event_backtest.json"
    _write_outcomes(outcomes)
    _write_feature_artifact(features)

    result = build_public_outcome_evaluation_dataset(
        outcomes_path=outcomes,
        feature_artifacts=[features],
        output_root=tmp_path / "out",
        run_id="audit_mode_clamped",
        audit_mode=True,
        audit_sample_rows=2,
        overwrite=True,
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert ((manifest.get("inputs") or {}).get("audit_mode")) is True
    assert ((manifest.get("inputs") or {}).get("audit_sample_rows")) == 5
    artifacts = manifest.get("artifacts") or {}
    assert isinstance(artifacts.get("pipeline_audit_report_markdown"), str)
