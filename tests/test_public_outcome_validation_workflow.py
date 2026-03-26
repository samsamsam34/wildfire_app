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
    assert "wildfire_risk_score_auc_confidence_interval_95" in (report.get("discrimination_metrics") or {})
    assert "wildfire_risk_score_pr_auc_confidence_interval_95" in (report.get("discrimination_metrics") or {})
    assert "wildfire_discrimination_stability" in (report.get("discrimination_metrics") or {})
    assert report["brier_scores"]["wildfire_probability_proxy"] is not None
    assert "wildfire_probability_proxy_confidence_interval_95" in (report.get("brier_scores") or {})
    assert "40" in report["threshold_metrics_wildfire_risk_score"]
    assert "70" in report["threshold_metrics_wildfire_risk_score"]
    assert report["calibration_metrics"]["wildfire_risk_score"]["bins"]
    assert "by_confidence_tier" in (report["slice_metrics"] or {})
    assert "by_evidence_group" in (report["slice_metrics"] or {})
    assert "by_validation_confidence_tier" in (report["slice_metrics"] or {})
    assert "by_hazard_level" in (report["slice_metrics"] or {})
    assert "by_vegetation_density" in (report["slice_metrics"] or {})
    assert "by_region" in (report["slice_metrics"] or {})
    assert "segment_performance_summary" in report
    assert "strongest_segments" in (report.get("segment_performance_summary") or {})
    assert "segment_strength_map" in (report.get("segment_performance_summary") or {})
    strength_map = (report.get("segment_performance_summary") or {}).get("segment_strength_map") or {}
    assert "hazard_level" in strength_map
    assert "vegetation_density" in strength_map
    assert "confidence_tier" in strength_map
    assert "region" in strength_map
    assert "subset_metrics" in report
    assert report["subset_metrics"]["full_dataset"]["count"] == report["row_count_labeled"]
    assert "medium_confidence_subset" in report["subset_metrics"]
    assert "confidence_tier_performance" in report
    tier_perf = report["confidence_tier_performance"]
    assert "tiers" in tier_perf
    assert "all_data" in tier_perf["tiers"]
    assert "high_confidence" in tier_perf["tiers"]
    assert "medium_confidence" in tier_perf["tiers"]
    assert "deltas_vs_all_data" in tier_perf
    assert "minimum_viable_metrics" in report
    assert report["minimum_viable_metrics"]["available"] is True
    assert "data_sufficiency_flags" in report
    assert "data_sufficiency_indicator" in report
    assert report["data_sufficiency_indicator"]["total_dataset"]["tier"] == "insufficient"
    assert report["data_sufficiency_indicator"]["high_confidence_subset"]["tier"] == "insufficient"
    assert "modeling_viability" in report
    viability = report["modeling_viability"] if isinstance(report.get("modeling_viability"), dict) else {}
    assert viability.get("dataset_viable_for_predictive_modeling") is False
    assert viability.get("classification") == "dataset_not_viable_for_predictive_modeling"
    checks = viability.get("checks") if isinstance(viability.get("checks"), dict) else {}
    assert checks.get("independent_sample_count") == report["row_count_labeled"]
    assert "feature_variation_ratio" in checks
    fallback_diagnostics = report.get("fallback_diagnostics") if isinstance(report.get("fallback_diagnostics"), dict) else {}
    assert 0.0 <= float(fallback_diagnostics.get("fallback_heavy_fraction") or 0.0) < 1.0
    assert "rows_with_elevated_fallback_weight" in fallback_diagnostics
    assert "narrative_summary" in report
    assert "proxy_validation" in report
    assert "synthetic_validation" in report
    assert "validation_streams" in report
    assert "feature_signal_diagnostics" in report
    assert (report["feature_signal_diagnostics"] or {}).get("available") is True
    assert isinstance((report["feature_signal_diagnostics"] or {}).get("top_predictive_features"), list)
    assert "baseline_model_comparison" in report
    baseline = report["baseline_model_comparison"] if isinstance(report.get("baseline_model_comparison"), dict) else {}
    assert baseline.get("available") is True
    assert "baselines" in baseline
    baseline_models = baseline.get("baselines") if isinstance(baseline.get("baselines"), dict) else {}
    assert "random" in baseline_models
    assert "hazard_only" in baseline_models
    assert "vegetation_only" in baseline_models
    assert "comparison" in baseline
    assert "false_review_sets" in report
    assert "metric_stability" in report
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
    assert "feature_signal_report_path" in first
    assert first["feature_signal_report_path"].endswith("/feature_signal_report.json")
    manifest_text_1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    manifest_text_2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert manifest_text_1 == manifest_text_2
    manifest_obj = json.loads(manifest_text_1)
    assert (
        ((manifest_obj.get("artifacts") or {}).get("feature_signal_report_json"))
        == first["feature_signal_report_path"]
    )

    evaluation_text_1 = Path(first["validation_metrics_path"]).read_text(encoding="utf-8")
    evaluation_text_2 = Path(second["validation_metrics_path"]).read_text(encoding="utf-8")
    assert evaluation_text_1 == evaluation_text_2
    assert Path(first["summary_path"]).exists()
    assert (output_root / "fixed_validation_run" / "calibration_table.json").exists()
    assert (output_root / "fixed_validation_run" / "threshold_metrics.json").exists()
    assert (output_root / "fixed_validation_run" / "false_low_review_set.jsonl").exists()
    assert (output_root / "fixed_validation_run" / "false_high_review_set.jsonl").exists()
    assert (output_root / "fixed_validation_run" / "feature_diagnostics.json").exists()
    assert (output_root / "fixed_validation_run" / "feature_signal_report.json").exists()
    assert (output_root / "fixed_validation_run" / "baseline_model_comparison.json").exists()
    assert (output_root / "fixed_validation_run" / "segment_metrics.json").exists()
    assert (output_root / "fixed_validation_run" / "segment_report.md").exists()
    assert (output_root / "fixed_validation_run" / "comparison_to_previous.json").exists()
    assert (output_root / "fixed_validation_run" / "comparison_to_previous.md").exists()
    feature_signal = json.loads((output_root / "fixed_validation_run" / "feature_signal_report.json").read_text(encoding="utf-8"))
    assert "top_predictive_features" in feature_signal
    assert "weak_or_noisy_features" in feature_signal


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
    assert "by_validation_confidence_tier" in (report["slice_metrics"] or {})
    assert "by_hazard_level" in (report["slice_metrics"] or {})
    assert "by_vegetation_density" in (report["slice_metrics"] or {})
    assert "by_region" in (report["slice_metrics"] or {})
    assert "high" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert "moderate" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert "low" in (report["slice_metrics"]["by_join_confidence_tier"] or {})
    assert "segment_strength_map" in (report.get("segment_performance_summary") or {})
    assert "subset_metrics" in report
    assert "medium_confidence_subset" in report["subset_metrics"]
    assert report["subset_metrics"]["high_evidence_subset"]["count"] >= 1
    tier_perf = report.get("confidence_tier_performance") or {}
    assert "tiers" in tier_perf
    assert (tier_perf.get("tiers") or {}).get("high_confidence", {}).get("count") >= 1
    assert (tier_perf.get("tiers") or {}).get("medium_confidence", {}).get("count") >= 1
    assert "proxy_validation" in report
    assert "synthetic_validation" in report
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


def test_evaluate_public_outcome_dataset_allows_small_usable_set_when_configured(tmp_path: Path) -> None:
    dataset_path = tmp_path / "small_dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "event": {"event_id": "evt-small"},
                "feature": {"record_id": "r-small-1"},
                "outcome": {
                    "damage_label": "destroyed",
                    "damage_severity_class": "destroyed",
                    "structure_loss_or_major_damage": 1,
                },
                "scores": {"wildfire_risk_score": 78.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report, rows = evaluate_public_outcome_dataset_file(
        dataset_path=dataset_path,
        min_labeled_rows=1,
    )
    assert report["row_count_labeled"] == 1
    assert report["sample_counts"]["row_count_usable"] == 1
    assert report["minimum_viable_metrics"]["available"] is True
    assert report["minimum_viable_metrics"]["simple_accuracy_at_threshold"]["accuracy"] in {0.0, 1.0}
    assert report["data_sufficiency_flags"]["flags"]["small_sample_size"] is True
    assert report["data_sufficiency_indicator"]["total_dataset"]["tier"] == "insufficient"
    assert report["data_sufficiency_indicator"]["high_confidence_subset"]["tier"] == "insufficient"
    assert report["metric_stability"]["auc_stable"] is False
    assert "Insufficient data for stable AUC/PR-AUC" in " ".join(report["narrative_summary"]["bullets"])
    assert "Dataset too small for calibration trust" in " ".join(report["narrative_summary"]["bullets"])
    assert len(rows) == 1


def test_small_sample_metrics_are_marked_unstable_even_with_perfect_auc(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tiny_perfect_auc.jsonl"
    rows = [
        {
            "event": {"event_id": "evt-small"},
            "feature": {"record_id": "a1"},
            "outcome": {"damage_label": "destroyed", "damage_severity_class": "destroyed", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 95.0},
        },
        {
            "event": {"event_id": "evt-small"},
            "feature": {"record_id": "a2"},
            "outcome": {"damage_label": "major_damage", "damage_severity_class": "major", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 90.0},
        },
        {
            "event": {"event_id": "evt-small"},
            "feature": {"record_id": "b1"},
            "outcome": {"damage_label": "no_damage", "damage_severity_class": "none", "structure_loss_or_major_damage": 0},
            "scores": {"wildfire_risk_score": 25.0},
        },
        {
            "event": {"event_id": "evt-small"},
            "feature": {"record_id": "b2"},
            "outcome": {"damage_label": "no_damage", "damage_severity_class": "none", "structure_loss_or_major_damage": 0},
            "scores": {"wildfire_risk_score": 20.0},
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report, _ = evaluate_public_outcome_dataset_file(dataset_path=dataset_path, min_labeled_rows=1)
    assert report["row_count_labeled"] == 4
    assert report["discrimination_metrics"]["wildfire_risk_score_auc"] == 1.0
    assert report["metric_stability"]["auc_stable"] is False
    assert report["discrimination_metrics"]["wildfire_discrimination_stability"] == "unstable_small_sample"
    warnings = report["metric_stability"]["warnings"]
    assert any("Insufficient data for stable AUC/PR-AUC interpretation" in warning for warning in warnings)
    guardrail_warnings = (report.get("guardrails") or {}).get("warnings") or []
    assert any("High-confidence slice is too small for stable interpretation" in warning for warning in guardrail_warnings)
    ci = report["discrimination_metrics"]["wildfire_risk_score_auc_confidence_interval_95"]
    assert isinstance(ci, dict)
    assert "low" in ci and "high" in ci


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
    assert metrics["minimum_viable_metrics"]["available"] is False
    assert (metrics.get("modeling_viability") or {}).get("dataset_viable_for_predictive_modeling") is False
    assert metrics["data_sufficiency_flags"]["flags"]["small_sample_size"] is True
    assert metrics["data_sufficiency_indicator"]["total_dataset"]["tier"] == "insufficient"
    assert metrics["data_sufficiency_indicator"]["high_confidence_subset"]["tier"] == "insufficient"
    assert "no usable labeled rows" in str((metrics.get("narrative_summary") or {}).get("headline", "")).lower()
    assert metrics["proxy_validation"]["available"] is False
    assert metrics["synthetic_validation"]["available"] is False
    assert metrics["validation_streams"]["real_outcome_validation"]["available"] is False
    assert metrics["feature_signal_diagnostics"]["available"] is False
    assert "Insufficient usable labeled rows" in " ".join((metrics.get("guardrails") or {}).get("warnings") or [])


def test_feature_signal_diagnostics_flags_direction_conflict(tmp_path: Path) -> None:
    rows = []
    for idx in range(10):
        adverse = 1 if idx < 5 else 0
        rows.append(
            {
                "event": {"event_id": "evt-c"},
                "feature": {"record_id": f"r-{idx}"},
                "outcome": {
                    "damage_label": "destroyed" if adverse else "no_damage",
                    "damage_severity_class": "destroyed" if adverse else "none",
                    "structure_loss_or_major_damage": adverse,
                },
                "scores": {"wildfire_risk_score": (85.0 if adverse else 30.0)},
                "feature_snapshot": {
                    "raw_feature_vector": {
                        "burn_probability": (0.9 if adverse else 0.2),
                        "nearest_vegetation_distance_ft": (400.0 if adverse else 40.0),
                    }
                },
                "join_metadata": {"join_confidence_tier": "high", "join_confidence_score": 0.95},
                "confidence": {"confidence_tier": "high"},
                "evidence": {"evidence_quality_tier": "high"},
            }
        )
    dataset_path = tmp_path / "direction_conflict_eval.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report, _ = evaluate_public_outcome_dataset_file(dataset_path=dataset_path, min_labeled_rows=1)
    diag = report.get("feature_signal_diagnostics") or {}
    harmful = diag.get("potentially_harmful_features") or []
    harmful_names = {str(row.get("feature")) for row in harmful if isinstance(row, dict)}
    assert "nearest_vegetation_distance_ft" in harmful_names


def test_validation_propagates_retention_fallback_warning_from_dataset_join_report(tmp_path: Path) -> None:
    dataset_root = tmp_path / "eval_ds"
    run_dir = dataset_root / "tiny_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "evaluation_dataset.jsonl"
    rows = [
        {
            "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
            "feature": {"record_id": "f1", "source_record_id": "1"},
            "outcome": {"record_id": "o1", "damage_label": "destroyed", "damage_severity_class": "destroyed", "structure_loss_or_major_damage": 1},
            "scores": {"wildfire_risk_score": 88.0},
            "join_metadata": {"join_confidence_tier": "low", "join_confidence_score": 0.42},
            "evaluation": {"row_confidence_tier": "low-confidence", "soft_filter_flags": ["retention_fallback_mode"]},
        },
        {
            "event": {"event_id": "evt-a", "event_name": "Event A", "event_date": "2021-08-01"},
            "feature": {"record_id": "f2", "source_record_id": "2"},
            "outcome": {"record_id": "o2", "damage_label": "no_damage", "damage_severity_class": "none", "structure_loss_or_major_damage": 0},
            "scores": {"wildfire_risk_score": 22.0},
            "join_metadata": {"join_confidence_tier": "low", "join_confidence_score": 0.39},
            "evaluation": {"row_confidence_tier": "low-confidence", "soft_filter_flags": ["retention_fallback_mode"]},
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (run_dir / "join_quality_report.json").write_text(
        json.dumps(
            {
                "total_outcomes_loaded": 2,
                "total_feature_rows_loaded": 2,
                "total_joined_records": 2,
                "excluded_row_count": 0,
                "join_rate": 1.0,
                "retention_fallback_triggered": True,
                "retention_fallback_used": True,
                "retention_fallback": {
                    "enabled": True,
                    "triggered": True,
                    "used": True,
                    "target_min_records": 20,
                    "primary_joined_records": 0,
                    "fallback_joined_records": 2,
                    "active_pass": "retention_fallback_relaxed",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "validation_out"
    result = run_public_outcome_validation(
        evaluation_dataset=dataset_path,
        output_root=output_root,
        run_id="retention_flag_validation",
        min_labeled_rows=1,
        overwrite=True,
    )
    metrics = json.loads(Path(result["validation_metrics_path"]).read_text(encoding="utf-8"))
    pipeline = metrics.get("pipeline_stage_counts") or {}
    assert pipeline.get("retention_fallback_triggered") is True
    assert pipeline.get("retention_fallback_used") is True
    warnings = (metrics.get("guardrails") or {}).get("warnings") or []
    assert any("minimum-retention fallback mode was triggered" in str(item) for item in warnings)


def test_proxy_validation_stream_available_when_proxy_features_present(tmp_path: Path) -> None:
    rows = []
    for idx, (score, label, burn, hazard, dist, slope) in enumerate(
        [
            (88.0, 1, 0.92, 4.7, 120.0, 34.0),
            (82.0, 1, 0.86, 4.4, 180.0, 31.0),
            (70.0, 1, 0.74, 3.9, 260.0, 28.0),
            (42.0, 0, 0.35, 2.2, 880.0, 14.0),
            (34.0, 0, 0.22, 1.6, 1200.0, 11.0),
            (28.0, 0, 0.18, 1.3, 1400.0, 9.0),
        ],
        start=1,
    ):
        rows.append(
            {
                "event": {"event_id": "evt-proxy", "event_date": "2021-08-01"},
                "feature": {"record_id": f"p-{idx}", "source_record_id": str(idx)},
                "outcome": {
                    "damage_label": "destroyed" if label == 1 else "no_damage",
                    "damage_severity_class": "destroyed" if label == 1 else "none",
                    "structure_loss_or_major_damage": label,
                },
                "scores": {"wildfire_risk_score": score},
                "confidence": {"confidence_tier": "moderate"},
                "evidence": {"evidence_quality_tier": "moderate", "evidence_quality_summary": {}, "coverage_summary": {}},
                "feature_snapshot": {
                    "raw_feature_vector": {
                        "burn_probability": burn,
                        "wildfire_hazard": hazard,
                        "wildland_distance_m": dist,
                        "slope": slope,
                    }
                },
                "join_metadata": {"join_confidence_tier": "moderate", "join_confidence_score": 0.8},
            }
        )
    dataset_path = tmp_path / "proxy_eval.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report, _ = evaluate_public_outcome_dataset_file(dataset_path=dataset_path, thresholds=[40.0, 70.0], bins=4)
    proxy = report.get("proxy_validation") or {}
    assert proxy.get("available") is True
    assert (proxy.get("alignment_metrics") or {}).get("spearman_model_vs_proxy_index") is not None
    assert (proxy.get("alignment_metrics") or {}).get("auc_model_vs_proxy_high_low") is not None
    synthetic = report.get("synthetic_validation") or {}
    assert "available" in synthetic
    if synthetic.get("available"):
        extreme = synthetic.get("extreme_scenario_ranking") or {}
        assert "passed" in extreme
        assert "high_risk_score" in extreme
        assert "low_risk_score" in extreme


def test_validation_retains_unusable_rows_with_flags(tmp_path: Path) -> None:
    dataset_path = tmp_path / "mixed_rows.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": {"event_id": "evt-a"},
                        "feature": {"record_id": "ok-1"},
                        "outcome": {
                            "damage_label": "destroyed",
                            "damage_severity_class": "destroyed",
                            "structure_loss_or_major_damage": 1,
                        },
                        "scores": {"wildfire_risk_score": 82.0},
                        "join_metadata": {"join_confidence_tier": "high", "join_confidence_score": 0.95},
                    }
                ),
                json.dumps(
                    {
                        "event": {"event_id": "evt-a"},
                        "feature": {"record_id": "bad-1"},
                        "outcome": {
                            "damage_label": "unknown",
                            "damage_severity_class": "unknown",
                            "structure_loss_or_major_damage": None,
                        },
                        "scores": {"wildfire_risk_score": None},
                        "join_metadata": {"join_confidence_tier": "low", "join_confidence_score": 0.35},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report, rows = evaluate_public_outcome_dataset_file(
        dataset_path=dataset_path,
        min_labeled_rows=1,
        retain_unusable_rows=True,
    )
    assert report["sample_counts"]["row_count_retained"] == 2
    assert report["sample_counts"]["row_count_usable"] == 1
    assert report["sample_counts"]["row_count_unusable"] == 1
    assert len(rows) == 2
    flagged = [row for row in rows if row.get("record_id") == "bad-1"][0]
    assert flagged["row_usable_for_metrics"] is False
    assert "missing_or_invalid_structure_loss_or_major_damage" in (flagged.get("exclusion_reasons") or [])
    assert "missing_scores.wildfire_risk_score" in (flagged.get("exclusion_reasons") or [])


def test_min_join_confidence_filter_is_configurable_and_tags_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "join_filter_rows.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": {"event_id": "evt-a"},
                        "feature": {"record_id": "high-join"},
                        "outcome": {
                            "damage_label": "major_damage",
                            "damage_severity_class": "major",
                            "structure_loss_or_major_damage": 1,
                        },
                        "scores": {"wildfire_risk_score": 75.0},
                        "join_metadata": {"join_confidence_tier": "high", "join_confidence_score": 0.91},
                    }
                ),
                json.dumps(
                    {
                        "event": {"event_id": "evt-a"},
                        "feature": {"record_id": "low-join"},
                        "outcome": {
                            "damage_label": "no_damage",
                            "damage_severity_class": "none",
                            "structure_loss_or_major_damage": 0,
                        },
                        "scores": {"wildfire_risk_score": 35.0},
                        "join_metadata": {"join_confidence_tier": "low", "join_confidence_score": 0.42},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report, rows = evaluate_public_outcome_dataset_file(
        dataset_path=dataset_path,
        min_labeled_rows=1,
        min_join_confidence_score_for_metrics=0.7,
        retain_unusable_rows=True,
    )
    assert report["sample_counts"]["row_count_retained"] == 2
    assert report["sample_counts"]["row_count_usable"] == 1
    low_join = [row for row in rows if row.get("record_id") == "low-join"][0]
    assert low_join["row_usable_for_metrics"] is False
    assert "join_confidence_below_min" in (low_join.get("exclusion_reasons") or [])
