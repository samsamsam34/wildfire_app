from __future__ import annotations

import json
from pathlib import Path

from backend.benchmarking import build_wildfire_context
from backend.evaluation.no_ground_truth import (
    build_no_ground_truth_summary_markdown,
    evaluate_counterfactual_groups,
    evaluate_confidence_diagnostics,
    evaluate_distribution,
    evaluate_monotonicity_rules,
    evaluate_stability,
)
from scripts.run_no_ground_truth_evaluation import run_no_ground_truth_evaluation


def test_monotonicity_rule_evaluation_flags_violation() -> None:
    snapshots = {
        "a": {"scores": {"wildfire_risk_score": 60.0}},
        "b": {"scores": {"wildfire_risk_score": 54.0}},
    }
    result = evaluate_monotonicity_rules(
        rules=[
            {
                "rule_id": "risk_should_not_drop",
                "baseline": "a",
                "variant": "b",
                "metric": "scores.wildfire_risk_score",
                "expected": "non_decrease",
            }
        ],
        snapshots_by_id=snapshots,
    )
    assert result["failed_count"] == 1
    assert result["violated_rules"] == ["risk_should_not_drop"]


def test_counterfactual_delta_generation() -> None:
    snapshots = {
        "base": {
            "scores": {
                "wildfire_risk_score": 70.0,
                "insurance_readiness_score": 35.0,
                "home_ignition_vulnerability_score": 68.0,
            }
        },
        "mitigation": {
            "scores": {
                "wildfire_risk_score": 62.0,
                "insurance_readiness_score": 44.0,
                "home_ignition_vulnerability_score": 60.0,
            }
        },
    }
    result = evaluate_counterfactual_groups(
        groups=[
            {
                "group_id": "g1",
                "baseline": "base",
                "variants": [{"scenario_id": "mitigation", "intervention": "fuel_reduction", "expected": "risk_down"}],
            }
        ],
        snapshots_by_id=snapshots,
    )
    group = result["groups"][0]
    row = group["interventions"][0]
    assert row["risk_delta"] == -8.0
    assert row["directionally_consistent"] is True


def test_stability_metrics_computation() -> None:
    base = {"s0": {"scores": {"wildfire_risk_score": 55.0, "risk_band": "medium"}, "confidence": {"confidence_score": 62.0, "confidence_tier": "moderate"}}}
    variants = {
        "v1": {"scores": {"wildfire_risk_score": 56.0, "risk_band": "medium"}, "confidence": {"confidence_score": 61.0, "confidence_tier": "moderate"}},
        "v2": {"scores": {"wildfire_risk_score": 71.0, "risk_band": "high"}, "confidence": {"confidence_score": 49.0, "confidence_tier": "low"}},
    }
    descriptors = [
        {"test_id": "t1", "base_scenario_id": "s0", "variant_scenario_id": "v1", "variant_type": "jitter", "detail": {}},
        {"test_id": "t1", "base_scenario_id": "s0", "variant_scenario_id": "v2", "variant_type": "jitter", "detail": {}},
    ]
    result = evaluate_stability(
        base_snapshots=base,
        variant_snapshots=variants,
        descriptors=descriptors,
    )
    assert result["test_count"] == 1
    assert result["tests"][0]["max_abs_score_swing"] == 16.0
    assert result["status"] == "warn"


def test_summary_markdown_contains_required_caveat() -> None:
    text = build_no_ground_truth_summary_markdown(
        run_id="r1",
        generated_at="r1",
        fixture_path=Path("benchmark/fixtures/no_ground_truth/scenario_pack_v1.json"),
        monotonicity={"status": "ok", "passed_count": 2, "rule_count": 2, "violated_rules": []},
        counterfactual={"status": "ok", "flagged_interventions": []},
        stability={"status": "ok", "warnings": []},
        distribution={"status": "ok", "warnings": []},
        benchmark_alignment={"status": "warn"},
        confidence_diagnostics={"status": "ok", "warnings": []},
    )
    assert "not ground-truth accuracy validation" in text
    assert "external alignment caveat" in text.lower()


def test_orchestration_is_deterministic_with_fixed_run_id(tmp_path: Path) -> None:
    fixture = Path("benchmark/fixtures/no_ground_truth/scenario_pack_v1.json")
    output_root = tmp_path / "no_gt_runs"
    first = run_no_ground_truth_evaluation(
        fixture_path=fixture,
        output_root=output_root,
        run_id="fixed_no_gt_run",
        seed=17,
        overwrite=True,
    )
    second = run_no_ground_truth_evaluation(
        fixture_path=fixture,
        output_root=output_root,
        run_id="fixed_no_gt_run",
        seed=17,
        overwrite=True,
    )
    manifest_1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    manifest_2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert manifest_1 == manifest_2
    mono_1 = (output_root / "fixed_no_gt_run" / "monotonicity_results.json").read_text(encoding="utf-8")
    mono_2 = (output_root / "fixed_no_gt_run" / "monotonicity_results.json").read_text(encoding="utf-8")
    assert mono_1 == mono_2
    distribution = json.loads((output_root / "fixed_no_gt_run" / "distribution_results.json").read_text(encoding="utf-8"))
    assert distribution["largest_bucket_fraction"] < 0.75
    assert distribution["occupied_risk_bucket_count"] >= 3
    assert Path(first["summary_path"]).exists()


def test_orchestration_writes_comparison_to_previous_when_baseline_exists(tmp_path: Path) -> None:
    fixture = Path("benchmark/fixtures/no_ground_truth/scenario_pack_v1.json")
    output_root = tmp_path / "no_gt_runs"
    run_no_ground_truth_evaluation(
        fixture_path=fixture,
        output_root=output_root,
        run_id="run_a",
        seed=17,
        overwrite=True,
    )
    run_no_ground_truth_evaluation(
        fixture_path=fixture,
        output_root=output_root,
        run_id="run_b",
        seed=17,
        overwrite=True,
    )
    comparison = json.loads(
        (output_root / "run_b" / "comparison_to_previous.json").read_text(encoding="utf-8")
    )
    assert comparison["available"] is True
    assert comparison["run_id"] == "run_b"
    assert comparison["baseline_run_id"] == "run_a"
    assert "monotonicity" in comparison
    assert "distribution" in comparison


def test_orchestration_marks_comparison_unavailable_without_baseline(tmp_path: Path) -> None:
    fixture = Path("benchmark/fixtures/no_ground_truth/scenario_pack_v1.json")
    output_root = tmp_path / "no_gt_runs"
    run_no_ground_truth_evaluation(
        fixture_path=fixture,
        output_root=output_root,
        run_id="only_run",
        seed=17,
        overwrite=True,
    )
    comparison = json.loads(
        (output_root / "only_run" / "comparison_to_previous.json").read_text(encoding="utf-8")
    )
    assert comparison["available"] is False
    assert comparison["reason"] == "no_previous_run_available"


def test_distribution_bucketing_uses_thresholds_and_reports_balance() -> None:
    scenarios = {
        "s_low": {"region": "r1", "segments": ["a"]},
        "s_medium": {"region": "r1", "segments": ["a"]},
        "s_high": {"region": "r1", "segments": ["b"]},
    }
    snapshots = {
        "s_low": {"scores": {"wildfire_risk_score": 35.0}, "confidence": {"confidence_tier": "moderate"}, "evidence_metrics": {}},
        "s_medium": {"scores": {"wildfire_risk_score": 50.0}, "confidence": {"confidence_tier": "moderate"}, "evidence_metrics": {}},
        "s_high": {"scores": {"wildfire_risk_score": 72.0}, "confidence": {"confidence_tier": "high"}, "evidence_metrics": {}},
    }
    distribution = evaluate_distribution(
        scenarios_by_id=scenarios,
        snapshots_by_id=snapshots,
    )
    assert distribution["risk_bucket_counts"]["low"] == 1
    assert distribution["risk_bucket_counts"]["medium"] == 1
    assert distribution["risk_bucket_counts"]["high"] == 1
    assert distribution["occupied_risk_bucket_count"] == 3
    assert distribution["largest_bucket_fraction"] == 0.3333


def test_confidence_diagnostics_warn_on_backwards_fallback_relationship() -> None:
    snapshots = {
        "a": {
            "confidence": {"confidence_score": 20.0, "confidence_tier": "low", "missing_critical_fields_count": 1, "inferred_fields_count": 1},
            "evidence_metrics": {"fallback_weight_fraction": 0.2, "observed_feature_count": 8},
        },
        "b": {
            "confidence": {"confidence_score": 50.0, "confidence_tier": "moderate", "missing_critical_fields_count": 2, "inferred_fields_count": 2},
            "evidence_metrics": {"fallback_weight_fraction": 0.5, "observed_feature_count": 5},
        },
        "c": {
            "confidence": {"confidence_score": 70.0, "confidence_tier": "high", "missing_critical_fields_count": 3, "inferred_fields_count": 3},
            "evidence_metrics": {"fallback_weight_fraction": 0.8, "observed_feature_count": 2},
        },
    }
    diag = evaluate_confidence_diagnostics(snapshots_by_id=snapshots)
    assert diag["status"] == "warn"
    assert any("increases with fallback weight" in row for row in diag["warnings"])
    assert "confidence_by_evidence_tier" in diag
    assert "missing_critical_field_count_distribution" in diag


def test_confidence_diagnostics_pass_when_confidence_tracks_evidence_quality() -> None:
    snapshots = {
        "a": {
            "confidence": {"confidence_score": 80.0, "confidence_tier": "high", "missing_critical_fields_count": 0, "inferred_fields_count": 0},
            "evidence_metrics": {"fallback_weight_fraction": 0.1, "observed_feature_count": 10},
        },
        "b": {
            "confidence": {"confidence_score": 58.0, "confidence_tier": "moderate", "missing_critical_fields_count": 1, "inferred_fields_count": 1},
            "evidence_metrics": {"fallback_weight_fraction": 0.4, "observed_feature_count": 6},
        },
        "c": {
            "confidence": {"confidence_score": 34.0, "confidence_tier": "low", "missing_critical_fields_count": 2, "inferred_fields_count": 2},
            "evidence_metrics": {"fallback_weight_fraction": 0.7, "observed_feature_count": 2},
        },
    }
    diag = evaluate_confidence_diagnostics(snapshots_by_id=snapshots)
    assert diag["status"] == "ok"
    assert not any("increases with fallback weight" in row for row in diag["warnings"])
    assert not any("increases with missing critical field count" in row for row in diag["warnings"])
    assert not any("increases with inferred-field count" in row for row in diag["warnings"])


def test_confidence_diagnostics_prefers_feature_fallback_fraction_over_weighted_fallback_fraction() -> None:
    snapshots = {
        "a": {
            "confidence": {"confidence_score": 80.0, "confidence_tier": "low", "missing_critical_fields_count": 0, "inferred_fields_count": 0},
            "evidence_metrics": {
                "fallback_weight_fraction": 0.7,
                "fallback_evidence_fraction": 0.1,
                "observed_feature_count": 8,
            },
        },
        "b": {
            "confidence": {"confidence_score": 55.0, "confidence_tier": "low", "missing_critical_fields_count": 0, "inferred_fields_count": 0},
            "evidence_metrics": {
                "fallback_weight_fraction": 0.5,
                "fallback_evidence_fraction": 0.4,
                "observed_feature_count": 8,
            },
        },
        "c": {
            "confidence": {"confidence_score": 30.0, "confidence_tier": "preliminary", "missing_critical_fields_count": 1, "inferred_fields_count": 0},
            "evidence_metrics": {
                "fallback_weight_fraction": 0.1,
                "fallback_evidence_fraction": 0.7,
                "observed_feature_count": 8,
            },
        },
    }
    diag = evaluate_confidence_diagnostics(snapshots_by_id=snapshots)
    assert not any("increases with fallback weight" in row for row in diag["warnings"])
    corr = diag.get("confidence_vs_fallback_weight_spearman")
    assert isinstance(corr, (int, float))
    assert float(corr) <= 0.0


def test_build_wildfire_context_syncs_property_ring_metrics_to_structure_ring_metrics() -> None:
    context = build_wildfire_context(
        {
            "property_level_context": {
                "ring_metrics": {
                    "ring_0_5_ft": {"vegetation_density": 12.0},
                    "ring_5_30_ft": {"vegetation_density": 28.0},
                }
            }
        }
    )
    assert context.structure_ring_metrics["ring_0_5_ft"]["vegetation_density"] == 12.0
    assert context.structure_ring_metrics["ring_5_30_ft"]["vegetation_density"] == 28.0


def test_build_wildfire_context_location_specific_structure_proxy_fields_vary() -> None:
    ctx_a = build_wildfire_context(latitude=39.7392, longitude=-104.9903)
    ctx_b = build_wildfire_context(latitude=47.6062, longitude=-122.3321)

    plc_a = ctx_a.property_level_context
    plc_b = ctx_b.property_level_context

    assert isinstance(plc_a.get("neighboring_structure_metrics"), dict)
    assert isinstance(plc_b.get("neighboring_structure_metrics"), dict)
    assert plc_a.get("structure_density") is not None
    assert plc_b.get("structure_density") is not None
    assert plc_a.get("clustering_index") is not None
    assert plc_b.get("clustering_index") is not None
    assert plc_a.get("building_age_proxy_year") is not None
    assert plc_b.get("building_age_proxy_year") is not None

    varying_fields = (
        "structure_density",
        "clustering_index",
        "building_age_proxy_year",
        "building_age_material_proxy_risk",
        "near_structure_connectivity_index",
        "near_structure_vegetation_0_5_pct",
        "canopy_adjacency_proxy_pct",
    )
    assert any((plc_a.get(field) != plc_b.get(field)) for field in varying_fields)
