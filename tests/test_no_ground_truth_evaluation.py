from __future__ import annotations

from pathlib import Path

from backend.evaluation.no_ground_truth import (
    build_no_ground_truth_summary_markdown,
    evaluate_counterfactual_groups,
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
    assert Path(first["summary_path"]).exists()
