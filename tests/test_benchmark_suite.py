from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.benchmarking import (
    compare_benchmark_artifacts,
    load_benchmark_pack,
    run_benchmark_suite,
)


def _write_pack(path: Path, *, passing: bool = True) -> None:
    pack = {
        "benchmark_pack_version": "test-1",
        "factor_schema_version": "test-factor-1",
        "scenarios": [
            {
                "scenario_id": "low_case",
                "description": "Low case",
                "input_payload": {
                    "address": "1 Test Way",
                    "attributes": {
                        "roof_type": "class a",
                        "vent_type": "ember-resistant",
                        "defensible_space_ft": 35,
                        "construction_year": 2018,
                    },
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft", "construction_year"],
                    "audience": "insurer",
                    "tags": ["benchmark-test"],
                },
                "location": {"lat": 39.7, "lon": -104.9, "geocode_source": "benchmark-test"},
                "context": {
                    "burn_probability_index": 22.0,
                    "hazard_severity_index": 28.0,
                    "slope_index": 24.0,
                    "fuel_index": 30.0,
                    "moisture_index": 35.0,
                    "canopy_index": 26.0,
                    "wildland_distance_index": 28.0,
                    "historic_fire_index": 22.0,
                    "structure_ring_metrics": {
                        "ring_0_5_ft": {"vegetation_density": 18.0},
                        "ring_5_30_ft": {"vegetation_density": 25.0},
                        "ring_30_100_ft": {"vegetation_density": 32.0},
                    },
                    "property_level_context": {
                        "footprint_used": True,
                        "footprint_status": "used",
                        "fallback_mode": "footprint",
                    },
                },
                "expected": {
                    "risk_band": "low",
                    "assessment_status_in": ["fully_scored", "partially_scored"],
                },
            },
            {
                "scenario_id": "high_case",
                "description": "High case",
                "input_payload": {
                    "address": "2 Test Way",
                    "attributes": {
                        "roof_type": "wood",
                        "vent_type": "standard",
                        "defensible_space_ft": 4,
                        "construction_year": 1990,
                    },
                    "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
                    "audience": "insurer",
                    "tags": ["benchmark-test"],
                },
                "location": {"lat": 39.71, "lon": -105.01, "geocode_source": "benchmark-test"},
                "context": {
                    "burn_probability_index": 82.0,
                    "hazard_severity_index": 86.0,
                    "slope_index": 72.0,
                    "fuel_index": 84.0,
                    "moisture_index": 78.0,
                    "canopy_index": 82.0,
                    "wildland_distance_index": 88.0,
                    "historic_fire_index": 80.0,
                    "structure_ring_metrics": {
                        "ring_0_5_ft": {"vegetation_density": 90.0},
                        "ring_5_30_ft": {"vegetation_density": 88.0},
                        "ring_30_100_ft": {"vegetation_density": 86.0},
                    },
                },
                "expected": {
                    "risk_band": "severe",
                    "assessment_status_in": ["fully_scored", "partially_scored"],
                },
            },
        ],
        "relative_assertions": [
            {
                "assertion_id": "high_gt_low",
                "left": "high_case",
                "metric": "scores.wildfire_risk_score",
                "op": ">" if passing else "<",
                "right": "low_case",
            }
        ],
        "monotonic_assertions": [
            {
                "assertion_id": "high_vulnerability_gt_low",
                "left": "high_case",
                "metric": "scores.home_ignition_vulnerability_score",
                "op": ">" if passing else "<",
                "right": "low_case",
            }
        ],
    }
    path.write_text(json.dumps(pack), encoding="utf-8")


def test_default_benchmark_pack_loads():
    pack = load_benchmark_pack()
    assert pack["benchmark_pack_version"]
    assert isinstance(pack["scenarios"], list)
    assert len(pack["scenarios"]) >= 8


def test_run_benchmark_suite_outputs_artifact_and_passes(tmp_path):
    pack_path = tmp_path / "pack.json"
    out_dir = tmp_path / "out"
    _write_pack(pack_path, passing=True)

    artifact = run_benchmark_suite(pack_path=pack_path, output_dir=out_dir)
    assert artifact["summary"]["passed"] is True
    assert artifact["summary"]["scenario_count"] == 2
    assert Path(artifact["artifact_path"]).exists()
    assert "governance" in artifact
    assert "relative_assertions" in artifact
    assert "monotonic_assertions" in artifact
    first_snapshot = artifact["scenario_results"][0]["snapshot"]
    assert "evidence_metrics" in first_snapshot
    assert "fallback_weight_fraction" in first_snapshot["evidence_metrics"]


def test_run_benchmark_suite_detects_monotonic_failure(tmp_path):
    pack_path = tmp_path / "pack_fail.json"
    out_dir = tmp_path / "out_fail"
    _write_pack(pack_path, passing=False)

    artifact = run_benchmark_suite(pack_path=pack_path, output_dir=out_dir)
    assert artifact["summary"]["passed"] is False
    assert artifact["summary"]["assertion_failures"]


def test_compare_benchmark_artifacts_detects_drift():
    baseline = {
        "artifact_path": "baseline.json",
        "governance": {"scoring_model_versions": ["1.0.0"]},
        "scenario_results": [
            {
                "scenario_id": "a",
                "snapshot": {
                    "scores": {"wildfire_risk_score": 50.0, "site_hazard_score": 55.0, "home_ignition_vulnerability_score": 45.0, "insurance_readiness_score": 60.0},
                    "confidence": {"confidence_score": 80.0},
                    "assessment_blockers": [],
                    "readiness_blockers": [],
                    "warnings": [],
                    "score_evidence_ledger_summary": {"wildfire_risk_score": {"site_hazard_component": 30.0}},
                },
            }
        ],
    }
    current = {
        "artifact_path": "current.json",
        "governance": {"scoring_model_versions": ["1.1.0"]},
        "scenario_results": [
            {
                "scenario_id": "a",
                "snapshot": {
                    "scores": {"wildfire_risk_score": 66.0, "site_hazard_score": 70.0, "home_ignition_vulnerability_score": 60.0, "insurance_readiness_score": 50.0},
                    "confidence": {"confidence_score": 68.0},
                    "assessment_blockers": ["new blocker"],
                    "readiness_blockers": [],
                    "warnings": ["new warning"],
                    "score_evidence_ledger_summary": {"wildfire_risk_score": {"site_hazard_component": 48.0}},
                },
            }
        ],
    }

    drift = compare_benchmark_artifacts(
        baseline_artifact=baseline,
        current_artifact=current,
        score_drift_threshold=8.0,
        confidence_drift_threshold=8.0,
        contribution_drift_threshold=5.0,
    )
    assert drift["material_drift_count"] == 1
    assert drift["scenario_deltas"][0]["material_drift"] is True


def test_run_benchmark_suite_script_exit_behavior(tmp_path):
    script = Path("scripts") / "run_benchmark_suite.py"
    passing_pack = tmp_path / "pack_pass.json"
    failing_pack = tmp_path / "pack_fail.json"
    out_dir = tmp_path / "artifacts"
    _write_pack(passing_pack, passing=True)
    _write_pack(failing_pack, passing=False)

    ok = subprocess.run(
        [sys.executable, str(script), "--pack", str(passing_pack), "--output-dir", str(out_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    bad = subprocess.run(
        [sys.executable, str(script), "--pack", str(failing_pack), "--output-dir", str(out_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert bad.returncode != 0, bad.stdout


def test_run_confidence_benchmark_pack_script_outputs_summary(tmp_path):
    script = Path("scripts") / "run_confidence_benchmark_pack.py"
    out_dir = tmp_path / "confidence_artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--pack",
            "benchmark/scenario_pack_confidence_v2.json",
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["summary"]["passed"] is True
    assert payload["distribution"]["fallback_weight_fraction"]["count"] >= 1
    assert payload["distribution"]["suppressed_factor_count"]["count"] >= 1
    assert payload["spread_checks"]["wildfire_risk_score_spread"] is not None
    assert "nearby_differentiation_performance" in payload


def test_nearby_differentiation_pack_reports_local_separation_and_caution(tmp_path):
    pack_path = Path("benchmark") / "scenario_pack_nearby_differentiation_v1.json"
    artifact = run_benchmark_suite(pack_path=pack_path, output_dir=tmp_path / "nearby")
    assert artifact["summary"]["passed"] is True

    nearby = artifact.get("nearby_differentiation_performance") or {}
    assert nearby.get("available") is True
    assert int(nearby.get("scenario_count") or 0) >= 8
    assert int(nearby.get("assertion_fail_count") or 0) == 0
    assert int((nearby.get("local_subscore_assertions") or {}).get("count") or 0) >= 3
    assert int((nearby.get("confidence_caution_assertions") or {}).get("count") or 0) >= 3

    snapshots = {
        row["scenario_id"]: row["snapshot"]
        for row in artifact.get("scenario_results", [])
        if isinstance(row, dict) and isinstance(row.get("snapshot"), dict)
    }
    dense = snapshots["nearby_dense_veg_0_5ft"]
    clear = snapshots["nearby_clear_veg_0_5ft"]
    assert float(dense["scores"]["home_ignition_vulnerability_score"]) > float(
        clear["scores"]["home_ignition_vulnerability_score"]
    )

    missing = snapshots["nearby_missing_geometry_point"]
    footprint = snapshots["nearby_footprint_geometry_available"]
    assert float(missing["differentiation"]["neighborhood_differentiation_confidence"]) < float(
        footprint["differentiation"]["neighborhood_differentiation_confidence"]
    )
    assert bool(missing["differentiation"]["nearby_home_comparison_safeguard_triggered"]) is True
    assert bool(footprint["differentiation"]["nearby_home_comparison_safeguard_triggered"]) is False


def test_nearby_differentiation_v2_pack_measures_separation_and_honest_abstention(tmp_path):
    pack_path = Path("benchmark") / "scenario_pack_nearby_differentiation_v2.json"
    artifact = run_benchmark_suite(pack_path=pack_path, output_dir=tmp_path / "nearby_v2")
    assert artifact["summary"]["passed"] is True

    nearby = artifact.get("nearby_differentiation_performance") or {}
    assert nearby.get("available") is True
    separation = nearby.get("separation_analysis") or {}
    assert int(separation.get("pair_count") or 0) >= 7
    assert int(separation.get("separation_achieved_count") or 0) >= 7
    assert int(separation.get("collapsed_toward_similarity_count") or 0) >= 1
    assert int(separation.get("collapsed_correctly_flagged_low_specificity_count") or 0) >= 1
    assert int(separation.get("collapsed_not_flagged_count") or 0) == 0

    snapshots = {
        row["scenario_id"]: row["snapshot"]
        for row in artifact.get("scenario_results", [])
        if isinstance(row, dict) and isinstance(row.get("snapshot"), dict)
    }

    dense = snapshots["nearby_v2_dense_0_5ft_footprint"]
    clear = snapshots["nearby_v2_clear_0_5ft_footprint"]
    assert float(dense["scores"]["wildfire_risk_score"]) > float(clear["scores"]["wildfire_risk_score"])
    assert float(dense["scores"]["home_ignition_vulnerability_score"]) > float(
        clear["scores"]["home_ignition_vulnerability_score"]
    )
    assert dense["specificity"]["comparison_allowed"] is True
    assert clear["specificity"]["comparison_allowed"] is True

    missing = snapshots["nearby_v2_footprint_missing_point"]
    available = snapshots["nearby_v2_footprint_available"]
    assert bool(missing["specificity"]["comparison_allowed"]) is False
    assert bool(available["specificity"]["comparison_allowed"]) is True
    assert str(missing["specificity"]["specificity_tier"]) in {"regional_estimate", "address_level", "insufficient_data"}
