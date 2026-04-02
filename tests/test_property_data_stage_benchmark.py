from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts") / "run_property_data_stage_benchmark.py"
PACK_PATH = Path("benchmark") / "scenario_pack_property_data_differentiation_v1.json"

STAGE_ORDER = [
    "geocode_only",
    "parcel_matched",
    "footprint_matched",
    "footprint_naip",
    "footprint_naip_enriched",
]


def _make_snapshot(
    *,
    stage: str,
    variant: str,
    property_confidence_score: float,
    property_confidence_level: str,
    specificity_tier: str,
    comparison_allowed: bool,
    local_differentiation_score: float,
    confidence_language: str,
    top_driver: str,
    top_action: str,
    safeguard_triggered: bool = False,
    safeguard_message: str | None = None,
) -> dict[str, object]:
    return {
        "profile_tags": ["property_data_stage", f"stage:{stage}", variant],
        "property_confidence_summary": {
            "score": property_confidence_score,
            "level": property_confidence_level,
            "key_reasons": [],
        },
        "specificity": {
            "specificity_tier": specificity_tier,
            "comparison_allowed": comparison_allowed,
        },
        "differentiation": {
            "local_differentiation_score": local_differentiation_score,
            "nearby_home_comparison_safeguard_triggered": safeguard_triggered,
            "nearby_home_comparison_safeguard_message": safeguard_message,
        },
        "confidence": {
            "confidence_language": confidence_language,
        },
        "top_risk_drivers": [top_driver, "backup driver"],
        "top_recommended_actions": [top_action, "backup action"],
    }


def _write_artifact(path: Path, *, regress_wording: bool = False) -> None:
    dense_progression = [
        # stage, property_confidence_score, property_confidence_level, specificity_tier, comparison_allowed, local_differentiation_score, confidence_language
        ("geocode_only", 30.0, "regional_estimate_with_anchor", "regional_estimate", False, 24.0, "limited confidence"),
        ("parcel_matched", 45.0, "address_level", "address_level", False, 38.0, "moderate confidence"),
        ("footprint_matched", 62.0, "strong_property_specific", "address_level", True, 56.0, "moderate confidence"),
        ("footprint_naip", 74.0, "strong_property_specific", "property_specific", True, 69.0, "moderate confidence"),
        ("footprint_naip_enriched", 86.0, "verified_property_specific", "property_specific", True, 79.0, "high confidence"),
    ]
    clear_progression = [
        ("geocode_only", 29.0, "regional_estimate_with_anchor", "regional_estimate", False, 22.0, "limited confidence"),
        ("parcel_matched", 43.0, "address_level", "address_level", False, 34.0, "moderate confidence"),
        ("footprint_matched", 59.0, "strong_property_specific", "address_level", True, 51.0, "moderate confidence"),
        ("footprint_naip", 72.0, "strong_property_specific", "property_specific", True, 64.0, "moderate confidence"),
        ("footprint_naip_enriched", 84.0, "verified_property_specific", "property_specific", True, 73.0, "high confidence"),
    ]

    if regress_wording:
        # Force a confidence-language regression despite stronger property confidence.
        dense_progression[0] = ("geocode_only", 30.0, "regional_estimate_with_anchor", "regional_estimate", False, 24.0, "moderate confidence")
        clear_progression[0] = ("geocode_only", 29.0, "regional_estimate_with_anchor", "regional_estimate", False, 22.0, "moderate confidence")
        dense_progression[1] = ("parcel_matched", 47.0, "address_level", "address_level", False, 39.0, "limited confidence")
        clear_progression[1] = ("parcel_matched", 46.0, "address_level", "address_level", False, 35.0, "limited confidence")

    scenario_results = []
    for stage, score, level, specificity, comparison_allowed, local_diff, conf_language in dense_progression:
        scenario_results.append(
            {
                "scenario_id": f"stage_{stage}_dense",
                "snapshot": _make_snapshot(
                    stage=stage,
                    variant="dense",
                    property_confidence_score=score,
                    property_confidence_level=level,
                    specificity_tier=specificity,
                    comparison_allowed=comparison_allowed,
                    local_differentiation_score=local_diff,
                    confidence_language=conf_language,
                    top_driver=f"dense_driver_{stage}",
                    top_action=f"dense_action_{stage}",
                    safeguard_triggered=(stage == "geocode_only"),
                    safeguard_message=(
                        "This estimate is not precise enough to compare adjacent homes."
                        if stage == "geocode_only"
                        else None
                    ),
                ),
            }
        )
    for stage, score, level, specificity, comparison_allowed, local_diff, conf_language in clear_progression:
        scenario_results.append(
            {
                "scenario_id": f"stage_{stage}_clear",
                "snapshot": _make_snapshot(
                    stage=stage,
                    variant="clear",
                    property_confidence_score=score,
                    property_confidence_level=level,
                    specificity_tier=specificity,
                    comparison_allowed=comparison_allowed,
                    local_differentiation_score=local_diff,
                    confidence_language=conf_language,
                    top_driver=f"clear_driver_{stage}",
                    top_action=f"clear_action_{stage}",
                    safeguard_triggered=(stage == "geocode_only"),
                    safeguard_message=(
                        "This estimate is not precise enough to compare adjacent homes."
                        if stage == "geocode_only"
                        else None
                    ),
                ),
            }
        )

    artifact = {
        "benchmark_pack_version": "property-data-stage-v1-test",
        "summary": {"passed": True, "scenario_count": len(scenario_results)},
        "scenario_results": scenario_results,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_property_data_stage_benchmark_reports_stage_progression(tmp_path: Path) -> None:
    artifact_path = tmp_path / "fixture_artifact.json"
    _write_artifact(artifact_path, regress_wording=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-artifact",
            str(artifact_path),
            "--fail-on-stage-regression",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads(proc.stdout)
    stage_quality = payload.get("stage_quality") or {}
    assert stage_quality.get("passed") is True
    assert int(stage_quality.get("transition_count") or 0) == 8
    assert int(stage_quality.get("trust_wording_regression_count") or 0) == 0

    summary_paths = payload.get("summary_artifacts") or {}
    report_path = Path(str(summary_paths.get("json")))
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    stage_summaries = report.get("stage_summaries") or []
    assert len(stage_summaries) == len(STAGE_ORDER)
    first_stage = stage_summaries[0]
    assert first_stage.get("stage_key") == "geocode_only"
    dense_summary = ((first_stage.get("variant_summaries") or {}).get("dense") or {})
    assert "property_confidence_summary" in dense_summary
    assert "assessment_specificity_tier" in dense_summary
    assert "local_differentiation_score" in dense_summary

    transitions = report.get("stage_transitions") or []
    assert transitions
    sample_transition = transitions[0]
    assert "top_risk_driver_stability" in sample_transition
    assert "recommendation_changes" in sample_transition
    assert "trust_wording_regression" in sample_transition


def test_property_data_stage_benchmark_flags_trust_wording_regression(tmp_path: Path) -> None:
    artifact_path = tmp_path / "fixture_artifact_regression.json"
    _write_artifact(artifact_path, regress_wording=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-artifact",
            str(artifact_path),
            "--fail-on-stage-regression",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr

    payload = json.loads(proc.stdout)
    stage_quality = payload.get("stage_quality") or {}
    assert stage_quality.get("passed") is False
    assert int(stage_quality.get("trust_wording_regression_count") or 0) > 0


def test_property_data_stage_pack_covers_all_requested_stages() -> None:
    payload = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    tags = []
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        profile_tags = row.get("profile_tags")
        if not isinstance(profile_tags, list):
            continue
        tags.extend(
            str(tag).split(":", 1)[1]
            for tag in profile_tags
            if isinstance(tag, str) and tag.startswith("stage:")
        )
    stages = set(tags)
    assert stages == {
        "geocode_only",
        "parcel_matched",
        "footprint_matched",
        "footprint_naip",
        "footprint_naip_enriched",
    }
