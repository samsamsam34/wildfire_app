from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import AddressRequest, AssessmentResult, UnderwritingRuleset
from backend.wildfire_data import WildfireContext
from backend.version import MODEL_VERSION

BENCHMARK_PACK_VERSION = "1.0.0"
FACTOR_SCHEMA_VERSION = "1.0.0"
DEFAULT_PACK_PATH = Path("benchmark") / "scenario_pack_v1.json"
DEFAULT_RESULTS_DIR = Path("benchmark") / "results"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _risk_band(score: float | None) -> str:
    if score is None:
        return "unscored"
    if score < 30:
        return "low"
    if score < 55:
        return "moderate"
    if score < 75:
        return "high"
    return "severe"


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _path_get(obj: dict[str, Any], path: str) -> Any:
    current: Any = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def _compare_values(left: Any, op: str, right: Any) -> bool:
    if op == ">":
        return left is not None and right is not None and left > right
    if op == ">=":
        return left is not None and right is not None and left >= right
    if op == "<":
        return left is not None and right is not None and left < right
    if op == "<=":
        return left is not None and right is not None and left <= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    raise ValueError(f"Unsupported operator: {op}")


def default_wildfire_context_dict() -> dict[str, Any]:
    return {
        "environmental_index": 50.0,
        "slope_index": 45.0,
        "aspect_index": 50.0,
        "fuel_index": 50.0,
        "moisture_index": 50.0,
        "canopy_index": 50.0,
        "wildland_distance_index": 50.0,
        "historic_fire_index": 45.0,
        "burn_probability_index": 50.0,
        "hazard_severity_index": 50.0,
        "access_exposure_index": 35.0,
        "burn_probability": 0.5,
        "wildfire_hazard": 3.0,
        "slope": 12.0,
        "fuel_model": 55.0,
        "canopy_cover": 45.0,
        "historic_fire_distance": 2.0,
        "wildland_distance": 250.0,
        "environmental_layer_status": {
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        "data_sources": ["benchmark-fixture"],
        "assumptions": [],
        "structure_ring_metrics": {
            "ring_0_5_ft": {"vegetation_density": 35.0},
            "ring_5_30_ft": {"vegetation_density": 45.0},
            "ring_30_100_ft": {"vegetation_density": 50.0},
            "ring_100_300_ft": {"vegetation_density": 55.0},
        },
        "property_level_context": {
            "footprint_used": True,
            "footprint_status": "used",
            "fallback_mode": "footprint",
            "ring_metrics": {
                "ring_0_5_ft": {"vegetation_density": 35.0},
                "ring_5_30_ft": {"vegetation_density": 45.0},
                "ring_30_100_ft": {"vegetation_density": 50.0},
                "ring_100_300_ft": {"vegetation_density": 55.0},
            },
            "region_status": "prepared",
            "region_id": "benchmark_region",
        },
        "region_context": {"region_status": "prepared", "region_id": "benchmark_region"},
        "hazard_context": {"status": "observed", "source": "benchmark_whp"},
        "moisture_context": {"status": "observed", "source": "benchmark_gridmet", "dryness_index": 45.0},
        "historical_fire_context": {"status": "observed", "source": "benchmark_mtbs"},
        "access_context": {"status": "ok", "source": "benchmark_osm"},
        "layer_coverage_audit": [],
        "coverage_summary": {},
    }


def build_wildfire_context(overrides: dict[str, Any] | None = None) -> WildfireContext:
    payload = default_wildfire_context_dict()
    if overrides:
        payload = _deep_update(payload, overrides)
    return WildfireContext(**payload)


def load_benchmark_pack(pack_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(pack_path or DEFAULT_PACK_PATH).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    validate_benchmark_pack(payload)
    return payload


def validate_benchmark_pack(pack: dict[str, Any]) -> None:
    if not isinstance(pack, dict):
        raise ValueError("Benchmark pack must be a JSON object.")
    if not isinstance(pack.get("benchmark_pack_version"), str):
        raise ValueError("benchmark_pack_version is required.")
    scenarios = pack.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list.")
    seen_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Each scenario must be an object.")
        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Every scenario must include scenario_id.")
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)
        payload = scenario.get("input_payload")
        if not isinstance(payload, dict) or not payload.get("address"):
            raise ValueError(f"Scenario {scenario_id} missing input_payload.address.")
        expected = scenario.get("expected")
        if expected is not None and not isinstance(expected, dict):
            raise ValueError(f"Scenario {scenario_id} expected must be an object if provided.")
    for block_name in ("relative_assertions", "monotonic_assertions"):
        block = pack.get(block_name, [])
        if not isinstance(block, list):
            raise ValueError(f"{block_name} must be a list.")


@contextmanager
def patched_runtime_inputs(
    *,
    latitude: float,
    longitude: float,
    geocode_source: str,
    context: WildfireContext,
):
    import backend.main as app_main  # lazy import to avoid module cycle

    original_geocode = app_main.geocoder.geocode
    original_collect = app_main.wildfire_data.collect_context
    app_main.geocoder.geocode = lambda _addr: (latitude, longitude, geocode_source)
    app_main.wildfire_data.collect_context = lambda _lat, _lon: context
    try:
        yield
    finally:
        app_main.geocoder.geocode = original_geocode
        app_main.wildfire_data.collect_context = original_collect


def _resolve_ruleset(ruleset_id: str | None) -> UnderwritingRuleset:
    import backend.main as app_main  # lazy import to avoid module cycle

    try:
        return app_main._get_ruleset_or_default(ruleset_id or "default")
    except Exception:
        return UnderwritingRuleset(
            ruleset_id=ruleset_id or "default",
            ruleset_name="Default Carrier Profile",
            ruleset_version="1.0",
            ruleset_description="Fallback ruleset for benchmark harness.",
            config={},
        )


def _ledger_contributions(result: AssessmentResult) -> dict[str, dict[str, float]]:
    families = {
        "site_hazard_score": result.score_evidence_ledger.site_hazard_score,
        "home_ignition_vulnerability_score": result.score_evidence_ledger.home_ignition_vulnerability_score,
        "insurance_readiness_score": result.score_evidence_ledger.insurance_readiness_score,
        "wildfire_risk_score": result.score_evidence_ledger.wildfire_risk_score,
    }
    out: dict[str, dict[str, float]] = {}
    for family, rows in families.items():
        out[family] = {row.factor_key: float(row.contribution) for row in rows}
    return out


def _scenario_snapshot(
    scenario: dict[str, Any],
    result: AssessmentResult,
    debug_payload: dict[str, Any],
) -> dict[str, Any]:
    warnings = [
        n
        for n in (result.scoring_notes or [])
        if any(
            token in n.lower()
            for token in [
                "fallback",
                "missing",
                "unavailable",
                "outside",
                "not configured",
                "partial",
            ]
        )
    ]
    governance = {
        "scoring_model_version": result.model_version,
        "ruleset_version": result.ruleset_version,
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "benchmark_pack_version": BENCHMARK_PACK_VERSION,
        "region_data_version": (result.property_level_context or {}).get("region_manifest_path"),
    }
    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario.get("description", ""),
        "profile_tags": scenario.get("profile_tags", []),
        "scores": {
            "wildfire_risk_score": result.wildfire_risk_score,
            "site_hazard_score": result.site_hazard_score,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "insurance_readiness_score": result.insurance_readiness_score,
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "site_hazard_score_available": result.site_hazard_score_available,
            "home_ignition_vulnerability_score_available": result.home_ignition_vulnerability_score_available,
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "risk_band": _risk_band(result.wildfire_risk_score),
        },
        "confidence": {
            "confidence_score": result.confidence_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
        },
        "assessment_status": result.assessment_status,
        "assessment_blockers": list(result.assessment_blockers or []),
        "readiness_blockers": list(result.readiness_blockers or []),
        "top_risk_drivers": list(result.top_risk_drivers or []),
        "warnings": warnings,
        "coverage_summary": result.coverage_summary.model_dump(),
        "evidence_quality_summary": result.evidence_quality_summary.model_dump(),
        "score_evidence_ledger_summary": _ledger_contributions(result),
        "fallback_mode": (result.property_level_context or {}).get("fallback_mode"),
        "footprint_used": bool((result.property_level_context or {}).get("footprint_used")),
        "debug_excerpt": {
            "eligibility": debug_payload.get("eligibility", {}),
            "coverage": debug_payload.get("coverage", {}),
        },
        "governance": governance,
    }


def _evaluate_scenario_expectations(snapshot: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add_check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    if not expected:
        return checks

    expected_band = expected.get("risk_band")
    if isinstance(expected_band, str):
        actual_band = snapshot["scores"]["risk_band"]
        add_check("risk_band", actual_band == expected_band, f"expected={expected_band} actual={actual_band}")

    expected_confidence = expected.get("confidence_tiers")
    if isinstance(expected_confidence, list) and expected_confidence:
        actual = snapshot["confidence"]["confidence_tier"]
        add_check("confidence_tier", actual in expected_confidence, f"expected_in={expected_confidence} actual={actual}")

    expected_restriction = expected.get("use_restrictions")
    if isinstance(expected_restriction, list) and expected_restriction:
        actual = snapshot["confidence"]["use_restriction"]
        add_check("use_restriction", actual in expected_restriction, f"expected_in={expected_restriction} actual={actual}")

    expected_status = expected.get("assessment_status_in")
    if isinstance(expected_status, list) and expected_status:
        actual = snapshot.get("assessment_status")
        add_check("assessment_status", actual in expected_status, f"expected_in={expected_status} actual={actual}")

    expected_fallback = expected.get("fallback_mode")
    if isinstance(expected_fallback, str):
        actual = snapshot.get("fallback_mode")
        add_check("fallback_mode", actual == expected_fallback, f"expected={expected_fallback} actual={actual}")

    for keyword in expected.get("driver_keywords", []) or []:
        haystack = " ".join(snapshot.get("top_risk_drivers", [])).lower()
        add_check(
            f"driver_keyword:{keyword}",
            str(keyword).lower() in haystack,
            f"keyword={keyword}",
        )
    for keyword in expected.get("warning_keywords", []) or []:
        haystack = " ".join(snapshot.get("warnings", [])).lower()
        add_check(
            f"warning_keyword:{keyword}",
            str(keyword).lower() in haystack,
            f"keyword={keyword}",
        )
    return checks


def _evaluate_pairwise_assertions(
    *,
    assertions: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    assertion_type: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for assertion in assertions:
        left_id = str(assertion.get("left") or "")
        right_id = str(assertion.get("right") or "")
        metric = str(assertion.get("metric") or "")
        op = str(assertion.get("op") or "")
        assertion_id = str(assertion.get("assertion_id") or f"{assertion_type}:{left_id}:{metric}:{op}:{right_id}")
        left_snapshot = snapshots.get(left_id)
        right_snapshot = snapshots.get(right_id)
        if not left_snapshot or not right_snapshot:
            results.append(
                {
                    "assertion_id": assertion_id,
                    "type": assertion_type,
                    "passed": False,
                    "detail": "Scenario missing for assertion.",
                }
            )
            continue
        left_val = _path_get(left_snapshot, metric)
        right_val = _path_get(right_snapshot, metric)
        passed = _compare_values(left_val, op, right_val)
        results.append(
            {
                "assertion_id": assertion_id,
                "type": assertion_type,
                "left": left_id,
                "right": right_id,
                "metric": metric,
                "op": op,
                "left_value": left_val,
                "right_value": right_val,
                "passed": passed,
                "detail": f"{left_id}.{metric}={left_val} {op} {right_id}.{metric}={right_val}",
            }
        )
    return results


def run_benchmark_suite(
    *,
    pack_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import backend.main as app_main  # lazy import to avoid module cycle

    pack = load_benchmark_pack(pack_path)
    scenarios = list(pack.get("scenarios", []))
    scenario_results: list[dict[str, Any]] = []
    snapshots_by_id: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        location = scenario.get("location", {}) if isinstance(scenario.get("location"), dict) else {}
        lat = float(location.get("lat", 39.7392))
        lon = float(location.get("lon", -104.9903))
        geocode_source = str(location.get("geocode_source", "benchmark-fixture"))
        context = build_wildfire_context(scenario.get("context") if isinstance(scenario.get("context"), dict) else {})
        payload = AddressRequest.model_validate(scenario.get("input_payload", {}))
        ruleset = _resolve_ruleset(payload.ruleset_id)
        org_id = str(scenario.get("organization_id") or "default_org")

        with patched_runtime_inputs(
            latitude=lat,
            longitude=lon,
            geocode_source=geocode_source,
            context=context,
        ):
            result, debug_payload = app_main._run_assessment(
                payload,
                organization_id=org_id,
                ruleset=ruleset,
            )
        snapshot = _scenario_snapshot(scenario, result, debug_payload)
        checks = _evaluate_scenario_expectations(snapshot, scenario.get("expected", {}))
        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "snapshot": snapshot,
                "expectation_checks": checks,
                "all_expectations_passed": all(c.get("passed", False) for c in checks) if checks else True,
            }
        )
        snapshots_by_id[scenario["scenario_id"]] = snapshot

    relative = _evaluate_pairwise_assertions(
        assertions=list(pack.get("relative_assertions", [])),
        snapshots=snapshots_by_id,
        assertion_type="relative_ordering",
    )
    monotonic = _evaluate_pairwise_assertions(
        assertions=list(pack.get("monotonic_assertions", [])),
        snapshots=snapshots_by_id,
        assertion_type="monotonic",
    )

    scenario_failures = [
        r["scenario_id"]
        for r in scenario_results
        if not bool(r.get("all_expectations_passed", False))
    ]
    assertion_failures = [
        r["assertion_id"]
        for r in (relative + monotonic)
        if not bool(r.get("passed", False))
    ]

    model_versions = sorted(
        {
            str((r.get("snapshot") or {}).get("governance", {}).get("scoring_model_version") or MODEL_VERSION)
            for r in scenario_results
        }
    )
    ruleset_versions = sorted(
        {
            str((r.get("snapshot") or {}).get("governance", {}).get("ruleset_version") or "unknown")
            for r in scenario_results
        }
    )
    region_versions = sorted(
        {
            str((r.get("snapshot") or {}).get("governance", {}).get("region_data_version") or "unknown")
            for r in scenario_results
        }
    )

    artifact = {
        "generated_at": _now_iso(),
        "benchmark_pack_version": str(pack.get("benchmark_pack_version") or BENCHMARK_PACK_VERSION),
        "factor_schema_version": str(pack.get("factor_schema_version") or FACTOR_SCHEMA_VERSION),
        "governance": {
            "scoring_model_versions": model_versions,
            "ruleset_versions": ruleset_versions,
            "factor_schema_version": str(pack.get("factor_schema_version") or FACTOR_SCHEMA_VERSION),
            "benchmark_pack_version": str(pack.get("benchmark_pack_version") or BENCHMARK_PACK_VERSION),
            "region_data_versions": region_versions,
        },
        "scenario_results": scenario_results,
        "relative_assertions": relative,
        "monotonic_assertions": monotonic,
        "summary": {
            "scenario_count": len(scenario_results),
            "scenario_failures": scenario_failures,
            "assertion_failures": assertion_failures,
            "passed": (not scenario_failures) and (not assertion_failures),
        },
    }

    out_dir = Path(output_dir or DEFAULT_RESULTS_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = out_dir / f"benchmark_run_{stamp}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def compare_benchmark_artifacts(
    *,
    baseline_artifact: dict[str, Any],
    current_artifact: dict[str, Any],
    score_drift_threshold: float = 8.0,
    confidence_drift_threshold: float = 8.0,
    contribution_drift_threshold: float = 10.0,
) -> dict[str, Any]:
    baseline_map = {
        str(item.get("scenario_id")): item.get("snapshot", {})
        for item in baseline_artifact.get("scenario_results", [])
        if isinstance(item, dict)
    }
    current_map = {
        str(item.get("scenario_id")): item.get("snapshot", {})
        for item in current_artifact.get("scenario_results", [])
        if isinstance(item, dict)
    }
    shared_ids = sorted(set(baseline_map.keys()) & set(current_map.keys()))
    scenario_deltas: list[dict[str, Any]] = []
    material_flags: list[str] = []

    for scenario_id in shared_ids:
        old = baseline_map[scenario_id]
        new = current_map[scenario_id]
        score_delta = {
            key: (
                (new.get("scores", {}).get(key) - old.get("scores", {}).get(key))
                if isinstance(new.get("scores", {}).get(key), (int, float))
                and isinstance(old.get("scores", {}).get(key), (int, float))
                else None
            )
            for key in [
                "wildfire_risk_score",
                "site_hazard_score",
                "home_ignition_vulnerability_score",
                "insurance_readiness_score",
            ]
        }
        confidence_delta = None
        old_conf = old.get("confidence", {}).get("confidence_score")
        new_conf = new.get("confidence", {}).get("confidence_score")
        if isinstance(old_conf, (int, float)) and isinstance(new_conf, (int, float)):
            confidence_delta = new_conf - old_conf

        old_blockers = sorted(set((old.get("assessment_blockers") or []) + (old.get("readiness_blockers") or [])))
        new_blockers = sorted(set((new.get("assessment_blockers") or []) + (new.get("readiness_blockers") or [])))
        old_warnings = sorted(set(old.get("warnings") or []))
        new_warnings = sorted(set(new.get("warnings") or []))

        contribution_delta_max = 0.0
        old_ledgers = old.get("score_evidence_ledger_summary", {})
        new_ledgers = new.get("score_evidence_ledger_summary", {})
        for family in set(old_ledgers.keys()) | set(new_ledgers.keys()):
            old_rows = old_ledgers.get(family, {}) if isinstance(old_ledgers, dict) else {}
            new_rows = new_ledgers.get(family, {}) if isinstance(new_ledgers, dict) else {}
            for factor_key in set(old_rows.keys()) | set(new_rows.keys()):
                old_val = old_rows.get(factor_key)
                new_val = new_rows.get(factor_key)
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    contribution_delta_max = max(contribution_delta_max, abs(float(new_val) - float(old_val)))

        exceeded = False
        if any(abs(v) > score_drift_threshold for v in score_delta.values() if isinstance(v, (int, float))):
            exceeded = True
        if isinstance(confidence_delta, (int, float)) and abs(confidence_delta) > confidence_drift_threshold:
            exceeded = True
        if contribution_delta_max > contribution_drift_threshold:
            exceeded = True

        if exceeded:
            material_flags.append(scenario_id)

        scenario_deltas.append(
            {
                "scenario_id": scenario_id,
                "score_deltas": score_delta,
                "confidence_delta": confidence_delta,
                "changed_blockers": old_blockers != new_blockers,
                "changed_warnings": old_warnings != new_warnings,
                "contribution_delta_max": round(contribution_delta_max, 3),
                "material_drift": exceeded,
            }
        )

    baseline_versions = set((baseline_artifact.get("governance") or {}).get("scoring_model_versions") or [])
    current_versions = set((current_artifact.get("governance") or {}).get("scoring_model_versions") or [])

    if not material_flags:
        summary_label = "no material drift"
    elif baseline_versions != current_versions and len(material_flags) <= max(1, len(shared_ids) // 2):
        summary_label = "expected drift due to changed factor weights"
    else:
        summary_label = "unexpected drift requiring review"

    return {
        "baseline_artifact_path": baseline_artifact.get("artifact_path"),
        "current_artifact_path": current_artifact.get("artifact_path"),
        "shared_scenarios": shared_ids,
        "scenario_deltas": scenario_deltas,
        "thresholds": {
            "score_drift_threshold": score_drift_threshold,
            "confidence_drift_threshold": confidence_drift_threshold,
            "contribution_drift_threshold": contribution_drift_threshold,
        },
        "material_drift_count": len(material_flags),
        "material_drift_scenarios": material_flags,
        "summary": summary_label,
    }


def _single_assessment_sanity_checks(result: AssessmentResult) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    facts = result.property_facts or {}
    def_space = facts.get("defensible_space_ft")
    roof = str(facts.get("roof_type") or "").lower()
    vent = str(facts.get("vent_type") or "").lower()

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    _add(
        "missing_evidence_not_low_risk",
        not (
            result.assessment_status == "insufficient_data"
            and bool(result.wildfire_risk_score_available)
            and isinstance(result.wildfire_risk_score, (int, float))
            and float(result.wildfire_risk_score) <= 20.0
        ),
        "Insufficient-data runs should not present as computed low risk.",
    )

    if isinstance(def_space, (int, float)) and isinstance(result.home_ignition_vulnerability_score, (int, float)):
        if float(def_space) <= 5:
            _add(
                "very_low_defensible_space_not_low_vulnerability",
                float(result.home_ignition_vulnerability_score) >= 20.0,
                "Very low defensible space should not map to low vulnerability.",
            )
        if float(def_space) >= 30:
            _add(
                "strong_defensible_space_not_extreme_vulnerability",
                float(result.home_ignition_vulnerability_score) <= 90.0,
                "Strong defensible space should not map to extreme vulnerability absent other severe signals.",
            )

    if roof in {"class a", "metal", "tile", "composite"} and "ember" in vent and isinstance(
        result.home_ignition_vulnerability_score, (int, float)
    ):
        _add(
            "hardened_envelope_consistency",
            float(result.home_ignition_vulnerability_score) < 90.0,
            "Hardened roof+vent should not produce near-max vulnerability by itself.",
        )

    return checks


def build_benchmark_hints_for_assessment(result: AssessmentResult, pack_path: str | Path | None = None) -> dict[str, Any]:
    try:
        pack = load_benchmark_pack(pack_path)
        scenarios = pack.get("scenarios", [])
    except Exception:
        scenarios = []
        pack = {"benchmark_pack_version": BENCHMARK_PACK_VERSION}

    result_band = _risk_band(result.wildfire_risk_score)
    fallback_mode = str((result.property_level_context or {}).get("fallback_mode") or "unknown")
    matched: list[dict[str, Any]] = []
    for scenario in scenarios:
        expected = scenario.get("expected") if isinstance(scenario.get("expected"), dict) else {}
        expected_band = expected.get("risk_band")
        expected_fallback = expected.get("fallback_mode")
        if expected_band and expected_band != result_band:
            continue
        if expected_fallback and expected_fallback != fallback_mode:
            continue
        matched.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "description": scenario.get("description", ""),
            }
        )
    matched = matched[:3]

    sanity_checks = _single_assessment_sanity_checks(result)
    return {
        "benchmark_pack_version": str(pack.get("benchmark_pack_version") or BENCHMARK_PACK_VERSION),
        "scoring_model_version": result.model_version,
        "resembles_scenarios": matched,
        "benchmark_style_sanity_checks": sanity_checks,
        "all_sanity_checks_passed": all(c.get("passed", False) for c in sanity_checks) if sanity_checks else True,
    }


def load_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["artifact_path"] = str(p)
    return payload
