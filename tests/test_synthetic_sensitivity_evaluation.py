from __future__ import annotations

from scripts.run_synthetic_sensitivity_evaluation import (
    SCENARIO_PROFILES,
    _build_synthetic_event_records,
    _summarize_response,
    apply_synthetic_profile,
)


def _base_raw_features() -> dict[str, float]:
    return {
        "ring_0_5_ft_vegetation_density": 32.0,
        "ring_5_30_ft_vegetation_density": 36.0,
        "near_structure_vegetation_0_5_pct": 28.0,
        "canopy_adjacency_proxy_pct": 24.0,
        "vegetation_continuity_proxy_pct": 21.0,
        "nearest_high_fuel_patch_distance_ft": 220.0,
        "burn_probability": 0.34,
        "slope": 11.0,
        "fuel_model": 33.0,
        "canopy_cover": 30.0,
        "wildland_distance_m": 650.0,
    }


def _base_transformed_features() -> dict[str, float]:
    return {
        "burn_probability_index": 34.0,
        "slope_index": 24.0,
        "fuel_index": 33.0,
        "canopy_index": 30.0,
        "wildland_distance_index": 62.0,
    }


def test_apply_synthetic_profile_is_directional() -> None:
    raw = _base_raw_features()
    transformed = _base_transformed_features()

    baseline_raw, baseline_tx = apply_synthetic_profile(
        base_id="p1",
        profile="baseline_observed",
        raw_features=raw,
        transformed_features=transformed,
    )
    veg_raw, veg_tx = apply_synthetic_profile(
        base_id="p1",
        profile="vegetation_up",
        raw_features=raw,
        transformed_features=transformed,
    )
    slope_raw, slope_tx = apply_synthetic_profile(
        base_id="p1",
        profile="slope_up",
        raw_features=raw,
        transformed_features=transformed,
    )
    fuel_raw, fuel_tx = apply_synthetic_profile(
        base_id="p1",
        profile="fuel_near",
        raw_features=raw,
        transformed_features=transformed,
    )
    mitigation_raw, mitigation_tx = apply_synthetic_profile(
        base_id="p1",
        profile="mitigation_low",
        raw_features=raw,
        transformed_features=transformed,
    )

    assert float(veg_raw["ring_0_5_ft_vegetation_density"]) > float(
        baseline_raw["ring_0_5_ft_vegetation_density"]
    )
    assert float(veg_raw["ring_5_30_ft_vegetation_density"]) > float(
        baseline_raw["ring_5_30_ft_vegetation_density"]
    )
    assert float(slope_tx["slope_index"]) > float(baseline_tx["slope_index"])
    assert float(fuel_tx["fuel_index"]) > float(baseline_tx["fuel_index"])
    assert float(fuel_tx["wildland_distance_index"]) > float(
        baseline_tx["wildland_distance_index"]
    )
    assert float(mitigation_raw["ring_0_5_ft_vegetation_density"]) < float(
        baseline_raw["ring_0_5_ft_vegetation_density"]
    )
    assert float(mitigation_tx["burn_probability_index"]) < float(
        baseline_tx["burn_probability_index"]
    )
    assert float(mitigation_raw["wildland_distance_m"]) > float(
        baseline_raw["wildland_distance_m"]
    )
    assert float(mitigation_tx["wildland_distance_index"]) < float(
        baseline_tx["wildland_distance_index"]
    )
    assert float(slope_raw["slope"]) > float(baseline_raw["slope"])


def test_build_synthetic_event_records_labels_and_metadata() -> None:
    base_rows = [
        {
            "property_event_id": "evtA::p123",
            "feature": {
                "record_id": "f123",
                "latitude": 47.0,
                "longitude": -120.5,
                "address_text": "123 Main St, Town, ST",
            },
            "event": {"event_id": "evtA", "event_name": "Event A", "event_date": "2020-09-01"},
            "outcome": {"damage_label": "minor_damage"},
            "feature_snapshot": {
                "raw_feature_vector": _base_raw_features(),
                "transformed_feature_vector": _base_transformed_features(),
            },
        }
    ]

    records = _build_synthetic_event_records(base_rows)
    assert len(records) == len(SCENARIO_PROFILES)

    profiles = {
        ((row.get("source_metadata") or {}).get("synthetic_profile")): row
        for row in records
    }
    assert set(profiles.keys()) == set(SCENARIO_PROFILES)
    assert all(str(row.get("record_id")).startswith("evtA::p123__syn__") for row in records)
    assert all(row.get("source_name") == "synthetic_sensitivity" for row in records)
    assert all(((row.get("source_metadata") or {}).get("synthetic_variation") is True) for row in records)

    assert profiles["vegetation_up"]["outcome_label"] == "major_damage"
    assert profiles["slope_up"]["outcome_label"] == "major_damage"
    assert profiles["fuel_near"]["outcome_label"] == "major_damage"
    assert profiles["combined_high"]["outcome_label"] == "major_damage"
    assert profiles["mitigation_low"]["outcome_label"] == "no_damage"
    assert profiles["baseline_observed"]["outcome_label"] == "minor_damage"


def test_summarize_response_reports_directionality_checks() -> None:
    scored_records = [
        {"record_id": "p1__syn__baseline_observed", "scores": {"wildfire_risk_score": 50.0}},
        {"record_id": "p1__syn__vegetation_up", "scores": {"wildfire_risk_score": 62.0}},
        {"record_id": "p1__syn__slope_up", "scores": {"wildfire_risk_score": 59.0}},
        {"record_id": "p1__syn__fuel_near", "scores": {"wildfire_risk_score": 58.0}},
        {"record_id": "p1__syn__combined_high", "scores": {"wildfire_risk_score": 70.0}},
        {"record_id": "p1__syn__mitigation_low", "scores": {"wildfire_risk_score": 39.0}},
    ]

    summary = _summarize_response(scored_records)
    checks = summary.get("directionality_checks") or {}
    rates = summary.get("directionality_pass_rates") or {}

    assert summary.get("base_property_count") == 1
    assert checks["vegetation_up_ge_baseline"]["pass"] == 1
    assert checks["slope_up_ge_baseline"]["pass"] == 1
    assert checks["fuel_near_ge_baseline"]["pass"] == 1
    assert checks["combined_high_ge_baseline"]["pass"] == 1
    assert checks["mitigation_low_le_baseline"]["pass"] == 1
    assert rates["vegetation_up_ge_baseline"] == 1.0
    assert rates["mitigation_low_le_baseline"] == 1.0
