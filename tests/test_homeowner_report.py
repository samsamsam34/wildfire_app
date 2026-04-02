from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.homeowner_report import export_homeowner_report
from backend.models import HomeownerPrioritizedAction, MitigationAction
from backend.wildfire_data import WildfireContext


client = TestClient(app_main.app)


def _ctx(
    env: float,
    wildland: float,
    historic: float,
    *,
    ring_metrics: dict[str, dict[str, float | None]] | None = None,
    environmental_layer_status: dict[str, str] | None = None,
) -> WildfireContext:
    ring_metrics = ring_metrics or {}
    environmental_layer_status = environmental_layer_status or {
        "burn_probability": "ok",
        "hazard": "ok",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }
    return WildfireContext(
        environmental_index=env,
        slope_index=env,
        aspect_index=50.0,
        fuel_index=env,
        moisture_index=env,
        canopy_index=env,
        wildland_distance_index=wildland,
        historic_fire_index=historic,
        burn_probability_index=env,
        hazard_severity_index=env,
        burn_probability=env,
        wildfire_hazard=env,
        slope=env,
        fuel_model=env,
        canopy_cover=env,
        historic_fire_distance=2.0,
        wildland_distance=100.0,
        environmental_layer_status=environmental_layer_status,
        data_sources=["burn_probability", "fuel", "canopy", "slope"],
        assumptions=[],
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": bool(ring_metrics),
            "footprint_status": "used" if ring_metrics else "not_found",
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
            "region_id": "missoula_pilot",
        },
    )


def _setup(monkeypatch, tmp_path: Path, context: WildfireContext) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _address: (46.8721, -113.9940, "test-geocoder"))
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: context)
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "homeowner_report.db")))


def _assess_payload(address: str) -> dict:
    return {
        "address": address,
        "attributes": {
            "roof_type": "class_a_asphalt_composition",
            "vent_type": "ember_resistant_vents",
            "defensible_space_ft": 28,
        },
        "confirmed_fields": ["roof_type", "vent_type", "defensible_space_ft"],
        "audience": "homeowner",
    }


def _run_assessment(address: str) -> dict:
    response = client.post("/risk/assess", json=_assess_payload(address))
    assert response.status_code == 200
    return response.json()


def test_homeowner_report_and_pdf_generate_for_complete_assessment(monkeypatch, tmp_path: Path):
    context = _ctx(
        env=44.0,
        wildland=36.0,
        historic=24.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 20.0, "coverage_pct": 18.0, "fuel_presence_proxy": 15.0},
            "zone_5_30_ft": {"vegetation_density": 42.0, "coverage_pct": 38.0, "fuel_presence_proxy": 35.0},
            "zone_30_100_ft": {"vegetation_density": 56.0, "coverage_pct": 54.0, "fuel_presence_proxy": 52.0},
        },
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("201 W Front St, Missoula, MT 59802")

    report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
    assert report_res.status_code == 200
    report = report_res.json()

    for key in (
        "first_screen",
        "headline_risk_summary",
        "top_risk_drivers",
        "prioritized_actions",
        "ranked_actions",
        "most_impactful_actions",
        "what_to_do_first",
        "limitations_notice",
        "report_header",
        "property_summary",
        "score_summary",
        "key_risk_drivers",
        "top_risk_drivers_detailed",
        "defensible_space_summary",
        "top_recommended_actions",
        "prioritized_mitigation_actions",
        "mitigation_plan",
        "home_hardening_readiness_summary",
        "insurance_readiness_summary",
        "confidence_summary",
        "confidence_and_limitations",
        "metadata",
        "specificity_summary",
    ):
        assert key in report

    assert report["score_summary"]["wildfire_risk_score"] is not None
    assert report["score_summary"]["overall_wildfire_risk"] is not None
    assert report["score_summary"]["home_hardening_readiness"] is not None
    assert report["score_summary"]["insurance_readiness_score"] is not None
    assert isinstance(report["top_recommended_actions"], list)
    assert isinstance(report["top_risk_drivers"], list)
    assert len(report["top_risk_drivers"]) <= 3
    assert isinstance(report["prioritized_actions"], list)
    assert len(report["prioritized_actions"]) <= 3
    assert isinstance(report["ranked_actions"], list)
    assert len(report["ranked_actions"]) <= 5
    assert isinstance(report["most_impactful_actions"], list)
    assert len(report["most_impactful_actions"]) <= 2
    assert all(bool(row.get("most_impactful")) for row in report["most_impactful_actions"])
    assert isinstance(report["what_to_do_first"], dict)
    assert isinstance(report["limitations_notice"], str)
    assert isinstance(report["prioritized_mitigation_actions"], list)
    assert isinstance(report["top_risk_drivers_detailed"], list)
    assert isinstance(report["confidence_summary"], dict)
    assert report["specificity_summary"]["specificity_tier"] == assessed["specificity_summary"]["specificity_tier"]
    assert isinstance(report["specificity_summary"]["comparison_allowed"], bool)
    first_screen = report.get("first_screen") or {}
    assert list(first_screen.keys()) == [
        "specificity_summary",
        "headline_risk_summary",
        "top_risk_drivers",
        "top_actions",
        "what_to_do_first",
        "limitations_note",
    ]
    assert len(first_screen.get("top_risk_drivers") or []) <= 3
    assert len(first_screen.get("top_actions") or []) <= 3
    assert len(str(first_screen.get("limitations_note") or "")) <= 320
    assert len(report["top_recommended_actions"]) <= 3
    for action in report["top_recommended_actions"]:
        assert isinstance(action.get("why_this_matters"), str)
        assert isinstance(action.get("what_it_reduces"), str)
        assert action.get("expected_effect") in {"small", "moderate", "significant"}
    for action in report["mitigation_plan"]:
        assert isinstance(action.get("why_this_matters"), str)
        assert isinstance(action.get("what_it_reduces"), str)
        assert action.get("expected_effect") in {"small", "moderate", "significant"}
    for action in report["ranked_actions"]:
        assert isinstance(action.get("prioritization_score"), float)
        assert isinstance(action.get("ranking_basis"), dict)
    assert "blockers" in report["home_hardening_readiness_summary"]
    assert "summary" in report["home_hardening_readiness_summary"]
    assert isinstance(report["defensible_space_summary"]["zone_findings"], list)
    assert report.get("professional_debug_metadata") is None
    trust_summary = (report.get("confidence_and_limitations") or {}).get("trust_summary") or {}
    assert "differentiation_mode" in trust_summary
    assert "neighborhood_differentiation_confidence" in trust_summary
    assert "property_specific_feature_count" in trust_summary
    assert "proxy_feature_count" in trust_summary
    assert "defaulted_feature_count" in trust_summary
    assert "regional_feature_count" in trust_summary


def test_homeowner_report_surfaces_mostly_regional_differentiation_mode(monkeypatch, tmp_path: Path):
    context = _ctx(
        env=52.0,
        wildland=49.0,
        historic=41.0,
        ring_metrics={},
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
    )
    context.burn_probability_index = None
    context.burn_probability = None
    context.hazard_severity_index = None
    context.wildfire_hazard = None
    context.property_level_context.update(
        {
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "parcel_geometry": None,
            "near_structure_vegetation_0_5_pct": None,
            "canopy_adjacency_proxy_pct": None,
            "vegetation_continuity_proxy_pct": None,
        }
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("299 Regional Estimate Rd, Missoula, MT 59802")
    report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
    assert report_res.status_code == 200
    report = report_res.json()
    specificity = report.get("specificity_summary") or {}
    assert specificity.get("specificity_tier") == "regional_estimate"
    assert specificity.get("comparison_allowed") is False
    assert "nearby homes may appear similar" in str(specificity.get("what_this_means") or "").lower()
    first_screen = report.get("first_screen") or {}
    assert "nearby homes may appear similar" in str(first_screen.get("limitations_note") or "").lower()
    assert "may have" in str(first_screen.get("headline_risk_summary") or "").lower()
    trust_summary = (report.get("confidence_and_limitations") or {}).get("trust_summary") or {}
    assert trust_summary.get("differentiation_mode") == "mostly_regional"
    assert float(trust_summary.get("neighborhood_differentiation_confidence") or 0.0) <= 40.0
    assert isinstance(trust_summary.get("differentiation_summary"), str)
    assert trust_summary.get("geometry_specificity_limited") is True
    geometry_summary = trust_summary.get("geometry_resolution_summary") or {}
    assert geometry_summary.get("ring_generation_mode") == "point_annulus_fallback"
    assert geometry_summary.get("footprint_match_status") in {"none", "ambiguous", "provider_unavailable", "error"}
    low_diff = trust_summary.get("low_differentiation_explanation") or {}
    assert low_diff.get("applies") is True
    assert isinstance(low_diff.get("why_nearby_properties_may_appear_similar"), str)
    assert isinstance(low_diff.get("what_would_make_this_more_property_specific") or [], list)

    pdf_res = client.get(f"/report/{assessed['assessment_id']}/homeowner/pdf")
    assert pdf_res.status_code == 200
    assert "application/pdf" in pdf_res.headers.get("content-type", "")
    assert pdf_res.content.startswith(b"%PDF-1.4")
    assert b"Specificity: Regional estimate" in pdf_res.content
    assert "attachment; filename=\"wildfire_homeowner_report_" in pdf_res.headers.get("content-disposition", "")


def test_homeowner_report_surfaces_address_level_specificity(monkeypatch, tmp_path: Path):
    context = _ctx(env=49.0, wildland=44.0, historic=35.0, ring_metrics={})
    context.access_exposure_index = 18.0
    context.access_context = {"status": "ok"}
    context.property_level_context.update(
        {
            "parcel_geometry": {
                "type": "Polygon",
                "coordinates": [[[-113.9943, 46.8719], [-113.9939, 46.8719], [-113.9939, 46.8722], [-113.9943, 46.8722], [-113.9943, 46.8719]]],
            },
            "region_property_specific_readiness": "address_level_only",
            "footprint_used": False,
            "fallback_mode": "point_based",
            "ring_metrics": {},
        }
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("422 Address Level Example Ave, Missoula, MT 59802")

    report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
    assert report_res.status_code == 200
    report = report_res.json()
    specificity = report.get("specificity_summary") or {}
    assert specificity.get("specificity_tier") == "address_level"
    assert "address-level" in str(specificity.get("headline") or "").lower()
    assert isinstance(specificity.get("comparison_allowed"), bool)


def test_homeowner_report_surfaces_fallback_limitations_and_optional_debug_block(monkeypatch, tmp_path: Path):
    context = _ctx(
        env=52.0,
        wildland=58.0,
        historic=62.0,
        ring_metrics={},
        environmental_layer_status={
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "ok",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "missing",
        },
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("221 Fallback Path, Missoula, MT 59802")

    report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
    assert report_res.status_code == 200
    report = report_res.json()
    limitations = report["confidence_and_limitations"].get("limitations") or []
    assert len(limitations) >= 1
    assert report["confidence_and_limitations"]["confidence_tier"] in {"high", "moderate", "low", "preliminary"}
    assert "fallback_decisions" not in report["confidence_and_limitations"]
    assert report["defensible_space_summary"]["analysis_status"] in {"partial", "unavailable", "complete"}

    debug_res = client.get(
        f"/report/{assessed['assessment_id']}/homeowner?include_professional_debug_metadata=true"
    )
    assert debug_res.status_code == 200
    debug_report = debug_res.json()
    assert isinstance(debug_report.get("professional_debug_metadata"), dict)
    assert "coverage_summary" in debug_report["professional_debug_metadata"]
    assert isinstance((debug_report.get("confidence_and_limitations") or {}).get("fallback_decisions"), list)


def test_homeowner_report_handles_unavailable_scores_and_long_text_deterministically(monkeypatch, tmp_path: Path):
    context = _ctx(env=47.0, wildland=50.0, historic=55.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("100 Stable Output Drive, Missoula, MT 59802")

    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    long_reason = (
        "Complete vegetation clearance and ember-resistant upgrades around the structure perimeter, "
        "including noncombustible zone creation, vent hardening, and recurring maintenance verification "
        "to reduce near-structure ignition pathways and improve readiness evidence."
    )

    patched = original.model_copy(deep=True)
    patched.wildfire_risk_score = None
    patched.wildfire_risk_score_available = False
    patched.insurance_readiness_score = None
    patched.insurance_readiness_score_available = False
    patched.home_hardening_readiness = None
    patched.home_hardening_readiness_score_available = False
    patched.address = (
        "9999 Extremely Long Address Lane With Many Unit Descriptors and Additional Context, "
        "Missoula, MT 59802-1234"
    )
    patched.mitigation_plan = [
        MitigationAction(
            title="Long Form Vegetation and Ember Mitigation Program",
            reason=long_reason,
            impacted_submodels=["defensible_space_risk"],
            impacted_readiness_factors=["defensible_space"],
            estimated_risk_reduction_band="high",
            estimated_readiness_improvement_band="high",
            priority=1,
            insurer_relevance="recommended",
        )
    ]

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id: patched)

    report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["score_summary"]["wildfire_risk_band"] == "unavailable"
    assert report["score_summary"]["home_hardening_readiness_band"] == "unavailable"
    assert report["score_summary"]["insurance_readiness_band"] == "unavailable"

    pdf_res_a = client.get(f"/report/{assessed['assessment_id']}/homeowner/pdf")
    pdf_res_b = client.get(f"/report/{assessed['assessment_id']}/homeowner/pdf")
    assert pdf_res_a.status_code == 200
    assert pdf_res_b.status_code == 200
    assert pdf_res_a.content == pdf_res_b.content
    assert len(pdf_res_a.content) > 800


def test_homeowner_report_homeowner_focused_fields_across_confidence_tiers(monkeypatch, tmp_path: Path):
    context = _ctx(
        env=58.0,
        wildland=62.0,
        historic=44.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 18.0, "coverage_pct": 16.0, "fuel_presence_proxy": 14.0},
            "zone_5_30_ft": {"vegetation_density": 41.0, "coverage_pct": 37.0, "fuel_presence_proxy": 34.0},
            "zone_30_100_ft": {"vegetation_density": 55.0, "coverage_pct": 52.0, "fuel_presence_proxy": 50.0},
        },
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("701 Confidence Tier Ln, Missoula, MT 59802")

    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    base_actions = [
        HomeownerPrioritizedAction(
            action="Remove leaves and debris within 5 feet of the home",
            explanation="Reduces immediate ignition pathways next to the structure.",
            impact_level="high",
            effort_level="low",
            estimated_cost_band="low",
            timeline="now",
            priority=1,
        ),
        HomeownerPrioritizedAction(
            action="Thin dense vegetation in the 5-30 foot zone",
            explanation="Lowers heat and ember exposure around the structure.",
            impact_level="medium",
            effort_level="medium",
            estimated_cost_band="medium",
            timeline="this_season",
            priority=2,
        ),
        HomeownerPrioritizedAction(
            action="Maintain tree spacing and prune branches near the roofline",
            explanation="Reduces fire spread into the home envelope.",
            impact_level="low",
            effort_level="medium",
            estimated_cost_band="medium",
            timeline="later",
            priority=3,
        ),
    ]

    for tier in ("high", "moderate", "low", "preliminary"):
        patched = original.model_copy(deep=True)
        patched.confidence_tier = tier
        patched.confidence_summary.missing_data = ["roof_type"] if tier in {"low", "preliminary"} else []
        patched.confidence_summary.fallback_assumptions = ["regional proxy"] if tier in {"low", "preliminary"} else []
        patched.fallback_weight_fraction = 0.65 if tier in {"low", "preliminary"} else 0.0
        patched.assessment_diagnostics.fallback_decisions = []
        patched.top_risk_drivers = [
            "dense vegetation close to the home",
            "high ember exposure",
            "slope/topography amplification",
            "fuel model pressure near structure",
            "limited defensible space within 30 feet",
        ]
        patched.prioritized_mitigation_actions = list(base_actions)

        monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _patched=patched: _patched)
        report_res = client.get(f"/report/{assessed['assessment_id']}/homeowner")
        assert report_res.status_code == 200
        report = report_res.json()

        assert "headline_risk_summary" in report
        assert report["headline_risk_summary"]
        assert len(report.get("top_risk_drivers") or []) <= 3
        assert all("fuel model" not in str(row).lower() for row in (report.get("top_risk_drivers") or []))
        assert isinstance((report.get("top_risk_drivers") or [None])[0], str)
        prioritized = report.get("prioritized_actions") or []
        assert 1 <= len(prioritized) <= 3
        first = prioritized[0]
        assert first.get("effort_level") in {"low", "medium", "high"}
        assert isinstance(first.get("estimated_benefit"), str) and first.get("estimated_benefit")
        assert isinstance(first.get("why_this_matters"), str) and first.get("why_this_matters")
        assert isinstance(first.get("what_it_reduces"), str) and first.get("what_it_reduces")
        assert first.get("expected_effect") in {"small", "moderate", "significant"}
        assert isinstance(report.get("what_to_do_first"), dict)
        assert str((report.get("what_to_do_first") or {}).get("action") or "").strip()
        assert isinstance(report.get("limitations_notice"), str)
        assert all(
            isinstance(row.get("why_this_matters"), str) and row.get("why_this_matters")
            and isinstance(row.get("what_it_reduces"), str) and row.get("what_it_reduces")
            and row.get("expected_effect") in {"small", "moderate", "significant"}
            for row in prioritized
        )

        if tier in {"low", "preliminary"}:
            assert "may have" in report["headline_risk_summary"].lower()
            assert "estimated" in report["limitations_notice"].lower() or "missing" in report["limitations_notice"].lower()
            assert "may be contributing to risk" in str((report.get("top_risk_drivers") or [""])[0]).lower()
            assert "may help reduce" in str(first.get("why_this_matters")).lower()
        if tier == "moderate":
            assert "appears to have" in report["headline_risk_summary"].lower()
            assert "appears to increase risk" in str((report.get("top_risk_drivers") or [""])[0]).lower()
            assert "appears to help reduce" in str(first.get("why_this_matters")).lower()
        if tier == "high":
            headline = str(report["headline_risk_summary"]).lower()
            assert headline.startswith("your home has ") or headline.startswith("your home appears to have ")
            driver_text = str((report.get("top_risk_drivers") or [""])[0]).lower()
            assert ("is a major risk factor" in driver_text) or ("appears to increase risk" in driver_text)
            why_text = str(first.get("why_this_matters")).lower()
            assert ("helps reduce" in why_text) or ("appears to help reduce" in why_text)
            assert "may help reduce" not in why_text


def test_homeowner_report_degraded_data_produces_more_cautious_tone(monkeypatch, tmp_path: Path):
    context = _ctx(
        env=61.0,
        wildland=58.0,
        historic=49.0,
        ring_metrics={
            "zone_0_5_ft": {"vegetation_density": 24.0, "coverage_pct": 20.0, "fuel_presence_proxy": 19.0},
            "zone_5_30_ft": {"vegetation_density": 46.0, "coverage_pct": 42.0, "fuel_presence_proxy": 38.0},
            "zone_30_100_ft": {"vegetation_density": 60.0, "coverage_pct": 57.0, "fuel_presence_proxy": 54.0},
        },
    )
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("888 Tone Shift Ave, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    high = original.model_copy(deep=True)
    high.confidence_tier = "high"
    high.confidence_summary.missing_data = []
    high.confidence_summary.fallback_assumptions = []
    high.fallback_weight_fraction = 0.0
    high.assessment_diagnostics.fallback_decisions = []
    high.top_risk_drivers = ["dense vegetation close to the home"]
    high.prioritized_mitigation_actions = [
        HomeownerPrioritizedAction(
            action="Clear vegetation within 5 feet",
            explanation="Creates a non-combustible area next to the home.",
            impact_level="high",
            effort_level="low",
            estimated_cost_band="low",
            timeline="now",
            priority=1,
        )
    ]

    low = high.model_copy(deep=True)
    low.confidence_tier = "low"
    low.confidence_summary.missing_data = ["roof_type", "vent_type", "defensible_space_ft", "structure_geometry"]
    low.confidence_summary.fallback_assumptions = ["regional vegetation proxy", "structure proxy"]
    low.fallback_weight_fraction = 0.75
    low.assessment_diagnostics.fallback_decisions = [{"fallback_type": "derived_proxy"}]

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=high: _obj)
    high_report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=low: _obj)
    low_report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()

    high_headline = str(high_report.get("headline_risk_summary", "")).lower()
    assert ("your home has" in high_headline) or ("your home appears to have" in high_headline)
    high_driver = str((high_report.get("top_risk_drivers") or [""])[0]).lower()
    assert ("is a major risk factor" in high_driver) or ("appears to increase risk" in high_driver)
    high_why = str(((high_report.get("prioritized_actions") or [{}])[0]).get("why_this_matters") or "").lower()
    assert ("helps reduce" in high_why) or ("appears to help reduce" in high_why)

    assert "your home may have" in str(low_report.get("headline_risk_summary", "")).lower()
    assert "may be contributing to risk" in str((low_report.get("top_risk_drivers") or [""])[0]).lower()
    assert "may help reduce" in str(((low_report.get("prioritized_actions") or [{}])[0]).get("why_this_matters") or "").lower()


def test_mitigation_ranking_prioritizes_near_structure_actions(monkeypatch, tmp_path: Path):
    context = _ctx(env=57.0, wildland=46.0, historic=31.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("120 Proximity Rank Ct, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    patched = original.model_copy(deep=True)
    patched.confidence_tier = "high"
    patched.confidence_summary.missing_data = []
    patched.confidence_summary.fallback_assumptions = []
    patched.fallback_weight_fraction = 0.0
    patched.assessment_diagnostics.fallback_decisions = []
    patched.prioritized_mitigation_actions = [
        HomeownerPrioritizedAction(
            action="Clear vegetation within 0-5 ft of the home",
            explanation="Removes ignition pathways right next to the structure.",
            impact_level="medium",
            effort_level="low",
            data_confidence="high",
            priority=2,
        ),
        HomeownerPrioritizedAction(
            action="Reduce vegetation around 50 ft from the home",
            explanation="Lowers fuel farther from the structure.",
            impact_level="medium",
            effort_level="low",
            data_confidence="high",
            priority=1,
        ),
    ]

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=patched: _obj)
    report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()
    ranked = report.get("ranked_actions") or []
    assert len(ranked) >= 2
    assert "0-5" in str(ranked[0].get("action") or "")
    assert float(ranked[0].get("proximity_score") or 0.0) > float(ranked[1].get("proximity_score") or 0.0)


def test_mitigation_ranking_prefers_high_confidence_over_low_confidence(monkeypatch, tmp_path: Path):
    context = _ctx(env=59.0, wildland=47.0, historic=33.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("121 Confidence Rank Ct, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    patched = original.model_copy(deep=True)
    patched.confidence_tier = "high"
    patched.confidence_summary.missing_data = []
    patched.confidence_summary.fallback_assumptions = []
    patched.fallback_weight_fraction = 0.0
    patched.assessment_diagnostics.fallback_decisions = []
    patched.prioritized_mitigation_actions = [
        HomeownerPrioritizedAction(
            action="Improve defensible space within 30 ft",
            explanation="Reduces flame exposure near the structure.",
            impact_level="medium",
            effort_level="low",
            data_confidence="high",
            priority=2,
        ),
        HomeownerPrioritizedAction(
            action="Major fuel reduction in nearby area",
            explanation="High-impact recommendation but based on limited local detail.",
            impact_level="high",
            effort_level="low",
            data_confidence="low",
            priority=1,
        ),
    ]

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=patched: _obj)
    report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()
    ranked = report.get("ranked_actions") or []
    assert len(ranked) >= 2
    assert "30 ft" in str(ranked[0].get("action") or "")
    assert float(ranked[0].get("data_confidence_score") or 0.0) > float(ranked[1].get("data_confidence_score") or 0.0)


def test_homeowner_report_composes_existing_outputs_without_mitigation_fallback_recompute(monkeypatch, tmp_path: Path):
    context = _ctx(env=62.0, wildland=52.0, historic=40.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("300 Compose Only Way, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    patched = original.model_copy(deep=True)
    patched.top_risk_drivers = [
        "dense vegetation close to the home",
        "high ember exposure",
        "limited defensible space within 30 feet",
        "close proximity to wildland fuels",
    ]
    patched.prioritized_mitigation_actions = [
        HomeownerPrioritizedAction(action="Action A", impact_level="high", effort_level="low", data_confidence="high", priority=1),
        HomeownerPrioritizedAction(action="Action B", impact_level="medium", effort_level="low", data_confidence="high", priority=2),
        HomeownerPrioritizedAction(action="Action C", impact_level="low", effort_level="medium", data_confidence="high", priority=3),
        HomeownerPrioritizedAction(action="Action D", impact_level="low", effort_level="medium", data_confidence="high", priority=4),
    ]
    patched.mitigation_plan = [
        MitigationAction(
            title="Fallback Only Action (Should Not Enter Top 3)",
            reason="This is present only in mitigation_plan.",
            impacted_submodels=["home_ignition_vulnerability"],
            estimated_risk_reduction_band="high",
            estimated_readiness_improvement_band="high",
            priority=1,
            insurer_relevance="recommended",
        )
    ]

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=patched: _obj)
    report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()

    assert len(report.get("top_risk_drivers") or []) == 3
    prioritized = report.get("prioritized_actions") or []
    assert len(prioritized) == 3
    action_names = [str(row.get("action") or "") for row in prioritized]
    assert "Fallback Only Action (Should Not Enter Top 3)" not in action_names
    first_action = str((report.get("what_to_do_first") or {}).get("action") or "")
    assert first_action in action_names


def test_export_homeowner_report_generates_clean_structured_output_across_confidence_tiers(monkeypatch, tmp_path: Path):
    context = _ctx(env=53.0, wildland=45.0, historic=30.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("55 Shareable Report Dr, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    for tier in ("high", "moderate", "low", "preliminary"):
        patched = original.model_copy(deep=True)
        patched.confidence_tier = tier
        patched.confidence_summary.missing_data = ["roof_type"] if tier in {"low", "preliminary"} else []
        patched.confidence_summary.fallback_assumptions = ["regional proxy"] if tier in {"low", "preliminary"} else []
        patched.fallback_weight_fraction = 0.7 if tier in {"low", "preliminary"} else 0.0
        patched.assessment_diagnostics.fallback_decisions = [{"fallback_type": "derived_proxy"}] if tier in {"low", "preliminary"} else []
        patched.prioritized_mitigation_actions = [
            HomeownerPrioritizedAction(
                action="Clear vegetation within 5 ft of the home",
                explanation="Removes ignition pathways next to the structure.",
                impact_level="high",
                effort_level="low",
                data_confidence="high" if tier in {"high", "moderate"} else "low",
                priority=1,
            )
        ]

        exported = export_homeowner_report(patched, output_format="structured")
        assert isinstance(exported, dict)
        for required in (
            "first_screen",
            "headline_risk_summary",
            "top_risk_drivers",
            "prioritized_actions",
            "what_to_do_first",
            "confidence_summary",
            "trust_summary",
            "improve_your_result",
            "limitations_notice",
            "disclaimer",
        ):
            assert required in exported
        assert isinstance(exported.get("top_risk_drivers"), list)
        assert isinstance(exported.get("prioritized_actions"), list)
        assert isinstance(exported.get("confidence_summary"), dict)
        assert isinstance(exported.get("trust_summary"), dict)
        assert isinstance(exported.get("improve_your_result"), dict)
        assert list((exported.get("first_screen") or {}).keys()) == [
            "specificity_summary",
            "headline_risk_summary",
            "top_risk_drivers",
            "top_actions",
            "what_to_do_first",
            "limitations_note",
        ]


def test_export_homeowner_report_low_confidence_includes_clear_limitations_and_pdf(monkeypatch, tmp_path: Path):
    context = _ctx(env=56.0, wildland=49.0, historic=32.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("56 Shareable Limitations Ln, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    low = original.model_copy(deep=True)
    low.confidence_tier = "low"
    low.confidence_summary.missing_data = ["roof_type", "vent_type", "structure_geometry"]
    low.confidence_summary.fallback_assumptions = ["point-based geometry fallback"]
    low.fallback_weight_fraction = 0.8
    low.assessment_diagnostics.fallback_decisions = [{"fallback_type": "derived_proxy"}]

    exported = export_homeowner_report(low, output_format="structured")
    assert isinstance(exported, dict)
    confidence_summary = exported.get("confidence_summary") or {}
    assert confidence_summary.get("confidence_tier") in {"low", "preliminary"}
    assert "estimated" in str(exported.get("limitations_notice") or "").lower() or "missing" in str(exported.get("limitations_notice") or "").lower()
    assert "not a guarantee" in str(exported.get("disclaimer") or "").lower()
    first_screen = exported.get("first_screen") or {}
    assert len(first_screen.get("top_actions") or []) >= 1
    assert str((first_screen.get("what_to_do_first") or {}).get("action") or "").strip()

    pdf_bytes = export_homeowner_report(low, output_format="pdf")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-1.4")


def test_homeowner_report_demotes_optional_calibration_metadata_in_consumer_view(monkeypatch, tmp_path: Path):
    context = _ctx(env=54.0, wildland=43.0, historic=29.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("77 Optional Calibration Ln, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    calibrated = original.model_copy(deep=True)
    calibrated.calibration_applied = True
    calibrated.calibration_status = "applied"
    calibrated.calibration_method = "logistic"
    calibrated.calibrated_damage_likelihood = 0.41
    calibrated.empirical_damage_likelihood_proxy = 0.41
    calibrated.empirical_loss_likelihood_proxy = 0.41
    calibrated.calibration_version = "0.3.0"

    monkeypatch.setattr(app_main.store, "get", lambda _assessment_id, _obj=calibrated: _obj)
    report = client.get(f"/report/{assessed['assessment_id']}/homeowner").json()

    first_screen = report.get("first_screen") or {}
    homeowner_text = " ".join(
        [
            str(first_screen.get("headline_risk_summary") or ""),
            " ".join(str(x) for x in (first_screen.get("top_risk_drivers") or [])),
            " ".join(str((x or {}).get("action") or "") for x in (first_screen.get("top_actions") or [])),
        ]
    ).lower()
    assert "calibrat" not in homeowner_text

    score_summary = report.get("score_summary") or {}
    assert score_summary.get("calibration_status") == "applied"
    assert score_summary.get("calibrated_damage_likelihood") == 0.41
    assert "optional/additive" in str(score_summary.get("public_outcome_calibration_note") or "").lower()

    metadata = report.get("metadata") or {}
    optional_calibration = metadata.get("optional_public_outcome_calibration") or {}
    assert optional_calibration.get("available") is True
    assert "additive context only" in str(optional_calibration.get("summary") or "").lower()
    assert "not be interpreted as insurer underwriting" in str(optional_calibration.get("caveat") or "").lower()

    assert report.get("professional_debug_metadata") is None

    debug_report = client.get(
        f"/report/{assessed['assessment_id']}/homeowner?include_professional_debug_metadata=true"
    ).json()
    gov_note = ((debug_report.get("professional_debug_metadata") or {}).get("public_outcome_governance_note") or {})
    assert "internal governance metadata" in str(gov_note.get("summary") or "").lower()
    docs = gov_note.get("docs") or []
    assert "docs/public_outcome_validation.md" in docs
    assert "docs/public_outcome_calibration.md" in docs


def test_export_homeowner_report_includes_optional_calibration_block_without_foregrounding(monkeypatch, tmp_path: Path):
    context = _ctx(env=51.0, wildland=40.0, historic=26.0)
    _setup(monkeypatch, tmp_path, context)
    assessed = _run_assessment("78 Optional Calibration Export Ln, Missoula, MT 59802")
    original = app_main.store.get(assessed["assessment_id"])
    assert original is not None

    calibrated = original.model_copy(deep=True)
    calibrated.calibration_applied = True
    calibrated.calibration_status = "applied"
    calibrated.calibrated_damage_likelihood = 0.36
    calibrated.empirical_damage_likelihood_proxy = 0.36
    calibrated.calibration_version = "0.3.0"

    exported = export_homeowner_report(calibrated, output_format="structured")
    assert isinstance(exported, dict)
    assert "headline_risk_summary" in exported
    assert "top_risk_drivers" in exported
    assert "prioritized_actions" in exported
    optional_calibration = exported.get("optional_public_outcome_calibration") or {}
    assert optional_calibration.get("available") is True
    assert optional_calibration.get("calibrated_public_outcome_probability") == 0.36
    assert "additive context only" in str(optional_calibration.get("summary") or "").lower()
