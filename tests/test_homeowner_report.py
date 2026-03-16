from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.models import MitigationAction
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
    ):
        assert key in report

    assert report["score_summary"]["wildfire_risk_score"] is not None
    assert report["score_summary"]["overall_wildfire_risk"] is not None
    assert report["score_summary"]["home_hardening_readiness"] is not None
    assert report["score_summary"]["insurance_readiness_score"] is not None
    assert isinstance(report["top_recommended_actions"], list)
    assert isinstance(report["prioritized_mitigation_actions"], list)
    assert isinstance(report["top_risk_drivers_detailed"], list)
    assert isinstance(report["confidence_summary"], dict)
    assert len(report["top_recommended_actions"]) <= 3
    assert "blockers" in report["home_hardening_readiness_summary"]
    assert "summary" in report["home_hardening_readiness_summary"]
    assert isinstance(report["defensible_space_summary"]["zone_findings"], list)
    assert report.get("professional_debug_metadata") is None

    pdf_res = client.get(f"/report/{assessed['assessment_id']}/homeowner/pdf")
    assert pdf_res.status_code == 200
    assert "application/pdf" in pdf_res.headers.get("content-type", "")
    assert pdf_res.content.startswith(b"%PDF-1.4")
    assert "attachment; filename=\"wildfire_homeowner_report_" in pdf_res.headers.get("content-disposition", "")


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
    assert report["defensible_space_summary"]["analysis_status"] in {"partial", "unavailable", "complete"}

    debug_res = client.get(
        f"/report/{assessed['assessment_id']}/homeowner?include_professional_debug_metadata=true"
    )
    assert debug_res.status_code == 200
    debug_report = debug_res.json()
    assert isinstance(debug_report.get("professional_debug_metadata"), dict)
    assert "coverage_summary" in debug_report["professional_debug_metadata"]


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
