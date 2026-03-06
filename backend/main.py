from __future__ import annotations

from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import require_api_key
from backend.database import AssessmentStore
from backend.geocoding import Geocoder
from backend.mitigation import build_mitigation_plan
from backend.models import (
    AddressRequest,
    AssessmentResult,
    DataQuality,
    EnvironmentalFactors,
)
from backend.risk_engine import RiskEngine
from backend.wildfire_data import WildfireDataClient

app = FastAPI(title="WildfireRisk Advisor API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

risk_engine = RiskEngine()
geocoder = Geocoder()
wildfire_data = WildfireDataClient()
store = AssessmentStore()


def _build_data_quality(assumptions: list[str], data_sources: list[str]) -> DataQuality:
    low_confidence_flags: list[str] = []

    for a in assumptions:
        low = a.lower()
        if "missing" in low or "unavailable" in low or "fallback" in low or "default" in low:
            low_confidence_flags.append(a)

    expected_sources = {
        "Burn probability raster",
        "Wildfire hazard severity raster",
        "Slope raster",
        "Aspect raster",
        "Fuel model raster",
        "Canopy density raster",
        "Moisture/fuel dryness raster",
        "Distance to wildland vegetation (derived)",
        "Historical fire perimeter recurrence",
    }

    present = len(expected_sources.intersection(set(data_sources)))
    completeness = round((present / len(expected_sources)) * 100.0, 1)
    requires_user_verification = completeness < 70.0 or len(low_confidence_flags) >= 3

    return DataQuality(
        data_completeness_score=completeness,
        assumption_count=len(assumptions),
        low_confidence_flags=sorted(set(low_confidence_flags)),
        requires_user_verification=requires_user_verification,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(payload: AddressRequest) -> AssessmentResult:
    assumptions = []

    try:
        lat, lon, geocode_source = geocoder.geocode(payload.address)
    except Exception:
        lat, lon = risk_engine.geocode_stub(payload.address)
        geocode_source = "Deterministic geocode fallback"
        assumptions.append("Geocoding provider unavailable; fallback coordinates used.")

    context = wildfire_data.collect_context(lat, lon)
    risk = risk_engine.score(payload.attributes, lat, lon, context)
    plan = build_mitigation_plan(payload.attributes, risk.total_score, context)

    mitigation_credit = sum(a.estimated_risk_reduction for a in plan[:3]) * 0.45
    readiness = max(0.0, min(100.0, round(100 - risk.total_score + mitigation_credit, 1)))

    environmental_factors = EnvironmentalFactors(
        burn_probability=context.burn_probability_index,
        hazard_severity=context.hazard_severity_index,
        slope_topography=context.slope_index,
        aspect_exposure=context.aspect_index,
        vegetation_fuel=context.fuel_index,
        drought_moisture=context.moisture_index,
        canopy_density=context.canopy_index,
        fuel_proximity=context.wildland_distance_index,
        historical_fire_recurrence=context.historic_fire_index,
    )

    all_assumptions = assumptions + risk.assumptions
    all_sources = [geocode_source] + context.data_sources
    data_quality = _build_data_quality(all_assumptions, all_sources)

    factor_items = [
        ("fuel proximity", environmental_factors.fuel_proximity),
        ("burn probability", environmental_factors.burn_probability),
        ("vegetation/fuel", environmental_factors.vegetation_fuel),
        ("hazard severity", environmental_factors.hazard_severity),
        ("drought/moisture", environmental_factors.drought_moisture),
        ("slope/topography", environmental_factors.slope_topography),
        ("canopy density", environmental_factors.canopy_density),
        ("historical fire recurrence", environmental_factors.historical_fire_recurrence),
    ]
    top_factors = sorted(factor_items, key=lambda x: x[1], reverse=True)[:3]
    factor_text = ", ".join(f"{name} ({score:.1f})" for name, score in top_factors)

    mitigation_text = ", ".join(f"{m.action} [{m.related_factor}]" for m in plan[:2]) if plan else "no immediate mitigations"

    explanation = (
        f"Wildfire risk score is {risk.total_score}. "
        f"Highest contributing factors: {factor_text}. "
        f"Top actions tied to these drivers: {mitigation_text}. "
        f"Data completeness is {data_quality.data_completeness_score:.1f}%"
        f" with {data_quality.assumption_count} assumptions."
    )

    result = AssessmentResult(
        assessment_id=str(uuid4()),
        address=payload.address,
        latitude=lat,
        longitude=lon,
        wildfire_risk_score=risk.total_score,
        insurance_readiness_score=readiness,
        risk_drivers=risk.drivers,
        environmental_factors=environmental_factors,
        data_quality=data_quality,
        assumptions_used=all_assumptions,
        data_sources=all_sources,
        mitigation_plan=plan,
        explanation=explanation,
    )
    store.save(result)
    return result


@app.get("/report/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def get_report(assessment_id: str) -> AssessmentResult:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result
