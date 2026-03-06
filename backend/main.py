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
    AssumptionsBlock,
    ConfidenceBlock,
    Coordinates,
    EnvironmentalFactors,
    FactorBreakdown,
    RiskDrivers,
    RiskScores,
)
from backend.risk_engine import RiskEngine
from backend.version import MODEL_VERSION
from backend.wildfire_data import WildfireDataClient

app = FastAPI(title="WildfireRisk Advisor API", version="0.5.0")

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


def _build_assumption_tracking(payload: AddressRequest, assumptions_used: list[str], data_sources: list[str]) -> AssumptionsBlock:
    observed_inputs: dict[str, object] = {"address": payload.address}
    inferred_inputs: dict[str, object] = {}
    missing_inputs: list[str] = []

    attrs = payload.attributes

    if attrs.roof_type is None:
        inferred_inputs["roof_type"] = "composition_shingle_baseline"
        missing_inputs.append("roof_type")
    else:
        observed_inputs["roof_type"] = attrs.roof_type

    if attrs.vent_type is None:
        inferred_inputs["vent_type"] = "standard"
        missing_inputs.append("vent_type")
    else:
        observed_inputs["vent_type"] = attrs.vent_type

    if attrs.defensible_space_ft is None:
        inferred_inputs["defensible_space_ft"] = 15
        missing_inputs.append("defensible_space_ft")
    else:
        observed_inputs["defensible_space_ft"] = attrs.defensible_space_ft

    if attrs.construction_year is None:
        inferred_inputs["construction_year"] = "pre_2008_proxy"
        missing_inputs.append("construction_year")
    else:
        observed_inputs["construction_year"] = attrs.construction_year

    source_blob = " ".join(data_sources).lower()
    if "burn probability raster" not in source_blob:
        missing_inputs.append("burn_probability_layer")
    if "wildfire hazard severity raster" not in source_blob:
        missing_inputs.append("hazard_severity_layer")
    if "fuel model raster" not in source_blob:
        missing_inputs.append("fuel_model_layer")
    if "historical fire perimeter recurrence" not in source_blob:
        missing_inputs.append("historical_fire_perimeter_layer")

    return AssumptionsBlock(
        observed_inputs=observed_inputs,
        inferred_inputs=inferred_inputs,
        missing_inputs=sorted(set(missing_inputs)),
        assumptions_used=assumptions_used,
    )


def _build_confidence(assumptions: AssumptionsBlock) -> ConfidenceBlock:
    important_missing = len(
        [
            m
            for m in assumptions.missing_inputs
            if m
            in {
                "roof_type",
                "vent_type",
                "defensible_space_ft",
                "burn_probability_layer",
                "hazard_severity_layer",
                "fuel_model_layer",
                "historical_fire_perimeter_layer",
            }
        ]
    )
    inferred_count = len(assumptions.inferred_inputs)
    external_fail_count = sum(
        1
        for note in assumptions.assumptions_used
        if any(k in note.lower() for k in ["unavailable", "failed", "fallback"])
    )

    confidence = 100.0
    confidence -= important_missing * 8.0
    confidence -= inferred_count * 4.0
    confidence -= external_fail_count * 7.0
    confidence -= max(0, len(assumptions.assumptions_used) - external_fail_count) * 2.0
    confidence = max(0.0, min(100.0, round(confidence, 1)))

    low_confidence_flags: list[str] = []
    if important_missing >= 3:
        low_confidence_flags.append("Multiple important inputs or layers are missing")
    if external_fail_count > 0:
        low_confidence_flags.append("At least one external provider or layer fetch failed")
    if inferred_count >= 3:
        low_confidence_flags.append("Several core property attributes were inferred")
    if confidence < 70:
        low_confidence_flags.append("Overall confidence below recommended underwriting threshold")

    data_completeness_score = round(
        max(0.0, min(100.0, 100.0 - (len(assumptions.missing_inputs) * 6.5))),
        1,
    )

    return ConfidenceBlock(
        confidence_score=confidence,
        data_completeness_score=data_completeness_score,
        assumption_count=len(assumptions.assumptions_used),
        low_confidence_flags=low_confidence_flags,
        requires_user_verification=confidence < 70.0 or len(low_confidence_flags) > 0,
    )


def _build_top_risk_drivers(factors: EnvironmentalFactors, factor_breakdown: FactorBreakdown) -> list[str]:
    candidates = [
        ("steep terrain slope", factors.slope_topography),
        ("dense vegetation/fuel near the property", factors.vegetation_fuel),
        ("high burn probability", factors.burn_probability),
        ("elevated wildfire hazard severity", factors.hazard_severity),
        ("historical wildfire activity nearby", factors.historical_fire_recurrence),
        ("close proximity to wildland vegetation", factors.fuel_proximity),
        ("dry fuel/moisture conditions", factors.drought_moisture),
        ("high access-related exposure", factor_breakdown.access_risk),
    ]
    return [name for name, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:3]]


def _build_top_protective_factors(payload: AddressRequest, factors: EnvironmentalFactors) -> list[str]:
    protective: list[str] = []
    attrs = payload.attributes

    if attrs.roof_type and attrs.roof_type.lower() in {"class a", "metal", "tile", "composite"}:
        protective.append("class A or equivalent fire-rated roof material")
    if attrs.vent_type and "ember" in attrs.vent_type.lower():
        protective.append("ember-resistant venting")
    if attrs.defensible_space_ft is not None and attrs.defensible_space_ft >= 30:
        protective.append("defensible space > 30 ft")
    if factors.fuel_proximity <= 35:
        protective.append("limited adjacent wildland fuel proximity")
    if factors.canopy_density <= 35:
        protective.append("lower canopy density in home ignition zone")

    if not protective:
        protective.append("no strong structural or site-level protective factors detected")
    return protective[:3]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(payload: AddressRequest) -> AssessmentResult:
    pre_assumptions: list[str] = []

    try:
        lat, lon, geocode_source = geocoder.geocode(payload.address)
    except Exception:
        lat, lon = risk_engine.geocode_stub(payload.address)
        geocode_source = "Deterministic geocode fallback"
        pre_assumptions.append("Geocoding provider unavailable; fallback coordinates used.")

    context = wildfire_data.collect_context(lat, lon)
    risk = risk_engine.score(payload.attributes, lat, lon, context)
    plan = build_mitigation_plan(payload.attributes, risk.total_score, context)

    mitigation_credit = sum(a.estimated_risk_reduction for a in plan[:3]) * 0.45
    readiness = max(0.0, min(100.0, round(100 - risk.total_score + mitigation_credit, 1)))

    factors = EnvironmentalFactors(
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

    breakdown = FactorBreakdown(
        environmental_risk=risk.drivers.environmental,
        structural_risk=risk.drivers.structural,
        access_risk=risk.drivers.access_exposure,
    )

    all_assumptions = pre_assumptions + risk.assumptions
    all_sources = [geocode_source] + context.data_sources
    assumptions_block = _build_assumption_tracking(payload, all_assumptions, all_sources)
    confidence_block = _build_confidence(assumptions_block)

    top_risk_drivers = _build_top_risk_drivers(factors, breakdown)
    top_protective_factors = _build_top_protective_factors(payload, factors)

    explanation_summary = (
        f"Risk is driven primarily by {', '.join(top_risk_drivers[:2])}. "
        f"Key protective factors: {', '.join(top_protective_factors[:2])}."
    )

    risk_scores = RiskScores(
        wildfire_risk_score=risk.total_score,
        insurance_readiness_score=readiness,
    )
    coordinates = Coordinates(latitude=lat, longitude=lon)

    result = AssessmentResult(
        assessment_id=str(uuid4()),
        address=payload.address,
        coordinates=coordinates,
        risk_scores=risk_scores,
        factor_breakdown=breakdown,
        mitigation_recommendations=plan,
        assumptions=assumptions_block,
        confidence=confidence_block,
        top_risk_drivers=top_risk_drivers,
        top_protective_factors=top_protective_factors,
        explanation_summary=explanation_summary,
        model_version=MODEL_VERSION,
        # Backward compatibility fields
        latitude=lat,
        longitude=lon,
        wildfire_risk_score=risk.total_score,
        insurance_readiness_score=readiness,
        risk_drivers=RiskDrivers(
            environmental=risk.drivers.environmental,
            structural=risk.drivers.structural,
            access_exposure=risk.drivers.access_exposure,
        ),
        environmental_factors=factors,
        assumptions_used=all_assumptions,
        data_sources=all_sources,
        mitigation_plan=plan,
        explanation=explanation_summary,
    )

    store.save(result)
    return result


@app.get("/report/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def get_report(assessment_id: str) -> AssessmentResult:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result
