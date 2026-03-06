from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.auth import require_api_key
from backend.database import AssessmentStore
from backend.geocoding import Geocoder
from backend.mitigation import build_mitigation_plan
from backend.models import (
    AddressRequest,
    AssessmentListItem,
    AssessmentResult,
    AssumptionsBlock,
    ConfidenceBlock,
    Coordinates,
    EnvironmentalFactors,
    FactorBreakdown,
    MitigationAction,
    PropertyAttributes,
    ReadinessFactor,
    ReassessmentRequest,
    ReportExport,
    RiskScores,
    SimulationDelta,
    SimulationRequest,
    SimulationResult,
    SubmodelScore,
    WeightedContribution,
)
from backend.risk_engine import RiskComputation, RiskEngine
from backend.scoring_config import load_scoring_config
from backend.version import MODEL_VERSION
from backend.wildfire_data import WildfireDataClient

app = FastAPI(title="WildfireRisk Advisor API", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

scoring_config = load_scoring_config()
risk_engine = RiskEngine(scoring_config)
geocoder = Geocoder()
wildfire_data = WildfireDataClient()
store = AssessmentStore()

ACCESS_PROVISIONAL_NOTE = "Access risk is provisional and not included in total scoring."

SUBMODEL_ALIASES = {
    "ember_exposure": "ember_exposure_risk",
    "flame_contact_exposure": "flame_contact_risk",
    "topography_risk": "slope_topography_risk",
    "home_hardening_risk": "structure_vulnerability_risk",
}

CANONICAL_SUBMODELS = [
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "ember_exposure_risk",
    "flame_contact_risk",
    "historic_fire_risk",
    "structure_vulnerability_risk",
    "defensible_space_risk",
]

ENVIRONMENTAL_SUBMODELS = [
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "ember_exposure_risk",
    "flame_contact_risk",
    "historic_fire_risk",
]

STRUCTURAL_SUBMODELS = [
    "structure_vulnerability_risk",
    "defensible_space_risk",
]

CORE_FACT_FIELDS = {"roof_type", "vent_type", "defensible_space_ft", "construction_year"}


def _attributes_to_dict(attrs: PropertyAttributes) -> Dict[str, object]:
    return {k: v for k, v in attrs.model_dump().items() if v is not None}


def _merge_attributes(base: PropertyAttributes, overrides: PropertyAttributes) -> PropertyAttributes:
    merged = base.model_dump(exclude_none=True)
    merged.update(overrides.model_dump(exclude_none=True))
    return PropertyAttributes.model_validate(merged)


def _build_assumption_tracking(payload: AddressRequest, assumptions_used: list[str], data_sources: list[str]) -> AssumptionsBlock:
    observed_inputs: dict[str, object] = {"address": payload.address}
    observed_inputs.update(_attributes_to_dict(payload.attributes))

    confirmed_inputs: dict[str, object] = {
        field: observed_inputs[field]
        for field in payload.confirmed_fields
        if field in observed_inputs
    }

    inferred_inputs: dict[str, object] = {}
    missing_inputs: list[str] = []

    attrs = payload.attributes

    if attrs.roof_type is None:
        inferred_inputs["roof_type"] = "composition_shingle_baseline"
        missing_inputs.append("roof_type")
    if attrs.vent_type is None:
        inferred_inputs["vent_type"] = "standard"
        missing_inputs.append("vent_type")
    if attrs.defensible_space_ft is None:
        inferred_inputs["defensible_space_ft"] = 15
        missing_inputs.append("defensible_space_ft")
    if attrs.construction_year is None:
        inferred_inputs["construction_year"] = "pre_2008_proxy"
        missing_inputs.append("construction_year")

    if attrs.siding_type is None:
        missing_inputs.append("siding_type")
    if attrs.window_type is None:
        missing_inputs.append("window_type")
    if attrs.vegetation_condition is None:
        missing_inputs.append("vegetation_condition")

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
        confirmed_inputs=confirmed_inputs,
        observed_inputs=observed_inputs,
        inferred_inputs=inferred_inputs,
        missing_inputs=sorted(set(missing_inputs)),
        assumptions_used=assumptions_used,
    )


def _build_confidence(assumptions: AssumptionsBlock) -> ConfidenceBlock:
    important_missing = len([m for m in assumptions.missing_inputs if m in CORE_FACT_FIELDS or m.endswith("_layer")])
    inferred_count = len(assumptions.inferred_inputs)
    confirmed_core_count = len([k for k in assumptions.confirmed_inputs if k in CORE_FACT_FIELDS])
    external_fail_count = sum(
        1
        for note in assumptions.assumptions_used
        if any(k in note.lower() for k in ["unavailable", "failed", "fallback"])
    )

    confidence = 100.0
    confidence -= important_missing * 7.5
    confidence -= inferred_count * 4.0
    confidence -= external_fail_count * 8.0
    confidence -= max(0, len(assumptions.assumptions_used) - external_fail_count) * 1.0
    confidence += min(10.0, confirmed_core_count * 2.5)
    confidence = max(0.0, min(100.0, round(confidence, 1)))

    low_confidence_flags: list[str] = []
    if important_missing >= 3:
        low_confidence_flags.append("Multiple important inputs or layers are missing")
    if external_fail_count > 0:
        low_confidence_flags.append("At least one external provider or layer fetch failed")
    if inferred_count >= 3:
        low_confidence_flags.append("Several core property attributes were inferred")
    if any("provisional" in note.lower() for note in assumptions.assumptions_used):
        low_confidence_flags.append("Access scoring is provisional and not yet parcel/egress-based")
    if confidence < 70:
        low_confidence_flags.append("Overall confidence below recommended underwriting threshold")

    data_completeness_score = round(max(0.0, min(100.0, 100.0 - (len(assumptions.missing_inputs) * 3.0))), 1)

    return ConfidenceBlock(
        confidence_score=confidence,
        data_completeness_score=data_completeness_score,
        assumption_count=len(assumptions.assumptions_used),
        low_confidence_flags=sorted(set(low_confidence_flags)),
        requires_user_verification=confidence < 70.0 or len(low_confidence_flags) > 0,
    )


def _build_top_risk_drivers(submodels: dict[str, SubmodelScore]) -> list[str]:
    labels = {
        "ember_exposure_risk": "high ember exposure",
        "flame_contact_risk": "high flame-contact exposure",
        "slope_topography_risk": "slope/topography amplification",
        "fuel_proximity_risk": "close proximity to wildland fuels",
        "vegetation_intensity_risk": "dense and dry vegetation",
        "historic_fire_risk": "recurring nearby fire history",
        "structure_vulnerability_risk": "high structure vulnerability",
        "defensible_space_risk": "insufficient defensible space",
    }
    ordered = sorted(
        [(k, v) for k, v in submodels.items() if k in labels],
        key=lambda kv: kv[1].score,
        reverse=True,
    )
    return [labels.get(name, name) for name, _ in ordered[:3]]


def _build_factor_breakdown(submodels: dict[str, SubmodelScore], risk: RiskComputation) -> FactorBreakdown:
    canonical = {name: round(submodels[name].score, 1) for name in CANONICAL_SUBMODELS if name in submodels}
    environmental = {name: canonical[name] for name in ENVIRONMENTAL_SUBMODELS if name in canonical}
    structural = {name: canonical[name] for name in STRUCTURAL_SUBMODELS if name in canonical}

    # Legacy compatibility fields are kept for older clients.
    return FactorBreakdown(
        submodels=canonical,
        environmental=environmental,
        structural=structural,
        environmental_risk=risk.drivers.environmental,
        structural_risk=risk.drivers.structural,
        access_risk=risk.drivers.access_exposure,
    )


def _build_top_protective_factors(payload: AddressRequest, submodels: dict[str, SubmodelScore]) -> list[str]:
    factors: list[str] = []
    attrs = payload.attributes

    if attrs.roof_type and attrs.roof_type.lower() in {"class a", "metal", "tile", "composite"}:
        factors.append("class A or equivalent fire-rated roof")
    if attrs.vent_type and "ember" in attrs.vent_type.lower():
        factors.append("ember-resistant venting")
    if attrs.defensible_space_ft is not None and attrs.defensible_space_ft >= 30:
        factors.append("defensible space >= 30 ft")
    if submodels["vegetation_intensity_risk"].score <= 35:
        factors.append("lower vegetation intensity near structure")
    if submodels["fuel_proximity_risk"].score <= 35:
        factors.append("limited adjacent wildland fuel proximity")

    if not factors:
        factors.append("no strong protective factors detected")
    return factors[:3]


def _run_assessment(payload: AddressRequest, *, assessment_id: str | None = None) -> tuple[AssessmentResult, dict]:
    pre_assumptions: list[str] = []

    try:
        lat, lon, geocode_source = geocoder.geocode(payload.address)
    except Exception:
        lat, lon = risk_engine.geocode_stub(payload.address)
        geocode_source = "Deterministic geocode fallback"
        pre_assumptions.append("Geocoding provider unavailable; fallback coordinates used.")

    context = wildfire_data.collect_context(lat, lon)
    risk: RiskComputation = risk_engine.score(payload.attributes, lat, lon, context)
    readiness = risk_engine.compute_insurance_readiness(payload.attributes, context, risk)

    submodel_scores: dict[str, SubmodelScore] = {}
    weighted_contributions: dict[str, WeightedContribution] = {}
    for name, result in risk.submodel_scores.items():
        contribution = risk.weighted_contributions[name]["contribution"]
        score_model = SubmodelScore(
            score=result.score,
            weighted_contribution=contribution,
            explanation=result.explanation,
            key_inputs=result.key_inputs,
            assumptions=result.assumptions,
            key_contributing_inputs=result.key_inputs,
        )
        submodel_scores[name] = score_model
        weighted_contributions[name] = WeightedContribution(
            weight=risk.weighted_contributions[name]["weight"],
            score=risk.weighted_contributions[name]["score"],
            contribution=contribution,
        )

    breakdown = _build_factor_breakdown(submodel_scores, risk)

    for legacy_key, canonical in SUBMODEL_ALIASES.items():
        submodel_scores[legacy_key] = submodel_scores[canonical]
        weighted_contributions[legacy_key] = weighted_contributions[canonical]

    mitigation_plan = build_mitigation_plan(
        payload.attributes,
        context,
        {k: v.score for k, v in submodel_scores.items()},
        readiness.readiness_blockers,
    )

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

    all_assumptions = sorted(set(pre_assumptions + risk.assumptions))
    all_sources = [geocode_source] + context.data_sources
    assumptions_block = _build_assumption_tracking(payload, all_assumptions, all_sources)
    confidence_block = _build_confidence(assumptions_block)

    top_risk_drivers = _build_top_risk_drivers(submodel_scores)
    top_protective_factors = _build_top_protective_factors(payload, submodel_scores)

    scoring_notes = [
        ACCESS_PROVISIONAL_NOTE,
        "Submodel/weight framework and readiness rules are deterministic MVP heuristics for calibration and explainability.",
    ]
    if any("fallback" in a.lower() or "unavailable" in a.lower() for a in all_assumptions):
        scoring_notes.append("One or more providers/layers required fallback assumptions.")

    explanation_summary = (
        f"Risk is driven by {', '.join(top_risk_drivers[:2])}. "
        f"Insurance readiness summary: {readiness.readiness_summary} "
        f"{ACCESS_PROVISIONAL_NOTE}"
    )

    risk_scores = RiskScores(wildfire_risk_score=risk.total_score, insurance_readiness_score=readiness.insurance_readiness_score)
    coordinates = Coordinates(latitude=lat, longitude=lon)

    readiness_factors = [
        ReadinessFactor(name=f["name"], status=f["status"], score_impact=f["score_impact"], detail=f["detail"])
        for f in readiness.readiness_factors
    ]

    submodel_explanations = {k: v.explanation for k, v in submodel_scores.items()}
    fact_map = _attributes_to_dict(payload.attributes)

    result = AssessmentResult(
        assessment_id=assessment_id or str(uuid4()),
        address=payload.address,
        audience=payload.audience,
        property_facts=fact_map,
        confirmed_fields=sorted(set(payload.confirmed_fields)),
        latitude=lat,
        longitude=lon,
        wildfire_risk_score=risk.total_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
        risk_drivers=risk.drivers,
        factor_breakdown=breakdown,
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        submodel_explanations=submodel_explanations,
        top_risk_drivers=top_risk_drivers,
        top_protective_factors=top_protective_factors,
        explanation_summary=explanation_summary,
        confirmed_inputs=assumptions_block.confirmed_inputs,
        observed_inputs=assumptions_block.observed_inputs,
        inferred_inputs=assumptions_block.inferred_inputs,
        missing_inputs=assumptions_block.missing_inputs,
        assumptions_used=all_assumptions,
        confidence_score=confidence_block.confidence_score,
        low_confidence_flags=confidence_block.low_confidence_flags,
        data_sources=all_sources,
        mitigation_plan=mitigation_plan,
        readiness_factors=readiness_factors,
        readiness_blockers=readiness.readiness_blockers,
        readiness_penalties=readiness.readiness_penalties,
        readiness_summary=readiness.readiness_summary,
        model_version=MODEL_VERSION,
        generated_at=datetime.now(tz=timezone.utc),
        scoring_notes=scoring_notes,
        coordinates=coordinates,
        risk_scores=risk_scores,
        assumptions=assumptions_block,
        confidence=confidence_block,
        mitigation_recommendations=mitigation_plan,
        environmental_factors=factors,
        explanation=explanation_summary,
    )

    debug_payload = {
        "address": payload.address,
        "coordinates": {"latitude": lat, "longitude": lon},
        "context_indices": {
            "burn_probability_index": context.burn_probability_index,
            "hazard_severity_index": context.hazard_severity_index,
            "slope_index": context.slope_index,
            "aspect_index": context.aspect_index,
            "fuel_index": context.fuel_index,
            "moisture_index": context.moisture_index,
            "canopy_index": context.canopy_index,
            "wildland_distance_index": context.wildland_distance_index,
            "historic_fire_index": context.historic_fire_index,
        },
        "submodel_scores": {
            name: {
                "score": sm.score,
                "weighted_contribution": sm.weighted_contribution,
                "explanation": sm.explanation,
                "key_inputs": sm.key_inputs,
                "assumptions": sm.assumptions,
            }
            for name, sm in submodel_scores.items()
        },
        "weighted_contributions": {
            name: {"weight": wc.weight, "score": wc.score, "contribution": wc.contribution}
            for name, wc in weighted_contributions.items()
        },
        "readiness": {
            "score": readiness.insurance_readiness_score,
            "blockers": readiness.readiness_blockers,
            "penalties": readiness.readiness_penalties,
            "factors": readiness.readiness_factors,
            "summary": readiness.readiness_summary,
        },
        "assumptions_used": all_assumptions,
        "data_sources": all_sources,
        "config": {
            "submodel_weights": scoring_config.submodel_weights,
            "readiness_penalties": scoring_config.readiness_penalties,
            "readiness_bonuses": scoring_config.readiness_bonuses,
        },
    }

    return result, debug_payload


def _payload_from_assessment(existing: AssessmentResult) -> AddressRequest:
    return AddressRequest(
        address=existing.address,
        attributes=PropertyAttributes.model_validate(existing.property_facts or {}),
        confirmed_fields=list(existing.confirmed_fields),
        audience=existing.audience,
    )


def _build_report_export(result: AssessmentResult) -> ReportExport:
    return ReportExport(
        assessment_id=result.assessment_id,
        generated_at=result.generated_at.isoformat(),
        model_version=result.model_version,
        property_summary={
            "address": result.address,
            "audience": result.audience,
            "property_facts": result.property_facts,
            "confirmed_fields": result.confirmed_fields,
        },
        location_summary={"latitude": result.latitude, "longitude": result.longitude, "data_sources": result.data_sources},
        wildfire_risk_summary={
            "wildfire_risk_score": result.wildfire_risk_score,
            "factor_breakdown": result.factor_breakdown.model_dump(),
            "top_risk_drivers": result.top_risk_drivers,
            "top_protective_factors": result.top_protective_factors,
        },
        insurance_readiness_summary={
            "insurance_readiness_score": result.insurance_readiness_score,
            "readiness_factors": [r.model_dump() for r in result.readiness_factors],
            "readiness_blockers": result.readiness_blockers,
            "readiness_penalties": result.readiness_penalties,
            "readiness_summary": result.readiness_summary,
        },
        top_risk_drivers=result.top_risk_drivers,
        top_protective_factors=result.top_protective_factors,
        assumptions_confidence={
            "confirmed_inputs": result.confirmed_inputs,
            "inferred_inputs": result.inferred_inputs,
            "missing_inputs": result.missing_inputs,
            "assumptions_used": result.assumptions_used,
            "confidence_score": result.confidence_score,
            "low_confidence_flags": result.low_confidence_flags,
        },
        mitigation_recommendations=result.mitigation_plan,
    )


def _build_report_html(result: AssessmentResult) -> str:
    blockers = "<li>None</li>" if not result.readiness_blockers else "".join(f"<li>{b}</li>" for b in result.readiness_blockers)
    mitigations = "".join(
        f"<li><strong>{m.title}</strong>: {m.reason}"
        f" <em>(risk: {m.estimated_risk_reduction_band}, readiness: {m.estimated_readiness_improvement_band})</em></li>"
        for m in result.mitigation_plan
    )
    drivers = "".join(f"<li>{d}</li>" for d in result.top_risk_drivers)
    protective = "".join(f"<li>{p}</li>" for p in result.top_protective_factors)

    return f"""
<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Wildfire Report {result.assessment_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:1rem; margin-bottom:1rem; }}
.row {{ display:flex; gap:1rem; flex-wrap:wrap; }}
.badge {{ display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#f3f4f6; font-size:0.8rem; }}
</style></head>
<body>
<h1>WildfireRisk Advisor Report</h1>
<p><span class=\"badge\">Assessment {result.assessment_id}</span> <span class=\"badge\">Model {result.model_version}</span> <span class=\"badge\">Generated {result.generated_at.isoformat()}</span></p>
<div class=\"card\"><h2>Property</h2><p>{result.address}</p><p>Audience: {result.audience}</p></div>
<div class=\"row\">
<div class=\"card\"><h3>Wildfire Risk Score</h3><p>{result.wildfire_risk_score}</p></div>
<div class=\"card\"><h3>Insurance Readiness Score</h3><p>{result.insurance_readiness_score}</p></div>
</div>
<div class=\"card\"><h3>Top Risk Drivers</h3><ul>{drivers}</ul></div>
<div class=\"card\"><h3>Top Protective Factors</h3><ul>{protective}</ul></div>
<div class=\"card\"><h3>Readiness Blockers</h3><ul>{blockers}</ul></div>
<div class=\"card\"><h3>Mitigation Recommendations</h3><ul>{mitigations}</ul></div>
<div class=\"card\"><h3>Assumptions & Confidence</h3>
<p>Confidence: {result.confidence_score}</p>
<p>Assumptions: {', '.join(result.assumptions_used) if result.assumptions_used else 'None'}</p>
</div>
</body></html>
"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(payload: AddressRequest) -> AssessmentResult:
    result, _ = _run_assessment(payload)
    store.save(result)
    return result


@app.post("/risk/reassess/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def reassess_risk(assessment_id: str, payload: ReassessmentRequest) -> AssessmentResult:
    existing = store.get(assessment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Assessment not found")

    base_req = _payload_from_assessment(existing)
    merged_attrs = _merge_attributes(base_req.attributes, payload.attributes)
    merged_confirmed = sorted(set(base_req.confirmed_fields + payload.confirmed_fields))

    req = AddressRequest(
        address=base_req.address,
        attributes=merged_attrs,
        confirmed_fields=merged_confirmed,
        audience=payload.audience or base_req.audience,
    )
    result, _ = _run_assessment(req)
    store.save(result)
    return result


@app.post("/risk/simulate", response_model=SimulationResult, dependencies=[Depends(require_api_key)])
def simulate_risk(payload: SimulationRequest) -> SimulationResult:
    if payload.assessment_id:
        existing = store.get(payload.assessment_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Assessment not found")
        base_req = _payload_from_assessment(existing)
    else:
        if not payload.address:
            raise HTTPException(status_code=400, detail="Provide assessment_id or address for simulation")
        base_req = AddressRequest(
            address=payload.address,
            attributes=payload.attributes,
            confirmed_fields=payload.confirmed_fields,
            audience=payload.audience,
        )

    baseline_attrs = _merge_attributes(base_req.attributes, payload.attributes)
    baseline_confirmed = sorted(set(base_req.confirmed_fields + payload.confirmed_fields))
    baseline_req = AddressRequest(
        address=base_req.address,
        attributes=baseline_attrs,
        confirmed_fields=baseline_confirmed,
        audience=base_req.audience,
    )

    baseline, _ = _run_assessment(baseline_req)

    simulated_attrs = _merge_attributes(baseline_attrs, payload.scenario_overrides)
    simulated_confirmed = sorted(set(baseline_confirmed + payload.scenario_confirmed_fields))
    simulated_req = AddressRequest(
        address=base_req.address,
        attributes=simulated_attrs,
        confirmed_fields=simulated_confirmed,
        audience=base_req.audience,
    )
    simulated, _ = _run_assessment(simulated_req)

    changed_inputs: Dict[str, Dict[str, object]] = {}
    before = baseline.property_facts
    after = simulated.property_facts
    for key in sorted(set(before.keys()) | set(after.keys())):
        if before.get(key) != after.get(key):
            changed_inputs[key] = {"before": before.get(key), "after": after.get(key)}

    return SimulationResult(
        scenario_name=payload.scenario_name,
        baseline=baseline,
        simulated=simulated,
        delta=SimulationDelta(
            wildfire_risk_score_delta=round(simulated.wildfire_risk_score - baseline.wildfire_risk_score, 1),
            insurance_readiness_score_delta=round(
                simulated.insurance_readiness_score - baseline.insurance_readiness_score, 1
            ),
        ),
        changed_inputs=changed_inputs,
        next_best_actions=simulated.mitigation_plan,
    )


@app.post("/risk/debug", dependencies=[Depends(require_api_key)])
def debug_risk(payload: AddressRequest) -> dict:
    _, debug_payload = _run_assessment(payload)
    return debug_payload


@app.get("/report/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def get_report(assessment_id: str) -> AssessmentResult:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result


@app.get("/report/{assessment_id}/export", response_model=ReportExport, dependencies=[Depends(require_api_key)])
def export_report(assessment_id: str) -> ReportExport:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _build_report_export(result)


@app.get("/report/{assessment_id}/view", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
def view_report(assessment_id: str) -> HTMLResponse:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return HTMLResponse(_build_report_html(result))


@app.get("/assessments", response_model=list[AssessmentListItem], dependencies=[Depends(require_api_key)])
def list_assessments(limit: int = Query(default=20, ge=1, le=200)) -> list[AssessmentListItem]:
    return store.list_assessments(limit=limit)
