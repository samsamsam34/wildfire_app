from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from backend.auth import require_api_key
from backend.database import AssessmentStore, DEFAULT_ORG_ID
from backend.geocoding import Geocoder
from backend.mitigation import build_mitigation_plan
from backend.models import (
    AddressRequest,
    AdminSummary,
    AssessmentAnnotation,
    AssessmentAnnotationCreate,
    AssessmentAssignmentRequest,
    AssessmentComparisonItem,
    AssessmentComparisonResponse,
    AssessmentComparisonResult,
    AssessmentListItem,
    AssessmentResult,
    AssessmentReviewStatus,
    AssessmentReviewStatusUpdate,
    AssessmentSummaryResponse,
    AssessmentWorkflowInfo,
    AssessmentWorkflowUpdateRequest,
    AssumptionsBlock,
    Audience,
    AuditEvent,
    BatchAssessmentRequest,
    BatchAssessmentResponse,
    BatchAssessmentResultItem,
    ConfidenceBlock,
    Coordinates,
    CSVImportError,
    CSVImportRequest,
    CSVImportResponse,
    EnvironmentalFactors,
    FactorBreakdown,
    Organization,
    OrganizationCreate,
    PortfolioJobCreate,
    PortfolioJobResultsResponse,
    PortfolioJobStatus,
    PortfolioJobsSummary,
    PortfolioResponse,
    PropertyAttributes,
    ReadinessFactor,
    ReassessmentRequest,
    ReportExport,
    RiskScores,
    SimulationDelta,
    SimulationRequest,
    SimulationResult,
    SimulationScenarioItem,
    ScoreSectionSummary,
    SubmodelScore,
    UnderwritingRuleset,
    UnderwritingRulesetCreate,
    UserRole,
    WeightedContribution,
    WorkflowState,
)
from backend.risk_engine import RiskComputation, RiskEngine
from backend.scoring_config import load_scoring_config
from backend.version import MODEL_VERSION
from backend.wildfire_data import WildfireDataClient

app = FastAPI(title="WildfireRisk Advisor API", version="0.9.0")

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

SITE_HAZARD_SUBMODELS = [
    "vegetation_intensity_risk",
    "fuel_proximity_risk",
    "slope_topography_risk",
    "historic_fire_risk",
]

CORE_FACT_FIELDS = {"roof_type", "vent_type", "defensible_space_ft", "construction_year"}

ALLOWED_ROLES: set[str] = {"admin", "underwriter", "broker", "inspector", "agent", "viewer"}
WRITE_ROLES: set[str] = {"admin", "underwriter", "broker", "inspector", "agent"}
REVIEW_EDIT_ROLES: set[str] = {"admin", "underwriter"}
WORKFLOW_EDIT_ROLES: set[str] = {"admin", "underwriter", "broker", "inspector"}

WORKFLOW_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    "new": {"triaged", "needs_inspection", "approved", "declined", "escalated"},
    "triaged": {"needs_inspection", "mitigation_pending", "ready_for_review", "escalated", "declined"},
    "needs_inspection": {"mitigation_pending", "ready_for_review", "escalated", "declined"},
    "mitigation_pending": {"ready_for_review", "needs_inspection", "escalated", "declined"},
    "ready_for_review": {"approved", "declined", "escalated", "mitigation_pending"},
    "approved": {"escalated"},
    "declined": {"escalated"},
    "escalated": {"triaged", "needs_inspection", "mitigation_pending", "ready_for_review", "approved", "declined"},
}


@dataclass
class ActorContext:
    user_role: UserRole
    organization_id: str
    user_id: str


def _default_audience_for_role(role: UserRole) -> Audience:
    if role in {"admin", "underwriter"}:
        return "insurer"
    if role == "inspector":
        return "inspector"
    if role in {"broker", "agent"}:
        return "agent"
    return "homeowner"


def _build_context(
    x_user_role: str | None,
    x_organization_id: str | None,
    x_user_id: str | None,
) -> ActorContext:
    role = (x_user_role or "admin").strip().lower()
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail="Unsupported user role")

    org_id = (x_organization_id or DEFAULT_ORG_ID).strip() or DEFAULT_ORG_ID
    user_id = (x_user_id or "api_user").strip() or "api_user"
    return ActorContext(user_role=role, organization_id=org_id, user_id=user_id)


def get_actor_context(
    x_user_role: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> ActorContext:
    return _build_context(x_user_role, x_organization_id, x_user_id)


def _require_role(ctx: ActorContext, allowed_roles: Iterable[str], detail: str = "Forbidden") -> None:
    if ctx.user_role not in set(allowed_roles):
        raise HTTPException(status_code=403, detail=detail)


def _resolve_org_id(request_org: str | None, ctx: ActorContext) -> str:
    return request_org or ctx.organization_id


def _enforce_org_scope(ctx: ActorContext, organization_id: str) -> None:
    if ctx.user_role == "admin":
        return
    if ctx.organization_id != organization_id:
        raise HTTPException(status_code=403, detail="Organization scope violation")


def _log_audit(
    *,
    ctx: ActorContext,
    entity_type: str,
    entity_id: str,
    action: str,
    organization_id: str,
    metadata: dict[str, object] | None = None,
) -> None:
    store.log_event(
        entity_type=entity_type,
        entity_id=entity_id,
        organization_id=organization_id,
        user_role=ctx.user_role,
        action=action,
        metadata=metadata or {},
    )


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
    missing_layer_count = len([m for m in assumptions.missing_inputs if m.endswith("_layer")])
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

    if confidence < 45 or important_missing >= 5 or external_fail_count >= 2:
        confidence_tier = "preliminary"
    elif confidence < 70 or important_missing >= 3 or external_fail_count >= 1:
        confidence_tier = "low"
    elif confidence < 85 or important_missing >= 1 or inferred_count >= 2:
        confidence_tier = "moderate"
    else:
        confidence_tier = "high"

    if confidence_tier == "preliminary" or missing_layer_count >= 2:
        use_restriction = "not_for_underwriting_or_binding"
    elif confidence < 70 and external_fail_count >= 1:
        use_restriction = "not_for_underwriting_or_binding"
    elif confidence_tier == "low":
        use_restriction = "agent_or_inspector_review_recommended"
    elif confidence_tier == "moderate":
        use_restriction = "homeowner_review_recommended"
    else:
        use_restriction = "shareable"

    return ConfidenceBlock(
        confidence_score=confidence,
        data_completeness_score=data_completeness_score,
        confidence_tier=confidence_tier,
        use_restriction=use_restriction,
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


def _density_from_ring(metrics: dict[str, object] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("vegetation_density")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_property_findings(property_level_context: dict[str, Any]) -> list[str]:
    if not isinstance(property_level_context, dict):
        return []

    rings = property_level_context.get("ring_metrics")
    if not isinstance(rings, dict) or not rings:
        return []

    ring_0_5 = _density_from_ring(rings.get("ring_0_5_ft"))
    ring_5_30 = _density_from_ring(rings.get("ring_5_30_ft"))
    ring_30_100 = _density_from_ring(rings.get("ring_30_100_ft"))
    findings: list[str] = []

    if ring_0_5 is not None and ring_0_5 >= 60:
        findings.append("Vegetation appears dense within 5 feet of the home.")
    elif ring_0_5 is not None and ring_0_5 <= 30:
        findings.append("The immediate 0-5 foot zone appears relatively clear.")

    if ring_5_30 is not None and ring_5_30 >= 60:
        findings.append("Tree or shrub cover is elevated within 30 feet of the structure.")
    elif ring_5_30 is not None and ring_5_30 <= 35:
        findings.append("Vegetation conditions within 30 feet look more manageable.")

    if ring_30_100 is not None and ring_30_100 >= 65:
        findings.append("Vegetation and fuels are elevated in the 30-100 foot zone around the home.")

    if ring_0_5 is not None and ring_5_30 is not None and (ring_5_30 - ring_0_5) >= 20:
        findings.append("Defensible space appears stronger very close to the home than farther out.")

    return findings[:3]


def _merge_property_drivers(base_drivers: list[str], property_findings: list[str]) -> list[str]:
    if not property_findings:
        return base_drivers[:3]

    mapped: list[str] = []
    for finding in property_findings:
        text = finding.lower()
        if "within 5 feet" in text:
            mapped.append("dense vegetation close to the home")
        elif "within 30 feet" in text:
            mapped.append("limited defensible space within 30 feet")
        elif "30-100 foot" in text:
            mapped.append("elevated vegetation and fuels within 100 feet")

    merged = mapped + base_drivers
    unique: list[str] = []
    for driver in merged:
        if driver not in unique:
            unique.append(driver)
    return unique[:3]


def _weighted_average(submodels: dict[str, SubmodelScore], keys: list[str]) -> float:
    present = [submodels[k].score for k in keys if k in submodels]
    if not present:
        return 0.0
    return round(sum(present) / len(present), 1)


def _build_score_decomposition(
    *,
    submodels: dict[str, SubmodelScore],
    context: object,
) -> tuple[float, float]:
    site_submodel_component = _weighted_average(submodels, SITE_HAZARD_SUBMODELS)
    site_hazard_score = round(0.75 * site_submodel_component + 0.25 * float(getattr(context, "environmental_index", 50.0)), 1)

    structure_component = _weighted_average(submodels, STRUCTURAL_SUBMODELS)
    near_structure_component = _weighted_average(submodels, ["flame_contact_risk", "ember_exposure_risk"])
    home_ignition_vulnerability_score = round(0.7 * structure_component + 0.3 * near_structure_component, 1)

    return (
        max(0.0, min(100.0, site_hazard_score)),
        max(0.0, min(100.0, home_ignition_vulnerability_score)),
    )


def _build_score_sections(
    *,
    site_hazard_score: float,
    home_ignition_vulnerability_score: float,
    insurance_readiness_score: float,
    top_risk_drivers: list[str],
    top_protective_factors: list[str],
    mitigation_plan,
    property_findings: list[str],
    readiness_blockers: list[str],
    confidence_block: ConfidenceBlock,
) -> tuple[ScoreSectionSummary, ScoreSectionSummary, ScoreSectionSummary]:
    site_actions = [
        m.title
        for m in mitigation_plan
        if any(
            k in (m.impacted_submodels or [])
            for k in ["vegetation_intensity_risk", "fuel_proximity_risk", "slope_topography_risk", "historic_fire_risk"]
        )
    ][:3]
    home_actions = [
        m.title
        for m in mitigation_plan
        if any(
            k in (m.impacted_submodels or [])
            for k in ["structure_vulnerability_risk", "defensible_space_risk", "flame_contact_risk", "ember_exposure_risk"]
        )
    ][:3]

    site_section = ScoreSectionSummary(
        summary=(
            f"Site Hazard {site_hazard_score:.1f}/100 reflects landscape fuel, slope, and nearby fire pressure around the home."
        ),
        key_drivers=top_risk_drivers[:3],
        protective_factors=top_protective_factors[:3],
        next_actions=site_actions,
    )

    home_section = ScoreSectionSummary(
        summary=(
            f"Home Ignition Vulnerability {home_ignition_vulnerability_score:.1f}/100 reflects structure hardening and near-home ignition pathways."
        ),
        key_drivers=(property_findings[:3] or top_risk_drivers[:3]),
        protective_factors=top_protective_factors[:3],
        next_actions=home_actions,
    )

    readiness_actions = [m.title for m in mitigation_plan[:3]]
    readiness_section = ScoreSectionSummary(
        summary=(
            f"Insurance Readiness {insurance_readiness_score:.1f}/100 is a heuristic advisory score and not carrier-approved underwriting."
        ),
        key_drivers=(readiness_blockers[:3] or ["No major readiness blockers detected"]),
        protective_factors=top_protective_factors[:3],
        next_actions=readiness_actions,
    )

    if confidence_block.use_restriction == "not_for_underwriting_or_binding":
        readiness_section.summary += " Current confidence gating: not for underwriting or binding decisions."

    return site_section, home_section, readiness_section


def _build_factor_breakdown(submodels: dict[str, SubmodelScore], risk: RiskComputation) -> FactorBreakdown:
    canonical = {name: round(submodels[name].score, 1) for name in CANONICAL_SUBMODELS if name in submodels}
    environmental = {name: canonical[name] for name in ENVIRONMENTAL_SUBMODELS if name in canonical}
    structural = {name: canonical[name] for name in STRUCTURAL_SUBMODELS if name in canonical}

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


def _apply_ruleset_to_result(result: AssessmentResult, ruleset: UnderwritingRuleset) -> AssessmentResult:
    config = ruleset.config or {}
    penalty_multiplier = float(config.get("penalty_multiplier", 1.0))
    risk_blocker_threshold = float(config.get("risk_blocker_threshold", 85.0))
    inspection_missing_threshold = int(config.get("inspection_missing_threshold", 5))
    required_priority_boost = int(config.get("mitigation_required_priority_boost", 0))

    result.ruleset_id = ruleset.ruleset_id
    result.ruleset_name = ruleset.ruleset_name
    result.ruleset_version = ruleset.ruleset_version
    result.ruleset_description = ruleset.ruleset_description

    extra_penalty = 0.0
    if penalty_multiplier > 1.0 and result.readiness_penalties:
        scaled: dict[str, float] = {}
        for key, value in result.readiness_penalties.items():
            new_value = round(float(value) * penalty_multiplier, 1)
            scaled[key] = new_value
            extra_penalty += max(0.0, new_value - float(value))
        result.readiness_penalties = scaled

    if result.wildfire_risk_score >= risk_blocker_threshold:
        blocker = f"Carrier profile threshold exceeded ({ruleset.ruleset_id})"
        if blocker not in result.readiness_blockers:
            result.readiness_blockers.append(blocker)
            extra_penalty += 5.0

    if len(result.missing_inputs) >= inspection_missing_threshold:
        blocker = "Inspection required before underwriting decision"
        if blocker not in result.readiness_blockers:
            result.readiness_blockers.append(blocker)
            extra_penalty += 4.0

    if extra_penalty > 0:
        result.insurance_readiness_score = round(max(0.0, min(100.0, result.insurance_readiness_score - extra_penalty)), 1)
        if result.risk_scores:
            result.risk_scores.insurance_readiness_score = result.insurance_readiness_score
        if result.insurance_readiness_section:
            result.insurance_readiness_section.summary = (
                f"Insurance Readiness {result.insurance_readiness_score:.1f}/100 is a heuristic advisory score "
                "and not carrier-approved underwriting."
            )

    if required_priority_boost > 0:
        for rec in result.mitigation_plan:
            if rec.insurer_relevance == "required":
                rec.priority = max(1, rec.priority - required_priority_boost)
        result.mitigation_plan.sort(key=lambda x: x.priority)
        result.mitigation_recommendations = list(result.mitigation_plan)

    result.readiness_blockers = sorted(set(result.readiness_blockers))
    result.readiness_summary = f"{result.readiness_summary} Ruleset: {ruleset.ruleset_name} ({ruleset.ruleset_id})."
    result.scoring_notes.append(
        f"Underwriting ruleset {ruleset.ruleset_id}@{ruleset.ruleset_version} applied (multiplier={penalty_multiplier})."
    )
    return result


def _run_assessment(
    payload: AddressRequest,
    *,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    assessment_id: str | None = None,
    portfolio_name: str | None = None,
    tags: list[str] | None = None,
) -> tuple[AssessmentResult, dict]:
    try:
        lat, lon, geocode_source = geocoder.geocode(payload.address)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Geocoding lookup failed; a trusted location match is required for this assessment. "
                "Please verify the address and try again."
            ),
        ) from exc

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

    all_assumptions = sorted(set(risk.assumptions))
    all_sources = [geocode_source] + context.data_sources
    assumptions_block = _build_assumption_tracking(payload, all_assumptions, all_sources)
    confidence_block = _build_confidence(assumptions_block)

    property_findings = _build_property_findings(context.property_level_context)
    top_risk_drivers = _merge_property_drivers(_build_top_risk_drivers(submodel_scores), property_findings)
    top_protective_factors = _build_top_protective_factors(payload, submodel_scores)

    site_hazard_score, home_ignition_vulnerability_score = _build_score_decomposition(
        submodels=submodel_scores,
        context=context,
    )
    legacy_weighted_wildfire_risk_score = risk.total_score
    blended_wildfire_risk_score = round(
        max(0.0, min(100.0, 0.6 * site_hazard_score + 0.4 * home_ignition_vulnerability_score)),
        1,
    )

    site_hazard_section, home_ignition_section, insurance_readiness_section = _build_score_sections(
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
        top_risk_drivers=top_risk_drivers,
        top_protective_factors=top_protective_factors,
        mitigation_plan=mitigation_plan,
        property_findings=property_findings,
        readiness_blockers=readiness.readiness_blockers,
        confidence_block=confidence_block,
    )

    scoring_notes = [
        ACCESS_PROVISIONAL_NOTE,
        "Submodel/weight framework and readiness rules are deterministic MVP heuristics for calibration and explainability.",
        "Scores are advisory heuristics and not carrier-approved underwriting or premium predictions.",
    ]
    if any("fallback" in a.lower() or "unavailable" in a.lower() for a in all_assumptions):
        scoring_notes.append("One or more providers/layers required fallback assumptions.")
    if confidence_block.use_restriction == "not_for_underwriting_or_binding":
        scoring_notes.append("Current confidence gating: not for underwriting or binding decisions.")

    explanation_summary = (
        f"Site Hazard: {site_hazard_score:.1f}/100. "
        f"Home Ignition Vulnerability: {home_ignition_vulnerability_score:.1f}/100. "
        f"Insurance Readiness: {readiness.insurance_readiness_score:.1f}/100. "
        f"Primary drivers: {', '.join(top_risk_drivers[:2])}."
    )

    risk_scores = RiskScores(
        wildfire_risk_score=blended_wildfire_risk_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
    )
    coordinates = Coordinates(latitude=lat, longitude=lon)

    readiness_factors = [
        ReadinessFactor(name=f["name"], status=f["status"], score_impact=f["score_impact"], detail=f["detail"])
        for f in readiness.readiness_factors
    ]

    submodel_explanations = {k: v.explanation for k, v in submodel_scores.items()}
    fact_map = _attributes_to_dict(payload.attributes)
    final_tags = sorted(set((payload.tags or []) + (tags or [])))

    result = AssessmentResult(
        assessment_id=assessment_id or str(uuid4()),
        address=payload.address,
        audience=payload.audience,
        report_audience=payload.audience,
        audience_highlights=[],
        organization_id=organization_id,
        portfolio_name=portfolio_name,
        tags=final_tags,
        ruleset_id=ruleset.ruleset_id,
        ruleset_name=ruleset.ruleset_name,
        ruleset_version=ruleset.ruleset_version,
        ruleset_description=ruleset.ruleset_description,
        review_status="pending",
        workflow_state="new",
        assigned_reviewer=None,
        assigned_role=None,
        property_facts=fact_map,
        confirmed_fields=sorted(set(payload.confirmed_fields)),
        latitude=lat,
        longitude=lon,
        wildfire_risk_score=blended_wildfire_risk_score,
        legacy_weighted_wildfire_risk_score=legacy_weighted_wildfire_risk_score,
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
        risk_drivers=risk.drivers,
        factor_breakdown=breakdown,
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        submodel_explanations=submodel_explanations,
        property_findings=property_findings,
        top_risk_drivers=top_risk_drivers,
        top_protective_factors=top_protective_factors,
        explanation_summary=explanation_summary,
        confirmed_inputs=assumptions_block.confirmed_inputs,
        observed_inputs=assumptions_block.observed_inputs,
        inferred_inputs=assumptions_block.inferred_inputs,
        missing_inputs=assumptions_block.missing_inputs,
        assumptions_used=all_assumptions,
        confidence_score=confidence_block.confidence_score,
        data_completeness_score=confidence_block.data_completeness_score,
        confidence_tier=confidence_block.confidence_tier,
        use_restriction=confidence_block.use_restriction,
        low_confidence_flags=confidence_block.low_confidence_flags,
        data_sources=all_sources,
        property_level_context=context.property_level_context,
        mitigation_plan=mitigation_plan,
        readiness_factors=readiness_factors,
        readiness_blockers=readiness.readiness_blockers,
        readiness_penalties=readiness.readiness_penalties,
        readiness_summary=readiness.readiness_summary,
        site_hazard_section=site_hazard_section,
        home_ignition_vulnerability_section=home_ignition_section,
        insurance_readiness_section=insurance_readiness_section,
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

    result = _apply_ruleset_to_result(result, ruleset)

    debug_payload = {
        "address": payload.address,
        "organization_id": organization_id,
        "ruleset_id": ruleset.ruleset_id,
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
        "property_level_context": context.property_level_context,
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
            "score": result.insurance_readiness_score,
            "blockers": result.readiness_blockers,
            "penalties": result.readiness_penalties,
            "factors": [f.model_dump() for f in result.readiness_factors],
            "summary": result.readiness_summary,
        },
        "score_decomposition": {
            "site_hazard_score": result.site_hazard_score,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "wildfire_risk_score": result.wildfire_risk_score,
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "insurance_readiness_score": result.insurance_readiness_score,
        },
        "confidence_gating": {
            "confidence_score": result.confidence_score,
            "data_completeness_score": result.data_completeness_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
        },
        "assumptions_used": all_assumptions,
        "data_sources": all_sources,
        "tags": final_tags,
        "portfolio_name": portfolio_name,
        "config": {
            "submodel_weights": scoring_config.submodel_weights,
            "readiness_penalties": scoring_config.readiness_penalties,
            "readiness_bonuses": scoring_config.readiness_bonuses,
            "ruleset": ruleset.model_dump(),
        },
    }

    return result, debug_payload


def _compute_assessment(
    payload: AddressRequest,
    *,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    portfolio_name: str | None = None,
    tags: list[str] | None = None,
) -> AssessmentResult:
    result, _ = _run_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        portfolio_name=portfolio_name,
        tags=tags,
    )
    return result


def _payload_from_assessment(existing: AssessmentResult) -> AddressRequest:
    return AddressRequest(
        address=existing.address,
        attributes=PropertyAttributes.model_validate(existing.property_facts or {}),
        confirmed_fields=list(existing.confirmed_fields),
        audience=existing.audience,
        tags=list(existing.tags),
        organization_id=existing.organization_id,
        ruleset_id=existing.ruleset_id,
    )


def _audience_highlights(result: AssessmentResult, audience_mode: Audience) -> list[str]:
    if audience_mode == "homeowner":
        return [
            "Focus first on the top two mitigation actions to reduce ignition pathways.",
            "Use readiness blockers as a practical home-hardening checklist.",
        ]
    if audience_mode == "agent":
        return [
            "Use risk/readiness and mitigation points in disclosure conversations.",
            "Document which blockers were resolved before closing timelines.",
        ]
    if audience_mode == "inspector":
        return [
            "Validate observed vs inferred inputs and capture evidence in inspection notes.",
            "Prioritize verification of high-impact factors tied to readiness blockers.",
        ]
    return [
        "Prioritize properties with severe blockers and low confidence for manual review.",
        "Use factorized submodel contributions to triage mitigation-driven eligibility improvements.",
    ]


def _audience_focus(result: AssessmentResult, audience_mode: Audience) -> dict[str, object]:
    if audience_mode == "homeowner":
        return {
            "next_steps": [m.model_dump() for m in result.mitigation_plan[:3]],
            "plain_language_summary": result.explanation_summary,
        }
    if audience_mode == "agent":
        return {
            "disclosure_summary": {
                "top_risk_drivers": result.top_risk_drivers,
                "top_protective_factors": result.top_protective_factors,
                "readiness_blockers": result.readiness_blockers,
            },
            "mitigation_talking_points": [m.title for m in result.mitigation_plan[:4]],
        }
    if audience_mode == "inspector":
        return {
            "observed_inputs": result.observed_inputs,
            "inferred_inputs": result.inferred_inputs,
            "missing_inputs": result.missing_inputs,
            "assumptions_used": result.assumptions_used,
            "inspection_notes": result.property_facts.get("inspection_notes"),
        }
    return {
        "readiness_blockers": result.readiness_blockers,
        "readiness_penalties": result.readiness_penalties,
        "confidence_score": result.confidence_score,
        "weighted_contributions": {k: v.model_dump() for k, v in result.weighted_contributions.items()},
    }


def _resolve_audience(
    assessment: AssessmentResult,
    actor: ActorContext,
    audience: Audience | None,
    audience_mode: Audience | None,
) -> Audience:
    return audience or audience_mode or _default_audience_for_role(actor.user_role)


def _apply_audience_view(
    assessment: AssessmentResult,
    *,
    actor: ActorContext,
    audience: Audience | None,
    audience_mode: Audience | None,
) -> AssessmentResult:
    mode = _resolve_audience(assessment, actor, audience, audience_mode)
    view = assessment.model_copy(deep=True)
    view.report_audience = mode
    view.audience_highlights = _audience_highlights(view, mode)
    return view


def _build_report_export(
    result: AssessmentResult,
    *,
    actor: ActorContext,
    audience: Audience | None = None,
    audience_mode: Audience | None = None,
) -> ReportExport:
    mode = _resolve_audience(result, actor, audience, audience_mode)
    return ReportExport(
        assessment_id=result.assessment_id,
        generated_at=result.generated_at.isoformat(),
        model_version=result.model_version,
        organization_id=result.organization_id,
        audience_mode=mode,
        audience_highlights=_audience_highlights(result, mode),
        audience_focus=_audience_focus(result, mode),
        ruleset={
            "ruleset_id": result.ruleset_id,
            "ruleset_name": result.ruleset_name,
            "ruleset_version": result.ruleset_version,
            "ruleset_description": result.ruleset_description,
        },
        property_summary={
            "address": result.address,
            "organization_id": result.organization_id,
            "audience": result.audience,
            "portfolio_name": result.portfolio_name,
            "tags": result.tags,
            "review_status": result.review_status,
            "workflow_state": result.workflow_state,
            "assigned_reviewer": result.assigned_reviewer,
            "assigned_role": result.assigned_role,
            "property_facts": result.property_facts,
            "confirmed_fields": result.confirmed_fields,
        },
        location_summary={
            "latitude": result.latitude,
            "longitude": result.longitude,
            "data_sources": result.data_sources,
            "property_level_context": result.property_level_context,
        },
        wildfire_risk_summary={
            "wildfire_risk_score": result.wildfire_risk_score,
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "site_hazard_score": result.site_hazard_score,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "site_hazard_section": result.site_hazard_section.model_dump(),
            "home_ignition_vulnerability_section": result.home_ignition_vulnerability_section.model_dump(),
            "factor_breakdown": result.factor_breakdown.model_dump(),
            "property_findings": result.property_findings,
            "top_risk_drivers": result.top_risk_drivers,
            "top_protective_factors": result.top_protective_factors,
            "weighted_contributions": {k: v.model_dump() for k, v in result.weighted_contributions.items()},
        },
        insurance_readiness_summary={
            "insurance_readiness_score": result.insurance_readiness_score,
            "insurance_readiness_section": result.insurance_readiness_section.model_dump(),
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
            "data_completeness_score": result.data_completeness_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
            "low_confidence_flags": result.low_confidence_flags,
        },
        mitigation_recommendations=result.mitigation_plan,
    )


def _build_report_html(
    result: AssessmentResult,
    *,
    actor: ActorContext,
    audience: Audience | None = None,
    audience_mode: Audience | None = None,
) -> str:
    mode = _resolve_audience(result, actor, audience, audience_mode)
    highlights = _audience_highlights(result, mode)
    focus = _audience_focus(result, mode)
    blockers = "<li>None</li>" if not result.readiness_blockers else "".join(f"<li>{b}</li>" for b in result.readiness_blockers)
    mitigations = "".join(
        f"<li><strong>{m.title}</strong>: {m.reason}"
        f" <em>(risk: {m.estimated_risk_reduction_band}, readiness: {m.estimated_readiness_improvement_band})</em></li>"
        for m in result.mitigation_plan
    )
    drivers = "".join(f"<li>{d}</li>" for d in result.top_risk_drivers)
    protective = "".join(f"<li>{p}</li>" for p in result.top_protective_factors)
    audience_notes = "".join(f"<li>{n}</li>" for n in highlights)

    return f"""
<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Wildfire Report {result.assessment_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
.card {{ border:1px solid #ddd; border-radius:10px; padding:1rem; margin-bottom:1rem; }}
.row {{ display:flex; gap:1rem; flex-wrap:wrap; }}
.badge {{ display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#f3f4f6; font-size:0.8rem; }}
pre {{ white-space: pre-wrap; }}
</style></head>
<body>
<h1>WildfireRisk Advisor Report</h1>
<p><span class=\"badge\">Assessment {result.assessment_id}</span> <span class=\"badge\">Model {result.model_version}</span> <span class=\"badge\">Generated {result.generated_at.isoformat()}</span> <span class=\"badge\">Audience View {mode}</span></p>
<div class=\"card\"><h2>Property</h2><p>{result.address}</p><p>Organization: {result.organization_id}</p><p>Audience: {result.audience}</p><p>Review Status: {result.review_status}</p><p>Workflow: {result.workflow_state}</p><p>Assigned: {result.assigned_reviewer or 'n/a'}</p><p>Portfolio: {result.portfolio_name or 'n/a'}</p><p>Tags: {', '.join(result.tags) if result.tags else 'none'}</p></div>
<div class=\"card\"><h3>Ruleset</h3><p>{result.ruleset_name} ({result.ruleset_id} v{result.ruleset_version})</p><p>{result.ruleset_description}</p></div>
<div class=\"row\">
<div class=\"card\"><h3>Wildfire Risk Score</h3><p>{result.wildfire_risk_score}</p></div>
<div class=\"card\"><h3>Insurance Readiness Score</h3><p>{result.insurance_readiness_score}</p></div>
</div>
<div class=\"card\"><h3>Top Risk Drivers</h3><ul>{drivers}</ul></div>
<div class=\"card\"><h3>Top Protective Factors</h3><ul>{protective}</ul></div>
<div class=\"card\"><h3>Readiness Blockers</h3><ul>{blockers}</ul></div>
<div class=\"card\"><h3>Mitigation Recommendations</h3><ul>{mitigations}</ul></div>
<div class=\"card\"><h3>Audience-Specific Highlights ({mode})</h3><ul>{audience_notes}</ul></div>
<div class=\"card\"><h3>Audience Focus Payload</h3><pre>{focus}</pre></div>
<div class=\"card\"><h3>Assumptions & Confidence</h3>
<p>Confidence: {result.confidence_score}</p>
<p>Assumptions: {', '.join(result.assumptions_used) if result.assumptions_used else 'None'}</p>
</div>
</body></html>
"""


def _to_comparison_item(result: AssessmentResult) -> AssessmentComparisonItem:
    return AssessmentComparisonItem(
        assessment_id=result.assessment_id,
        address=result.address,
        wildfire_risk_score=result.wildfire_risk_score,
        insurance_readiness_score=result.insurance_readiness_score,
        top_risk_drivers=result.top_risk_drivers,
        readiness_blockers=result.readiness_blockers,
        mitigation_titles=[m.title for m in result.mitigation_plan],
    )


def _compare_results(base: AssessmentResult, other: AssessmentResult) -> AssessmentComparisonResult:
    base_drivers = set(base.top_risk_drivers)
    other_drivers = set(other.top_risk_drivers)
    base_blockers = set(base.readiness_blockers)
    other_blockers = set(other.readiness_blockers)
    base_mitigations = {m.title for m in base.mitigation_plan}
    other_mitigations = {m.title for m in other.mitigation_plan}

    return AssessmentComparisonResult(
        base=_to_comparison_item(base),
        other=_to_comparison_item(other),
        wildfire_risk_delta=round(other.wildfire_risk_score - base.wildfire_risk_score, 1),
        insurance_readiness_delta=round(other.insurance_readiness_score - base.insurance_readiness_score, 1),
        driver_differences={
            "added": sorted(other_drivers - base_drivers),
            "removed": sorted(base_drivers - other_drivers),
        },
        blocker_differences={
            "added": sorted(other_blockers - base_blockers),
            "removed": sorted(base_blockers - other_blockers),
        },
        mitigation_differences={
            "added": sorted(other_mitigations - base_mitigations),
            "removed": sorted(base_mitigations - other_mitigations),
        },
    )


def _get_ruleset_or_default(ruleset_id: str | None) -> UnderwritingRuleset:
    selected = store.get_ruleset(ruleset_id or "default")
    if selected:
        return selected
    fallback = store.get_ruleset("default")
    if not fallback:
        raise HTTPException(status_code=500, detail="Default ruleset is not configured")
    return fallback


def _assessment_to_csv(rows: list[BatchAssessmentResultItem]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=[
            "address",
            "assessment_id",
            "status",
            "error",
            "wildfire_risk_score",
            "insurance_readiness_score",
            "confidence_score",
            "top_risk_drivers",
            "readiness_blockers",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "address": row.address,
                "assessment_id": row.assessment_id or "",
                "status": row.status,
                "error": row.error or "",
                "wildfire_risk_score": row.wildfire_risk_score if row.wildfire_risk_score is not None else "",
                "insurance_readiness_score": row.insurance_readiness_score if row.insurance_readiness_score is not None else "",
                "confidence_score": row.confidence_score if row.confidence_score is not None else "",
                "top_risk_drivers": " | ".join(row.top_risk_drivers),
                "readiness_blockers": " | ".join(row.readiness_blockers),
            }
        )
    return out.getvalue()


def _list_item_to_csv(rows: list[AssessmentListItem]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=[
            "assessment_id",
            "created_at",
            "organization_id",
            "address",
            "wildfire_risk_score",
            "insurance_readiness_score",
            "confidence_score",
            "top_risk_drivers",
            "readiness_blockers",
            "review_status",
            "workflow_state",
            "assigned_reviewer",
            "ruleset_id",
        ],
    )
    writer.writeheader()
    for item in rows:
        assessment = store.get(item.assessment_id)
        writer.writerow(
            {
                "assessment_id": item.assessment_id,
                "created_at": item.created_at,
                "organization_id": item.organization_id,
                "address": item.address,
                "wildfire_risk_score": item.wildfire_risk_score,
                "insurance_readiness_score": item.insurance_readiness_score,
                "confidence_score": item.confidence_score,
                "top_risk_drivers": " | ".join(assessment.top_risk_drivers if assessment else []),
                "readiness_blockers": " | ".join(item.readiness_blockers),
                "review_status": item.review_status,
                "workflow_state": item.workflow_state,
                "assigned_reviewer": item.assigned_reviewer or "",
                "ruleset_id": item.ruleset_id,
            }
        )
    return out.getvalue()


def _to_job_status(record: dict) -> PortfolioJobStatus:
    result = record.get("result") or {}
    req = record.get("request") or {}
    return PortfolioJobStatus(
        job_id=record["job_id"],
        organization_id=record["organization_id"],
        portfolio_name=req.get("portfolio_name"),
        ruleset_id=req.get("ruleset_id", "default"),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        status=record["status"],
        total_properties=result.get("total_properties", len((req.get("items") or []))),
        completed_count=result.get("completed_count", 0),
        failed_count=result.get("failed_count", 0),
        high_risk_count=result.get("high_risk_count", 0),
        blocker_count=result.get("blocker_count", 0),
        average_wildfire_risk=result.get("average_wildfire_risk", 0.0),
        average_insurance_readiness=result.get("average_insurance_readiness", 0.0),
        error_summary=record.get("error_summary"),
    )


def _run_batch(
    payload: BatchAssessmentRequest,
    *,
    actor: ActorContext,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    job_id: str | None = None,
) -> BatchAssessmentResponse:
    results: list[BatchAssessmentResultItem] = []
    success_count = 0

    for idx, item in enumerate(payload.items):
        row_id = item.row_id or str(idx + 1)
        if not item.address or not item.address.strip():
            results.append(
                BatchAssessmentResultItem(
                    row_id=row_id,
                    address=item.address,
                    status="failed",
                    error="Address is required",
                )
            )
            continue

        req = AddressRequest(
            address=item.address,
            attributes=item.attributes,
            confirmed_fields=item.confirmed_fields,
            audience=item.audience,
            tags=item.tags,
            organization_id=organization_id,
            ruleset_id=ruleset.ruleset_id,
        )
        try:
            assessment, _ = _run_assessment(
                req,
                organization_id=organization_id,
                ruleset=ruleset,
                portfolio_name=payload.portfolio_name,
                tags=item.tags,
            )
            store.save(assessment)
            success_count += 1
            results.append(
                BatchAssessmentResultItem(
                    row_id=row_id,
                    address=item.address,
                    status="success",
                    assessment_id=assessment.assessment_id,
                    wildfire_risk_score=assessment.wildfire_risk_score,
                    insurance_readiness_score=assessment.insurance_readiness_score,
                    top_risk_drivers=assessment.top_risk_drivers,
                    readiness_blockers=assessment.readiness_blockers,
                    confidence_score=assessment.confidence_score,
                )
            )
            _log_audit(
                ctx=actor,
                entity_type="assessment",
                entity_id=assessment.assessment_id,
                action="assessment_created",
                organization_id=organization_id,
                metadata={"source": "batch", "job_id": job_id, "ruleset_id": ruleset.ruleset_id},
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                BatchAssessmentResultItem(
                    row_id=row_id,
                    address=item.address,
                    status="failed",
                    error=str(exc),
                )
            )

    successful = [r for r in results if r.status == "success"]
    avg_risk = round(sum((r.wildfire_risk_score or 0.0) for r in successful) / len(successful), 1) if successful else 0.0
    avg_readiness = (
        round(sum((r.insurance_readiness_score or 0.0) for r in successful) / len(successful), 1) if successful else 0.0
    )

    high_risk_count = sum(1 for r in successful if (r.wildfire_risk_score or 0.0) >= 70.0)
    blocker_count = sum(1 for r in successful if r.readiness_blockers)

    return BatchAssessmentResponse(
        portfolio_name=payload.portfolio_name,
        organization_id=organization_id,
        ruleset_id=ruleset.ruleset_id,
        job_id=job_id,
        total_properties=len(payload.items),
        completed_count=success_count,
        failed_count=len(payload.items) - success_count,
        high_risk_count=high_risk_count,
        blocker_count=blocker_count,
        average_wildfire_risk=avg_risk,
        average_insurance_readiness=avg_readiness,
        total=len(payload.items),
        succeeded=success_count,
        failed=len(payload.items) - success_count,
        results=results,
    )


def _execute_portfolio_job(
    *,
    job_id: str,
    payload: BatchAssessmentRequest,
    actor: ActorContext,
    organization_id: str,
    ruleset: UnderwritingRuleset,
) -> None:
    store.update_portfolio_job(job_id, status="running")

    try:
        batch = _run_batch(
            payload,
            actor=actor,
            organization_id=organization_id,
            ruleset=ruleset,
            job_id=job_id,
        )
        if batch.failed_count and batch.completed_count:
            status = "partial"
            error_summary = f"{batch.failed_count} row(s) failed"
        elif batch.failed_count and not batch.completed_count:
            status = "failed"
            error_summary = f"{batch.failed_count} row(s) failed"
        else:
            status = "completed"
            error_summary = None

        store.update_portfolio_job(
            job_id,
            status=status,
            result=batch.model_dump(mode="json"),
            error_summary=error_summary,
        )

        _log_audit(
            ctx=actor,
            entity_type="portfolio_job",
            entity_id=job_id,
            action="portfolio_job_completed",
            organization_id=organization_id,
            metadata={
                "status": status,
                "completed_count": batch.completed_count,
                "failed_count": batch.failed_count,
                "ruleset_id": ruleset.ruleset_id,
            },
        )
    except Exception as exc:  # pragma: no cover
        store.update_portfolio_job(job_id, status="failed", error_summary=str(exc))
        _log_audit(
            ctx=actor,
            entity_type="portfolio_job",
            entity_id=job_id,
            action="portfolio_job_failed",
            organization_id=organization_id,
            metadata={"error": str(exc)},
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/organizations", response_model=list[Organization], dependencies=[Depends(require_api_key)])
def list_organizations(ctx: ActorContext = Depends(get_actor_context)) -> list[Organization]:
    if ctx.user_role != "admin":
        org = store.get_organization(ctx.organization_id)
        return [org] if org else []
    return store.list_organizations()


@app.post("/organizations", response_model=Organization, dependencies=[Depends(require_api_key)])
def create_organization(
    payload: OrganizationCreate,
    ctx: ActorContext = Depends(get_actor_context),
) -> Organization:
    _require_role(ctx, {"admin"}, "Only admin can create organizations")
    created = store.create_organization(payload)
    _log_audit(
        ctx=ctx,
        entity_type="organization",
        entity_id=created.organization_id,
        action="organization_created",
        organization_id=created.organization_id,
        metadata=created.model_dump(mode="json"),
    )
    return created


@app.get("/organizations/{organization_id}", response_model=Organization, dependencies=[Depends(require_api_key)])
def get_organization(
    organization_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> Organization:
    _enforce_org_scope(ctx, organization_id)
    org = store.get_organization(organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org


@app.get("/underwriting/rulesets", response_model=list[UnderwritingRuleset], dependencies=[Depends(require_api_key)])
def list_rulesets(_: ActorContext = Depends(get_actor_context)) -> list[UnderwritingRuleset]:
    return store.list_rulesets()


@app.get(
    "/underwriting/rulesets/{ruleset_id}",
    response_model=UnderwritingRuleset,
    dependencies=[Depends(require_api_key)],
)
def get_ruleset(ruleset_id: str, _: ActorContext = Depends(get_actor_context)) -> UnderwritingRuleset:
    ruleset = store.get_ruleset(ruleset_id)
    if not ruleset:
        raise HTTPException(status_code=404, detail="Ruleset not found")
    return ruleset


@app.post(
    "/underwriting/rulesets",
    response_model=UnderwritingRuleset,
    dependencies=[Depends(require_api_key)],
)
def create_ruleset(
    payload: UnderwritingRulesetCreate,
    ctx: ActorContext = Depends(get_actor_context),
) -> UnderwritingRuleset:
    _require_role(ctx, {"admin"}, "Only admin can create rulesets")
    created = store.create_ruleset(payload)
    _log_audit(
        ctx=ctx,
        entity_type="ruleset",
        entity_id=created.ruleset_id,
        action="ruleset_created",
        organization_id=ctx.organization_id,
        metadata=created.model_dump(mode="json"),
    )
    return created


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(
    payload: AddressRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot create assessments")

    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)

    ruleset = _get_ruleset_or_default(payload.ruleset_id)

    result = _compute_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
    )
    store.save(result)
    _log_audit(
        ctx=ctx,
        entity_type="assessment",
        entity_id=result.assessment_id,
        action="assessment_created",
        organization_id=organization_id,
        metadata={"ruleset_id": ruleset.ruleset_id},
    )
    return result


@app.post("/risk/reassess/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def reassess_risk(
    assessment_id: str,
    payload: ReassessmentRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot create reassessments")

    existing = store.get(assessment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, existing.organization_id)

    base_req = _payload_from_assessment(existing)
    merged_attrs = _merge_attributes(base_req.attributes, payload.attributes)
    merged_confirmed = sorted(set(base_req.confirmed_fields + payload.confirmed_fields))

    ruleset = _get_ruleset_or_default(payload.ruleset_id or existing.ruleset_id)

    req = AddressRequest(
        address=base_req.address,
        attributes=merged_attrs,
        confirmed_fields=merged_confirmed,
        audience=payload.audience or base_req.audience,
        tags=base_req.tags,
        organization_id=existing.organization_id,
        ruleset_id=ruleset.ruleset_id,
    )
    result = _compute_assessment(
        req,
        organization_id=existing.organization_id,
        ruleset=ruleset,
        portfolio_name=existing.portfolio_name,
        tags=existing.tags,
    )
    result.review_status = existing.review_status
    result.workflow_state = existing.workflow_state
    result.assigned_reviewer = existing.assigned_reviewer
    result.assigned_role = existing.assigned_role

    store.save(result)
    _log_audit(
        ctx=ctx,
        entity_type="assessment",
        entity_id=result.assessment_id,
        action="reassessment_created",
        organization_id=result.organization_id,
        metadata={"source_assessment_id": assessment_id, "ruleset_id": ruleset.ruleset_id},
    )
    return result


@app.post("/risk/simulate", response_model=SimulationResult, dependencies=[Depends(require_api_key)])
def simulate_risk(
    payload: SimulationRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> SimulationResult:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot run simulations")

    if payload.assessment_id:
        existing = store.get(payload.assessment_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Assessment not found")
        _enforce_org_scope(ctx, existing.organization_id)
        base_req = _payload_from_assessment(existing)
        org_id = existing.organization_id
        ruleset = _get_ruleset_or_default(existing.ruleset_id)
    else:
        if not payload.address:
            raise HTTPException(status_code=400, detail="Provide assessment_id or address for simulation")
        base_req = AddressRequest(
            address=payload.address,
            attributes=payload.attributes,
            confirmed_fields=payload.confirmed_fields,
            audience=payload.audience,
            organization_id=ctx.organization_id,
            ruleset_id="default",
        )
        org_id = ctx.organization_id
        ruleset = _get_ruleset_or_default("default")

    scenario_override_values = _attributes_to_dict(payload.scenario_overrides)
    if not scenario_override_values:
        raise HTTPException(status_code=400, detail="Simulation requires at least one scenario override")

    baseline_attrs = _merge_attributes(base_req.attributes, payload.attributes)
    baseline_confirmed = sorted(set(base_req.confirmed_fields + payload.confirmed_fields))
    baseline_req = AddressRequest(
        address=base_req.address,
        attributes=baseline_attrs,
        confirmed_fields=baseline_confirmed,
        audience=base_req.audience,
        tags=base_req.tags,
        organization_id=org_id,
        ruleset_id=ruleset.ruleset_id,
    )

    baseline = _compute_assessment(
        baseline_req,
        organization_id=org_id,
        ruleset=ruleset,
    )

    simulated_attrs = _merge_attributes(baseline_attrs, payload.scenario_overrides)
    simulated_confirmed = sorted(set(baseline_confirmed + payload.scenario_confirmed_fields))
    simulated_req = AddressRequest(
        address=base_req.address,
        attributes=simulated_attrs,
        confirmed_fields=simulated_confirmed,
        audience=base_req.audience,
        tags=base_req.tags,
        organization_id=org_id,
        ruleset_id=ruleset.ruleset_id,
    )
    simulated = _compute_assessment(
        simulated_req,
        organization_id=org_id,
        ruleset=ruleset,
    )

    # Persist simulation outputs for homeowner follow-up/reporting flows.
    if not payload.assessment_id:
        store.save(baseline)
    store.save(simulated)

    changed_inputs: Dict[str, Dict[str, object]] = {}
    before = baseline.property_facts
    after = simulated.property_facts
    for key in sorted(set(before.keys()) | set(after.keys())):
        if before.get(key) != after.get(key):
            changed_inputs[key] = {"before": before.get(key), "after": after.get(key)}

    sim_result = SimulationResult(
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
        base_assessment_id=payload.assessment_id or baseline.assessment_id,
        simulated_assessment_id=simulated.assessment_id,
        base_scores=baseline.risk_scores,
        simulated_scores=simulated.risk_scores,
        score_delta=SimulationDelta(
            wildfire_risk_score_delta=round(simulated.wildfire_risk_score - baseline.wildfire_risk_score, 1),
            insurance_readiness_score_delta=round(
                simulated.insurance_readiness_score - baseline.insurance_readiness_score, 1
            ),
        ),
        base_confidence=baseline.confidence,
        simulated_confidence=simulated.confidence,
        base_assumptions=baseline.assumptions,
        simulated_assumptions=simulated.assumptions,
        summary=(
            f"Wildfire risk changed by {round(simulated.wildfire_risk_score - baseline.wildfire_risk_score, 1)} "
            f"and insurance readiness changed by "
            f"{round(simulated.insurance_readiness_score - baseline.insurance_readiness_score, 1)}."
        ),
    )

    if payload.assessment_id:
        store.save_simulation(payload.assessment_id, payload.scenario_name, sim_result.model_dump(mode="json"))

    _log_audit(
        ctx=ctx,
        entity_type="simulation",
        entity_id=payload.assessment_id or baseline.assessment_id,
        action="simulation_created",
        organization_id=org_id,
        metadata={"scenario_name": payload.scenario_name},
    )

    return sim_result


@app.post("/risk/debug", dependencies=[Depends(require_api_key)])
def debug_risk(payload: AddressRequest, ctx: ActorContext = Depends(get_actor_context)) -> dict:
    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)
    _, debug_payload = _run_assessment(payload, organization_id=organization_id, ruleset=ruleset)
    return debug_payload


@app.post("/portfolio/assess", response_model=BatchAssessmentResponse, dependencies=[Depends(require_api_key)])
def portfolio_assess(
    payload: BatchAssessmentRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> BatchAssessmentResponse:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot run batch assessments")
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch request must include at least one property item")

    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)

    batch = _run_batch(payload, actor=ctx, organization_id=organization_id, ruleset=ruleset)
    _log_audit(
        ctx=ctx,
        entity_type="portfolio",
        entity_id=payload.portfolio_name or "adhoc_batch",
        action="portfolio_batch_completed",
        organization_id=organization_id,
        metadata={
            "completed_count": batch.completed_count,
            "failed_count": batch.failed_count,
            "ruleset_id": ruleset.ruleset_id,
        },
    )
    return batch


@app.post("/portfolio/jobs", response_model=PortfolioJobStatus, dependencies=[Depends(require_api_key)])
def create_portfolio_job(
    payload: PortfolioJobCreate,
    background_tasks: BackgroundTasks,
    ctx: ActorContext = Depends(get_actor_context),
) -> PortfolioJobStatus:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot create portfolio jobs")
    if not payload.items:
        raise HTTPException(status_code=400, detail="Job request must include at least one property item")

    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)

    request_blob = payload.model_dump(mode="json")
    request_blob["organization_id"] = organization_id
    request_blob["ruleset_id"] = ruleset.ruleset_id

    job_id = store.create_portfolio_job(organization_id=organization_id, payload=request_blob, status="queued")

    batch_payload = BatchAssessmentRequest(
        portfolio_name=payload.portfolio_name,
        organization_id=organization_id,
        ruleset_id=ruleset.ruleset_id,
        items=payload.items,
    )

    _log_audit(
        ctx=ctx,
        entity_type="portfolio_job",
        entity_id=job_id,
        action="portfolio_job_created",
        organization_id=organization_id,
        metadata={"ruleset_id": ruleset.ruleset_id, "item_count": len(payload.items)},
    )

    if payload.process_immediately or len(payload.items) <= 25:
        _execute_portfolio_job(
            job_id=job_id,
            payload=batch_payload,
            actor=ctx,
            organization_id=organization_id,
            ruleset=ruleset,
        )
    else:
        background_tasks.add_task(
            _execute_portfolio_job,
            job_id=job_id,
            payload=batch_payload,
            actor=ctx,
            organization_id=organization_id,
            ruleset=ruleset,
        )

    record = store.get_portfolio_job(job_id)
    if not record:
        raise HTTPException(status_code=500, detail="Failed to load created job")
    return _to_job_status(record)


@app.get("/portfolio/jobs/{job_id}", response_model=PortfolioJobStatus, dependencies=[Depends(require_api_key)])
def get_portfolio_job(job_id: str, ctx: ActorContext = Depends(get_actor_context)) -> PortfolioJobStatus:
    record = store.get_portfolio_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _enforce_org_scope(ctx, record["organization_id"])
    return _to_job_status(record)


@app.get(
    "/portfolio/jobs/{job_id}/results",
    response_model=PortfolioJobResultsResponse,
    dependencies=[Depends(require_api_key)],
)
def get_portfolio_job_results(job_id: str, ctx: ActorContext = Depends(get_actor_context)) -> PortfolioJobResultsResponse:
    record = store.get_portfolio_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _enforce_org_scope(ctx, record["organization_id"])

    status = _to_job_status(record)
    result = record.get("result") or {}
    rows = [BatchAssessmentResultItem.model_validate(x) for x in result.get("results", [])]
    return PortfolioJobResultsResponse(job=status, results=rows)


@app.get(
    "/portfolio/jobs/{job_id}/export/csv",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
)
def export_portfolio_job_csv(job_id: str, ctx: ActorContext = Depends(get_actor_context)) -> PlainTextResponse:
    record = store.get_portfolio_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _enforce_org_scope(ctx, record["organization_id"])

    result = record.get("result") or {}
    rows = [BatchAssessmentResultItem.model_validate(x) for x in result.get("results", [])]
    csv_text = _assessment_to_csv(rows)
    _log_audit(
        ctx=ctx,
        entity_type="portfolio_job",
        entity_id=job_id,
        action="job_csv_exported",
        organization_id=record["organization_id"],
        metadata={"row_count": len(rows)},
    )
    return PlainTextResponse(content=csv_text, media_type="text/csv")


@app.get("/portfolio/jobs/{job_id}/report-pack", dependencies=[Depends(require_api_key)])
def portfolio_job_report_pack(job_id: str, ctx: ActorContext = Depends(get_actor_context)) -> dict:
    record = store.get_portfolio_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    _enforce_org_scope(ctx, record["organization_id"])

    status = _to_job_status(record)
    result = record.get("result") or {}
    rows = [BatchAssessmentResultItem.model_validate(x) for x in result.get("results", [])]

    reports = []
    for row in rows:
        if row.assessment_id:
            assessment = store.get(row.assessment_id)
            if assessment:
                reports.append(
                    {
                        "assessment_id": assessment.assessment_id,
                        "address": assessment.address,
                        "wildfire_risk_score": assessment.wildfire_risk_score,
                        "insurance_readiness_score": assessment.insurance_readiness_score,
                        "top_risk_drivers": assessment.top_risk_drivers,
                        "readiness_blockers": assessment.readiness_blockers,
                        "review_status": assessment.review_status,
                    }
                )

    return {
        "job": status.model_dump(mode="json"),
        "summary": {
            "total_properties": status.total_properties,
            "completed_count": status.completed_count,
            "failed_count": status.failed_count,
            "high_risk_count": status.high_risk_count,
            "blocker_count": status.blocker_count,
            "average_wildfire_risk": status.average_wildfire_risk,
            "average_insurance_readiness": status.average_insurance_readiness,
        },
        "reports": reports,
    }


@app.get("/portfolio/jobs/summary", response_model=PortfolioJobsSummary, dependencies=[Depends(require_api_key)])
def portfolio_jobs_summary(
    organization_id: str | None = Query(default=None),
    ctx: ActorContext = Depends(get_actor_context),
) -> PortfolioJobsSummary:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)
    return store.summarize_portfolio_jobs(organization_id=org)


@app.post("/portfolio/import/csv", response_model=CSVImportResponse, dependencies=[Depends(require_api_key)])
def import_portfolio_csv(
    payload: CSVImportRequest,
    background_tasks: BackgroundTasks,
    ctx: ActorContext = Depends(get_actor_context),
) -> CSVImportResponse:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot import portfolio CSV")

    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)

    reader = csv.DictReader(io.StringIO(payload.csv_text))
    validation_errors: list[CSVImportError] = []
    items = []

    for idx, row in enumerate(reader, start=2):
        address = (row.get("address") or "").strip()
        if not address:
            validation_errors.append(CSVImportError(row_number=idx, address=None, error="Missing address"))
            continue

        attrs = PropertyAttributes(
            roof_type=(row.get("roof_type") or None),
            vent_type=(row.get("vent_type") or None),
            siding_type=(row.get("siding_type") or None),
            window_type=(row.get("window_type") or None),
            defensible_space_ft=float(row["defensible_space_ft"]) if row.get("defensible_space_ft") else None,
            vegetation_condition=(row.get("vegetation_condition") or None),
            driveway_access_notes=(row.get("driveway_access_notes") or None),
            construction_year=int(row["construction_year"]) if row.get("construction_year") else None,
            inspection_notes=(row.get("inspection_notes") or None),
        )
        confirmed = [f.strip() for f in (row.get("confirmed_fields") or "").split(",") if f.strip()]
        row_tags = [f.strip() for f in (row.get("tags") or "").split(",") if f.strip()]

        items.append(
            {
                "row_id": str(idx),
                "address": address,
                "attributes": attrs.model_dump(mode="json", exclude_none=True),
                "confirmed_fields": confirmed,
                "audience": (row.get("audience") or payload.audience),
                "tags": sorted(set(payload.tags + row_tags)),
            }
        )

    if not items:
        raise HTTPException(status_code=400, detail="CSV import had no valid rows")

    job_payload = PortfolioJobCreate(
        portfolio_name=payload.portfolio_name,
        organization_id=organization_id,
        ruleset_id=ruleset.ruleset_id,
        process_immediately=payload.process_immediately,
        items=items,
    )
    job_status = create_portfolio_job(job_payload, background_tasks, ctx)

    return CSVImportResponse(
        row_count=len(items) + len(validation_errors),
        accepted_count=len(items),
        rejected_count=len(validation_errors),
        validation_errors=validation_errors,
        job=job_status,
    )


@app.get(
    "/portfolio/{portfolio_name}/export/csv",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
)
def export_portfolio_csv(
    portfolio_name: str,
    organization_id: str | None = Query(default=None),
    ctx: ActorContext = Depends(get_actor_context),
) -> PlainTextResponse:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)

    rows = store.list_assessments_by_portfolio(portfolio_name=portfolio_name, organization_id=org)
    csv_text = _list_item_to_csv(rows)
    _log_audit(
        ctx=ctx,
        entity_type="portfolio",
        entity_id=portfolio_name,
        action="portfolio_csv_exported",
        organization_id=org,
        metadata={"row_count": len(rows)},
    )
    return PlainTextResponse(content=csv_text, media_type="text/csv")


@app.get("/report/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def get_report(
    assessment_id: str,
    audience: Audience | None = Query(default=None),
    audience_mode: Audience | None = Query(default=None),
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    return _apply_audience_view(result, actor=ctx, audience=audience, audience_mode=audience_mode)


@app.get("/report/{assessment_id}/export", response_model=ReportExport, dependencies=[Depends(require_api_key)])
def export_report(
    assessment_id: str,
    audience: Audience | None = Query(default=None),
    audience_mode: Audience | None = Query(default=None),
    ctx: ActorContext = Depends(get_actor_context),
) -> ReportExport:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    export = _build_report_export(result, actor=ctx, audience=audience, audience_mode=audience_mode)
    _log_audit(
        ctx=ctx,
        entity_type="report",
        entity_id=assessment_id,
        action="report_exported",
        organization_id=result.organization_id,
        metadata={"audience": export.audience_mode},
    )
    return export


@app.get("/report/{assessment_id}/view", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
def view_report(
    assessment_id: str,
    audience: Audience | None = Query(default=None),
    audience_mode: Audience | None = Query(default=None),
    ctx: ActorContext = Depends(get_actor_context),
) -> HTMLResponse:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    html = _build_report_html(result, actor=ctx, audience=audience, audience_mode=audience_mode)
    _log_audit(
        ctx=ctx,
        entity_type="report",
        entity_id=assessment_id,
        action="report_viewed",
        organization_id=result.organization_id,
        metadata={"audience": audience or audience_mode or _default_audience_for_role(ctx.user_role)},
    )
    return HTMLResponse(html)


@app.get("/portfolio", response_model=PortfolioResponse, dependencies=[Depends(require_api_key)])
def get_portfolio(
    organization_id: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    min_risk: float | None = Query(default=None, ge=0, le=100),
    max_risk: float | None = Query(default=None, ge=0, le=100),
    min_readiness: float | None = Query(default=None, ge=0, le=100),
    max_readiness: float | None = Query(default=None, ge=0, le=100),
    readiness_blocker: str | None = Query(default=None),
    confidence_min: float | None = Query(default=None, ge=0, le=100),
    audience: Audience | None = Query(default=None),
    tag: str | None = Query(default=None),
    portfolio_name: str | None = Query(default=None),
    workflow_state: WorkflowState | None = Query(default=None),
    assigned_reviewer: str | None = Query(default=None),
    created_after: str | None = Query(default=None),
    created_before: str | None = Query(default=None),
    recent_days: int | None = Query(default=None, ge=0),
    limit: int = Query(default=20, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    ctx: ActorContext = Depends(get_actor_context),
) -> PortfolioResponse:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)

    items, total, summary = store.query_assessments(
        organization_id=org,
        sort_by=sort_by,
        sort_dir=sort_dir,
        min_risk=min_risk,
        max_risk=max_risk,
        min_readiness=min_readiness,
        max_readiness=max_readiness,
        readiness_blocker=readiness_blocker,
        confidence_min=confidence_min,
        audience=audience,
        tag=tag,
        portfolio_name=portfolio_name,
        workflow_state=workflow_state,
        assigned_reviewer=assigned_reviewer,
        created_after=created_after,
        created_before=created_before,
        recent_days=recent_days,
        limit=limit,
        offset=offset,
    )
    return PortfolioResponse(limit=limit, offset=offset, total=total, items=items, summary=summary)


@app.get("/assessments", response_model=list[AssessmentListItem], dependencies=[Depends(require_api_key)])
def list_assessments(
    organization_id: str | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    min_risk: float | None = Query(default=None, ge=0, le=100),
    max_risk: float | None = Query(default=None, ge=0, le=100),
    min_readiness: float | None = Query(default=None, ge=0, le=100),
    max_readiness: float | None = Query(default=None, ge=0, le=100),
    readiness_blocker: str | None = Query(default=None),
    confidence_min: float | None = Query(default=None, ge=0, le=100),
    audience: Audience | None = Query(default=None),
    tag: str | None = Query(default=None),
    portfolio_name: str | None = Query(default=None),
    workflow_state: WorkflowState | None = Query(default=None),
    assigned_reviewer: str | None = Query(default=None),
    created_after: str | None = Query(default=None),
    created_before: str | None = Query(default=None),
    recent_days: int | None = Query(default=None, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    ctx: ActorContext = Depends(get_actor_context),
) -> list[AssessmentListItem]:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)
    return store.list_assessments(
        limit=limit,
        offset=offset,
        organization_id=org,
        sort_by=sort_by,
        sort_dir=sort_dir,
        min_risk=min_risk,
        max_risk=max_risk,
        min_readiness=min_readiness,
        max_readiness=max_readiness,
        readiness_blocker=readiness_blocker,
        confidence_min=confidence_min,
        audience=audience,
        tag=tag,
        portfolio_name=portfolio_name,
        workflow_state=workflow_state,
        assigned_reviewer=assigned_reviewer,
        created_after=created_after,
        created_before=created_before,
        recent_days=recent_days,
    )


@app.get("/assessments/summary", response_model=AssessmentSummaryResponse, dependencies=[Depends(require_api_key)])
def assessments_summary(
    organization_id: str | None = Query(default=None),
    min_risk: float | None = Query(default=None, ge=0, le=100),
    max_risk: float | None = Query(default=None, ge=0, le=100),
    min_readiness: float | None = Query(default=None, ge=0, le=100),
    max_readiness: float | None = Query(default=None, ge=0, le=100),
    readiness_blocker: str | None = Query(default=None),
    confidence_min: float | None = Query(default=None, ge=0, le=100),
    audience: Audience | None = Query(default=None),
    tag: str | None = Query(default=None),
    portfolio_name: str | None = Query(default=None),
    workflow_state: WorkflowState | None = Query(default=None),
    assigned_reviewer: str | None = Query(default=None),
    created_after: str | None = Query(default=None),
    created_before: str | None = Query(default=None),
    recent_days: int | None = Query(default=None, ge=0),
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentSummaryResponse:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)

    summary = store.summary_assessments(
        organization_id=org,
        min_risk=min_risk,
        max_risk=max_risk,
        min_readiness=min_readiness,
        max_readiness=max_readiness,
        readiness_blocker=readiness_blocker,
        confidence_min=confidence_min,
        audience=audience,
        tag=tag,
        portfolio_name=portfolio_name,
        workflow_state=workflow_state,
        assigned_reviewer=assigned_reviewer,
        created_after=created_after,
        created_before=created_before,
        recent_days=recent_days,
    )
    return AssessmentSummaryResponse(summary=summary)


@app.post(
    "/assessments/{assessment_id}/annotations",
    response_model=AssessmentAnnotation,
    dependencies=[Depends(require_api_key)],
)
def add_assessment_annotation(
    assessment_id: str,
    payload: AssessmentAnnotationCreate,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentAnnotation:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot add annotations")

    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    if payload.author_role == "insurer" and ctx.user_role not in {"admin", "underwriter"}:
        raise HTTPException(status_code=403, detail="Only admin/underwriter can create insurer annotations")

    annotation = store.save_annotation(
        assessment_id=assessment_id,
        organization_id=assessment.organization_id,
        author_role=payload.author_role,
        note=payload.note,
        tags=payload.tags,
        visibility=payload.visibility,
        review_status=payload.review_status,
    )

    _log_audit(
        ctx=ctx,
        entity_type="annotation",
        entity_id=annotation.annotation_id,
        action="annotation_added",
        organization_id=assessment.organization_id,
        metadata={"assessment_id": assessment_id, "review_status": annotation.review_status},
    )
    return annotation


@app.get(
    "/assessments/{assessment_id}/annotations",
    response_model=list[AssessmentAnnotation],
    dependencies=[Depends(require_api_key)],
)
def list_assessment_annotations(
    assessment_id: str,
    visibility: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    ctx: ActorContext = Depends(get_actor_context),
) -> list[AssessmentAnnotation]:
    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    annotations = store.list_annotations(
        assessment_id=assessment_id,
        organization_id=assessment.organization_id,
        limit=limit,
    )
    if visibility:
        annotations = [a for a in annotations if a.visibility == visibility]
    return annotations


@app.post(
    "/assessment/{assessment_id}/annotations",
    response_model=AssessmentAnnotation,
    dependencies=[Depends(require_api_key)],
)
def add_assessment_annotation_alias(
    assessment_id: str,
    payload: AssessmentAnnotationCreate,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentAnnotation:
    return add_assessment_annotation(assessment_id, payload, ctx)


@app.get(
    "/assessment/{assessment_id}/annotations",
    response_model=list[AssessmentAnnotation],
    dependencies=[Depends(require_api_key)],
)
def list_assessment_annotations_alias(
    assessment_id: str,
    visibility: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    ctx: ActorContext = Depends(get_actor_context),
) -> list[AssessmentAnnotation]:
    return list_assessment_annotations(assessment_id, visibility, limit, ctx)


@app.put(
    "/assessments/{assessment_id}/review-status",
    response_model=AssessmentReviewStatus,
    dependencies=[Depends(require_api_key)],
)
def update_review_status(
    assessment_id: str,
    payload: AssessmentReviewStatusUpdate,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentReviewStatus:
    _require_role(ctx, REVIEW_EDIT_ROLES, "Only admin/underwriter can change review status")

    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    updated = store.set_review_status(assessment_id, assessment.organization_id, payload.review_status)
    _log_audit(
        ctx=ctx,
        entity_type="review",
        entity_id=assessment_id,
        action="review_status_changed",
        organization_id=assessment.organization_id,
        metadata={"review_status": payload.review_status},
    )
    return updated


@app.get(
    "/assessments/{assessment_id}/review-status",
    response_model=AssessmentReviewStatus,
    dependencies=[Depends(require_api_key)],
)
def get_review_status(
    assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentReviewStatus:
    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    status = store.get_review_status(assessment_id)
    if status:
        return status
    return store.set_review_status(assessment_id, assessment.organization_id, "pending")


@app.post(
    "/assessment/{assessment_id}/assign",
    response_model=AssessmentWorkflowInfo,
    dependencies=[Depends(require_api_key)],
)
def assign_assessment(
    assessment_id: str,
    payload: AssessmentAssignmentRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentWorkflowInfo:
    _require_role(ctx, WORKFLOW_EDIT_ROLES, "Role cannot assign reviewers")

    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    updated = store.set_assignment(
        assessment_id=assessment_id,
        organization_id=assessment.organization_id,
        assigned_reviewer=payload.assigned_reviewer,
        assigned_role=payload.assigned_role,
    )
    _log_audit(
        ctx=ctx,
        entity_type="workflow",
        entity_id=assessment_id,
        action="assignment_changed",
        organization_id=assessment.organization_id,
        metadata=payload.model_dump(mode="json"),
    )
    return updated


@app.post(
    "/assessment/{assessment_id}/workflow",
    response_model=AssessmentWorkflowInfo,
    dependencies=[Depends(require_api_key)],
)
def update_assessment_workflow(
    assessment_id: str,
    payload: AssessmentWorkflowUpdateRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentWorkflowInfo:
    _require_role(ctx, WORKFLOW_EDIT_ROLES, "Role cannot update workflow state")

    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    current = store.get_workflow(assessment_id)
    current_state = current.workflow_state if current else "new"
    if payload.workflow_state not in WORKFLOW_TRANSITIONS[current_state]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workflow transition: {current_state} -> {payload.workflow_state}",
        )

    updated = store.set_workflow(
        assessment_id=assessment_id,
        organization_id=assessment.organization_id,
        workflow_state=payload.workflow_state,
    )
    _log_audit(
        ctx=ctx,
        entity_type="workflow",
        entity_id=assessment_id,
        action="workflow_state_changed",
        organization_id=assessment.organization_id,
        metadata={"from": current_state, "to": payload.workflow_state},
    )
    return updated


@app.get(
    "/assessment/{assessment_id}/workflow",
    response_model=AssessmentWorkflowInfo,
    dependencies=[Depends(require_api_key)],
)
def get_assessment_workflow(
    assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentWorkflowInfo:
    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)

    workflow = store.get_workflow(assessment_id)
    if workflow:
        return workflow

    return store.set_workflow(
        assessment_id=assessment_id,
        organization_id=assessment.organization_id,
        workflow_state="new",
        assigned_reviewer=None,
        assigned_role=None,
    )


@app.get(
    "/assessments/{assessment_id}/compare/{other_assessment_id}",
    response_model=AssessmentComparisonResult,
    dependencies=[Depends(require_api_key)],
)
def compare_assessments_pair(
    assessment_id: str,
    other_assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentComparisonResult:
    base = store.get(assessment_id)
    if not base:
        raise HTTPException(status_code=404, detail="Base assessment not found")
    other = store.get(other_assessment_id)
    if not other:
        raise HTTPException(status_code=404, detail="Other assessment not found")

    _enforce_org_scope(ctx, base.organization_id)
    _enforce_org_scope(ctx, other.organization_id)

    return _compare_results(base, other)


@app.get(
    "/assessment/{assessment_id}/compare/{other_assessment_id}",
    response_model=AssessmentComparisonResult,
    dependencies=[Depends(require_api_key)],
)
def compare_assessments_pair_alias(
    assessment_id: str,
    other_assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentComparisonResult:
    return compare_assessments_pair(assessment_id, other_assessment_id, ctx)


@app.get(
    "/assessments/compare",
    response_model=AssessmentComparisonResponse,
    dependencies=[Depends(require_api_key)],
)
def compare_assessments_multi(
    ids: str = Query(..., min_length=3),
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentComparisonResponse:
    requested_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if len(requested_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two assessment ids")

    loaded: list[AssessmentResult] = []
    for assessment_id in requested_ids:
        row = store.get(assessment_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"Assessment not found: {assessment_id}")
        _enforce_org_scope(ctx, row.organization_id)
        loaded.append(row)

    base = loaded[0]
    comparisons = [_compare_results(base, other) for other in loaded[1:]]
    return AssessmentComparisonResponse(requested_ids=requested_ids, comparisons=comparisons)


@app.get(
    "/assessments/{assessment_id}/scenarios",
    response_model=list[SimulationScenarioItem],
    dependencies=[Depends(require_api_key)],
)
def list_assessment_scenarios(
    assessment_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    ctx: ActorContext = Depends(get_actor_context),
) -> list[SimulationScenarioItem]:
    assessment = store.get(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, assessment.organization_id)
    return store.list_scenarios(assessment_id=assessment_id, limit=limit)


@app.get("/audit/events", response_model=list[AuditEvent], dependencies=[Depends(require_api_key)])
def list_audit_events(
    organization_id: str | None = Query(default=None),
    entity_type: str | None = Query(default=None),
    action: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ctx: ActorContext = Depends(get_actor_context),
) -> list[AuditEvent]:
    org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, org)
    return store.list_audit_events(
        organization_id=org,
        entity_type=entity_type,
        action=action,
        limit=limit,
        offset=offset,
    )


@app.get("/admin/summary", response_model=AdminSummary, dependencies=[Depends(require_api_key)])
def admin_summary(
    organization_id: str | None = Query(default=None),
    recent_days: int = Query(default=30, ge=0, le=3650),
    ctx: ActorContext = Depends(get_actor_context),
) -> AdminSummary:
    if ctx.user_role == "admin":
        return store.build_admin_summary(organization_id=organization_id, recent_days=recent_days)

    target_org = organization_id or ctx.organization_id
    _enforce_org_scope(ctx, target_org)
    return store.build_admin_summary(organization_id=target_org, recent_days=recent_days)


@app.get(
    "/organizations/{organization_id}/summary",
    response_model=AdminSummary,
    dependencies=[Depends(require_api_key)],
)
def organization_summary(
    organization_id: str,
    recent_days: int = Query(default=30, ge=0, le=3650),
    ctx: ActorContext = Depends(get_actor_context),
) -> AdminSummary:
    _enforce_org_scope(ctx, organization_id)
    return store.build_admin_summary(organization_id=organization_id, recent_days=recent_days)
