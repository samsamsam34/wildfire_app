from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from backend.auth import require_api_key
from backend.assessment_map import build_assessment_map_payload
from backend.benchmarking import (
    build_benchmark_hints_for_assessment,
)
from backend.defensible_space import (
    build_defensible_space_analysis,
    build_prioritized_vegetation_actions,
    build_top_near_structure_risk_drivers,
)
from backend.database import AssessmentStore, DEFAULT_ORG_ID
from backend.data_prep.region_lookup import (
    find_region_for_point as lookup_region_for_point,
)
from backend.geocoding import Geocoder, GeocodingError, normalize_address
from backend.homeowner_report import build_homeowner_report, render_homeowner_report_pdf
from backend.layer_diagnostics import LAYER_SPECS
from backend.mitigation import build_mitigation_plan
from backend.models import (
    AssessmentDiagnostics,
    AssessmentStatus,
    AddressRequest,
    AdminSummary,
    AssessmentAnnotation,
    AssessmentAnnotationCreate,
    AssessmentAssignmentRequest,
    AssessmentComparisonItem,
    AssessmentComparisonResponse,
    AssessmentComparisonResult,
    AssessmentListItem,
    AssessmentMapPayload,
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
    ConfidencePenalty,
    DataProvenanceBlock,
    DataProvenanceSummary,
    EvidenceQualitySummary,
    EligibilityStatus,
    EnvironmentalFactors,
    FreshnessStatus,
    InputSourceMetadata,
    GeocodeDebugRequest,
    GeocodingDetails,
    HomeownerReport,
    ModelGovernanceInfo,
    NearStructureAction,
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
    RegionBoundingBox,
    RegionCoverageRequest,
    RegionCoverageStatus,
    RegionResolution,
    RegionPrepareRequest,
    RegionPrepJobStatus,
    ReassessmentRequest,
    ReportExport,
    RiskScores,
    SimulationDelta,
    SimulationRequest,
    SimulationResult,
    SimulationScenarioItem,
    SourceType,
    ProviderStatus,
    ScoreFamilyInputQuality,
    ScoreEligibility,
    ScoreEvidenceFactor,
    ScoreEvidenceLedger,
    LayerCoverageAuditItem,
    LayerCoverageSummary,
    ScoreSummaries,
    ScoreSectionSummary,
    SubmodelScore,
    UnderwritingRuleset,
    UnderwritingRulesetCreate,
    UserRole,
    WeightedContribution,
    WorkflowState,
)
from backend.normalization import normalize_property_attributes, normalized_attribute_changes
from backend.risk_engine import RiskComputation, RiskEngine
from backend.scoring_config import load_scoring_config
from backend.version import (
    API_VERSION,
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    DEFAULT_RULESET_VERSION,
    FACTOR_SCHEMA_VERSION,
    MODEL_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    build_model_governance,
    compare_model_governance,
)
from backend.wildfire_data import (
    WildfireDataClient,
    compute_environmental_data_completeness,
)

app = FastAPI(title="WildfireRisk Advisor API", version=PRODUCT_VERSION)
LOGGER = logging.getLogger("wildfire_app.assessment")

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

ACCESS_PROVISIONAL_NOTE = "Access exposure is advisory and excluded from weighted wildfire scoring."

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
DIRECT_SOURCE_TYPES = {"observed", "footprint_derived"}
INFERRED_SOURCE_TYPES = {"user_provided", "public_record_inferred"}
LOW_QUALITY_SOURCE_TYPES = {"missing", "heuristic"}

FRESHNESS_POLICY_DEFAULTS_DAYS: dict[str, tuple[int, int]] = {
    "environmental_raster": (180, 365),
    "fire_history_layer": (365, 730),
    "user_provided": (30, 180),
    "public_record_inferred": (365, 1095),
    "footprint_derived": (365, 1095),
}

CRITICAL_PROVENANCE_FIELDS = {
    "burn_probability",
    "wildfire_hazard",
    "slope",
    "fuel_model",
    "canopy_cover",
    "historic_fire_distance",
    "wildland_distance",
    "roof_type",
    "vent_type",
    "defensible_space_ft",
    "zone_0_5_ft",
    "zone_5_30_ft",
}

SCORE_FAMILY_FIELDS = {
    "site_hazard": [
        "burn_probability",
        "wildfire_hazard",
        "slope",
        "fuel_model",
        "canopy_cover",
        "historic_fire_distance",
        "wildland_distance",
    ],
    "home_vulnerability": [
        "roof_type",
        "vent_type",
        "defensible_space_ft",
        "vegetation_condition",
        "construction_year",
        "zone_0_5_ft",
        "zone_5_30_ft",
        "zone_30_100_ft",
        "footprint_used",
    ],
    "insurance_readiness": [
        "roof_type",
        "vent_type",
        "defensible_space_ft",
        "construction_year",
        "vegetation_condition",
        "zone_0_5_ft",
        "zone_5_30_ft",
        "fuel_model",
        "wildland_distance",
        "historic_fire_distance",
    ],
}

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


def _build_result_governance(
    *,
    ruleset_version: str,
    region_data_version: str | None,
    benchmark_pack_version: str | None = None,
    data_bundle_version: str | None = None,
) -> dict[str, str | None]:
    return build_model_governance(
        ruleset_version=ruleset_version,
        benchmark_pack_version=benchmark_pack_version or BENCHMARK_PACK_VERSION,
        region_data_version=region_data_version,
        data_bundle_version=data_bundle_version,
    )


def _refresh_result_governance(result: AssessmentResult) -> AssessmentResult:
    governance = _build_result_governance(
        ruleset_version=result.ruleset_version,
        benchmark_pack_version=result.benchmark_pack_version or BENCHMARK_PACK_VERSION,
        region_data_version=result.region_data_version,
        data_bundle_version=result.data_bundle_version,
    )
    result.model_governance = ModelGovernanceInfo.model_validate(governance)
    result.model_version = str(governance["scoring_model_version"] or MODEL_VERSION)
    result.product_version = str(governance["product_version"] or PRODUCT_VERSION)
    result.api_version = str(governance["api_version"] or API_VERSION)
    result.scoring_model_version = str(governance["scoring_model_version"] or MODEL_VERSION)
    result.ruleset_version = str(governance["ruleset_version"] or result.ruleset_version)
    result.rules_logic_version = str(governance["rules_logic_version"] or RULESET_LOGIC_VERSION)
    result.factor_schema_version = str(governance["factor_schema_version"] or FACTOR_SCHEMA_VERSION)
    result.benchmark_pack_version = governance.get("benchmark_pack_version")
    result.calibration_version = str(governance["calibration_version"] or CALIBRATION_VERSION)
    result.region_data_version = governance.get("region_data_version")
    result.data_bundle_version = governance.get("data_bundle_version")
    return result


def _merge_attributes(base: PropertyAttributes, overrides: PropertyAttributes) -> PropertyAttributes:
    merged = base.model_dump(exclude_none=True)
    merged.update(overrides.model_dump(exclude_none=True))
    return PropertyAttributes.model_validate(merged)


def _ring_density_value(property_level_context: dict[str, Any], *keys: str) -> float | None:
    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    if not isinstance(rings, dict):
        return None
    for key in keys:
        zone = rings.get(key) or rings.get(key.replace("ring_", "zone_"))
        if isinstance(zone, dict):
            value = zone.get("vegetation_density")
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _estimate_defensible_space_proxy(property_level_context: dict[str, Any]) -> float | None:
    near = _ring_density_value(property_level_context, "ring_0_5_ft")
    zone1 = _ring_density_value(property_level_context, "ring_5_30_ft")
    if near is None and zone1 is None:
        return None
    samples = [v for v in [near, zone1] if v is not None]
    mean_density = sum(samples) / max(1, len(samples))
    if mean_density >= 75:
        return 5.0
    if mean_density >= 60:
        return 12.0
    if mean_density >= 45:
        return 24.0
    return 40.0


def _apply_attribute_fallbacks(
    attrs: PropertyAttributes,
    *,
    property_level_context: dict[str, Any],
    context: object,
) -> tuple[PropertyAttributes, list[dict[str, object]]]:
    updated = attrs.model_copy(deep=True)
    fallback_decisions: list[dict[str, object]] = []

    if updated.defensible_space_ft is None:
        proxy = _estimate_defensible_space_proxy(property_level_context)
        if proxy is not None:
            updated.defensible_space_ft = round(proxy, 1)
            fallback_decisions.append(
                {
                    "fallback_type": "derived_proxy",
                    "missing_input": "defensible_space_ft",
                    "substitute_input": "structure_ring_vegetation_density",
                    "confidence_penalty_hint": 3.0,
                    "quality_label": "inferred",
                    "note": (
                        "Defensible space was estimated from near-structure vegetation rings to keep vulnerability/readiness "
                        "scoring available."
                    ),
                }
            )
        else:
            updated.defensible_space_ft = 15.0
            fallback_decisions.append(
                {
                    "fallback_type": "conservative_default",
                    "missing_input": "defensible_space_ft",
                    "substitute_input": "default_15ft_proxy",
                    "confidence_penalty_hint": 4.0,
                    "quality_label": "conservative",
                    "note": "Defensible space defaulted to a conservative baseline because no ring proxy was available.",
                }
            )

    if updated.roof_type is None:
        updated.roof_type = "composite"
        fallback_decisions.append(
            {
                "fallback_type": "neutral_default",
                "missing_input": "roof_type",
                "substitute_input": "composite_baseline",
                "confidence_penalty_hint": 2.0,
                "quality_label": "inferred",
                "note": "Roof type missing; neutral composite baseline used for structure vulnerability scoring.",
            }
        )

    if updated.vent_type is None:
        updated.vent_type = "standard"
        fallback_decisions.append(
            {
                "fallback_type": "neutral_default",
                "missing_input": "vent_type",
                "substitute_input": "standard_vent_baseline",
                "confidence_penalty_hint": 2.0,
                "quality_label": "inferred",
                "note": "Vent type missing; standard vent baseline used for structure vulnerability scoring.",
            }
        )

    if updated.vegetation_condition is None:
        ring_mean = _ring_density_value(property_level_context, "ring_5_30_ft")
        if ring_mean is not None:
            updated.vegetation_condition = "dense" if ring_mean >= 60 else ("moderate" if ring_mean >= 40 else "managed")
            fallback_decisions.append(
                {
                    "fallback_type": "derived_proxy",
                    "missing_input": "vegetation_condition",
                    "substitute_input": "ring_5_30_ft_vegetation_density",
                    "confidence_penalty_hint": 1.5,
                    "quality_label": "inferred",
                    "note": "Vegetation condition inferred from ring vegetation density.",
                }
            )
        elif getattr(context, "fuel_index", None) is not None:
            fuel_idx = float(getattr(context, "fuel_index"))
            updated.vegetation_condition = "dense" if fuel_idx >= 65 else ("moderate" if fuel_idx >= 45 else "managed")
            fallback_decisions.append(
                {
                    "fallback_type": "derived_proxy",
                    "missing_input": "vegetation_condition",
                    "substitute_input": "fuel_index_proxy",
                    "confidence_penalty_hint": 1.5,
                    "quality_label": "inferred",
                    "note": "Vegetation condition inferred from regional fuel context.",
                }
            )

    return updated, fallback_decisions


def _build_assumption_tracking(
    payload: AddressRequest,
    assumptions_used: list[str],
    data_sources: list[str],
    environmental_layer_status: dict[str, str],
    property_level_context: dict[str, Any],
    geocode_verified: bool = True,
) -> AssumptionsBlock:
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
        defensible_proxy = _estimate_defensible_space_proxy(property_level_context)
        if defensible_proxy is not None:
            inferred_inputs["defensible_space_ft"] = round(defensible_proxy, 1)
            assumptions_used.append(
                "Defensible space was inferred from structure-ring vegetation context."
            )
        else:
            inferred_inputs["defensible_space_ft"] = 15
            assumptions_used.append(
                "Defensible space missing; conservative default baseline was used."
            )
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

    layer_to_input = {
        "burn_probability": "burn_probability_layer",
        "hazard": "hazard_severity_layer",
        "slope": "slope_layer",
        "fuel": "fuel_model_layer",
        "canopy": "canopy_layer",
        "fire_history": "historical_fire_perimeter_layer",
    }
    for layer, status in (environmental_layer_status or {}).items():
        if status != "ok":
            missing_inputs.append(layer_to_input.get(layer, f"{layer}_layer"))

    if not geocode_verified:
        missing_inputs.append("geocode_verification")
        assumptions_used.append("Geocoding fallback used; coordinates were not provider-verified.")

    if not property_level_context.get("footprint_used"):
        missing_inputs.append("building_footprint")
        assumptions_used.append("Building footprint not found; vulnerability estimated using point context.")

    return AssumptionsBlock(
        confirmed_inputs=confirmed_inputs,
        observed_inputs=observed_inputs,
        inferred_inputs=inferred_inputs,
        missing_inputs=sorted(set(missing_inputs)),
        assumptions_used=assumptions_used,
    )


def _build_confidence(
    assumptions: AssumptionsBlock,
    *,
    environmental_data_completeness: float,
    geocode_verified: bool,
    property_level_context: dict[str, Any],
    environmental_layer_status: dict[str, str],
    data_provenance: DataProvenanceBlock | None = None,
) -> ConfidenceBlock:
    missing_inputs_set = set(assumptions.missing_inputs)
    important_missing = len([m for m in assumptions.missing_inputs if m in CORE_FACT_FIELDS or m.endswith("_layer")])
    missing_layer_count = len([m for m in assumptions.missing_inputs if m.endswith("_layer")])
    inferred_count = len(assumptions.inferred_inputs)
    observed_core_count = len(
        [k for k in CORE_FACT_FIELDS if assumptions.observed_inputs.get(k) is not None and k not in missing_inputs_set]
    )
    confirmed_core_count = len(
        [
            k
            for k in assumptions.confirmed_inputs
            if k in CORE_FACT_FIELDS and k not in missing_inputs_set
        ]
    )
    external_fail_count = sum(
        1
        for note in assumptions.assumptions_used
        if any(k in note.lower() for k in ["unavailable", "failed", "fallback"])
    )

    data_completeness_score = round(max(0.0, min(100.0, 100.0 - (len(assumptions.missing_inputs) * 3.0))), 1)

    provider_error_count = sum(1 for status in (environmental_layer_status or {}).values() if status == "error")
    provider_health_score = 100.0
    provider_health_score -= provider_error_count * 25.0
    provider_health_score -= external_fail_count * 10.0
    if not geocode_verified:
        provider_health_score -= 35.0
    if not property_level_context.get("footprint_used"):
        provider_health_score -= 10.0
    provider_health_score = max(0.0, min(100.0, provider_health_score))

    provenance_summary = data_provenance.summary if data_provenance else DataProvenanceSummary()
    stale_share = float(provenance_summary.stale_data_share or 0.0)
    heuristic_count = int(provenance_summary.heuristic_input_count or 0)
    critical_unknown_or_stale = 0
    critical_provider_errors = 0
    if data_provenance:
        for meta in data_provenance.inputs:
            if meta.field_name in CRITICAL_PROVENANCE_FIELDS:
                if meta.provider_status == "error":
                    critical_provider_errors += 1
                if meta.source_type not in LOW_QUALITY_SOURCE_TYPES and meta.freshness_status in {"stale", "unknown"}:
                    critical_unknown_or_stale += 1

    ring_metrics = property_level_context.get("ring_metrics")
    has_ring_metrics = isinstance(ring_metrics, dict) and bool(ring_metrics)
    environmental_data_present = (
        environmental_data_completeness > 0.0
        or any(status == "ok" for status in (environmental_layer_status or {}).values())
    )
    property_context_present = bool(
        has_ring_metrics
        or property_level_context.get("footprint_used")
        or str(property_level_context.get("fallback_mode") or "") == "point_based"
    )

    # Confidence intentionally weights multiple evidence classes so baseline runs with real
    # geospatial context can be non-zero even before optional homeowner facts are confirmed.
    core_quality_by_field: list[float] = []
    for field in sorted(CORE_FACT_FIELDS):
        if field in assumptions.confirmed_inputs and field not in missing_inputs_set:
            core_quality_by_field.append(100.0)
        elif assumptions.observed_inputs.get(field) is not None and field not in missing_inputs_set:
            core_quality_by_field.append(78.0)
        elif field in assumptions.inferred_inputs:
            core_quality_by_field.append(52.0)
        else:
            core_quality_by_field.append(18.0)
    structural_signal_score = (
        round(sum(core_quality_by_field) / len(core_quality_by_field), 1)
        if core_quality_by_field
        else 0.0
    )
    property_context_score = 90.0 if has_ring_metrics else (68.0 if property_context_present else 35.0)

    confidence = (
        0.35 * environmental_data_completeness
        + 0.20 * property_context_score
        + 0.20 * structural_signal_score
        + 0.15 * provider_health_score
        + 0.10 * data_completeness_score
    )

    confidence -= max(0.0, (missing_layer_count - 1) * 3.5)
    confidence -= max(0.0, (inferred_count - observed_core_count) * 1.8)
    confidence -= min(12.0, stale_share * 0.2)
    confidence -= critical_unknown_or_stale * 2.5
    confidence -= min(6.0, heuristic_count * 1.2)
    confidence += min(8.0, confirmed_core_count * 1.4)

    has_meaningful_environment = environmental_data_completeness >= 25.0 or missing_layer_count <= 2
    has_meaningful_property = has_ring_metrics or observed_core_count > 0 or confirmed_core_count > 0
    if geocode_verified and has_meaningful_environment:
        contextual_floor = min(
            45.0,
            6.0
            + (0.22 * environmental_data_completeness)
            + (6.0 if property_context_present else 0.0)
            + (4.0 if has_ring_metrics else 0.0),
        )
        confidence = max(confidence, contextual_floor)

    critical_near_structure_missing = (
        not has_ring_metrics
        and all(field in missing_inputs_set for field in {"roof_type", "vent_type", "defensible_space_ft"})
    )
    if critical_near_structure_missing:
        confidence = min(confidence, 62.0)
    if inferred_count >= 3:
        confidence = min(confidence, 74.0)
    if missing_layer_count >= 3:
        confidence = min(confidence, 58.0)

    if not geocode_verified or (not has_meaningful_environment and not has_meaningful_property):
        confidence = 0.0

    confidence = max(0.0, min(100.0, round(confidence, 1)))

    missing_critical_fields = sorted(
        {
            missing
            for missing in assumptions.missing_inputs
            if missing in CORE_FACT_FIELDS
            or missing.endswith("_layer")
            or missing in {"geocode_verification", "building_footprint"}
        }
    )

    confidence_drivers: list[str] = []
    confidence_limiters: list[str] = []
    if environmental_data_present:
        confidence_drivers.append(
            f"Environmental/geospatial context available ({environmental_data_completeness:.1f}% layer completeness)."
        )
    if property_context_present:
        if has_ring_metrics:
            confidence_drivers.append("Building-footprint ring context is available.")
        else:
            confidence_drivers.append("Property context available via point-based fallback.")
    if confirmed_core_count > 0:
        confidence_drivers.append(f"{confirmed_core_count} core home fact(s) were confirmed by the user.")
    elif observed_core_count > 0:
        confidence_drivers.append(f"{observed_core_count} core home fact(s) were provided without confirmation.")

    if missing_layer_count > 0:
        confidence_limiters.append(f"{missing_layer_count} environmental layer(s) missing or unavailable.")
    if inferred_count > 0:
        confidence_limiters.append(f"{inferred_count} core property input(s) are inferred.")
    if not has_ring_metrics:
        confidence_limiters.append("Building footprint rings unavailable; using point-based property context.")
    if critical_unknown_or_stale > 0:
        confidence_limiters.append("Critical inputs have stale or unknown freshness metadata.")
    if heuristic_count > 0:
        confidence_limiters.append("Heuristic inputs are present in scoring context.")

    low_confidence_flags: list[str] = []
    if important_missing >= 3:
        low_confidence_flags.append("Multiple important inputs or layers are missing")
    if external_fail_count > 0:
        low_confidence_flags.append("At least one external provider or layer fetch failed")
    if environmental_data_completeness < 80:
        low_confidence_flags.append("Environmental layer coverage is incomplete")
    if inferred_count >= 3:
        low_confidence_flags.append("Several core property attributes were inferred")
    if not geocode_verified:
        low_confidence_flags.append("Address geocoding was not provider-verified")
    if not property_level_context.get("footprint_used"):
        low_confidence_flags.append("Building footprint unavailable; point-based property context used")
    if stale_share > 0:
        low_confidence_flags.append("Some inputs are stale relative to freshness policy")
    if critical_unknown_or_stale > 0:
        low_confidence_flags.append("Critical inputs have stale or unknown freshness metadata")
    if heuristic_count > 0:
        low_confidence_flags.append("Heuristic inputs are present in the scoring context")
    if critical_near_structure_missing:
        low_confidence_flags.append("Critical near-structure evidence is missing; confidence is capped")
    if any("provisional" in note.lower() for note in assumptions.assumptions_used):
        low_confidence_flags.append("Access scoring is provisional and not yet parcel/egress-based")
    if confidence < 70:
        low_confidence_flags.append("Overall confidence below recommended underwriting threshold")

    severe_layer_failure = external_fail_count >= 2 or missing_layer_count >= 4 or provider_error_count >= 1
    major_layer_failure = external_fail_count >= 1 or missing_layer_count >= 1 or provider_error_count >= 1
    multiple_critical_missing = important_missing >= 4

    if (
        confidence < 50
        or not geocode_verified
        or severe_layer_failure
        or multiple_critical_missing
        or critical_provider_errors >= 1
        or (not has_meaningful_environment and not has_meaningful_property)
    ):
        confidence_tier = "preliminary"
    elif confidence < 70 or stale_share >= 25.0 or critical_unknown_or_stale >= 3:
        confidence_tier = "low"
    elif (
        confidence < 85
        or stale_share > 10.0
        or critical_unknown_or_stale > 0
        or heuristic_count > 1
    ):
        confidence_tier = "moderate"
    elif major_layer_failure:
        confidence_tier = "moderate"
    else:
        confidence_tier = "high"

    if confidence_tier == "high" and (
        environmental_data_completeness < 90.0
        or not property_level_context.get("footprint_used")
        or provider_error_count > 0
        or stale_share > 5.0
        or critical_unknown_or_stale > 0
        or heuristic_count > 1
    ):
        confidence_tier = "moderate"

    if confidence_tier == "high":
        use_restriction = "shareable"
    elif confidence_tier == "moderate":
        use_restriction = "homeowner_review_recommended"
    elif confidence_tier == "low":
        use_restriction = "agent_or_inspector_review_recommended"
    else:
        use_restriction = "not_for_underwriting_or_binding"

    if severe_layer_failure or missing_layer_count >= 2 or stale_share >= 25.0 or critical_unknown_or_stale >= 3:
        use_restriction = "not_for_underwriting_or_binding"

    return ConfidenceBlock(
        confidence_score=confidence,
        data_completeness_score=data_completeness_score,
        environmental_data_completeness_score=environmental_data_completeness,
        confidence_tier=confidence_tier,
        use_restriction=use_restriction,
        assumption_count=len(assumptions.assumptions_used),
        low_confidence_flags=sorted(set(low_confidence_flags)),
        requires_user_verification=confidence < 70.0 or len(low_confidence_flags) > 0,
        environmental_data_present=environmental_data_present,
        property_context_present=property_context_present,
        confirmed_fields_count=len(assumptions.confirmed_inputs),
        inferred_fields_count=inferred_count,
        missing_critical_fields=missing_critical_fields,
        confidence_drivers=sorted(set(confidence_drivers)),
        confidence_limiters=sorted(set(confidence_limiters)),
    )


def _derive_confidence_penalties(
    assumptions: AssumptionsBlock,
    *,
    environmental_data_completeness: float,
    geocode_verified: bool,
    property_level_context: dict[str, Any],
    environmental_layer_status: dict[str, str],
    data_provenance: DataProvenanceBlock | None = None,
    coverage_summary: LayerCoverageSummary | None = None,
) -> list[ConfidencePenalty]:
    penalties: list[ConfidencePenalty] = []
    missing_inputs_set = set(assumptions.missing_inputs)
    missing_layer_count = len([m for m in assumptions.missing_inputs if m.endswith("_layer")])
    inferred_count = len(assumptions.inferred_inputs)
    observed_core_count = len(
        [k for k in CORE_FACT_FIELDS if assumptions.observed_inputs.get(k) is not None and k not in missing_inputs_set]
    )
    provider_error_count = sum(1 for status in (environmental_layer_status or {}).values() if status == "error")
    ring_metrics = property_level_context.get("ring_metrics")
    has_ring_metrics = isinstance(ring_metrics, dict) and bool(ring_metrics)
    stale_share = float((data_provenance.summary.stale_data_share if data_provenance else 0.0) or 0.0)
    heuristic_count = int((data_provenance.summary.heuristic_input_count if data_provenance else 0) or 0)
    coverage_failed_count = int((coverage_summary.failed_count if coverage_summary else 0) or 0)
    critical_missing_count = int(len((coverage_summary.critical_missing_layers if coverage_summary else []) or []))
    not_configured_count = int((coverage_summary.not_configured_count if coverage_summary else 0) or 0)

    if not geocode_verified:
        penalties.append(
            ConfidencePenalty(
                penalty_key="unverified_geocode",
                reason="Address geocoding could not be provider-verified.",
                amount=35.0,
            )
        )
    if environmental_data_completeness < 100.0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="environmental_incompleteness",
                reason="One or more environmental layers are missing or degraded.",
                amount=round(max(0.0, (100.0 - environmental_data_completeness) * 0.12), 1),
            )
        )
    if missing_layer_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="missing_environmental_layers",
                reason=f"{missing_layer_count} environmental layer(s) missing from this run.",
                amount=round(missing_layer_count * 3.5, 1),
            )
        )
    if inferred_count > observed_core_count:
        penalties.append(
            ConfidencePenalty(
                penalty_key="inferred_structure_facts",
                reason="Core structure attributes rely on inferred/default assumptions.",
                amount=round((inferred_count - observed_core_count) * 1.8, 1),
            )
        )
    if not has_ring_metrics:
        penalties.append(
            ConfidencePenalty(
                penalty_key="missing_ring_context",
                reason="Building footprint/ring metrics unavailable; using point-based fallback.",
                amount=6.0,
            )
        )
    if provider_error_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="provider_errors",
                reason="One or more providers/layers returned runtime errors.",
                amount=round(provider_error_count * 10.0, 1),
            )
        )
    if stale_share > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="stale_inputs",
                reason="Some inputs are stale relative to freshness policy.",
                amount=round(min(12.0, stale_share * 0.2), 1),
            )
        )
    if heuristic_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="heuristic_inputs",
                reason="Heuristic inputs were used where direct evidence was unavailable.",
                amount=round(min(6.0, heuristic_count * 1.2), 1),
            )
        )
    if coverage_failed_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="layer_sampling_failures",
                reason="One or more runtime layers failed extent/sampling checks.",
                amount=round(min(18.0, coverage_failed_count * 4.0), 1),
            )
        )
    if critical_missing_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="critical_layer_gaps",
                reason="Required prepared-region layers are missing or unusable.",
                amount=round(min(20.0, critical_missing_count * 5.0), 1),
            )
        )
    if not_configured_count > 0:
        penalties.append(
            ConfidencePenalty(
                penalty_key="unconfigured_layers",
                reason="Some optional/open-data layers were not configured.",
                amount=round(min(6.0, not_configured_count * 1.0), 1),
            )
        )

    return [p for p in penalties if p.amount > 0]


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _average_numeric_values(values: list[object]) -> float | None:
    nums = [v for v in (_safe_float(val) for val in values) if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 2)


def _source_type_to_evidence_status(source_type: str) -> str:
    if source_type in {"observed", "footprint_derived", "user_provided"}:
        return "observed"
    if source_type == "public_record_inferred":
        return "inferred"
    if source_type == "heuristic":
        return "fallback"
    return "missing"


def _map_key_input_to_source_field(key: str) -> str | None:
    k = key.lower()
    if "burn_probability" in k:
        return "burn_probability"
    if "hazard" in k:
        return "wildfire_hazard"
    if "slope" in k or "aspect" in k:
        return "slope"
    if "fuel" in k:
        return "fuel_model"
    if "canopy" in k:
        return "canopy_cover"
    if "wildland_distance" in k:
        return "wildland_distance"
    if "historic_fire" in k:
        return "historic_fire_distance"
    if "roof" in k:
        return "roof_type"
    if "vent" in k:
        return "vent_type"
    if "construction" in k:
        return "construction_year"
    if "defensible_space" in k:
        return "defensible_space_ft"
    if "ring_0_5" in k or "zone_0_5" in k:
        return "zone_0_5_ft"
    if "ring_5_30" in k or "zone_5_30" in k:
        return "zone_5_30_ft"
    if "ring_30_100" in k or "zone_30_100" in k:
        return "zone_30_100_ft"
    return None


def _combine_evidence_status(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    rank = {"observed": 1, "inferred": 2, "fallback": 3, "missing": 4}
    return max(statuses, key=lambda s: rank.get(s, 4))


def _status_from_source_fields(
    source_fields: list[str],
    input_source_metadata: dict[str, InputSourceMetadata],
) -> tuple[str, str]:
    statuses: list[str] = []
    resolved_fields: list[str] = []
    for field in source_fields:
        meta = input_source_metadata.get(field)
        if meta:
            statuses.append(_source_type_to_evidence_status(meta.source_type))
            resolved_fields.append(field)
    return _combine_evidence_status(statuses), ", ".join(sorted(set(resolved_fields))) or "derived"


def _build_score_evidence_ledger(
    *,
    submodel_scores: dict[str, SubmodelScore],
    weighted_contributions: dict[str, WeightedContribution],
    readiness_factors: list[ReadinessFactor],
    readiness_score: float | None,
    site_hazard_score: float | None,
    home_ignition_vulnerability_score: float | None,
    wildfire_risk_score: float | None,
    input_source_metadata: dict[str, InputSourceMetadata],
) -> ScoreEvidenceLedger:
    category_map = {
        "vegetation_intensity_risk": "environmental",
        "fuel_proximity_risk": "environmental",
        "slope_topography_risk": "environmental",
        "ember_exposure_risk": "environmental",
        "flame_contact_risk": "structural",
        "historic_fire_risk": "historical_fire",
        "structure_vulnerability_risk": "structural",
        "defensible_space_risk": "defensible_space",
    }
    display_map = {
        "vegetation_intensity_risk": "Vegetation Intensity",
        "fuel_proximity_risk": "Fuel Proximity",
        "slope_topography_risk": "Slope & Topography",
        "ember_exposure_risk": "Ember Exposure",
        "flame_contact_risk": "Flame Contact Exposure",
        "historic_fire_risk": "Historic Fire Exposure",
        "structure_vulnerability_risk": "Structure Vulnerability",
        "defensible_space_risk": "Defensible Space",
    }

    site_entries: list[ScoreEvidenceFactor] = []
    home_entries: list[ScoreEvidenceFactor] = []

    for submodel in CANONICAL_SUBMODELS:
        score = submodel_scores.get(submodel)
        weight = weighted_contributions.get(submodel)
        if not score or not weight:
            continue
        source_fields = [
            field for field in (_map_key_input_to_source_field(k) for k in score.key_inputs.keys()) if field
        ]
        evidence_status, resolved_source_fields = _status_from_source_fields(source_fields, input_source_metadata)
        factor = ScoreEvidenceFactor(
            factor_key=submodel,
            display_name=display_map.get(submodel, submodel),
            category=category_map.get(submodel, "data_quality"),
            raw_value=_average_numeric_values(list(score.key_inputs.values())),
            normalized_value=round(score.score, 1),
            weight=round(weight.weight, 4),
            contribution=round(weight.contribution, 2),
            direction="increases_risk",
            evidence_status=evidence_status,
            source_field=resolved_source_fields,
            source_layer=resolved_source_fields,
            notes=[score.explanation] + score.assumptions[:2],
        )
        if submodel in ENVIRONMENTAL_SUBMODELS:
            site_entries.append(factor)
        else:
            home_entries.append(factor)

    struct_weight = sum(sc.weight for key, sc in weighted_contributions.items() if key in STRUCTURAL_SUBMODELS)
    struct_contrib = sum(sc.contribution for key, sc in weighted_contributions.items() if key in STRUCTURAL_SUBMODELS)
    struct_base = round(struct_contrib / struct_weight, 1) if struct_weight > 0 else None
    if struct_base is not None and home_ignition_vulnerability_score is not None:
        ring_modifier = round(home_ignition_vulnerability_score - struct_base, 2)
        if abs(ring_modifier) >= 0.01:
            ring_status, ring_sources = _status_from_source_fields(
                ["zone_0_5_ft", "zone_5_30_ft", "zone_30_100_ft"],
                input_source_metadata,
            )
            home_entries.append(
                ScoreEvidenceFactor(
                    factor_key="structure_ring_modifier",
                    display_name="Structure Ring Context Modifier",
                    category="defensible_space",
                    raw_value=struct_base,
                    normalized_value=home_ignition_vulnerability_score,
                    weight=1.0,
                    contribution=ring_modifier,
                    direction="increases_risk" if ring_modifier >= 0 else "reduces_risk",
                    evidence_status=ring_status,
                    source_field=ring_sources,
                    source_layer=ring_sources,
                    notes=[
                        "Captures additional near-structure ring influence beyond base structural submodels."
                    ],
                )
            )

    readiness_field_map = {
        "roof_material": "roof_type",
        "vent_quality": "vent_type",
        "defensible_space": "defensible_space_ft",
        "structure_vulnerability": "structure_vulnerability_risk",
        "adjacent_fuel_pressure": "fuel_model",
        "vegetation_intensity": "zone_5_30_ft",
        "severe_ember_exposure": "burn_probability",
        "severe_environmental_hazard": "wildfire_hazard",
    }
    readiness_entries: list[ScoreEvidenceFactor] = []
    for factor in readiness_factors:
        source_field = readiness_field_map.get(factor.name)
        status, resolved = _status_from_source_fields(
            [source_field] if source_field else [],
            input_source_metadata,
        )
        direction = "blocks_readiness"
        if factor.status == "pass" and factor.score_impact >= 0:
            direction = "improves_readiness"
        readiness_entries.append(
            ScoreEvidenceFactor(
                factor_key=factor.name,
                display_name=factor.name.replace("_", " ").title(),
                category="data_quality" if factor.status == "watch" else "structural",
                raw_value=None,
                normalized_value=readiness_score,
                weight=1.0,
                contribution=round(factor.score_impact, 2),
                direction=direction,
                evidence_status=status,
                source_field=resolved or source_field,
                source_layer=resolved or source_field,
                notes=[factor.detail],
            )
        )

    env_weight = sum(weighted_contributions[key].weight for key in weighted_contributions if key in ENVIRONMENTAL_SUBMODELS)
    struct_weight_total = sum(
        weighted_contributions[key].weight for key in weighted_contributions if key in STRUCTURAL_SUBMODELS
    )
    denom = env_weight + struct_weight_total
    wildfire_entries: list[ScoreEvidenceFactor] = []
    if denom > 0 and wildfire_risk_score is not None:
        if site_hazard_score is not None:
            site_ratio = env_weight / denom
            wildfire_entries.append(
                ScoreEvidenceFactor(
                    factor_key="site_hazard_component",
                    display_name="Site Hazard Component",
                    category="environmental",
                    raw_value=site_hazard_score,
                    normalized_value=site_hazard_score,
                    weight=round(site_ratio, 4),
                    contribution=round(site_hazard_score * site_ratio, 2),
                    direction="composes_score",
                    evidence_status=_combine_evidence_status([f.evidence_status for f in site_entries]),
                    source_field="site_hazard_score",
                    source_layer="site_hazard_score",
                    notes=["Weighted contribution of Site Hazard to blended wildfire score."],
                )
            )
        if home_ignition_vulnerability_score is not None:
            home_ratio = struct_weight_total / denom
            wildfire_entries.append(
                ScoreEvidenceFactor(
                    factor_key="home_ignition_component",
                    display_name="Home Ignition Vulnerability Component",
                    category="structural",
                    raw_value=home_ignition_vulnerability_score,
                    normalized_value=home_ignition_vulnerability_score,
                    weight=round(home_ratio, 4),
                    contribution=round(home_ignition_vulnerability_score * home_ratio, 2),
                    direction="composes_score",
                    evidence_status=_combine_evidence_status([f.evidence_status for f in home_entries]),
                    source_field="home_ignition_vulnerability_score",
                    source_layer="home_ignition_vulnerability_score",
                    notes=["Weighted contribution of Home Ignition Vulnerability to blended wildfire score."],
                )
            )

    return ScoreEvidenceLedger(
        site_hazard_score=site_entries,
        home_ignition_vulnerability_score=home_entries,
        insurance_readiness_score=readiness_entries,
        wildfire_risk_score=wildfire_entries,
    )


def _build_evidence_quality_summary(
    *,
    ledger: ScoreEvidenceLedger,
    confidence_penalties: list[ConfidencePenalty],
    confidence: ConfidenceBlock,
    assessment_status: AssessmentStatus,
) -> EvidenceQualitySummary:
    factors = (
        list(ledger.site_hazard_score)
        + list(ledger.home_ignition_vulnerability_score)
        + list(ledger.insurance_readiness_score)
    )

    observed_count = sum(1 for f in factors if f.evidence_status == "observed")
    inferred_count = sum(1 for f in factors if f.evidence_status == "inferred")
    missing_count = sum(1 for f in factors if f.evidence_status == "missing")
    fallback_count = sum(1 for f in factors if f.evidence_status == "fallback")
    total = max(1, len(factors))

    raw_quality = (
        observed_count
        + (0.65 * inferred_count)
        + (0.35 * fallback_count)
    ) / total
    penalty_adjustment = min(25.0, sum(p.amount for p in confidence_penalties) * 0.2)
    evidence_quality_score = round(max(0.0, min(100.0, raw_quality * 100.0 - penalty_adjustment)), 1)

    if assessment_status == "insufficient_data" or confidence.confidence_tier == "preliminary":
        evidence_use_restriction = "screening_only"
    elif confidence.confidence_tier == "low" or (missing_count + fallback_count) >= max(3, observed_count):
        evidence_use_restriction = "review_required"
    else:
        evidence_use_restriction = "consumer_estimate"

    return EvidenceQualitySummary(
        observed_factor_count=observed_count,
        inferred_factor_count=inferred_count,
        missing_factor_count=missing_count,
        fallback_factor_count=fallback_count,
        evidence_quality_score=evidence_quality_score,
        confidence_penalties=confidence_penalties,
        use_restriction=evidence_use_restriction,  # type: ignore[arg-type]
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

    ring_0_5 = _density_from_ring(rings.get("ring_0_5_ft") or rings.get("zone_0_5_ft"))
    ring_5_30 = _density_from_ring(rings.get("ring_5_30_ft") or rings.get("zone_5_30_ft"))
    ring_30_100 = _density_from_ring(rings.get("ring_30_100_ft") or rings.get("zone_30_100_ft"))
    ring_100_300 = _density_from_ring(rings.get("ring_100_300_ft") or rings.get("zone_100_300_ft"))
    findings: list[str] = []

    if ring_0_5 is not None and ring_0_5 >= 60:
        findings.append("Vegetation appears dense within 5 feet of the home.")
    elif ring_0_5 is not None and ring_0_5 <= 30:
        findings.append("The immediate 0-5 foot zone appears relatively clear.")

    if ring_5_30 is not None and ring_5_30 >= 60:
        findings.append("Defensible space appears limited within 30 feet of the structure.")
    elif ring_5_30 is not None and ring_5_30 <= 35:
        findings.append("Vegetation conditions within 30 feet look more manageable.")

    if ring_30_100 is not None and ring_30_100 >= 65:
        findings.append("Vegetation and fuels are elevated in the 30-100 foot zone around the home.")
    if ring_100_300 is not None and ring_100_300 >= 70:
        findings.append("Surrounding fuels remain elevated in the wider 100-300 foot zone.")

    if ring_0_5 is not None and ring_5_30 is not None and (ring_5_30 - ring_0_5) >= 20:
        findings.append("Defensible space appears stronger very close to the home than farther out.")

    return findings[:3]


def _normalize_property_level_context(raw_context: object) -> dict[str, Any]:
    if not isinstance(raw_context, dict):
        return {
            "footprint_used": False,
            "footprint_status": "not_found",
            "structure_match_status": "none",
            "structure_match_method": None,
            "structure_match_confidence": 0.0,
            "building_source": None,
            "building_source_version": None,
            "building_source_confidence": 0.0,
            "structure_match_distance_m": None,
            "candidate_structure_count": 0,
            "structure_match_candidates": [],
            "structure_geometry_source": "auto_detected",
            "selection_mode": "polygon",
            "property_anchor_point": None,
            "user_selected_point": None,
            "selected_structure_id": None,
            "selected_structure_geometry": None,
            "final_structure_geometry_source": "auto_detected",
            "structure_geometry_confidence": 0.0,
            "snapped_structure_distance_m": None,
            "user_selected_point_in_footprint": False,
            "display_point_source": "property_anchor_point",
            "property_anchor_source": "geocoded_address_point",
            "property_anchor_precision": "unknown",
            "geocoded_address_point": None,
            "assessed_property_display_point": None,
            "matched_structure_centroid": None,
            "matched_structure_footprint": None,
            "parcel_id": None,
            "parcel_lookup_method": None,
            "parcel_lookup_distance_m": None,
            "parcel_geometry": None,
            "parcel_address_point": None,
            "alignment_notes": [],
            "source_conflict_flag": False,
            "fallback_mode": "point_based",
            "ring_metrics": None,
        }

    normalized = dict(raw_context)
    footprint_used = bool(normalized.get("footprint_used"))
    normalized["footprint_used"] = footprint_used
    normalized.setdefault("footprint_status", "used" if footprint_used else "not_found")
    if normalized.get("footprint_status") == "source_unavailable":
        normalized["footprint_status"] = "provider_unavailable"
    normalized.setdefault("fallback_mode", "footprint" if footprint_used else "point_based")
    normalized.setdefault("structure_match_status", "matched" if footprint_used else "none")
    normalized.setdefault(
        "structure_match_method",
        (
            "parcel_intersection"
            if (footprint_used and normalized.get("parcel_id"))
            else ("nearest_building_fallback" if footprint_used else None)
        ),
    )
    normalized.setdefault("structure_match_confidence", float(normalized.get("footprint_confidence") or (0.9 if footprint_used else 0.0)))
    normalized.setdefault("building_source", str(normalized.get("footprint_source_name") or "") or None)
    normalized.setdefault("building_source_version", str(normalized.get("footprint_source_vintage") or "") or None)
    normalized.setdefault("building_source_confidence", float(normalized.get("structure_match_confidence") or 0.0))
    normalized.setdefault("structure_match_distance_m", 0.0 if footprint_used else None)
    normalized.setdefault("candidate_structure_count", 1 if footprint_used else 0)
    candidates = normalized.get("structure_match_candidates")
    normalized["structure_match_candidates"] = candidates if isinstance(candidates, list) else []
    geometry_source = str(normalized.get("structure_geometry_source") or "").strip().lower()
    if geometry_source not in {"auto_detected", "user_selected", "user_modified"}:
        geometry_source = "auto_detected"
    normalized["structure_geometry_source"] = geometry_source
    selection_mode = str(normalized.get("selection_mode") or "").strip().lower()
    if selection_mode not in {"polygon", "point"}:
        selection_mode = "polygon"
    normalized["selection_mode"] = selection_mode
    user_selected_point = normalized.get("user_selected_point")
    if isinstance(user_selected_point, dict):
        try:
            us_lat = float(user_selected_point.get("latitude"))
            us_lon = float(user_selected_point.get("longitude"))
            if -90.0 <= us_lat <= 90.0 and -180.0 <= us_lon <= 180.0:
                normalized["user_selected_point"] = {"latitude": us_lat, "longitude": us_lon}
            else:
                normalized["user_selected_point"] = None
        except (TypeError, ValueError):
            normalized["user_selected_point"] = None
    else:
        normalized["user_selected_point"] = None
    selected_structure_id = normalized.get("selected_structure_id")
    normalized["selected_structure_id"] = (
        str(selected_structure_id).strip() if selected_structure_id is not None and str(selected_structure_id).strip() else None
    )
    selected_structure_geometry = normalized.get("selected_structure_geometry")
    normalized["selected_structure_geometry"] = (
        selected_structure_geometry if isinstance(selected_structure_geometry, dict) else None
    )
    final_source = str(normalized.get("final_structure_geometry_source") or "").strip().lower()
    if final_source not in {
        "auto_detected",
        "user_selected_polygon",
        "user_selected_point_snapped",
        "user_selected_point_unsnapped",
    }:
        final_source = (
            "user_selected_polygon"
            if geometry_source in {"user_selected", "user_modified"} and normalized["selected_structure_geometry"] is not None
            else "auto_detected"
        )
    normalized["final_structure_geometry_source"] = final_source
    try:
        normalized["structure_geometry_confidence"] = float(normalized.get("structure_geometry_confidence") or 0.0)
    except (TypeError, ValueError):
        normalized["structure_geometry_confidence"] = 0.0
    try:
        snapped_distance = normalized.get("snapped_structure_distance_m")
        normalized["snapped_structure_distance_m"] = (
            float(snapped_distance) if snapped_distance is not None else None
        )
    except (TypeError, ValueError):
        normalized["snapped_structure_distance_m"] = None
    normalized["user_selected_point_in_footprint"] = bool(normalized.get("user_selected_point_in_footprint"))
    normalized.setdefault("display_point_source", "matched_structure_centroid" if footprint_used else "property_anchor_point")
    normalized.setdefault(
        "property_anchor_point",
        {
            "latitude": None,
            "longitude": None,
        },
    )
    normalized.setdefault("property_anchor_source", "geocoded_address_point")
    normalized.setdefault("property_anchor_precision", "unknown")
    normalized.setdefault("geocoded_address_point", {"latitude": None, "longitude": None})
    normalized.setdefault("assessed_property_display_point", normalized.get("property_anchor_point"))
    matched_structure_centroid = normalized.get("matched_structure_centroid")
    if isinstance(matched_structure_centroid, dict):
        try:
            centroid_lat = float(matched_structure_centroid.get("latitude"))
            centroid_lon = float(matched_structure_centroid.get("longitude"))
            if -90.0 <= centroid_lat <= 90.0 and -180.0 <= centroid_lon <= 180.0:
                normalized["matched_structure_centroid"] = {"latitude": centroid_lat, "longitude": centroid_lon}
            else:
                normalized["matched_structure_centroid"] = None
        except (TypeError, ValueError):
            normalized["matched_structure_centroid"] = None
    else:
        normalized["matched_structure_centroid"] = None
    normalized["matched_structure_footprint"] = (
        normalized.get("matched_structure_footprint")
        if isinstance(normalized.get("matched_structure_footprint"), dict)
        else None
    )
    normalized.setdefault("parcel_id", None)
    normalized.setdefault("parcel_lookup_method", None)
    normalized.setdefault("parcel_lookup_distance_m", None)
    normalized.setdefault("parcel_geometry", None)
    normalized.setdefault("parcel_address_point", None)
    normalized.setdefault("matched_structure_id", None)
    notes = normalized.get("alignment_notes")
    normalized["alignment_notes"] = notes if isinstance(notes, list) else []
    normalized.setdefault("source_conflict_flag", False)
    ring_metrics = normalized.get("ring_metrics")
    if isinstance(ring_metrics, dict):
        # Canonical zone keys for API consumers.
        if "zone_0_5_ft" not in ring_metrics and "ring_0_5_ft" in ring_metrics:
            ring_metrics["zone_0_5_ft"] = ring_metrics.get("ring_0_5_ft")
        if "zone_5_30_ft" not in ring_metrics and "ring_5_30_ft" in ring_metrics:
            ring_metrics["zone_5_30_ft"] = ring_metrics.get("ring_5_30_ft")
        if "zone_30_100_ft" not in ring_metrics and "ring_30_100_ft" in ring_metrics:
            ring_metrics["zone_30_100_ft"] = ring_metrics.get("ring_30_100_ft")

        # Backward-compatible legacy aliases.
        if "ring_0_5_ft" not in ring_metrics and "zone_0_5_ft" in ring_metrics:
            ring_metrics["ring_0_5_ft"] = ring_metrics.get("zone_0_5_ft")
        if "ring_5_30_ft" not in ring_metrics and "zone_5_30_ft" in ring_metrics:
            ring_metrics["ring_5_30_ft"] = ring_metrics.get("zone_5_30_ft")
        if "ring_30_100_ft" not in ring_metrics and "zone_30_100_ft" in ring_metrics:
            ring_metrics["ring_30_100_ft"] = ring_metrics.get("zone_30_100_ft")

        normalized["ring_metrics"] = ring_metrics
    else:
        normalized["ring_metrics"] = None
    return normalized


def _normalize_layer_coverage(
    property_level_context: dict[str, Any],
    *,
    environmental_layer_status: dict[str, str],
) -> tuple[list[LayerCoverageAuditItem], LayerCoverageSummary]:
    raw_rows = property_level_context.get("layer_coverage_audit")
    rows: list[LayerCoverageAuditItem] = []
    if isinstance(raw_rows, list):
        for item in raw_rows:
            try:
                rows.append(LayerCoverageAuditItem.model_validate(item))
            except Exception:
                continue

    if not rows:
        fallback_rows: list[LayerCoverageAuditItem] = []
        footprint_used = bool(property_level_context.get("footprint_used"))
        footprint_status = str(property_level_context.get("footprint_status") or "not_found")
        has_hazard_context = isinstance(property_level_context.get("hazard_context"), dict) and bool(
            property_level_context.get("hazard_context")
        )
        has_moisture_context = isinstance(property_level_context.get("moisture_context"), dict) and bool(
            property_level_context.get("moisture_context")
        )
        has_historical_context = isinstance(property_level_context.get("historical_fire_context"), dict) and bool(
            property_level_context.get("historical_fire_context")
        )
        has_access_context = isinstance(property_level_context.get("access_context"), dict) and bool(
            property_level_context.get("access_context")
        )
        inferred_layer_states = {
            "dem": "ok" if environmental_layer_status else "missing",
            "slope": str(environmental_layer_status.get("slope") or "missing"),
            "fuel": str(environmental_layer_status.get("fuel") or "missing"),
            "canopy": str(environmental_layer_status.get("canopy") or "missing"),
            "fire_perimeters": str(environmental_layer_status.get("fire_history") or "missing"),
            "building_footprints": "ok" if footprint_used else "missing",
            "whp": "ok" if has_hazard_context else "missing",
            "mtbs_severity": "ok" if has_historical_context else "missing",
            "gridmet_dryness": "ok" if has_moisture_context else "missing",
            "roads": "ok" if has_access_context else "missing",
        }
        for key in [
            "dem",
            "slope",
            "fuel",
            "canopy",
            "fire_perimeters",
            "building_footprints",
            "whp",
            "mtbs_severity",
            "gridmet_dryness",
            "roads",
        ]:
            layer_state = inferred_layer_states.get(key, "missing")
            if layer_state == "ok":
                coverage_status = "observed"
            elif layer_state == "error":
                coverage_status = "sampling_failed"
            elif key in {"whp", "mtbs_severity", "gridmet_dryness", "roads"}:
                coverage_status = "not_configured"
            else:
                coverage_status = "fallback_used"
            spec = LAYER_SPECS.get(key, {"display_name": key.replace("_", " "), "required_for": []})
            fallback_rows.append(
                LayerCoverageAuditItem(
                    layer_key=key,
                    display_name=str(spec.get("display_name") or key.replace("_", " ")),
                    required_for=[str(x) for x in (spec.get("required_for") or [])],
                    configured=layer_state != "missing",
                    present_in_region=layer_state != "missing",
                    sample_attempted=layer_state in {"ok", "error"},
                    sample_succeeded=layer_state == "ok",
                    coverage_status=coverage_status,
                    source_type="open_data" if key in {"whp", "mtbs_severity", "gridmet_dryness", "roads"} else "runtime_env",
                    failure_reason=(
                        f"footprint_status={footprint_status}"
                        if key == "building_footprints" and not footprint_used
                        else None
                    ),
                    notes=["Generated fallback diagnostics from legacy status fields."],
                )
            )
        rows = fallback_rows

    raw_summary = property_level_context.get("coverage_summary")
    if isinstance(raw_summary, dict):
        try:
            summary = LayerCoverageSummary.model_validate(raw_summary)
            return rows, summary
        except Exception:
            pass

    observed_count = sum(1 for row in rows if row.coverage_status == "observed")
    partial_count = sum(1 for row in rows if row.coverage_status == "partial")
    fallback_count = sum(1 for row in rows if row.coverage_status == "fallback_used")
    failed_count = sum(1 for row in rows if row.coverage_status in {"missing_file", "outside_extent", "sampling_failed"})
    not_configured_count = sum(1 for row in rows if row.coverage_status == "not_configured")
    critical_missing_layers = sorted(
        {
            row.layer_key
            for row in rows
            if row.layer_key in {"dem", "slope", "fuel", "canopy", "fire_perimeters", "building_footprints"}
            and row.coverage_status not in {"observed", "partial"}
        }
    )
    recommended_actions: list[str] = []
    for row in rows:
        if row.coverage_status == "missing_file":
            recommended_actions.append(f"{row.layer_key} file is missing; verify prepared region files.")
        elif row.coverage_status == "not_configured":
            recommended_actions.append(f"{row.layer_key} is not configured; add a data source or treat it as optional.")
        elif row.coverage_status == "outside_extent":
            recommended_actions.append(f"{row.layer_key} does not cover this property location; prepare correct regional coverage.")
        elif row.coverage_status == "sampling_failed":
            recommended_actions.append(f"{row.layer_key} sampling failed; validate layer integrity and CRS.")
        elif row.coverage_status == "fallback_used":
            recommended_actions.append(f"{row.layer_key} used fallback behavior; improve data coverage for stronger confidence.")
    dedup_actions = list(dict.fromkeys(recommended_actions))[:10]
    summary = LayerCoverageSummary(
        total_layers_checked=len(rows),
        observed_count=observed_count,
        partial_count=partial_count,
        fallback_count=fallback_count,
        failed_count=failed_count,
        not_configured_count=not_configured_count,
        critical_missing_layers=critical_missing_layers,
        recommended_actions=dedup_actions,
    )
    return rows, summary


def _build_data_provenance(
    *,
    payload: AddressRequest,
    assumptions: AssumptionsBlock,
    context: object,
    property_level_context: dict[str, Any],
) -> tuple[dict[str, InputSourceMetadata], DataProvenanceBlock, float, float, float]:
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    env_status = getattr(context, "environmental_layer_status", {}) or {}

    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(raw)
            return parsed if parsed > 0 else default
        except ValueError:
            return default

    def _freshness_thresholds(source_class: str | None) -> tuple[int, int] | None:
        if not source_class or source_class not in FRESHNESS_POLICY_DEFAULTS_DAYS:
            return None
        default_current, default_aging = FRESHNESS_POLICY_DEFAULTS_DAYS[source_class]
        prefix = source_class.upper()
        current_days = _int_env(f"WF_FRESHNESS_{prefix}_CURRENT_DAYS", default_current)
        aging_days = _int_env(f"WF_FRESHNESS_{prefix}_AGING_DAYS", default_aging)
        if aging_days < current_days:
            aging_days = current_days
        return current_days, aging_days

    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        v = value.strip()
        if not v:
            return None
        try:
            if len(v) == 10:
                parsed = datetime.fromisoformat(v + "T00:00:00+00:00")
            else:
                parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    def _freshness_status(source_class: str | None, observed_at: str | None) -> FreshnessStatus:
        if source_class in {"missing", "heuristic"}:
            return "unknown"
        thresholds = _freshness_thresholds(source_class)
        observed_dt = _parse_iso(observed_at)
        if thresholds is None or observed_dt is None:
            return "unknown"
        age_days = max(0.0, (datetime.now(tz=timezone.utc) - observed_dt).total_seconds() / 86400.0)
        current_days, aging_days = thresholds
        if age_days <= current_days:
            return "current"
        if age_days <= aging_days:
            return "aging"
        return "stale"

    def _confidence_weight(source_type: str, provider_status: str, freshness: FreshnessStatus) -> float:
        base = {
            "observed": 1.0,
            "footprint_derived": 0.95,
            "user_provided": 0.8,
            "public_record_inferred": 0.65,
            "heuristic": 0.35,
            "missing": 0.0,
        }.get(source_type, 0.0)
        provider_mult = {"ok": 1.0, "missing": 0.65, "error": 0.45}.get(provider_status, 0.6)
        freshness_mult = {"current": 1.0, "aging": 0.85, "stale": 0.55, "unknown": 0.7}.get(freshness, 0.7)
        return round(max(0.0, min(1.0, base * provider_mult * freshness_mult)), 2)

    def _provider_status(layer_state: str) -> str:
        if layer_state in {"ok", "missing", "error"}:
            return layer_state
        return "missing"

    def _metadata(
        *,
        field_name: str,
        source_type: str,
        source_name: str,
        provider_status: str = "ok",
        source_class: str | None = None,
        dataset_version: str | None = None,
        observed_at: str | None = None,
        spatial_resolution_m: float | None = None,
        used_in_scoring: bool = True,
        details: str | None = None,
    ) -> InputSourceMetadata:
        freshness = _freshness_status(source_class or source_type, observed_at)
        return InputSourceMetadata(
            field_name=field_name,
            source_type=source_type,
            source_name=source_name,
            provider_status=provider_status,  # type: ignore[arg-type]
            freshness_status=freshness,
            used_in_scoring=used_in_scoring,
            confidence_weight=_confidence_weight(source_type, provider_status, freshness),
            observed_at=observed_at,
            loaded_at=now_iso,
            dataset_version=dataset_version,
            spatial_resolution_m=spatial_resolution_m,
            source_class=source_class,
            details=details,
        )

    env_dataset_meta = {
        "burn_probability": (
            "Burn probability raster",
            "environmental_raster",
            os.getenv("WF_LAYER_BURN_PROB_VERSION"),
            os.getenv("WF_LAYER_BURN_PROB_DATE"),
            30.0,
        ),
        "wildfire_hazard": (
            "Wildfire hazard severity raster",
            "environmental_raster",
            os.getenv("WF_LAYER_HAZARD_SEVERITY_VERSION"),
            os.getenv("WF_LAYER_HAZARD_SEVERITY_DATE"),
            30.0,
        ),
        "slope": (
            "Slope raster/DEM",
            "environmental_raster",
            os.getenv("WF_LAYER_SLOPE_VERSION") or os.getenv("WF_LAYER_DEM_VERSION"),
            os.getenv("WF_LAYER_SLOPE_DATE") or os.getenv("WF_LAYER_DEM_DATE"),
            30.0,
        ),
        "fuel_model": (
            "Fuel model raster",
            "environmental_raster",
            os.getenv("WF_LAYER_FUEL_VERSION"),
            os.getenv("WF_LAYER_FUEL_DATE"),
            100.0,
        ),
        "canopy_cover": (
            "Canopy cover raster",
            "environmental_raster",
            os.getenv("WF_LAYER_CANOPY_VERSION"),
            os.getenv("WF_LAYER_CANOPY_DATE"),
            100.0,
        ),
        "historic_fire_distance": (
            "Historical fire perimeter layer",
            "fire_history_layer",
            os.getenv("WF_LAYER_FIRE_PERIMETERS_VERSION"),
            os.getenv("WF_LAYER_FIRE_PERIMETERS_DATE"),
            250.0,
        ),
        "wildland_distance": (
            "Distance-to-wildland derivation",
            "environmental_raster",
            os.getenv("WF_LAYER_FUEL_VERSION") or os.getenv("WF_LAYER_CANOPY_VERSION"),
            os.getenv("WF_LAYER_FUEL_DATE") or os.getenv("WF_LAYER_CANOPY_DATE"),
            100.0,
        ),
    }

    env_value_getters = {
        "burn_probability": getattr(context, "burn_probability", None),
        "wildfire_hazard": getattr(context, "wildfire_hazard", None),
        "slope": getattr(context, "slope", None),
        "fuel_model": getattr(context, "fuel_model", None),
        "canopy_cover": getattr(context, "canopy_cover", None),
        "historic_fire_distance": getattr(context, "historic_fire_distance", None),
        "wildland_distance": getattr(context, "wildland_distance", None),
    }
    env_status_keys = {
        "burn_probability": "burn_probability",
        "wildfire_hazard": "hazard",
        "slope": "slope",
        "fuel_model": "fuel",
        "canopy_cover": "canopy",
        "historic_fire_distance": "fire_history",
        "wildland_distance": "fuel",
    }
    hazard_context = getattr(context, "hazard_context", {}) or {}
    if str(hazard_context.get("status") or "").lower() == "observed":
        source_name = str(hazard_context.get("source") or "USFS Wildfire Hazard Potential (WHP)")
        whp_version = os.getenv("WF_LAYER_WHP_VERSION")
        whp_date = os.getenv("WF_LAYER_WHP_DATE")
        env_dataset_meta["burn_probability"] = (
            source_name,
            "environmental_raster",
            whp_version,
            whp_date,
            270.0,
        )
        env_dataset_meta["wildfire_hazard"] = (
            source_name,
            "environmental_raster",
            whp_version,
            whp_date,
            270.0,
        )

    env_sources: dict[str, InputSourceMetadata] = {}
    for field, value in env_value_getters.items():
        status_key = env_status_keys[field]
        layer_state = env_status.get(status_key, "missing")
        provider = _provider_status(layer_state)
        if field == "wildland_distance" and value is not None:
            provider = "ok"
        source_name, source_class, dataset_version, observed_at, resolution_m = env_dataset_meta[field]
        if value is not None and provider == "ok":
            source_type = "observed"
            obs_date = observed_at or now_iso
        else:
            source_type = "missing"
            obs_date = observed_at
        env_sources[field] = _metadata(
            field_name=field,
            source_type=source_type,
            source_name=source_name,
            provider_status=provider,
            source_class=source_class,
            dataset_version=dataset_version,
            observed_at=obs_date,
            spatial_resolution_m=resolution_m,
            used_in_scoring=True,
            details=f"layer_status={layer_state}",
        )

    property_sources: dict[str, InputSourceMetadata] = {}
    attr_fields = [
        "roof_type",
        "vent_type",
        "siding_type",
        "window_type",
        "defensible_space_ft",
        "construction_year",
        "vegetation_condition",
    ]
    for field in attr_fields:
        value = getattr(payload.attributes, field, None)
        if value is not None:
            source_type = "user_provided"
            source_class = "user_provided"
            details = "Provided by homeowner/user."
            observed_at = now_iso
        elif field in assumptions.inferred_inputs:
            source_type = "public_record_inferred" if field in {"construction_year"} else "heuristic"
            source_class = source_type
            details = "Inferred due to missing explicit input."
            observed_at = None
        else:
            source_type = "missing"
            source_class = "missing"
            details = "Field not provided."
            observed_at = None

        property_sources[field] = _metadata(
            field_name=field,
            source_type=source_type,
            source_name="property_facts_form",
            provider_status="ok",
            source_class=source_class,
            observed_at=observed_at,
            used_in_scoring=field in {"roof_type", "vent_type", "defensible_space_ft", "construction_year"},
            details=details,
        )

    footprint_status = str(property_level_context.get("footprint_status", "not_found"))
    footprint_provider_status = "ok"
    if footprint_status in {"provider_unavailable", "not_found"}:
        footprint_provider_status = "missing"
    elif footprint_status == "error":
        footprint_provider_status = "error"

    footprint_used = bool(property_level_context.get("footprint_used"))
    property_sources["footprint_used"] = _metadata(
        field_name="footprint_used",
        source_type="footprint_derived" if footprint_used else "missing",
        source_name="building_footprint_lookup",
        provider_status=footprint_provider_status,
        source_class="footprint_derived",
        dataset_version=os.getenv("WF_BUILDING_FOOTPRINT_VERSION"),
        observed_at=os.getenv("WF_BUILDING_FOOTPRINT_DATE") or (now_iso if footprint_used else None),
        used_in_scoring=False,
        details=f"footprint_status={footprint_status}",
    )

    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    ring_basis = str(property_level_context.get("fallback_mode") or "point_based") if isinstance(property_level_context, dict) else "point_based"
    for zone_key in ["zone_0_5_ft", "zone_5_30_ft", "zone_30_100_ft", "zone_100_300_ft"]:
        zone_data = rings.get(zone_key) if isinstance(rings, dict) else None
        density = zone_data.get("vegetation_density") if isinstance(zone_data, dict) else None
        if density is not None and footprint_used:
            source_type: SourceType = "footprint_derived"
            source_name = "building_footprint_ring_analysis"
            source_class = "footprint_derived"
            details = "Measured from building-footprint buffer ring."
            provider_status = "ok"
        elif density is not None:
            source_type = "heuristic"
            source_name = "point_proxy_ring_analysis"
            source_class = "heuristic"
            details = "Approximated from point-based annulus vegetation/fuel sampling."
            provider_status = "ok"
        else:
            source_type = "missing"
            source_name = "building_footprint_ring_analysis"
            source_class = "footprint_derived"
            details = "Ring metric unavailable."
            provider_status = footprint_provider_status
        property_sources[zone_key] = _metadata(
            field_name=zone_key,
            source_type=source_type,
            source_name=source_name,
            provider_status=provider_status,  # type: ignore[arg-type]
            source_class=source_class,
            dataset_version=os.getenv("WF_BUILDING_FOOTPRINT_VERSION"),
            observed_at=os.getenv("WF_BUILDING_FOOTPRINT_DATE") or (now_iso if density is not None else None),
            used_in_scoring=True,
            spatial_resolution_m={
                "zone_0_5_ft": 1.5,
                "zone_5_30_ft": 9.1,
                "zone_30_100_ft": 30.5,
                "zone_100_300_ft": 91.4,
            }[zone_key],
            details=f"{details} basis={ring_basis}",
        )

    access_context = getattr(context, "access_context", {}) or {}
    access_status = str(access_context.get("status") or "missing")
    access_source_type: SourceType = "missing"
    access_provider_status: ProviderStatus = "missing"
    access_source_name = str(access_context.get("source") or "osm_road_network")
    access_details = "Road-network access context unavailable."
    access_observed_at: str | None = None
    if access_status == "ok" and getattr(context, "access_exposure_index", None) is not None:
        access_source_type = "observed"
        access_provider_status = "ok"
        access_observed_at = now_iso
        access_details = "Observed from OSM road-network features near the property."
    elif access_status == "partial":
        access_source_type = "heuristic"
        access_provider_status = "ok"
        access_observed_at = now_iso
        access_details = "Partially observed road-network evidence; treated as advisory."
    elif access_status == "error":
        access_source_type = "missing"
        access_provider_status = "error"
        access_details = "Road-network source failed during access context extraction."

    property_sources["access_exposure"] = _metadata(
        field_name="access_exposure",
        source_type=access_source_type,
        source_name=access_source_name,
        provider_status=access_provider_status,
        source_class="observed" if access_source_type == "observed" else "missing",
        observed_at=access_observed_at,
        used_in_scoring=False,
        details=access_details,
    )

    merged = {**env_sources, **property_sources}
    inputs = [merged[key] for key in sorted(merged.keys())]
    total = max(1, len(inputs))

    direct_count = sum(1 for meta in inputs if meta.source_type in DIRECT_SOURCE_TYPES)
    inferred_count = sum(1 for meta in inputs if meta.source_type in INFERRED_SOURCE_TYPES)
    missing_count = sum(1 for meta in inputs if meta.source_type in LOW_QUALITY_SOURCE_TYPES)
    stale_count = sum(1 for meta in inputs if meta.freshness_status == "stale")
    current_count = sum(1 for meta in inputs if meta.freshness_status == "current")

    direct_score = round((direct_count / total) * 100.0, 1)
    inferred_score = round((inferred_count / total) * 100.0, 1)
    missing_share = round((missing_count / total) * 100.0, 1)
    stale_share = round((stale_count / total) * 100.0, 1)

    summary = DataProvenanceSummary(
        direct_data_coverage_score=direct_score,
        inferred_data_coverage_score=inferred_score,
        missing_data_share=missing_share,
        stale_data_share=stale_share,
        heuristic_input_count=sum(1 for meta in inputs if meta.source_type == "heuristic"),
        current_input_count=current_count,
    )
    provenance = DataProvenanceBlock(
        inputs=inputs,
        summary=summary,
        environmental_inputs_used=env_sources,
        property_inputs_used=property_sources,
        inferred_inputs_used=sorted(
            key
            for key, meta in merged.items()
            if meta.source_type in INFERRED_SOURCE_TYPES
        ),
        missing_inputs=sorted(key for key, meta in merged.items() if meta.source_type == "missing"),
        heuristic_inputs_used=sorted(key for key, meta in merged.items() if meta.source_type == "heuristic"),
    )
    return merged, provenance, direct_score, inferred_score, missing_share


def _score_family_quality(
    input_source_metadata: dict[str, InputSourceMetadata],
    fields: list[str],
) -> ScoreFamilyInputQuality:
    selected = [input_source_metadata[k] for k in fields if k in input_source_metadata]
    if not selected:
        return ScoreFamilyInputQuality()

    total = float(len(selected))
    direct = sum(1 for meta in selected if meta.source_type in DIRECT_SOURCE_TYPES)
    inferred = sum(1 for meta in selected if meta.source_type in INFERRED_SOURCE_TYPES)
    stale = sum(1 for meta in selected if meta.freshness_status == "stale")
    missing = sum(1 for meta in selected if meta.source_type == "missing")
    heuristic = sum(1 for meta in selected if meta.source_type == "heuristic")

    return ScoreFamilyInputQuality(
        direct_coverage=round((direct / total) * 100.0, 1),
        inferred_coverage=round((inferred / total) * 100.0, 1),
        stale_share=round((stale / total) * 100.0, 1),
        missing_share=round((missing / total) * 100.0, 1),
        heuristic_count=heuristic,
    )


def _build_score_family_input_quality(
    input_source_metadata: dict[str, InputSourceMetadata],
) -> tuple[ScoreFamilyInputQuality, ScoreFamilyInputQuality, ScoreFamilyInputQuality]:
    site_quality = _score_family_quality(input_source_metadata, SCORE_FAMILY_FIELDS["site_hazard"])
    home_quality = _score_family_quality(input_source_metadata, SCORE_FAMILY_FIELDS["home_vulnerability"])
    readiness_quality = _score_family_quality(input_source_metadata, SCORE_FAMILY_FIELDS["insurance_readiness"])
    return site_quality, home_quality, readiness_quality


def _structure_fact_known(payload: AddressRequest, assumptions: AssumptionsBlock, field: str) -> bool:
    if getattr(payload.attributes, field, None) is not None:
        return True
    if field in assumptions.inferred_inputs:
        return True
    return False


def _build_score_eligibility(
    *,
    payload: AddressRequest,
    context: object,
    property_level_context: dict[str, Any],
    assumptions: AssumptionsBlock,
    geocode_verified: bool,
) -> tuple[ScoreEligibility, ScoreEligibility, ScoreEligibility, AssessmentStatus, list[str]]:
    region_status = ""
    if isinstance(property_level_context, dict):
        region_status = str(property_level_context.get("region_status") or "")

    burn_or_hazard = getattr(context, "burn_probability_index", None) is not None or getattr(
        context, "hazard_severity_index", None
    ) is not None
    slope_ok = getattr(context, "slope_index", None) is not None
    fuel_or_canopy = getattr(context, "fuel_index", None) is not None or getattr(context, "canopy_index", None) is not None

    site_blockers: list[str] = []
    site_caveats: list[str] = []
    if not geocode_verified:
        site_blockers.append("Verified geocode unavailable")
    if region_status == "region_not_prepared":
        site_blockers.append("Region not prepared for this location; initialize regional layers.")
        site_caveats.append("Site Hazard is unavailable until regional layers are prepared.")
    elif region_status == "invalid_manifest":
        site_caveats.append("Prepared region manifest is invalid or incomplete; using partial environmental evidence.")
        site_caveats.append("Site Hazard is limited by incomplete prepared-region files.")
    elif region_status == "legacy_fallback":
        site_caveats.append("Site Hazard used legacy direct layer paths instead of prepared region data.")
    if not burn_or_hazard:
        site_caveats.append("Burn probability/hazard unavailable; using conservative environmental proxy behavior.")
    if not slope_ok:
        site_caveats.append("Slope layer missing; topography contribution is partially inferred.")
    if not fuel_or_canopy:
        site_caveats.append("Fuel/canopy context unavailable; vegetation pressure uses conservative fallback.")

    available_site_evidence = sum([burn_or_hazard, slope_ok, fuel_or_canopy])
    site_status: EligibilityStatus
    if not geocode_verified or region_status == "region_not_prepared":
        site_status = "insufficient"
    elif geocode_verified and burn_or_hazard and slope_ok and fuel_or_canopy:
        site_status = "full"
    elif geocode_verified and available_site_evidence >= 1:
        site_status = "partial"
        site_caveats.append("Site Hazard is based on partial environmental coverage.")
    else:
        site_status = "insufficient"
        site_blockers.append("No usable environmental evidence was available for Site Hazard scoring.")

    site_eligibility = ScoreEligibility(
        eligible=site_status != "insufficient",
        eligibility_status=site_status,
        blocking_reasons=sorted(set(site_blockers)),
        caveats=sorted(set(site_caveats)),
    )

    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    ring_signal = False
    if isinstance(rings, dict):
        for key in ["zone_0_5_ft", "zone_5_30_ft", "zone_30_100_ft", "ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft"]:
            zone = rings.get(key)
            if isinstance(zone, dict) and zone.get("vegetation_density") is not None:
                ring_signal = True
                break
    footprint_used = bool(property_level_context.get("footprint_used"))

    structure_context = any(
        _structure_fact_known(payload, assumptions, field)
        for field in ["roof_type", "vent_type", "defensible_space_ft", "construction_year"]
    )
    near_structure_signal = ring_signal or getattr(context, "fuel_index", None) is not None or getattr(
        context, "canopy_index", None
    ) is not None

    home_blockers: list[str] = []
    home_caveats: list[str] = []
    if not geocode_verified:
        home_blockers.append("Verified geocode unavailable")
    if region_status == "region_not_prepared":
        home_blockers.append("Region not prepared for this location; property-level context unavailable")
    if not structure_context and not ring_signal:
        home_caveats.append("Structure details and ring context are limited; vulnerability uses conservative defaults.")
    if not near_structure_signal:
        home_caveats.append("No near-structure vegetation/fuel signal; vulnerability relies on limited context.")
    if not footprint_used:
        home_caveats.append("Building footprint not found; using point-based fallback")
        home_caveats.append("Home Ignition Vulnerability used point-based fallback instead of footprint rings.")

    home_status: EligibilityStatus
    if not geocode_verified or region_status == "region_not_prepared":
        home_status = "insufficient"
    elif geocode_verified and near_structure_signal and (structure_context or ring_signal) and footprint_used and ring_signal:
        home_status = "full"
    elif geocode_verified and near_structure_signal and (structure_context or ring_signal):
        home_status = "partial"
    elif geocode_verified and (near_structure_signal or ring_signal) and (structure_context or ring_signal):
        home_status = "partial"
        home_caveats.append("Home Ignition Vulnerability used fallback structure/ring evidence.")
    else:
        home_status = "insufficient"
        home_blockers.append("No usable structure or near-structure context was available for Home Ignition Vulnerability.")

    home_eligibility = ScoreEligibility(
        eligible=home_status != "insufficient",
        eligibility_status=home_status,
        blocking_reasons=sorted(set(home_blockers)),
        caveats=sorted(set(home_caveats)),
    )

    known_structure_count = sum(
        1
        for field in ["roof_type", "vent_type", "defensible_space_ft"]
        if _structure_fact_known(payload, assumptions, field)
    )
    readiness_blockers: list[str] = []
    readiness_caveats: list[str] = []
    if site_status == "insufficient":
        readiness_blockers.append("Site Hazard evidence is insufficient")
    if home_status == "insufficient":
        readiness_blockers.append("Home Ignition Vulnerability evidence is insufficient")
    if region_status == "region_not_prepared":
        readiness_blockers.append("Region not prepared for this location")
    if known_structure_count < 1:
        readiness_blockers.append("Too many unknown structure facts for readiness rules")
    elif known_structure_count < 3:
        readiness_caveats.append("Insurance Readiness is limited by unknown roof/vent/defensible-space attributes.")

    readiness_status: EligibilityStatus
    if site_status == "full" and home_status == "full" and known_structure_count >= 2:
        readiness_status = "full"
    elif site_status != "insufficient" and home_status != "insufficient" and known_structure_count >= 1:
        readiness_status = "partial"
    else:
        readiness_status = "insufficient"

    readiness_eligibility = ScoreEligibility(
        eligible=readiness_status != "insufficient",
        eligibility_status=readiness_status,
        blocking_reasons=sorted(set(readiness_blockers)),
        caveats=sorted(set(readiness_caveats)),
    )

    if (not geocode_verified) or region_status == "region_not_prepared":
        assessment_status: AssessmentStatus = "insufficient_data"
    elif site_status == "insufficient" and home_status == "insufficient":
        assessment_status = "insufficient_data"
    elif "insufficient" in {site_status, home_status, readiness_status}:
        assessment_status = "partially_scored"
    elif {site_status, home_status, readiness_status} == {"full"}:
        assessment_status = "fully_scored"
    else:
        assessment_status = "partially_scored"

    assessment_blockers = sorted(
        set(site_eligibility.blocking_reasons + home_eligibility.blocking_reasons + readiness_eligibility.blocking_reasons)
    )
    return site_eligibility, home_eligibility, readiness_eligibility, assessment_status, assessment_blockers


def _apply_hard_trust_guardrails(
    confidence: ConfidenceBlock,
    *,
    site_eligibility: ScoreEligibility,
    home_eligibility: ScoreEligibility,
    readiness_eligibility: ScoreEligibility,
    assessment_status: AssessmentStatus,
    coverage_summary: LayerCoverageSummary | None = None,
) -> tuple[ConfidenceBlock, list[str], list[str]]:
    updated = confidence.model_copy(deep=True)
    downgrade_reasons: list[str] = []
    trust_tier_blockers: list[str] = []

    if assessment_status == "insufficient_data":
        updated.confidence_tier = "preliminary"
        updated.use_restriction = "not_for_underwriting_or_binding"
        downgrade_reasons.append("Minimum evidence requirements not met for one or more score families.")
        trust_tier_blockers.append("Insufficient evidence for at least one score family.")
    elif assessment_status == "partially_scored" and updated.confidence_tier == "high":
        updated.confidence_tier = "moderate"
        downgrade_reasons.append("Partial evidence path prevents high confidence.")
        trust_tier_blockers.append("Assessment is partially scored.")

    if site_eligibility.eligibility_status != "full":
        trust_tier_blockers.append("Site Hazard evidence is partial or insufficient.")
    if home_eligibility.eligibility_status != "full":
        trust_tier_blockers.append("Home Ignition Vulnerability evidence is partial or insufficient.")
    if readiness_eligibility.eligibility_status != "full":
        trust_tier_blockers.append("Insurance Readiness evidence is partial or insufficient.")

    if any("verified geocode unavailable" in r.lower() for r in site_eligibility.blocking_reasons + home_eligibility.blocking_reasons):
        updated.confidence_tier = "preliminary"
        updated.use_restriction = "not_for_underwriting_or_binding"
        downgrade_reasons.append("Verified geocode unavailable.")
        trust_tier_blockers.append("Verified geocode unavailable.")

    if updated.confidence_score <= 0.0:
        updated.confidence_tier = "preliminary"
        updated.use_restriction = "not_for_underwriting_or_binding"
        downgrade_reasons.append("Confidence score is zero due to missing critical evidence.")
        trust_tier_blockers.append("Confidence score is zero due to missing critical evidence.")

    if any("point-based fallback" in r.lower() for r in home_eligibility.blocking_reasons):
        if updated.use_restriction == "shareable":
            updated.use_restriction = "homeowner_review_recommended"
        downgrade_reasons.append("Property-level footprint/ring context unavailable; point-based fallback used.")

    if coverage_summary:
        if coverage_summary.failed_count > 0:
            if updated.use_restriction == "shareable":
                updated.use_restriction = "agent_or_inspector_review_recommended"
            downgrade_reasons.append("One or more layers failed sampling/extent checks.")
            trust_tier_blockers.append("Layer sampling failures detected.")
        if coverage_summary.critical_missing_layers:
            updated.use_restriction = "not_for_underwriting_or_binding"
            if updated.confidence_tier != "preliminary":
                updated.confidence_tier = "low"
            downgrade_reasons.append("Critical required layers are missing or not usable.")
            trust_tier_blockers.append(
                "Critical missing layers: " + ", ".join(coverage_summary.critical_missing_layers[:6])
            )

    if readiness_eligibility.eligibility_status == "insufficient":
        updated.use_restriction = "not_for_underwriting_or_binding"
    elif readiness_eligibility.eligibility_status == "partial" and updated.use_restriction == "shareable":
        updated.use_restriction = "agent_or_inspector_review_recommended"

    if downgrade_reasons:
        updated.low_confidence_flags = sorted(set(updated.low_confidence_flags + downgrade_reasons))
    return updated, sorted(set(downgrade_reasons)), sorted(set(trust_tier_blockers))


def _build_assessment_diagnostics(
    *,
    data_provenance: DataProvenanceBlock,
    confidence_downgrade_reasons: list[str],
    trust_tier_blockers: list[str],
    fallback_decisions: list[dict[str, object]] | None = None,
) -> AssessmentDiagnostics:
    critical_present: list[str] = []
    critical_missing: list[str] = []
    stale_inputs: list[str] = []
    inferred_inputs: list[str] = []
    heuristic_inputs: list[str] = []

    for meta in data_provenance.inputs:
        if meta.field_name in CRITICAL_PROVENANCE_FIELDS:
            if meta.source_type in LOW_QUALITY_SOURCE_TYPES:
                critical_missing.append(meta.field_name)
            else:
                critical_present.append(meta.field_name)
        if meta.freshness_status == "stale":
            stale_inputs.append(meta.field_name)
        if meta.source_type in INFERRED_SOURCE_TYPES:
            inferred_inputs.append(meta.field_name)
        if meta.source_type == "heuristic":
            heuristic_inputs.append(meta.field_name)

    return AssessmentDiagnostics(
        critical_inputs_present=sorted(set(critical_present)),
        critical_inputs_missing=sorted(set(critical_missing)),
        stale_inputs=sorted(set(stale_inputs)),
        inferred_inputs=sorted(set(inferred_inputs)),
        heuristic_inputs=sorted(set(heuristic_inputs)),
        confidence_downgrade_reasons=sorted(set(confidence_downgrade_reasons)),
        trust_tier_blockers=sorted(set(trust_tier_blockers)),
        fallback_decisions=list(fallback_decisions or []),
    )


def _build_fallback_decisions(
    *,
    attribute_fallbacks: list[dict[str, object]],
    environmental_layer_status: dict[str, str],
    property_level_context: dict[str, Any],
    confidence_penalties: list[ConfidencePenalty],
    score_availability_notes: list[str],
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = list(attribute_fallbacks)
    penalty_by_key = {p.penalty_key: float(p.amount) for p in confidence_penalties}

    env_proxy_map = {
        "burn_probability": "hazard/wildland proxy",
        "hazard": "burn probability/wildland proxy",
        "slope": "terrain default weighting",
        "fuel": "ring/canopy proxy",
        "canopy": "fuel/ring proxy",
        "fire_history": "regional recurrence baseline",
    }
    for layer, status in (environmental_layer_status or {}).items():
        if status == "ok":
            continue
        confidence_penalty = 0.0
        if status == "error":
            confidence_penalty = penalty_by_key.get("provider_errors", 0.0)
        else:
            confidence_penalty = penalty_by_key.get("missing_environmental_layers", 0.0)
        decisions.append(
            {
                "fallback_type": "layer_proxy" if status in {"missing", "error"} else "partial_layer",
                "missing_input": f"{layer}_layer",
                "substitute_input": env_proxy_map.get(layer, "conservative_layer_proxy"),
                "confidence_penalty_hint": round(confidence_penalty, 1),
                "quality_label": "conservative",
                "note": f"{layer.replace('_', ' ').title()} layer status is '{status}', so scoring used fallback behavior.",
            }
        )

    if not bool(property_level_context.get("footprint_used")):
        decisions.append(
            {
                "fallback_type": "point_based_context",
                "missing_input": "building_footprint",
                "substitute_input": "point_neighborhood_context",
                "confidence_penalty_hint": round(penalty_by_key.get("missing_ring_context", 6.0), 1),
                "quality_label": "inferred",
                "note": "Building footprint rings unavailable; point-based near-structure context was used.",
            }
        )

    for note in score_availability_notes:
        lowered = str(note).lower()
        if "component only" in lowered:
            decisions.append(
                {
                    "fallback_type": "partial_component_blend",
                    "missing_input": "one_score_component_unavailable",
                    "substitute_input": "available_component_reweighted",
                    "confidence_penalty_hint": 2.5,
                    "quality_label": "neutral",
                    "note": str(note),
                }
            )

    unique: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in decisions:
        key = (
            str(row.get("fallback_type") or ""),
            str(row.get("missing_input") or ""),
            str(row.get("substitute_input") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _build_assessment_limitations_summary(
    *,
    fallback_decisions: list[dict[str, object]],
    score_availability_notes: list[str],
    coverage_summary: LayerCoverageSummary,
) -> list[str]:
    summary: list[str] = []
    for decision in fallback_decisions:
        note = str(decision.get("note") or "").strip()
        quality = str(decision.get("quality_label") or "").strip()
        if not note:
            continue
        if quality:
            summary.append(f"{note} ({quality}).")
        else:
            summary.append(note)
    for note in score_availability_notes:
        summary.append(str(note))
    if coverage_summary.critical_missing_layers:
        summary.append(
            "Critical layers still missing: "
            + ", ".join(coverage_summary.critical_missing_layers[:6])
            + "."
        )
    if coverage_summary.not_configured_count > 0:
        summary.append(
            f"{coverage_summary.not_configured_count} optional layer(s) are not configured; assessment used core data and fallbacks."
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in summary:
        trimmed = item.strip()
        if not trimmed:
            continue
        if trimmed in seen:
            continue
        seen.add(trimmed)
        deduped.append(trimmed)
    return deduped[:6]


def _apply_score_availability(
    *,
    site_hazard_score: float,
    home_ignition_vulnerability_score: float,
    insurance_readiness_score: float,
    blended_wildfire_risk_score: float,
    legacy_weighted_wildfire_risk_score: float,
    site_hazard_eligibility: ScoreEligibility,
    home_vulnerability_eligibility: ScoreEligibility,
    insurance_readiness_eligibility: ScoreEligibility,
) -> tuple[dict[str, float | None | bool], list[str]]:
    site_available = site_hazard_eligibility.eligibility_status != "insufficient"
    home_available = home_vulnerability_eligibility.eligibility_status != "insufficient"
    readiness_available = insurance_readiness_eligibility.eligibility_status != "insufficient"
    wildfire_available = site_available or home_available

    notes: list[str] = []
    if not site_available:
        notes.append("Wildfire score not computed because required environmental layers were unavailable.")
    if not home_available:
        notes.append(
            "Home vulnerability score not computed because no footprint/ring context or confirmed structure facts were available."
        )
    if not readiness_available:
        notes.append(
            "Insurance readiness score not computed because site hazard and structure evidence were insufficient."
        )
    if site_available and home_available:
        wildfire_score = blended_wildfire_risk_score
        legacy_score = legacy_weighted_wildfire_risk_score
    elif site_available:
        wildfire_score = round(site_hazard_score, 1)
        legacy_score = round(site_hazard_score, 1)
        notes.append(
            "Wildfire score used Site Hazard component only because Home Ignition Vulnerability evidence was insufficient."
        )
    elif home_available:
        wildfire_score = round(home_ignition_vulnerability_score, 1)
        legacy_score = round(home_ignition_vulnerability_score, 1)
        notes.append(
            "Wildfire score used Home Ignition Vulnerability component only because Site Hazard evidence was insufficient."
        )
    else:
        wildfire_score = None
        legacy_score = None
        notes.append(
            "Wildfire score not computed because both Site Hazard and Home Ignition Vulnerability lacked minimum evidence."
        )

    return (
        {
            "site_hazard_score": site_hazard_score if site_available else None,
            "home_ignition_vulnerability_score": home_ignition_vulnerability_score if home_available else None,
            "insurance_readiness_score": insurance_readiness_score if readiness_available else None,
            "wildfire_risk_score": wildfire_score if wildfire_available else None,
            "legacy_weighted_wildfire_risk_score": legacy_score if wildfire_available else None,
            "site_hazard_score_available": site_available,
            "home_ignition_vulnerability_score_available": home_available,
            "insurance_readiness_score_available": readiness_available,
            "wildfire_risk_score_available": wildfire_available,
        },
        notes,
    )


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


def _build_score_decomposition(
    *,
    risk: RiskComputation,
) -> tuple[float, float]:
    site_hazard_score = risk_engine.compute_site_hazard_score(risk)
    home_ignition_vulnerability_score = risk_engine.compute_home_ignition_vulnerability_score(risk)
    return site_hazard_score, home_ignition_vulnerability_score


def _build_score_sections(
    *,
    site_hazard_score: float | None,
    home_ignition_vulnerability_score: float | None,
    insurance_readiness_score: float | None,
    top_risk_drivers: list[str],
    top_protective_factors: list[str],
    mitigation_plan,
    property_findings: list[str],
    readiness_blockers: list[str],
    confidence_block: ConfidenceBlock,
) -> tuple[ScoreSectionSummary, ScoreSectionSummary, ScoreSectionSummary]:
    def _score_text(value: float | None) -> str:
        return f"{value:.1f}/100" if value is not None else "not computed"

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
        label="Site Hazard",
        score=site_hazard_score,
        summary=(
            f"Site Hazard {_score_text(site_hazard_score)} reflects landscape fuel, slope, and nearby fire pressure around the home."
        ),
        explanation="What the landscape is doing around your property.",
        top_drivers=top_risk_drivers[:3],
        key_drivers=top_risk_drivers[:3],
        protective_factors=top_protective_factors[:3],
        top_next_actions=site_actions,
        next_actions=site_actions,
    )

    home_section = ScoreSectionSummary(
        label="Home Ignition Vulnerability",
        score=home_ignition_vulnerability_score,
        summary=(
            f"Home Ignition Vulnerability {_score_text(home_ignition_vulnerability_score)} reflects structure hardening and near-home ignition pathways."
        ),
        explanation="What the home and immediate surroundings are contributing.",
        top_drivers=(property_findings[:3] or top_risk_drivers[:3]),
        key_drivers=(property_findings[:3] or top_risk_drivers[:3]),
        protective_factors=top_protective_factors[:3],
        top_next_actions=home_actions,
        next_actions=home_actions,
    )

    readiness_actions = [m.title for m in mitigation_plan[:3]]
    readiness_section = ScoreSectionSummary(
        label="Insurance Readiness",
        score=insurance_readiness_score,
        summary=(
            f"Insurance Readiness {_score_text(insurance_readiness_score)} is a heuristic advisory score and not carrier-approved underwriting."
        ),
        explanation="What an insurer is likely to care about next.",
        top_drivers=(readiness_blockers[:3] or ["No major readiness blockers detected"]),
        key_drivers=(readiness_blockers[:3] or ["No major readiness blockers detected"]),
        protective_factors=top_protective_factors[:3],
        top_next_actions=readiness_actions,
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
        access_risk_provisional=risk.access_provisional,
        access_included_in_total=False,
        access_risk_note=risk.access_note,
    )


def _build_top_protective_factors(attrs: PropertyAttributes, submodels: dict[str, SubmodelScore]) -> list[str]:
    factors: list[str] = []

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

    if result.wildfire_risk_score is not None and result.wildfire_risk_score >= risk_blocker_threshold:
        blocker = f"Carrier profile threshold exceeded ({ruleset.ruleset_id})"
        if blocker not in result.readiness_blockers:
            result.readiness_blockers.append(blocker)
            extra_penalty += 5.0

    if len(result.missing_inputs) >= inspection_missing_threshold:
        blocker = "Inspection required before underwriting decision"
        if blocker not in result.readiness_blockers:
            result.readiness_blockers.append(blocker)
            extra_penalty += 4.0

    if extra_penalty > 0 and result.insurance_readiness_score is not None:
        result.insurance_readiness_score = round(max(0.0, min(100.0, result.insurance_readiness_score - extra_penalty)), 1)
        if result.risk_scores:
            result.risk_scores.insurance_readiness_score = result.insurance_readiness_score
        if result.insurance_readiness_section:
            result.insurance_readiness_section.score = result.insurance_readiness_score
            result.insurance_readiness_section.summary = (
                f"Insurance Readiness {result.insurance_readiness_score:.1f}/100 is a heuristic advisory score "
                "and not carrier-approved underwriting."
            )
        if result.score_summaries:
            result.score_summaries.insurance_readiness.score = result.insurance_readiness_score
            result.score_summaries.insurance_readiness.summary = (
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
    if result.confidence:
        carried_penalties = list(result.evidence_quality_summary.confidence_penalties or [])
        result.score_evidence_ledger = _build_score_evidence_ledger(
            submodel_scores=result.submodel_scores,
            weighted_contributions=result.weighted_contributions,
            readiness_factors=result.readiness_factors,
            readiness_score=result.insurance_readiness_score,
            site_hazard_score=result.site_hazard_score,
            home_ignition_vulnerability_score=result.home_ignition_vulnerability_score,
            wildfire_risk_score=result.wildfire_risk_score,
            input_source_metadata=result.input_source_metadata,
        )
        result.evidence_quality_summary = _build_evidence_quality_summary(
            ledger=result.score_evidence_ledger,
            confidence_penalties=carried_penalties,
            confidence=result.confidence,
            assessment_status=result.assessment_status,
        )
    return _refresh_result_governance(result)


def _run_assessment(
    payload: AddressRequest,
    *,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    assessment_id: str | None = None,
    portfolio_name: str | None = None,
    tags: list[str] | None = None,
    geocode_resolution: GeocodeResolution | None = None,
    coverage_resolution: RegionCoverageResolution | None = None,
) -> tuple[AssessmentResult, dict]:
    geocode_resolution = geocode_resolution or _resolve_trusted_geocode(
        address_input=payload.address,
        purpose="assessment",
        route_name="assessment_core",
    )
    lat = float(geocode_resolution.latitude)
    lon = float(geocode_resolution.longitude)
    geocode_source = geocode_resolution.geocode_source
    geocode_meta = dict(geocode_resolution.geocode_meta or {})
    geocode_precision = str(geocode_meta.get("geocode_precision") or "unknown")
    requested_structure_source = str(payload.structure_geometry_source or "auto_detected").strip().lower()
    if requested_structure_source not in {"auto_detected", "user_selected", "user_modified"}:
        requested_structure_source = "auto_detected"
    requested_structure_id = (
        str(payload.selected_structure_id).strip()
        if payload.selected_structure_id is not None and str(payload.selected_structure_id).strip()
        else None
    )
    requested_structure_geometry = (
        payload.selected_structure_geometry if isinstance(payload.selected_structure_geometry, dict) else None
    )
    requested_selection_mode = str(payload.selection_mode or "polygon").strip().lower()
    if requested_selection_mode not in {"polygon", "point"}:
        requested_selection_mode = "polygon"
    requested_property_anchor_point: dict[str, float] | None = None
    raw_property_anchor_point = payload.property_anchor_point or payload.user_selected_point
    if raw_property_anchor_point is not None:
        try:
            requested_property_anchor_point = {
                "latitude": float(raw_property_anchor_point.latitude),
                "longitude": float(raw_property_anchor_point.longitude),
            }
            if not (
                -90.0 <= requested_property_anchor_point["latitude"] <= 90.0
                and -180.0 <= requested_property_anchor_point["longitude"] <= 180.0
            ):
                requested_property_anchor_point = None
        except (TypeError, ValueError):
            requested_property_anchor_point = None
    requested_user_selected_point: dict[str, float] | None = None
    if payload.user_selected_point is not None:
        try:
            requested_user_selected_point = {
                "latitude": float(payload.user_selected_point.latitude),
                "longitude": float(payload.user_selected_point.longitude),
            }
            if not (
                -90.0 <= requested_user_selected_point["latitude"] <= 90.0
                and -180.0 <= requested_user_selected_point["longitude"] <= 180.0
            ):
                requested_user_selected_point = None
        except (TypeError, ValueError):
            requested_user_selected_point = None
    collect_kwargs = {
        "geocode_precision": geocode_precision,
        "structure_geometry_source": requested_structure_source,
        "selection_mode": requested_selection_mode,
        "property_anchor_point": requested_property_anchor_point,
        "user_selected_point": requested_user_selected_point,
        "selected_structure_id": requested_structure_id,
        "selected_structure_geometry": requested_structure_geometry,
    }
    try:
        context = wildfire_data.collect_context(lat, lon, **collect_kwargs)
    except TypeError:
        # Backward-compatible for tests/mocks still using collect_context(lat, lon).
        context = wildfire_data.collect_context(lat, lon)
    scoring_attrs = normalize_property_attributes(payload.attributes)
    normalization_changes = normalized_attribute_changes(payload.attributes, scoring_attrs)
    property_level_context = _normalize_property_level_context(context.property_level_context)
    geocode_provider = str(geocode_meta.get("geocode_provider") or geocode_meta.get("provider") or geocode_source or "")
    property_level_context["geocode_provider"] = geocode_provider or None
    property_level_context["geocoded_address"] = geocode_meta.get("geocoded_address") or geocode_meta.get("matched_address")
    property_level_context["geocode_precision"] = geocode_precision
    if property_level_context.get("property_anchor_source") in {
        "geocoded_address_point",
        "rooftop_geocode",
        "address_point_geocode",
        "interpolated_geocode",
        "approximate_geocode",
    }:
        property_level_context["property_anchor_precision"] = geocode_precision
    if not isinstance(property_level_context.get("geocoded_address_point"), dict):
        property_level_context["geocoded_address_point"] = {"latitude": lat, "longitude": lon}
    if not isinstance(property_level_context.get("property_anchor_point"), dict):
        property_level_context["property_anchor_point"] = {"latitude": lat, "longitude": lon}
    if not isinstance(property_level_context.get("assessed_property_display_point"), dict):
        property_level_context["assessed_property_display_point"] = dict(
            property_level_context.get("property_anchor_point") or {"latitude": lat, "longitude": lon}
        )
    resolved_structure_geometry_source = str(
        property_level_context.get("structure_geometry_source") or "auto_detected"
    ).strip().lower()
    if requested_structure_source in {"user_selected", "user_modified"}:
        resolved_structure_geometry_source = requested_structure_source
    property_level_context["structure_geometry_source"] = (
        resolved_structure_geometry_source
        if resolved_structure_geometry_source in {"auto_detected", "user_selected", "user_modified"}
        else "auto_detected"
    )
    if requested_structure_id:
        property_level_context["selected_structure_id"] = requested_structure_id
    if requested_structure_geometry is not None:
        property_level_context["selected_structure_geometry"] = requested_structure_geometry
    property_level_context["selection_mode"] = str(
        property_level_context.get("selection_mode") or requested_selection_mode
    )
    if requested_property_anchor_point is not None:
        property_level_context["property_anchor_point"] = requested_property_anchor_point
    if requested_user_selected_point is not None:
        property_level_context["user_selected_point"] = requested_user_selected_point
    if _is_dev_mode():
        LOGGER.info(
            "structure_selection_context %s",
            json.dumps(
                {
                    "event": "structure_selection_context",
                    "selection_mode": property_level_context.get("selection_mode"),
                    "user_selected_point": property_level_context.get("user_selected_point"),
                    "candidate_structure_count": property_level_context.get("candidate_structure_count"),
                    "final_structure_geometry_source": property_level_context.get("final_structure_geometry_source"),
                    "structure_match_status": property_level_context.get("structure_match_status"),
                    "structure_match_distance_m": property_level_context.get("structure_match_distance_m"),
                    "footprint_source_name": property_level_context.get("footprint_source_name"),
                    "footprint_source_vintage": property_level_context.get("footprint_source_vintage"),
                    "user_selected_point_in_footprint": property_level_context.get("user_selected_point_in_footprint"),
                },
                sort_keys=True,
                default=str,
            ),
        )
    context.property_level_context = property_level_context
    scoring_attrs, attribute_fallbacks = _apply_attribute_fallbacks(
        scoring_attrs,
        property_level_context=property_level_context,
        context=context,
    )
    risk: RiskComputation = risk_engine.score(scoring_attrs, lat, lon, context)
    readiness = risk_engine.compute_insurance_readiness(scoring_attrs, context, risk)

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
        scoring_attrs,
        context,
        {k: v.score for k, v in submodel_scores.items()},
        readiness.readiness_blockers,
    )

    coverage_resolution = coverage_resolution or _resolve_prepared_region(
        latitude=lat,
        longitude=lon,
        route_name="assessment_core",
        address_input=payload.address,
    )
    coverage_lookup = dict(coverage_resolution.coverage or {})
    layer_coverage_audit, coverage_summary = _normalize_layer_coverage(
        property_level_context,
        environmental_layer_status=context.environmental_layer_status,
    )
    defensible_space_analysis = build_defensible_space_analysis(
        property_level_context=property_level_context,
        layer_coverage_audit=[row.model_dump() for row in layer_coverage_audit],
    )
    top_near_structure_risk_drivers = build_top_near_structure_risk_drivers(defensible_space_analysis)
    prioritized_vegetation_actions_raw = build_prioritized_vegetation_actions(defensible_space_analysis)
    prioritized_vegetation_actions = [NearStructureAction.model_validate(row) for row in prioritized_vegetation_actions_raw]
    defensible_space_limitations_summary = list(
        (defensible_space_analysis.get("data_quality") or {}).get("limitations") or []
    )[:4]
    property_level_context["layer_coverage_audit"] = [row.model_dump() for row in layer_coverage_audit]
    property_level_context["coverage_summary"] = coverage_summary.model_dump()
    property_level_context["defensible_space_analysis"] = defensible_space_analysis
    region_resolution = _build_region_resolution(
        property_level_context=property_level_context,
        coverage_lookup=coverage_lookup,
    )
    factors = EnvironmentalFactors(
        burn_probability=context.burn_probability,
        wildfire_hazard=context.wildfire_hazard,
        slope=context.slope,
        fuel_model=context.fuel_model,
        canopy_cover=context.canopy_cover,
        historic_fire_distance=context.historic_fire_distance,
        wildland_distance=context.wildland_distance,
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
    environmental_data_completeness = compute_environmental_data_completeness(context)
    assumptions_block = _build_assumption_tracking(
        payload,
        list(all_assumptions),
        all_sources,
        context.environmental_layer_status,
        property_level_context,
        geocode_verified=True,
    )
    all_assumptions = assumptions_block.assumptions_used
    input_source_metadata, data_provenance, direct_data_coverage_score, inferred_data_coverage_score, missing_data_share = (
        _build_data_provenance(
            payload=payload,
            assumptions=assumptions_block,
            context=context,
            property_level_context=property_level_context,
        )
    )
    (
        site_hazard_eligibility,
        home_vulnerability_eligibility,
        insurance_readiness_eligibility,
        assessment_status,
        assessment_blockers,
    ) = _build_score_eligibility(
        payload=payload,
        context=context,
        property_level_context=property_level_context,
        assumptions=assumptions_block,
        geocode_verified=True,
    )
    (
        site_hazard_input_quality,
        home_vulnerability_input_quality,
        insurance_readiness_input_quality,
    ) = _build_score_family_input_quality(input_source_metadata)
    confidence_block = _build_confidence(
        assumptions_block,
        environmental_data_completeness=environmental_data_completeness,
        geocode_verified=True,
        property_level_context=property_level_context,
        environmental_layer_status=context.environmental_layer_status,
        data_provenance=data_provenance,
    )
    confidence_block, confidence_downgrade_reasons, trust_tier_blockers = _apply_hard_trust_guardrails(
        confidence_block,
        site_eligibility=site_hazard_eligibility,
        home_eligibility=home_vulnerability_eligibility,
        readiness_eligibility=insurance_readiness_eligibility,
        assessment_status=assessment_status,
        coverage_summary=coverage_summary,
    )
    confidence_penalties = _derive_confidence_penalties(
        assumptions_block,
        environmental_data_completeness=environmental_data_completeness,
        geocode_verified=True,
        property_level_context=property_level_context,
        environmental_layer_status=context.environmental_layer_status,
        data_provenance=data_provenance,
        coverage_summary=coverage_summary,
    )
    property_findings = _build_property_findings(property_level_context)
    top_risk_drivers = _merge_property_drivers(_build_top_risk_drivers(submodel_scores), property_findings)
    for driver in top_near_structure_risk_drivers:
        if driver not in top_risk_drivers:
            top_risk_drivers.insert(0, driver)
    top_risk_drivers = top_risk_drivers[:3]
    top_protective_factors = _build_top_protective_factors(scoring_attrs, submodel_scores)

    raw_site_hazard_score, raw_home_ignition_vulnerability_score = _build_score_decomposition(risk=risk)
    raw_legacy_weighted_wildfire_risk_score = risk.total_score
    raw_blended_wildfire_risk_score = risk_engine.compute_blended_wildfire_score(
        site_hazard_score=raw_site_hazard_score,
        home_ignition_vulnerability_score=raw_home_ignition_vulnerability_score,
    )
    score_outputs, score_availability_notes = _apply_score_availability(
        site_hazard_score=raw_site_hazard_score,
        home_ignition_vulnerability_score=raw_home_ignition_vulnerability_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
        blended_wildfire_risk_score=raw_blended_wildfire_risk_score,
        legacy_weighted_wildfire_risk_score=raw_legacy_weighted_wildfire_risk_score,
        site_hazard_eligibility=site_hazard_eligibility,
        home_vulnerability_eligibility=home_vulnerability_eligibility,
        insurance_readiness_eligibility=insurance_readiness_eligibility,
    )
    site_hazard_score = score_outputs["site_hazard_score"]
    home_ignition_vulnerability_score = score_outputs["home_ignition_vulnerability_score"]
    insurance_readiness_score = score_outputs["insurance_readiness_score"]
    blended_wildfire_risk_score = score_outputs["wildfire_risk_score"]
    legacy_weighted_wildfire_risk_score = score_outputs["legacy_weighted_wildfire_risk_score"]
    fallback_decisions = _build_fallback_decisions(
        attribute_fallbacks=attribute_fallbacks,
        environmental_layer_status=context.environmental_layer_status,
        property_level_context=property_level_context,
        confidence_penalties=confidence_penalties,
        score_availability_notes=score_availability_notes,
    )
    assessment_diagnostics = _build_assessment_diagnostics(
        data_provenance=data_provenance,
        confidence_downgrade_reasons=confidence_downgrade_reasons,
        trust_tier_blockers=trust_tier_blockers + assessment_blockers,
        fallback_decisions=fallback_decisions,
    )
    assessment_limitations_summary = _build_assessment_limitations_summary(
        fallback_decisions=fallback_decisions,
        score_availability_notes=score_availability_notes,
        coverage_summary=coverage_summary,
    )
    readiness_factors = [
        ReadinessFactor(name=f["name"], status=f["status"], score_impact=f["score_impact"], detail=f["detail"])
        for f in readiness.readiness_factors
    ]

    score_evidence_ledger = _build_score_evidence_ledger(
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        readiness_factors=readiness_factors,
        readiness_score=insurance_readiness_score,
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        wildfire_risk_score=blended_wildfire_risk_score,
        input_source_metadata=input_source_metadata,
    )
    evidence_quality_summary = _build_evidence_quality_summary(
        ledger=score_evidence_ledger,
        confidence_penalties=confidence_penalties,
        confidence=confidence_block,
        assessment_status=assessment_status,
    )

    site_hazard_section, home_ignition_section, insurance_readiness_section = _build_score_sections(
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        insurance_readiness_score=insurance_readiness_score,
        top_risk_drivers=top_risk_drivers,
        top_protective_factors=top_protective_factors,
        mitigation_plan=mitigation_plan,
        property_findings=property_findings,
        readiness_blockers=readiness.readiness_blockers,
        confidence_block=confidence_block,
    )
    if site_hazard_eligibility.caveats:
        site_hazard_section.summary += " " + " ".join(site_hazard_eligibility.caveats[:2])
    if home_vulnerability_eligibility.caveats:
        home_ignition_section.summary += " " + " ".join(home_vulnerability_eligibility.caveats[:2])
    if insurance_readiness_eligibility.caveats:
        insurance_readiness_section.summary += " " + " ".join(insurance_readiness_eligibility.caveats[:2])

    scoring_notes = [
        ACCESS_PROVISIONAL_NOTE,
        "Submodel/weight framework and readiness rules are deterministic MVP heuristics for calibration and explainability.",
        "Scores are advisory heuristics and not carrier-approved underwriting or premium predictions.",
    ]
    for fallback in attribute_fallbacks:
        note = str(fallback.get("note") or "").strip()
        if note:
            scoring_notes.append(note)
    if any("fallback" in a.lower() or "unavailable" in a.lower() for a in all_assumptions):
        scoring_notes.append("One or more providers/layers required fallback assumptions.")
    for layer, status in (context.environmental_layer_status or {}).items():
        if status != "ok":
            scoring_notes.append(
                f"Environmental {layer.replace('_', ' ')} layer {status} — score uses partial data."
            )
    if not property_level_context.get("footprint_used"):
        scoring_notes.append("Building footprint not found — vulnerability estimated using point context.")
    else:
        scoring_notes.append("Structure ring metrics were derived from building-footprint context.")
    region_status = str(property_level_context.get("region_status") or "")
    if region_status == "region_not_prepared":
        scoring_notes.append(
            "Region not prepared for this location — initialize regional layers before high-confidence scoring."
        )
    elif region_status == "legacy_fallback":
        scoring_notes.append("Assessment used legacy direct layer paths because no prepared region matched this location.")
    elif region_status == "invalid_manifest":
        scoring_notes.append("Prepared region manifest is incomplete; assessment used partial regional coverage.")
    if data_provenance.heuristic_inputs_used:
        scoring_notes.append(
            "Heuristic inputs used: " + ", ".join(data_provenance.heuristic_inputs_used[:6]) + "."
        )
    if prioritized_vegetation_actions:
        scoring_notes.append(
            "Near-structure vegetation analysis identified prioritized zone actions: "
            + ", ".join(a.title for a in prioritized_vegetation_actions[:3])
            + "."
        )
    elif defensible_space_limitations_summary:
        scoring_notes.append(
            "Near-structure vegetation analysis limitations: "
            + "; ".join(defensible_space_limitations_summary[:2])
            + "."
        )
    access_context = getattr(context, "access_context", {}) or {}
    access_status = str(access_context.get("status") or "missing")
    if access_status == "ok":
        scoring_notes.append(
            "Access exposure derived from observable OSM road-network features (advisory and excluded from wildfire total weighting)."
        )
    else:
        scoring_notes.append(
            "Access exposure unavailable from open road-network context; advisory access metric could not be computed."
        )
    if data_provenance.summary.stale_data_share > 0:
        scoring_notes.append(
            f"{data_provenance.summary.stale_data_share:.1f}% of tracked inputs are stale per freshness policy."
        )
    if coverage_summary.critical_missing_layers:
        scoring_notes.append(
            "Critical layer gaps: " + ", ".join(coverage_summary.critical_missing_layers[:6]) + "."
        )
    optional_not_configured_layers = sorted(
        {
            row.layer_key
            for row in layer_coverage_audit
            if row.coverage_status == "not_configured"
            and row.layer_key in {"whp", "mtbs_severity", "gridmet_dryness", "roads", "fema_structures"}
        }
    )
    for recommendation in coverage_summary.recommended_actions[:5]:
        lower_reco = recommendation.lower()
        if (
            " is not configured;" in lower_reco
            and any(lower_reco.startswith(f"{layer} ") for layer in optional_not_configured_layers)
        ):
            continue
        scoring_notes.append(recommendation)
    if optional_not_configured_layers:
        scoring_notes.append(
            "Optional enrichment layers not configured: "
            + ", ".join(optional_not_configured_layers[:6])
            + ". Scores remain available using core prepared-region layers."
        )
    if normalization_changes:
        notes = ", ".join(
            f"{field}: '{change['input']}' -> '{change['normalized']}'"
            for field, change in sorted(normalization_changes.items())
        )
        scoring_notes.append(f"Categorical inputs were normalized for scoring ({notes}).")
    stale_fields = [meta.field_name for meta in data_provenance.inputs if meta.freshness_status == "stale"]
    if stale_fields:
        scoring_notes.append("Stale inputs: " + ", ".join(sorted(stale_fields)[:6]) + ".")
    critical_unknown = [
        meta.field_name
        for meta in data_provenance.inputs
        if meta.field_name in CRITICAL_PROVENANCE_FIELDS
        and meta.freshness_status == "unknown"
        and meta.source_type not in LOW_QUALITY_SOURCE_TYPES
    ]
    if critical_unknown:
        scoring_notes.append(
            "Critical inputs with unknown freshness: " + ", ".join(sorted(critical_unknown)[:6]) + "."
        )
    user_unverified_fields = [
        meta.field_name
        for meta in data_provenance.inputs
        if meta.source_type == "user_provided" and meta.field_name in {"roof_type", "vent_type", "defensible_space_ft"}
    ]
    if user_unverified_fields:
        scoring_notes.append(
            "User-provided structure details are unverified: " + ", ".join(sorted(user_unverified_fields)) + "."
        )
    if confidence_block.use_restriction == "not_for_underwriting_or_binding":
        scoring_notes.append("Current confidence gating: not for underwriting or binding decisions.")
    scoring_notes.extend(score_availability_notes)
    if assessment_status != "fully_scored":
        scoring_notes.append(
            f"Assessment status is {assessment_status}; review blockers before using this output for high-trust decisions."
        )
    if assessment_blockers:
        scoring_notes.append("Assessment blockers: " + ", ".join(assessment_blockers[:6]) + ".")

    def _score_phrase(label: str, value: float | None) -> str:
        return f"{label}: {value:.1f}/100" if value is not None else f"{label}: not computed"

    explanation_summary = (
        f"{_score_phrase('Site Hazard', site_hazard_score)}. "
        f"{_score_phrase('Home Ignition Vulnerability', home_ignition_vulnerability_score)}. "
        f"{_score_phrase('Insurance Readiness', insurance_readiness_score)}. "
        f"Primary drivers: {', '.join(top_risk_drivers[:2])}. "
        f"Near-structure summary: {str(defensible_space_analysis.get('summary') or 'Not available')}."
    )

    risk_scores = RiskScores(
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        wildfire_risk_score=blended_wildfire_risk_score,
        insurance_readiness_score=insurance_readiness_score,
        site_hazard_score_available=bool(score_outputs["site_hazard_score_available"]),
        home_ignition_vulnerability_score_available=bool(score_outputs["home_ignition_vulnerability_score_available"]),
        wildfire_risk_score_available=bool(score_outputs["wildfire_risk_score_available"]),
        insurance_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
    )
    score_summaries = ScoreSummaries(
        site_hazard=site_hazard_section,
        home_ignition_vulnerability=home_ignition_section,
        insurance_readiness=insurance_readiness_section,
    )
    coordinates = Coordinates(latitude=lat, longitude=lon)

    submodel_explanations = {k: v.explanation for k, v in submodel_scores.items()}
    fact_map = _attributes_to_dict(payload.attributes)
    final_tags = sorted(set((payload.tags or []) + (tags or [])))

    def _clean_point_dict(raw: object) -> dict[str, float] | None:
        if not isinstance(raw, dict):
            return None
        try:
            lat_v = float(raw.get("latitude"))
            lon_v = float(raw.get("longitude"))
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat_v <= 90.0 and -180.0 <= lon_v <= 180.0):
            return None
        return {"latitude": lat_v, "longitude": lon_v}

    def _clean_feature_dict(raw: object) -> dict[str, Any] | None:
        return raw if isinstance(raw, dict) else None

    region_data_version = str(property_level_context.get("region_manifest_path") or "") or None
    governance = _build_result_governance(
        ruleset_version=ruleset.ruleset_version,
        region_data_version=region_data_version,
        benchmark_pack_version=BENCHMARK_PACK_VERSION,
    )

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
        geocoding=GeocodingDetails.model_validate(geocode_meta),
        wildfire_risk_score=blended_wildfire_risk_score,
        legacy_weighted_wildfire_risk_score=legacy_weighted_wildfire_risk_score,
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        insurance_readiness_score=insurance_readiness_score,
        wildfire_risk_score_available=bool(score_outputs["wildfire_risk_score_available"]),
        site_hazard_score_available=bool(score_outputs["site_hazard_score_available"]),
        home_ignition_vulnerability_score_available=bool(score_outputs["home_ignition_vulnerability_score_available"]),
        insurance_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
        risk_drivers=risk.drivers,
        factor_breakdown=breakdown,
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        submodel_explanations=submodel_explanations,
        property_findings=property_findings,
        defensible_space_analysis=defensible_space_analysis,
        top_near_structure_risk_drivers=top_near_structure_risk_drivers,
        prioritized_vegetation_actions=prioritized_vegetation_actions,
        defensible_space_limitations_summary=defensible_space_limitations_summary,
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
        environmental_data_completeness_score=confidence_block.environmental_data_completeness_score,
        confidence_tier=confidence_block.confidence_tier,
        use_restriction=confidence_block.use_restriction,
        low_confidence_flags=confidence_block.low_confidence_flags,
        data_sources=all_sources,
        environmental_layer_status=context.environmental_layer_status,
        input_source_metadata=input_source_metadata,
        direct_data_coverage_score=direct_data_coverage_score,
        inferred_data_coverage_score=inferred_data_coverage_score,
        missing_data_share=missing_data_share,
        data_provenance=data_provenance,
        site_hazard_input_quality=site_hazard_input_quality,
        home_vulnerability_input_quality=home_vulnerability_input_quality,
        insurance_readiness_input_quality=insurance_readiness_input_quality,
        score_evidence_ledger=score_evidence_ledger,
        evidence_quality_summary=evidence_quality_summary,
        layer_coverage_audit=layer_coverage_audit,
        coverage_summary=coverage_summary,
        region_resolution=region_resolution,
        coverage_available=bool(region_resolution.coverage_available),
        resolved_region_id=region_resolution.resolved_region_id,
        property_anchor_point=_clean_point_dict(property_level_context.get("property_anchor_point")),
        property_anchor_source=(
            str(property_level_context.get("property_anchor_source"))
            if property_level_context.get("property_anchor_source")
            else None
        ),
        property_anchor_precision=(
            str(property_level_context.get("property_anchor_precision"))
            if property_level_context.get("property_anchor_precision")
            else None
        ),
        assessed_property_display_point=_clean_point_dict(property_level_context.get("assessed_property_display_point")),
        parcel_id=(
            str(property_level_context.get("parcel_id"))
            if property_level_context.get("parcel_id")
            else None
        ),
        parcel_lookup_method=(
            str(property_level_context.get("parcel_lookup_method"))
            if property_level_context.get("parcel_lookup_method")
            else None
        ),
        parcel_lookup_distance_m=(
            float(property_level_context.get("parcel_lookup_distance_m"))
            if property_level_context.get("parcel_lookup_distance_m") is not None
            else None
        ),
        source_conflict_flag=bool(property_level_context.get("source_conflict_flag")),
        alignment_notes=[
            str(note) for note in (property_level_context.get("alignment_notes") or []) if str(note).strip()
        ],
        display_point_source=str(property_level_context.get("display_point_source") or "property_anchor_point"),
        structure_match_status=str(property_level_context.get("structure_match_status") or "none"),
        structure_match_method=(
            str(property_level_context.get("structure_match_method"))
            if property_level_context.get("structure_match_method")
            else None
        ),
        matched_structure_id=(
            str(property_level_context.get("matched_structure_id"))
            if property_level_context.get("matched_structure_id")
            else None
        ),
        structure_match_confidence=(
            float(property_level_context.get("structure_match_confidence"))
            if property_level_context.get("structure_match_confidence") is not None
            else None
        ),
        building_source=(
            str(property_level_context.get("building_source"))
            if property_level_context.get("building_source")
            else None
        ),
        building_source_version=(
            str(property_level_context.get("building_source_version"))
            if property_level_context.get("building_source_version")
            else None
        ),
        building_source_confidence=(
            float(property_level_context.get("building_source_confidence"))
            if property_level_context.get("building_source_confidence") is not None
            else None
        ),
        structure_match_distance_m=(
            float(property_level_context.get("structure_match_distance_m"))
            if property_level_context.get("structure_match_distance_m") is not None
            else None
        ),
        candidate_structure_count=(
            int(property_level_context.get("candidate_structure_count"))
            if property_level_context.get("candidate_structure_count") is not None
            else None
        ),
        final_structure_geometry_source=(
            str(property_level_context.get("final_structure_geometry_source"))
            if property_level_context.get("final_structure_geometry_source")
            else None
        ),
        structure_geometry_confidence=(
            float(property_level_context.get("structure_geometry_confidence"))
            if property_level_context.get("structure_geometry_confidence") is not None
            else None
        ),
        snapped_structure_distance_m=(
            float(property_level_context.get("snapped_structure_distance_m"))
            if property_level_context.get("snapped_structure_distance_m") is not None
            else None
        ),
        selection_mode=(
            str(property_level_context.get("selection_mode"))
            if property_level_context.get("selection_mode") in {"polygon", "point"}
            else "polygon"
        ),
        matched_structure_centroid=_clean_point_dict(property_level_context.get("matched_structure_centroid")),
        matched_structure_footprint=_clean_feature_dict(property_level_context.get("matched_structure_footprint")),
        user_selected_point=_clean_point_dict(property_level_context.get("user_selected_point")),
        site_hazard_eligibility=site_hazard_eligibility,
        home_vulnerability_eligibility=home_vulnerability_eligibility,
        insurance_readiness_eligibility=insurance_readiness_eligibility,
        assessment_status=assessment_status,
        assessment_blockers=assessment_blockers,
        assessment_limitations_summary=assessment_limitations_summary,
        assessment_diagnostics=assessment_diagnostics,
        property_level_context=property_level_context,
        mitigation_plan=mitigation_plan,
        readiness_factors=readiness_factors,
        readiness_blockers=readiness.readiness_blockers,
        readiness_penalties=readiness.readiness_penalties,
        readiness_summary=readiness.readiness_summary,
        score_summaries=score_summaries,
        site_hazard_section=site_hazard_section,
        home_ignition_vulnerability_section=home_ignition_section,
        insurance_readiness_section=insurance_readiness_section,
        model_version=str(governance["scoring_model_version"] or MODEL_VERSION),
        product_version=str(governance["product_version"] or PRODUCT_VERSION),
        api_version=str(governance["api_version"] or API_VERSION),
        scoring_model_version=str(governance["scoring_model_version"] or MODEL_VERSION),
        rules_logic_version=str(governance["rules_logic_version"] or RULESET_LOGIC_VERSION),
        factor_schema_version=str(governance["factor_schema_version"] or FACTOR_SCHEMA_VERSION),
        benchmark_pack_version=governance.get("benchmark_pack_version"),
        calibration_version=str(governance["calibration_version"] or CALIBRATION_VERSION),
        region_data_version=governance.get("region_data_version"),
        data_bundle_version=governance.get("data_bundle_version"),
        model_governance=ModelGovernanceInfo.model_validate(governance),
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

    _log_region_resolution_event(
        address=payload.address,
        latitude=lat,
        longitude=lon,
        region_resolution=region_resolution,
        manifest_path=str(property_level_context.get("region_manifest_path") or "") or None,
    )

    result = _apply_ruleset_to_result(result, ruleset)

    debug_payload = {
        "address": payload.address,
        "organization_id": organization_id,
        "ruleset_id": ruleset.ruleset_id,
        "coordinates": {"latitude": lat, "longitude": lon},
        "geocoding": geocode_meta,
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
        "environmental_layer_status": context.environmental_layer_status,
        "environmental_data_completeness": environmental_data_completeness,
        "property_level_context": property_level_context,
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
            "site_hazard_score_available": result.site_hazard_score_available,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "home_ignition_vulnerability_score_available": result.home_ignition_vulnerability_score_available,
            "wildfire_risk_score": result.wildfire_risk_score,
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "insurance_readiness_score": result.insurance_readiness_score,
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
        },
        "normalized_property_facts": _attributes_to_dict(scoring_attrs),
        "normalization_changes": normalization_changes,
        "confidence_gating": {
            "confidence_score": result.confidence_score,
            "data_completeness_score": result.data_completeness_score,
            "environmental_data_completeness_score": result.environmental_data_completeness_score,
            "confidence_tier": result.confidence_tier,
            "use_restriction": result.use_restriction,
        },
        "eligibility": {
            "site_hazard": result.site_hazard_eligibility.model_dump(),
            "home_ignition_vulnerability": result.home_vulnerability_eligibility.model_dump(),
            "insurance_readiness": result.insurance_readiness_eligibility.model_dump(),
            "assessment_status": result.assessment_status,
            "assessment_blockers": result.assessment_blockers,
            "assessment_limitations_summary": result.assessment_limitations_summary,
        },
        "assessment_limitations_summary": list(result.assessment_limitations_summary),
        "coverage": {
            "direct_data_coverage_score": result.direct_data_coverage_score,
            "inferred_data_coverage_score": result.inferred_data_coverage_score,
            "missing_data_share": result.missing_data_share,
            "stale_data_share": result.data_provenance.summary.stale_data_share,
            "heuristic_input_count": result.data_provenance.summary.heuristic_input_count,
        },
        "layer_coverage_audit": [row.model_dump() for row in result.layer_coverage_audit],
        "coverage_summary": result.coverage_summary.model_dump(),
        "defensible_space_analysis": result.defensible_space_analysis,
        "top_near_structure_risk_drivers": result.top_near_structure_risk_drivers,
        "prioritized_vegetation_actions": [a.model_dump() for a in result.prioritized_vegetation_actions],
        "defensible_space_limitations_summary": result.defensible_space_limitations_summary,
        "region_resolution": result.region_resolution.model_dump(),
        "score_evidence_ledger": result.score_evidence_ledger.model_dump(),
        "evidence_quality_summary": result.evidence_quality_summary.model_dump(),
        "score_family_input_quality": {
            "site_hazard": result.site_hazard_input_quality.model_dump(),
            "home_vulnerability": result.home_vulnerability_input_quality.model_dump(),
            "insurance_readiness": result.insurance_readiness_input_quality.model_dump(),
        },
        "assessment_diagnostics": result.assessment_diagnostics.model_dump(),
        "data_provenance": result.data_provenance.model_dump(),
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
        "governance": result.model_governance.model_dump(),
        "model_governance": result.model_governance.model_dump(),
    }

    return result, debug_payload


def _compute_assessment(
    payload: AddressRequest,
    *,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    portfolio_name: str | None = None,
    tags: list[str] | None = None,
    geocode_resolution: GeocodeResolution | None = None,
    coverage_resolution: RegionCoverageResolution | None = None,
) -> AssessmentResult:
    result, _ = _run_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        portfolio_name=portfolio_name,
        tags=tags,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
    )
    return result


def _payload_from_assessment(existing: AssessmentResult) -> AddressRequest:
    property_ctx = existing.property_level_context if isinstance(existing.property_level_context, dict) else {}
    selection_mode = str(property_ctx.get("selection_mode") or "polygon").strip().lower()
    if selection_mode not in {"polygon", "point"}:
        selection_mode = "polygon"
    selected_point_payload = None
    if isinstance(property_ctx.get("user_selected_point"), dict):
        raw_point = property_ctx.get("user_selected_point") or {}
        try:
            selected_point_payload = Coordinates(
                latitude=float(raw_point.get("latitude")),
                longitude=float(raw_point.get("longitude")),
            )
        except (TypeError, ValueError):
            selected_point_payload = None
    return AddressRequest(
        address=existing.address,
        attributes=PropertyAttributes.model_validate(existing.property_facts or {}),
        confirmed_fields=list(existing.confirmed_fields),
        structure_geometry_source=str(property_ctx.get("structure_geometry_source") or "auto_detected"),
        selection_mode=selection_mode,
        property_anchor_point=selected_point_payload,
        user_selected_point=selected_point_payload,
        selected_structure_id=(
            str(property_ctx.get("selected_structure_id"))
            if property_ctx.get("selected_structure_id")
            else None
        ),
        selected_structure_geometry=(
            property_ctx.get("selected_structure_geometry")
            if isinstance(property_ctx.get("selected_structure_geometry"), dict)
            else None
        ),
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
            "near_structure_summary": result.defensible_space_analysis.get("summary"),
            "prioritized_vegetation_actions": [a.model_dump() for a in result.prioritized_vegetation_actions[:3]],
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
    include_benchmark_hints: bool = False,
) -> ReportExport:
    mode = _resolve_audience(result, actor, audience, audience_mode)
    result = _refresh_result_governance(result)
    benchmark_hints = build_benchmark_hints_for_assessment(result) if include_benchmark_hints else None
    assumptions_confidence = {
        "confirmed_inputs": result.confirmed_inputs,
        "inferred_inputs": result.inferred_inputs,
        "missing_inputs": result.missing_inputs,
        "assumptions_used": result.assumptions_used,
        "confidence_score": result.confidence_score,
        "data_completeness_score": result.data_completeness_score,
        "environmental_data_completeness_score": result.environmental_data_completeness_score,
        "direct_data_coverage_score": result.direct_data_coverage_score,
        "inferred_data_coverage_score": result.inferred_data_coverage_score,
        "missing_data_share": result.missing_data_share,
        "stale_data_share": result.data_provenance.summary.stale_data_share,
        "heuristic_input_count": result.data_provenance.summary.heuristic_input_count,
        "current_input_count": result.data_provenance.summary.current_input_count,
        "wildfire_risk_score_available": result.wildfire_risk_score_available,
        "site_hazard_score_available": result.site_hazard_score_available,
        "home_ignition_vulnerability_score_available": result.home_ignition_vulnerability_score_available,
        "insurance_readiness_score_available": result.insurance_readiness_score_available,
        "confidence_tier": result.confidence_tier,
        "use_restriction": result.use_restriction,
        "low_confidence_flags": result.low_confidence_flags,
        "assessment_status": result.assessment_status,
        "assessment_blockers": result.assessment_blockers,
        "site_hazard_input_quality": result.site_hazard_input_quality.model_dump(),
        "home_vulnerability_input_quality": result.home_vulnerability_input_quality.model_dump(),
        "insurance_readiness_input_quality": result.insurance_readiness_input_quality.model_dump(),
        "evidence_quality_summary": result.evidence_quality_summary.model_dump(),
    }
    if benchmark_hints is not None:
        assumptions_confidence["benchmark_hints"] = benchmark_hints

    return ReportExport(
        assessment_id=result.assessment_id,
        generated_at=result.generated_at.isoformat(),
        model_version=result.model_version,
        organization_id=result.organization_id,
        audience_mode=mode,
        audience_highlights=_audience_highlights(result, mode),
        audience_focus=_audience_focus(result, mode),
        governance_metadata=result.model_governance.model_dump(),
        model_governance=result.model_governance,
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
            "environmental_layer_status": result.environmental_layer_status,
            "data_provenance": result.data_provenance.model_dump(),
            "property_level_context": result.property_level_context,
            "region_resolution": result.region_resolution.model_dump(),
            "assessment_diagnostics": result.assessment_diagnostics.model_dump(),
            "layer_coverage_audit": [row.model_dump() for row in result.layer_coverage_audit],
            "coverage_summary": result.coverage_summary.model_dump(),
            "benchmark_hints": benchmark_hints,
        },
        wildfire_risk_summary={
            "wildfire_risk_score": result.wildfire_risk_score,
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "site_hazard_score": result.site_hazard_score,
            "site_hazard_score_available": result.site_hazard_score_available,
            "home_ignition_vulnerability_score": result.home_ignition_vulnerability_score,
            "home_ignition_vulnerability_score_available": result.home_ignition_vulnerability_score_available,
            "site_hazard_eligibility": result.site_hazard_eligibility.model_dump(),
            "home_ignition_vulnerability_eligibility": result.home_vulnerability_eligibility.model_dump(),
            "assessment_status": result.assessment_status,
            "assessment_blockers": result.assessment_blockers,
            "score_summaries": result.score_summaries.model_dump(),
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
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "insurance_readiness_section": result.insurance_readiness_section.model_dump(),
            "insurance_readiness_eligibility": result.insurance_readiness_eligibility.model_dump(),
            "readiness_factors": [r.model_dump() for r in result.readiness_factors],
            "readiness_blockers": result.readiness_blockers,
            "readiness_penalties": result.readiness_penalties,
            "readiness_summary": result.readiness_summary,
        },
        defensible_space_analysis=result.defensible_space_analysis,
        top_near_structure_risk_drivers=result.top_near_structure_risk_drivers,
        prioritized_vegetation_actions=result.prioritized_vegetation_actions,
        defensible_space_limitations_summary=result.defensible_space_limitations_summary,
        top_risk_drivers=result.top_risk_drivers,
        top_protective_factors=result.top_protective_factors,
        assumptions_confidence=assumptions_confidence,
        score_evidence_ledger=result.score_evidence_ledger,
        evidence_quality_summary=result.evidence_quality_summary,
        layer_coverage_audit=result.layer_coverage_audit,
        coverage_summary=result.coverage_summary,
        mitigation_recommendations=result.mitigation_plan,
    )


def _build_report_html(
    result: AssessmentResult,
    *,
    actor: ActorContext,
    audience: Audience | None = None,
    audience_mode: Audience | None = None,
) -> str:
    def _score_html(value: float | None) -> str:
        return f"{value:.1f}" if value is not None else "Not computed"

    mode = _resolve_audience(result, actor, audience, audience_mode)
    highlights = _audience_highlights(result, mode)
    focus = _audience_focus(result, mode)
    blockers = "<li>None</li>" if not result.readiness_blockers else "".join(f"<li>{b}</li>" for b in result.readiness_blockers)
    mitigations = "".join(
        f"<li><strong>{m.title}</strong>: {m.reason}"
        f" <em>(risk: {m.estimated_risk_reduction_band}, readiness: {m.estimated_readiness_improvement_band})</em></li>"
        for m in result.mitigation_plan
    )
    near_structure_drivers = "".join(f"<li>{d}</li>" for d in result.top_near_structure_risk_drivers) or "<li>None</li>"
    vegetation_actions = "".join(
        f"<li><strong>{a.title}</strong> ({a.target_zone}): {a.explanation}</li>"
        for a in result.prioritized_vegetation_actions
    ) or "<li>None</li>"
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
<div class=\"card\"><h3>Wildfire Risk Score</h3><p>{_score_html(result.wildfire_risk_score)}</p></div>
<div class=\"card\"><h3>Insurance Readiness Score</h3><p>{_score_html(result.insurance_readiness_score)}</p></div>
</div>
<div class=\"card\"><h3>Top Risk Drivers</h3><ul>{drivers}</ul></div>
<div class=\"card\"><h3>Near-Structure Drivers</h3><ul>{near_structure_drivers}</ul></div>
<div class=\"card\"><h3>Top Protective Factors</h3><ul>{protective}</ul></div>
<div class=\"card\"><h3>Readiness Blockers</h3><ul>{blockers}</ul></div>
<div class=\"card\"><h3>Mitigation Recommendations</h3><ul>{mitigations}</ul></div>
<div class=\"card\"><h3>Prioritized Vegetation Actions</h3><ul>{vegetation_actions}</ul></div>
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
    def _delta(before: float | None, after: float | None) -> float | None:
        if before is None or after is None:
            return None
        return round(after - before, 1)

    base_drivers = set(base.top_risk_drivers)
    other_drivers = set(other.top_risk_drivers)
    base_blockers = set(base.readiness_blockers)
    other_blockers = set(other.readiness_blockers)
    base_mitigations = {m.title for m in base.mitigation_plan}
    other_mitigations = {m.title for m in other.mitigation_plan}
    version_comparison = compare_model_governance(
        base.model_governance.model_dump() if base.model_governance else {},
        other.model_governance.model_dump() if other.model_governance else {},
    )

    return AssessmentComparisonResult(
        base=_to_comparison_item(base),
        other=_to_comparison_item(other),
        wildfire_risk_delta=_delta(base.wildfire_risk_score, other.wildfire_risk_score),
        insurance_readiness_delta=_delta(base.insurance_readiness_score, other.insurance_readiness_score),
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
        version_comparison=version_comparison,
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
                "wildfire_risk_score": item.wildfire_risk_score if item.wildfire_risk_score is not None else "",
                "insurance_readiness_score": item.insurance_readiness_score if item.insurance_readiness_score is not None else "",
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_dev_mode() -> bool:
    env = str(os.getenv("WF_ENV") or os.getenv("APP_ENV") or "").strip().lower()
    return env in {"dev", "development", "local", "test"} or _env_flag("WF_DEBUG_MODE", False)


def _normalize_geocode_status(status: str | None) -> str:
    value = str(status or "").strip().lower()
    if value == "matched":
        return "accepted"
    if value in {
        "accepted",
        "geocode_succeeded_untrusted",
        "geocode_succeeded_trusted",
        "no_match",
        "ambiguous_match",
        "low_confidence",
        "trust_filter_rejected",
        "missing_coordinates",
        "provider_error",
        "parser_error",
    }:
        return value
    return "parser_error"


def _geocode_error_message(status: str, purpose: str) -> str:
    context = "assessment" if purpose == "assessment" else "region coverage check"
    mapping = {
        "no_match": "No trusted location match was found. Please verify the address and try again.",
        "ambiguous_match": "Address matched multiple possible locations. Add city/state or ZIP and try again.",
        "low_confidence": "Address match confidence was below policy threshold. Add more address detail and retry.",
        "trust_filter_rejected": "Address was rejected by trust filters. Add city/state or ZIP and try again.",
        "missing_coordinates": "Address matched but coordinates were missing from provider response.",
        "provider_error": "Geocoding provider is temporarily unavailable. Please retry shortly.",
        "parser_error": "Address format could not be parsed. Please correct the address and retry.",
    }
    base = mapping.get(status, "Geocoding lookup failed. Please verify the address and try again.")
    return f"Geocoding failed for {context}. {base}"


def _normalize_address_for_component_compare(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    replacements = {
        r"\broad\b": "rd",
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bcourt\b": "ct",
        r"\bplace\b": "pl",
        r"\bnorth\b": "n",
        r"\bsouth\b": "s",
        r"\beast\b": "e",
        r"\bwest\b": "w",
        r"\bapartment\b": "apt",
        r"\bunit\b": "apt",
    }
    for pattern, repl in replacements.items():
        normalized = re.sub(pattern, repl, normalized)
    return " ".join(normalized.split())


def _build_address_component_comparison(submitted_address: str, candidate_address: str | None) -> dict[str, Any]:
    submitted_norm = _normalize_address_for_component_compare(submitted_address)
    candidate_norm = _normalize_address_for_component_compare(candidate_address or "")
    submitted_tokens = [tok for tok in submitted_norm.split() if tok]
    candidate_tokens = [tok for tok in candidate_norm.split() if tok]
    submitted_set = set(submitted_tokens)
    candidate_set = set(candidate_tokens)
    overlap = sorted(submitted_set & candidate_set)
    union_count = len(submitted_set | candidate_set)
    similarity = (len(overlap) / union_count) if union_count else 0.0
    return {
        "submitted_normalized": submitted_norm,
        "candidate_normalized": candidate_norm,
        "exact_normalized_match": submitted_norm == candidate_norm if submitted_norm and candidate_norm else False,
        "token_similarity_ratio": round(similarity, 3),
        "matched_tokens": overlap[:12],
        "missing_from_candidate": sorted(submitted_set - candidate_set)[:8],
        "extra_in_candidate": sorted(candidate_set - submitted_set)[:8],
    }


def _extract_candidate_from_preview(raw_preview: Any) -> dict[str, Any] | None:
    if not isinstance(raw_preview, dict):
        return None
    top = raw_preview.get("top_candidate")
    if isinstance(top, dict):
        return dict(top)
    parsed = raw_preview.get("parsed_candidates")
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return dict(parsed[0])
    return None


def _extract_lat_lon_from_candidate(candidate: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(candidate, dict):
        return None
    for lat_key in ("lat", "latitude"):
        for lon_key in ("lon", "lng", "longitude"):
            try:
                lat = float(candidate.get(lat_key))
                lon = float(candidate.get(lon_key))
            except (TypeError, ValueError):
                continue
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return float(lat), float(lon)
    return None


def _allow_untrusted_geocode_fallback(status: str) -> bool:
    allow_low = _env_flag("WF_GEOCODE_ALLOW_LOW_CONFIDENCE_FALLBACK", True)
    allow_ambiguous = _env_flag("WF_GEOCODE_ALLOW_AMBIGUOUS_FALLBACK", False)
    if status in {"low_confidence", "trust_filter_rejected"}:
        return allow_low
    if status == "ambiguous_match":
        return allow_ambiguous
    return False


def _geocode_http_status_for_status(status: str) -> int:
    if status == "provider_error":
        return 503
    return 422


def _log_geocode_event(
    *,
    purpose: str,
    status: str,
    submitted_address: str,
    normalized_address: str,
    latitude: float | None = None,
    longitude: float | None = None,
    provider: str | None = None,
    source: str | None = None,
    reason: str | None = None,
    candidate_count: int | None = None,
    trust_filter_rule: str | None = None,
    raw_response_preview: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "assessment_geocoding",
        "purpose": purpose,
        "geocode_status": status,
        "submitted_address": submitted_address,
        "normalized_address": normalized_address,
        "provider": provider,
        "source": source,
        "reason": reason,
    }
    if candidate_count is not None:
        payload["candidate_count"] = int(candidate_count)
    if trust_filter_rule:
        payload["trust_filter_rule"] = trust_filter_rule
    if latitude is not None and longitude is not None:
        payload["latitude"] = round(float(latitude), 6)
        payload["longitude"] = round(float(longitude), 6)
    if (_env_flag("WF_GEOCODE_DEBUG_LOG", False) or _is_dev_mode()) and raw_response_preview:
        payload["raw_response_preview"] = raw_response_preview

    level = logging.INFO if status == "accepted" else logging.WARNING
    LOGGER.log(level, "assessment_geocoding %s", json.dumps(payload, sort_keys=True))


@dataclass(frozen=True)
class GeocodeResolution:
    raw_input: str
    normalized_address: str
    geocode_status: str
    candidate_count: int
    selected_candidate: dict[str, Any] | None
    confidence_score: float | None
    latitude: float
    longitude: float
    geocode_source: str
    geocode_meta: dict[str, Any]
    geocode_outcome: str = "geocode_succeeded_trusted"
    trusted_match_status: str = "trusted"
    rejection_reason: str | None = None


@dataclass(frozen=True)
class RegionCoverageResolution:
    coverage_available: bool
    resolved_region_id: str | None
    reason: str
    diagnostics: list[str]
    coverage: dict[str, Any]


def _geocode_address_or_raise(*, address: str, purpose: str) -> tuple[float, float, str, dict[str, Any]]:
    submitted_address = str(address or "")
    normalized_address = normalize_address(submitted_address)
    provider = "OpenStreetMap Nominatim"

    try:
        lat, lon, geocode_source = geocoder.geocode(submitted_address)
    except GeocodingError as exc:
        status = _normalize_geocode_status(str(exc.status or "provider_error"))
        raw_preview = exc.raw_response_preview if isinstance(exc.raw_response_preview, dict) else {}
        fallback_candidate = _extract_candidate_from_preview(raw_preview)
        fallback_point = _extract_lat_lon_from_candidate(fallback_candidate)
        fallback_allowed = _allow_untrusted_geocode_fallback(status)
        if fallback_allowed and fallback_point is not None:
            fallback_lat, fallback_lon = fallback_point
            candidate_display = (
                fallback_candidate.get("display_name")
                if isinstance(fallback_candidate, dict)
                else None
            )
            address_compare = _build_address_component_comparison(
                submitted_address,
                str(candidate_display or ""),
            )
            geocode_meta = {
                "geocode_status": "accepted",
                "geocode_outcome": "geocode_succeeded_untrusted",
                "trusted_match_status": "untrusted_fallback",
                "submitted_address": submitted_address,
                "normalized_address": exc.normalized_address or normalized_address,
                "geocode_source": exc.provider or provider,
                "geocode_provider": exc.provider or provider,
                "provider": exc.provider or provider,
                "geocoded_address": candidate_display,
                "matched_address": candidate_display,
                "geocoded_point": {"latitude": float(fallback_lat), "longitude": float(fallback_lon)},
                "confidence_score": None,
                "candidate_count": int(raw_preview.get("candidate_count") or 1),
                "geocode_location_type": (
                    f"{str(fallback_candidate.get('class') or '').strip().lower()}:"
                    f"{str(fallback_candidate.get('type') or '').strip().lower()}"
                    if isinstance(fallback_candidate, dict)
                    and (
                        str(fallback_candidate.get("class") or "").strip()
                        or str(fallback_candidate.get("type") or "").strip()
                    )
                    else None
                ),
                "geocode_precision": "interpolated",
                "parsed_candidates": raw_preview.get("parsed_candidates"),
                "trust_filter_rule": raw_preview.get("trust_filter_rule"),
                "resolved_latitude": float(fallback_lat),
                "resolved_longitude": float(fallback_lon),
                "rejection_reason": exc.rejection_reason,
                "trusted_match_failure_reason": exc.rejection_reason,
                "fallback_eligibility": True,
                "address_component_comparison": address_compare,
            }
            if isinstance(fallback_candidate, dict) and fallback_candidate.get("importance") is not None:
                try:
                    geocode_meta["confidence_score"] = float(fallback_candidate.get("importance"))
                except (TypeError, ValueError):
                    geocode_meta["confidence_score"] = None
            if _env_flag("WF_GEOCODE_DEBUG_LOG", False) and raw_preview:
                geocode_meta["raw_response_preview"] = raw_preview
            _log_geocode_event(
                purpose=purpose,
                status="accepted",
                submitted_address=submitted_address,
                normalized_address=str(geocode_meta.get("normalized_address") or normalized_address),
                latitude=float(fallback_lat),
                longitude=float(fallback_lon),
                provider=str(geocode_meta.get("provider") or provider),
                source=str(geocode_meta.get("geocode_source") or provider),
                reason=f"untrusted_fallback:{status}:{exc.rejection_reason}",
                candidate_count=int(geocode_meta.get("candidate_count") or 1),
                trust_filter_rule=(
                    str(geocode_meta.get("trust_filter_rule"))
                    if geocode_meta.get("trust_filter_rule")
                    else None
                ),
                raw_response_preview=raw_preview,
            )
            return float(fallback_lat), float(fallback_lon), str(geocode_meta.get("geocode_source") or provider), geocode_meta

        detail: dict[str, Any] = {
            "error": "geocoding_failed",
            "geocode_status": status,
            "geocode_outcome": "geocode_failed",
            "trusted_match_status": "rejected",
            "rejection_category": (
                "trust_filter_rejected"
                if status in {"low_confidence", "ambiguous_match", "trust_filter_rejected"}
                else status
            ),
            "message": _geocode_error_message(status, purpose),
            "submitted_address": submitted_address,
            "normalized_address": exc.normalized_address or normalized_address,
            "provider": exc.provider or provider,
            "rejection_reason": exc.rejection_reason,
            "fallback_eligibility": bool(fallback_allowed and fallback_point is not None),
            "trusted_match_failure_reason": exc.rejection_reason,
            "candidate_count": int(raw_preview.get("candidate_count") or 0),
            "top_candidate_formatted_address": (
                fallback_candidate.get("display_name") if isinstance(fallback_candidate, dict) else None
            ),
            "top_candidate_lat": (
                fallback_candidate.get("lat") or fallback_candidate.get("latitude")
                if isinstance(fallback_candidate, dict)
                else None
            ),
            "top_candidate_lng": (
                fallback_candidate.get("lon")
                or fallback_candidate.get("lng")
                or fallback_candidate.get("longitude")
                if isinstance(fallback_candidate, dict)
                else None
            ),
            "geocode_precision": "unknown",
            "address_component_comparison": _build_address_component_comparison(
                submitted_address,
                str((fallback_candidate or {}).get("display_name") or ""),
            )
            if fallback_candidate
            else None,
        }
        if _env_flag("WF_GEOCODE_DEBUG_LOG", False) and raw_preview:
            detail["raw_response_preview"] = raw_preview
        _log_geocode_event(
            purpose=purpose,
            status=status,
            submitted_address=submitted_address,
            normalized_address=detail["normalized_address"],
            provider=detail["provider"],
            reason=exc.rejection_reason,
            candidate_count=(
                int(raw_preview.get("candidate_count"))
                if raw_preview.get("candidate_count") is not None
                else None
            ),
            trust_filter_rule=(
                str(raw_preview.get("trust_filter_rule"))
                if raw_preview.get("trust_filter_rule")
                else None
            ),
            raw_response_preview=raw_preview,
        )
        raise HTTPException(status_code=_geocode_http_status_for_status(status), detail=detail) from exc
    except Exception as exc:
        status = "provider_error"
        detail = {
            "error": "geocoding_failed",
            "geocode_status": status,
            "geocode_outcome": "geocode_failed",
            "trusted_match_status": "rejected",
            "rejection_category": status,
            "message": _geocode_error_message(status, purpose),
            "submitted_address": submitted_address,
            "normalized_address": normalized_address,
            "provider": provider,
            "rejection_reason": str(exc),
            "fallback_eligibility": False,
            "trusted_match_failure_reason": str(exc),
        }
        _log_geocode_event(
            purpose=purpose,
            status=status,
            submitted_address=submitted_address,
            normalized_address=normalized_address,
            provider=provider,
            reason=str(exc),
        )
        raise HTTPException(status_code=503, detail=detail) from exc

    geocoder_meta = getattr(geocoder, "last_result", None)
    geocode_meta: dict[str, Any] = {
        "geocode_status": "accepted",
        "geocode_outcome": "geocode_succeeded_trusted",
        "trusted_match_status": "trusted",
        "submitted_address": submitted_address,
        "normalized_address": normalized_address,
        "geocode_source": geocode_source,
        "geocode_provider": provider,
        "provider": provider,
        "geocoded_address": None,
        "matched_address": None,
        "geocoded_point": {"latitude": float(lat), "longitude": float(lon)},
        "confidence_score": None,
        "candidate_count": None,
        "geocode_location_type": None,
        "geocode_precision": "unknown",
        "parsed_candidates": None,
        "trust_filter_rule": None,
        "resolved_latitude": float(lat),
        "resolved_longitude": float(lon),
        "rejection_reason": None,
        "trusted_match_failure_reason": None,
        "fallback_eligibility": False,
        "address_component_comparison": None,
    }
    if isinstance(geocoder_meta, dict):
        matched_address = geocoder_meta.get("matched_address")
        geocode_meta.update(
            {
                "geocode_status": _normalize_geocode_status(str(geocoder_meta.get("geocode_status") or "accepted")),
                "geocode_outcome": "geocode_succeeded_trusted",
                "trusted_match_status": "trusted",
                "normalized_address": geocoder_meta.get("normalized_address") or normalized_address,
                "provider": geocoder_meta.get("provider") or provider,
                "geocode_provider": geocoder_meta.get("provider") or provider,
                "geocoded_address": matched_address,
                "matched_address": matched_address,
                "confidence_score": geocoder_meta.get("confidence_score"),
                "candidate_count": geocoder_meta.get("candidate_count"),
                "geocode_location_type": geocoder_meta.get("geocode_location_type"),
                "geocode_precision": geocoder_meta.get("geocode_precision") or "unknown",
                "rejection_reason": geocoder_meta.get("rejection_reason"),
                "trust_filter_rule": geocoder_meta.get("raw_response_preview", {}).get("trust_filter_rule")
                if isinstance(geocoder_meta.get("raw_response_preview"), dict)
                else None,
                "parsed_candidates": geocoder_meta.get("raw_response_preview", {}).get("parsed_candidates")
                if isinstance(geocoder_meta.get("raw_response_preview"), dict)
                else None,
                "address_component_comparison": _build_address_component_comparison(
                    submitted_address,
                    str(matched_address or ""),
                ),
            }
        )
        if _env_flag("WF_GEOCODE_DEBUG_LOG", False) and geocoder_meta.get("raw_response_preview"):
            geocode_meta["raw_response_preview"] = geocoder_meta.get("raw_response_preview")

    _log_geocode_event(
        purpose=purpose,
        status="accepted",
        submitted_address=submitted_address,
        normalized_address=str(geocode_meta.get("normalized_address") or normalized_address),
        latitude=float(lat),
        longitude=float(lon),
        provider=str(geocode_meta.get("provider") or provider),
        source=geocode_source,
        candidate_count=(
            int(geocode_meta.get("candidate_count"))
            if geocode_meta.get("candidate_count") is not None
            else None
        ),
        trust_filter_rule=(
            str(geocode_meta.get("trust_filter_rule"))
            if geocode_meta.get("trust_filter_rule")
            else None
        ),
        raw_response_preview=geocode_meta.get("raw_response_preview"),
    )
    return float(lat), float(lon), geocode_source, geocode_meta


def _resolve_trusted_geocode(
    *,
    address_input: str,
    purpose: str,
    route_name: str,
) -> GeocodeResolution:
    try:
        lat, lon, source, geocode_meta = _geocode_address_or_raise(address=address_input, purpose=purpose)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        if _is_dev_mode():
            LOGGER.warning(
                "route_geocode_resolution %s",
                json.dumps(
                    {
                        "event": "route_geocode_resolution",
                        "route_name": route_name,
                        "status_code": exc.status_code,
                        "submitted_address": str(detail.get("submitted_address") or address_input or ""),
                        "normalized_address": str(detail.get("normalized_address") or normalize_address(address_input or "")),
                        "geocode_status": detail.get("geocode_status") or "provider_error",
                        "geocode_outcome": detail.get("geocode_outcome") or "geocode_failed",
                        "trusted_match_status": detail.get("trusted_match_status") or "rejected",
                        "rejection_reason": detail.get("rejection_reason"),
                        "trusted_match_failure_reason": detail.get("trusted_match_failure_reason"),
                        "fallback_eligibility": detail.get("fallback_eligibility"),
                    },
                    sort_keys=True,
                ),
            )
        raise

    selected_candidate = {
        "display_name": geocode_meta.get("matched_address"),
        "confidence_score": geocode_meta.get("confidence_score"),
        "provider": geocode_meta.get("provider"),
    }
    if not selected_candidate["display_name"] and not selected_candidate["confidence_score"]:
        selected_candidate = None

    resolution = GeocodeResolution(
        raw_input=str(geocode_meta.get("submitted_address") or address_input or ""),
        normalized_address=str(geocode_meta.get("normalized_address") or normalize_address(address_input or "")),
        geocode_status=str(geocode_meta.get("geocode_status") or "accepted"),
        candidate_count=int(geocode_meta.get("candidate_count") or 1),
        selected_candidate=selected_candidate,
        confidence_score=(
            float(geocode_meta["confidence_score"])
            if geocode_meta.get("confidence_score") is not None
            else None
        ),
        latitude=float(lat),
        longitude=float(lon),
        geocode_source=str(source),
        geocode_meta=dict(geocode_meta or {}),
        geocode_outcome=str(geocode_meta.get("geocode_outcome") or "geocode_succeeded_trusted"),
        trusted_match_status=str(geocode_meta.get("trusted_match_status") or "trusted"),
        rejection_reason=geocode_meta.get("rejection_reason"),
    )
    if _is_dev_mode():
        LOGGER.info(
            "route_geocode_resolution %s",
            json.dumps(
                {
                    "event": "route_geocode_resolution",
                    "route_name": route_name,
                    "submitted_address": resolution.raw_input,
                    "normalized_address": resolution.normalized_address,
                    "geocode_status": resolution.geocode_status,
                    "geocode_outcome": resolution.geocode_outcome,
                    "trusted_match_status": resolution.trusted_match_status,
                    "candidate_count": resolution.candidate_count,
                    "confidence_score": resolution.confidence_score,
                    "latitude": round(resolution.latitude, 6),
                    "longitude": round(resolution.longitude, 6),
                    "trust_filter_rule": geocode_meta.get("trust_filter_rule"),
                    "trusted_match_failure_reason": geocode_meta.get("trusted_match_failure_reason"),
                },
                sort_keys=True,
            ),
        )
    return resolution


def _resolve_prepared_region(
    *,
    latitude: float,
    longitude: float,
    route_name: str,
    address_input: str = "",
) -> RegionCoverageResolution:
    coverage = _region_coverage_for_coordinates(lat=float(latitude), lon=float(longitude))
    resolution = RegionCoverageResolution(
        coverage_available=bool(coverage.get("coverage_available", False)),
        resolved_region_id=coverage.get("resolved_region_id"),
        reason=str(coverage.get("reason") or "unknown"),
        diagnostics=list(coverage.get("diagnostics") or []),
        coverage=dict(coverage),
    )
    if _is_dev_mode():
        LOGGER.info(
            "route_region_resolution %s",
            json.dumps(
                {
                    "event": "route_region_resolution",
                    "route_name": route_name,
                    "submitted_address": address_input,
                    "latitude": round(float(latitude), 6),
                    "longitude": round(float(longitude), 6),
                    "coverage_available": resolution.coverage_available,
                    "resolved_region_id": resolution.resolved_region_id,
                    "reason": resolution.reason,
                    "diagnostics": resolution.diagnostics[:4],
                    "region_distance_to_boundary_m": coverage.get("region_distance_to_boundary_m"),
                    "nearest_region_id": coverage.get("nearest_region_id"),
                },
                sort_keys=True,
            ),
        )
    return resolution


def _build_geocode_debug_payload(address: str) -> dict[str, Any]:
    submitted_address = str(address or "")
    normalized = normalize_address(submitted_address)
    try:
        geocode_resolution = _resolve_trusted_geocode(
            address_input=submitted_address,
            purpose="assessment",
            route_name="/risk/geocode-debug",
        )
        meta = dict(geocode_resolution.geocode_meta or {})
        geocoder_last = dict(getattr(geocoder, "last_result", {}) or {})
        raw_preview = meta.get("raw_response_preview")
        if raw_preview is None and isinstance(geocoder_last.get("raw_response_preview"), dict):
            raw_preview = geocoder_last.get("raw_response_preview")
        lat = float(geocode_resolution.latitude)
        lon = float(geocode_resolution.longitude)
        coverage = _region_coverage_for_coordinates(lat, lon)
        return {
            "raw_input_address": submitted_address,
            "normalized_address": meta.get("normalized_address") or normalized,
            "geocode_status": meta.get("geocode_status") or "accepted",
            "geocode_outcome": meta.get("geocode_outcome") or geocode_resolution.geocode_outcome,
            "accepted": True,
            "geocode_provider": meta.get("provider") or geocode_resolution.geocode_source,
            "geocode_location_type": meta.get("geocode_location_type"),
            "geocode_precision": meta.get("geocode_precision") or "unknown",
            "match_count": int(meta.get("candidate_count") or geocode_resolution.candidate_count or 1),
            "parsed_candidates": meta.get("parsed_candidates")
            or (raw_preview or {}).get("parsed_candidates"),
            "selected_match": {
                "display_name": (geocode_resolution.selected_candidate or {}).get("display_name")
                or meta.get("matched_address"),
                "latitude": float(lat),
                "longitude": float(lon),
                "confidence_score": meta.get("confidence_score"),
                "provider": meta.get("provider") or geocode_resolution.geocode_source,
            },
            "trust": {
                "confidence_score": meta.get("confidence_score"),
                "min_importance_threshold": getattr(geocoder, "min_importance", None),
                "ambiguity_delta": getattr(geocoder, "ambiguity_delta", None),
                "trusted_match_status": meta.get("trusted_match_status") or geocode_resolution.trusted_match_status,
                "trusted_match_failure_reason": meta.get("trusted_match_failure_reason"),
                "fallback_eligibility": bool(meta.get("fallback_eligibility")),
                "trust_filter_rule": meta.get("trust_filter_rule"),
                "trust_filter_rejected": bool(meta.get("trusted_match_status") == "untrusted_fallback"),
                "address_component_comparison": meta.get("address_component_comparison"),
            },
            "resolved_latitude": float(lat),
            "resolved_longitude": float(lon),
            "geocode_source": geocode_resolution.geocode_source,
            "rejection_reason": None,
            "rejection_category": None,
            "raw_response_preview": raw_preview,
            "region_resolution": {
                "coverage_available": bool(coverage.get("coverage_available", False)),
                "resolved_region_id": coverage.get("resolved_region_id"),
                "reason": coverage.get("reason"),
                "diagnostics": list(coverage.get("diagnostics") or []),
                "region_distance_to_boundary_m": coverage.get("region_distance_to_boundary_m"),
                "nearest_region_id": coverage.get("nearest_region_id"),
            },
        }
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return {
            "raw_input_address": submitted_address,
            "normalized_address": detail.get("normalized_address") or normalized,
            "geocode_status": detail.get("geocode_status") or "provider_error",
            "geocode_outcome": detail.get("geocode_outcome") or "geocode_failed",
            "accepted": False,
            "geocode_provider": detail.get("provider") or "OpenStreetMap Nominatim",
            "geocode_location_type": None,
            "geocode_precision": "unknown",
            "match_count": int(((detail.get("raw_response_preview") or {}).get("candidate_count")) or 0),
            "parsed_candidates": (detail.get("raw_response_preview") or {}).get("parsed_candidates"),
            "selected_match": None,
            "trust": {
                "confidence_score": None,
                "min_importance_threshold": getattr(geocoder, "min_importance", None),
                "ambiguity_delta": getattr(geocoder, "ambiguity_delta", None),
                "trusted_match_status": detail.get("trusted_match_status") or "rejected",
                "trusted_match_failure_reason": detail.get("trusted_match_failure_reason") or detail.get("rejection_reason"),
                "fallback_eligibility": bool(detail.get("fallback_eligibility")),
                "trust_filter_rule": (detail.get("raw_response_preview") or {}).get("trust_filter_rule"),
                "trust_filter_rejected": detail.get("rejection_category") == "trust_filter_rejected",
                "address_component_comparison": detail.get("address_component_comparison"),
            },
            "resolved_latitude": None,
            "resolved_longitude": None,
            "geocode_source": detail.get("provider") or "OpenStreetMap Nominatim",
            "rejection_reason": detail.get("rejection_reason"),
            "rejection_category": detail.get("rejection_category") or detail.get("geocode_status"),
            "raw_response_preview": detail.get("raw_response_preview"),
            "region_resolution": None,
        }
    except Exception as exc:
        return {
            "raw_input_address": submitted_address,
            "normalized_address": normalized,
            "geocode_status": "provider_error",
            "geocode_outcome": "geocode_failed",
            "accepted": False,
            "geocode_provider": "OpenStreetMap Nominatim",
            "geocode_location_type": None,
            "geocode_precision": "unknown",
            "match_count": 0,
            "parsed_candidates": None,
            "selected_match": None,
            "trust": {
                "confidence_score": None,
                "min_importance_threshold": getattr(geocoder, "min_importance", None),
                "ambiguity_delta": getattr(geocoder, "ambiguity_delta", None),
                "trusted_match_status": "rejected",
                "trusted_match_failure_reason": str(exc),
                "fallback_eligibility": False,
                "trust_filter_rule": None,
                "trust_filter_rejected": False,
                "address_component_comparison": None,
            },
            "resolved_latitude": None,
            "resolved_longitude": None,
            "geocode_source": "OpenStreetMap Nominatim",
            "rejection_reason": str(exc),
            "rejection_category": "provider_error",
            "raw_response_preview": None,
            "region_resolution": None,
        }


@app.post("/risk/geocode-debug", dependencies=[Depends(require_api_key)])
def geocode_debug(payload: GeocodeDebugRequest, _: ActorContext = Depends(get_actor_context)) -> dict[str, Any]:
    if _is_dev_mode():
        LOGGER.info(
            "geocode_debug_request %s",
            json.dumps(
                {
                    "event": "geocode_debug_request",
                    "submitted_address": payload.address,
                    "normalized_address": normalize_address(payload.address),
                },
                sort_keys=True,
            ),
        )
    return _build_geocode_debug_payload(payload.address)


@app.post("/debug/geocode", dependencies=[Depends(require_api_key)])
def geocode_debug_alias(payload: GeocodeDebugRequest, _: ActorContext = Depends(get_actor_context)) -> dict[str, Any]:
    # Alias for direct development debugging without risk-route naming.
    return _build_geocode_debug_payload(payload.address)


def _region_data_root() -> str:
    return os.getenv("WF_REGION_DATA_DIR", str(wildfire_data.region_data_dir))


def _region_coverage_for_coordinates(lat: float, lon: float) -> dict[str, Any]:
    lookup = lookup_region_for_point(lat=lat, lon=lon, regions_root=_region_data_root())
    if lookup.get("covered"):
        resolved_region_id = lookup.get("region_id")
        return {
            "covered": True,
            "region_id": lookup.get("region_id"),
            "display_name": lookup.get("display_name"),
            "latitude": lat,
            "longitude": lon,
            "message": "Prepared region coverage is available for this location.",
            "diagnostics": [],
            "regions_root": _region_data_root(),
            "coverage_available": True,
            "resolved_region_id": resolved_region_id,
            "reason": "prepared_region_found",
            "recommended_action": None,
            "region_check_result": "inside_prepared_region",
            "region_distance_to_boundary_m": 0.0,
            "nearest_region_id": resolved_region_id,
            "edge_tolerance_m": lookup.get("edge_tolerance_m"),
        }
    diagnostics = list(lookup.get("diagnostics") or [])
    nearest_region_id = lookup.get("nearest_region_id")
    region_distance_to_boundary_m = lookup.get("region_distance_to_boundary_m")
    return {
        "covered": False,
        "region_id": None,
        "display_name": None,
        "latitude": lat,
        "longitude": lon,
        "message": (
            "No prepared region covers this location. Run scripts/prepare_region_from_catalog_or_sources.py "
            "for the requested bbox, validate it, then retry assessment."
        ),
        "diagnostics": diagnostics,
        "regions_root": _region_data_root(),
        "coverage_available": False,
        "resolved_region_id": None,
        "reason": "no_prepared_region_for_location",
        "recommended_action": (
            "Prepare and validate a region for this location using "
            "scripts/prepare_region_from_catalog_or_sources.py, then retry assessment."
        ),
        "region_check_result": "outside_prepared_regions",
        "region_distance_to_boundary_m": (
            float(region_distance_to_boundary_m)
            if region_distance_to_boundary_m is not None
            else None
        ),
        "nearest_region_id": str(nearest_region_id) if nearest_region_id else None,
        "edge_tolerance_m": lookup.get("edge_tolerance_m"),
    }


def _log_region_resolution_event(
    *,
    address: str,
    latitude: float,
    longitude: float,
    region_resolution: RegionResolution | dict[str, Any],
    manifest_path: str | None = None,
) -> None:
    resolution = (
        region_resolution.model_dump()
        if isinstance(region_resolution, RegionResolution)
        else dict(region_resolution or {})
    )
    payload = {
        "event": "assessment_region_resolution",
        "address": address,
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "coverage_available": bool(resolution.get("coverage_available", False)),
        "resolved_region_id": resolution.get("resolved_region_id"),
        "reason": resolution.get("reason"),
        "manifest_path": manifest_path,
        "diagnostics": list(resolution.get("diagnostics") or []),
        "region_distance_to_boundary_m": resolution.get("region_distance_to_boundary_m"),
        "nearest_region_id": resolution.get("nearest_region_id"),
    }
    LOGGER.info("assessment_region_resolution %s", json.dumps(payload, sort_keys=True))


def _build_region_resolution(
    *,
    property_level_context: dict[str, Any],
    coverage_lookup: dict[str, Any] | None = None,
) -> RegionResolution:
    region_status = str(property_level_context.get("region_status") or "")
    region_id = property_level_context.get("region_id")
    region_display_name = property_level_context.get("region_display_name")
    coverage_lookup = dict(coverage_lookup or {})
    diagnostics = list(coverage_lookup.get("diagnostics") or [])
    lookup_covered = bool(
        coverage_lookup.get("coverage_available")
        or coverage_lookup.get("covered")
    )
    lookup_region_id = coverage_lookup.get("resolved_region_id") or coverage_lookup.get("region_id")
    lookup_display_name = coverage_lookup.get("resolved_region_display_name") or coverage_lookup.get("display_name")

    if region_status == "invalid_manifest":
        return RegionResolution(
            coverage_available=False,
            resolved_region_id=str(region_id) if region_id else None,
            resolved_region_display_name=str(region_display_name) if region_display_name else None,
            reason="prepared_region_manifest_invalid",
            recommended_action="Run scripts/validate_prepared_region.py for this region and repair missing files.",
            diagnostics=diagnostics,
        )

    if lookup_covered or region_status == "prepared":
        return RegionResolution(
            coverage_available=True,
            resolved_region_id=str(lookup_region_id or region_id) if (lookup_region_id or region_id) else None,
            resolved_region_display_name=(
                str(lookup_display_name or region_display_name)
                if (lookup_display_name or region_display_name)
                else None
            ),
            reason="prepared_region_found",
            recommended_action=None,
            diagnostics=diagnostics,
        )
    if region_status == "legacy_fallback":
        return RegionResolution(
            coverage_available=False,
            resolved_region_id=None,
            resolved_region_display_name=None,
            reason="legacy_fallback_used",
            recommended_action=(
                "Prepare and validate a region for this location to enable prepared-region runtime scoring."
            ),
            diagnostics=diagnostics,
        )
    return RegionResolution(
        coverage_available=False,
        resolved_region_id=None,
        resolved_region_display_name=None,
        reason="no_prepared_region_for_location",
        recommended_action=(
            "Prepare and validate a region for this location using "
            "scripts/prepare_region_from_catalog_or_sources.py, then retry assessment."
        ),
        diagnostics=diagnostics,
    )


def _derive_region_bbox_from_point(lat: float, lon: float) -> dict[str, float]:
    tile_deg_raw = os.getenv("WF_AUTO_REGION_PREP_TILE_DEG", "0.25")
    try:
        tile_deg = max(0.05, min(5.0, float(tile_deg_raw)))
    except ValueError:
        tile_deg = 0.25

    min_lon = math.floor(lon / tile_deg) * tile_deg
    min_lat = math.floor(lat / tile_deg) * tile_deg
    max_lon = min_lon + tile_deg
    max_lat = min_lat + tile_deg
    return {
        "min_lon": round(max(-180.0, min_lon), 6),
        "min_lat": round(max(-90.0, min_lat), 6),
        "max_lon": round(min(180.0, max_lon), 6),
        "max_lat": round(min(90.0, max_lat), 6),
    }


def _auto_region_id_for_bbox(bbox: dict[str, float]) -> str:
    digest = AssessmentStore.build_region_prep_dedupe_key("auto", bbox)
    return f"auto_{digest[:12]}"


def _to_region_job_status(job: dict[str, Any]) -> RegionPrepJobStatus:
    requested_bbox = job.get("requested_bbox") or {}
    bbox = RegionBoundingBox(
        min_lon=float(requested_bbox.get("min_lon", 0.0)),
        min_lat=float(requested_bbox.get("min_lat", 0.0)),
        max_lon=float(requested_bbox.get("max_lon", 0.0)),
        max_lat=float(requested_bbox.get("max_lat", 0.0)),
    )
    return RegionPrepJobStatus(
        job_id=str(job.get("job_id") or ""),
        region_id=str(job.get("region_id") or ""),
        display_name=str(job.get("display_name") or ""),
        requested_bbox=bbox,
        requested_address=job.get("requested_address"),
        point_lat=job.get("point_lat"),
        point_lon=job.get("point_lon"),
        status=str(job.get("status") or "queued"),  # type: ignore[arg-type]
        created_at=str(job.get("created_at") or ""),
        updated_at=str(job.get("updated_at") or ""),
        error_message=job.get("error_message"),
        manifest_path=job.get("manifest_path"),
        dedupe_key=str(job.get("dedupe_key") or ""),
        reused_existing_job=bool(job.get("reused_existing_job", False)),
        result=job.get("result"),
    )


def _build_region_prepare_request_payload(
    *,
    region_id: str,
    display_name: str,
    bbox: dict[str, float],
    source_config_path: str | None = None,
    validate: bool | None = None,
    overwrite: bool | None = None,
    allow_partial_coverage_fill: bool | None = None,
    skip_optional_layers: bool | None = None,
    prefer_bbox_downloads: bool | None = None,
    allow_full_download_fallback: bool | None = None,
    require_core_layers: bool | None = None,
    target_resolution: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "region_id": region_id,
        "display_name": display_name,
        "bbox": bbox,
        "source_config_path": source_config_path
        if source_config_path is not None
        else os.getenv("WF_REGION_PREP_SOURCE_CONFIG") or None,
        "validate": validate if validate is not None else _env_flag("WF_REGION_PREP_VALIDATE", True),
        "overwrite": overwrite if overwrite is not None else _env_flag("WF_REGION_PREP_OVERWRITE", False),
        "allow_partial_coverage_fill": (
            allow_partial_coverage_fill
            if allow_partial_coverage_fill is not None
            else _env_flag("WF_REGION_PREP_ALLOW_PARTIAL_COVERAGE_FILL", True)
        ),
        "skip_optional_layers": (
            skip_optional_layers
            if skip_optional_layers is not None
            else _env_flag("WF_REGION_PREP_SKIP_OPTIONAL_LAYERS", False)
        ),
        "prefer_bbox_downloads": (
            prefer_bbox_downloads
            if prefer_bbox_downloads is not None
            else _env_flag("WF_REGION_PREP_PREFER_BBOX_DOWNLOADS", True)
        ),
        "allow_full_download_fallback": (
            allow_full_download_fallback
            if allow_full_download_fallback is not None
            else _env_flag("WF_REGION_PREP_ALLOW_FULL_DOWNLOAD_FALLBACK", True)
        ),
        "require_core_layers": (
            require_core_layers
            if require_core_layers is not None
            else _env_flag("WF_REGION_PREP_REQUIRE_CORE_LAYERS", True)
        ),
        "target_resolution": target_resolution,
    }
    return payload


def _enqueue_region_prep_job(
    *,
    region_id: str,
    display_name: str,
    bbox: dict[str, float],
    requested_address: str | None = None,
    point_lat: float | None = None,
    point_lon: float | None = None,
    request_payload: dict[str, Any],
) -> RegionPrepJobStatus:
    job, _ = store.create_or_get_region_prep_job(
        region_id=region_id,
        display_name=display_name,
        requested_bbox=bbox,
        requested_address=requested_address,
        point_lat=point_lat,
        point_lon=point_lon,
        request_payload=request_payload,
    )
    return _to_region_job_status(job)


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
    return {
        "status": "ok",
        "product_version": PRODUCT_VERSION,
        "api_version": API_VERSION,
        "model_governance": _build_result_governance(
            ruleset_version=DEFAULT_RULESET_VERSION,
            region_data_version=None,
        ),
    }


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


@app.post("/regions/prepare", response_model=RegionPrepJobStatus, dependencies=[Depends(require_api_key)])
def create_region_prepare_job(
    payload: RegionPrepareRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> RegionPrepJobStatus:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot create region preparation jobs")

    bbox = payload.bbox.model_dump(mode="json")
    request_payload = _build_region_prepare_request_payload(
        region_id=payload.region_id,
        display_name=payload.display_name or payload.region_id.replace("_", " ").title(),
        bbox=bbox,
        source_config_path=payload.source_config_path,
        validate=payload.run_validation,
        overwrite=payload.overwrite,
        allow_partial_coverage_fill=payload.allow_partial_coverage_fill,
        skip_optional_layers=payload.skip_optional_layers,
        prefer_bbox_downloads=payload.prefer_bbox_downloads,
        allow_full_download_fallback=payload.allow_full_download_fallback,
        require_core_layers=payload.require_core_layers,
        target_resolution=payload.target_resolution,
    )

    status = _enqueue_region_prep_job(
        region_id=request_payload["region_id"],
        display_name=request_payload["display_name"],
        bbox=bbox,
        request_payload=request_payload,
    )
    _log_audit(
        ctx=ctx,
        entity_type="region_prep_job",
        entity_id=status.job_id,
        action="region_prep_job_queued",
        organization_id=ctx.organization_id,
        metadata={
            "region_id": status.region_id,
            "status": status.status,
            "reused_existing_job": status.reused_existing_job,
            "requested_bbox": bbox,
        },
    )
    return status


@app.get("/regions/prepare/{job_id}", response_model=RegionPrepJobStatus, dependencies=[Depends(require_api_key)])
def get_region_prepare_job(job_id: str, _: ActorContext = Depends(get_actor_context)) -> RegionPrepJobStatus:
    job = store.get_region_prep_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Region prep job not found")
    return _to_region_job_status(job)


@app.post("/regions/coverage-check", response_model=RegionCoverageStatus, dependencies=[Depends(require_api_key)])
def check_region_coverage(
    payload: RegionCoverageRequest,
    _: ActorContext = Depends(get_actor_context),
) -> RegionCoverageStatus:
    if _is_dev_mode():
        LOGGER.info(
            "coverage_check_request %s",
            json.dumps(
                {
                    "event": "coverage_check_request",
                    "address_raw": payload.address,
                    "address_normalized": normalize_address(payload.address or ""),
                    "has_lat_lon": payload.latitude is not None and payload.longitude is not None,
                    "payload_shape": sorted(list(payload.model_dump(exclude_none=True).keys())),
                },
                sort_keys=True,
            ),
        )
    lat = payload.latitude
    lon = payload.longitude
    geocode_resolution: GeocodeResolution | None = None
    if lat is None or lon is None:
        if not payload.address or len(payload.address.strip()) < 5:
            raise HTTPException(status_code=400, detail="Provide either latitude/longitude or a valid address.")
        geocode_resolution = _resolve_trusted_geocode(
            address_input=payload.address,
            purpose="region_coverage_check",
            route_name="/regions/coverage-check",
        )
        lat = geocode_resolution.latitude
        lon = geocode_resolution.longitude

    coverage_resolution = _resolve_prepared_region(
        latitude=float(lat),
        longitude=float(lon),
        route_name="/regions/coverage-check",
        address_input=str(payload.address or ""),
    )
    coverage = dict(coverage_resolution.coverage)
    if geocode_resolution:
        geocode_meta = geocode_resolution.geocode_meta
        coverage["geocode_status"] = geocode_meta.get("geocode_status")
        coverage["geocode_outcome"] = geocode_meta.get("geocode_outcome")
        coverage["trusted_match_status"] = geocode_meta.get("trusted_match_status")
        coverage["trusted_match_failure_reason"] = geocode_meta.get("trusted_match_failure_reason")
        coverage["fallback_eligibility"] = geocode_meta.get("fallback_eligibility")
        coverage["normalized_address"] = geocode_meta.get("normalized_address")
        coverage["geocode_source"] = geocode_meta.get("geocode_source")
        coverage["geocode_precision"] = geocode_meta.get("geocode_precision")
        coverage["geocode_location_type"] = geocode_meta.get("geocode_location_type")
    return RegionCoverageStatus.model_validate(coverage)


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(
    payload: AddressRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot create assessments")
    if _is_dev_mode():
        LOGGER.info(
            "assessment_request %s",
            json.dumps(
                {
                    "event": "assessment_request",
                    "address_raw": payload.address,
                    "address_normalized": normalize_address(payload.address or ""),
                    "payload_shape": sorted(list(payload.model_dump(exclude_none=True).keys())),
                    "attribute_keys": sorted(list((payload.attributes.model_dump(exclude_none=True) or {}).keys())),
                    "confirmed_fields_count": len(payload.confirmed_fields or []),
                },
                sort_keys=True,
            ),
        )

    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)

    ruleset = _get_ruleset_or_default(payload.ruleset_id)

    auto_queue_on_uncovered = _env_flag("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", False)
    require_prepared_region = _env_flag("WF_REQUIRE_PREPARED_REGION_COVERAGE", False)
    geocode_resolution = _resolve_trusted_geocode(
        address_input=payload.address,
        purpose="assessment",
        route_name="/risk/assess",
    )
    coverage_resolution = _resolve_prepared_region(
        latitude=geocode_resolution.latitude,
        longitude=geocode_resolution.longitude,
        route_name="/risk/assess",
        address_input=payload.address,
    )
    if not coverage_resolution.coverage_available and (auto_queue_on_uncovered or require_prepared_region):
        lat = geocode_resolution.latitude
        lon = geocode_resolution.longitude
        geocode_meta = geocode_resolution.geocode_meta
        coverage = coverage_resolution.coverage
        if auto_queue_on_uncovered:
            requested_bbox = _derive_region_bbox_from_point(lat=lat, lon=lon)
            region_id = _auto_region_id_for_bbox(requested_bbox)
            display_name = f"Auto Prepared Region {region_id.replace('auto_', '').upper()}"
            request_payload = _build_region_prepare_request_payload(
                region_id=region_id,
                display_name=display_name,
                bbox=requested_bbox,
            )
            status = _enqueue_region_prep_job(
                region_id=region_id,
                display_name=display_name,
                bbox=requested_bbox,
                requested_address=payload.address,
                point_lat=lat,
                point_lon=lon,
                request_payload=request_payload,
            )
            _log_audit(
                ctx=ctx,
                entity_type="region_prep_job",
                entity_id=status.job_id,
                action="region_prep_job_auto_enqueued",
                organization_id=organization_id,
                metadata={
                    "address": payload.address,
                    "region_id": region_id,
                    "requested_bbox": requested_bbox,
                    "reused_existing_job": status.reused_existing_job,
                },
            )
            _log_region_resolution_event(
                address=payload.address,
                latitude=lat,
                longitude=lon,
                region_resolution={
                    "coverage_available": False,
                    "resolved_region_id": None,
                    "reason": "no_prepared_region_for_location",
                    "diagnostics": coverage.get("diagnostics", []),
                },
                manifest_path=None,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "region_not_ready": True,
                    "geocode_status": geocode_meta.get("geocode_status"),
                    "geocode_outcome": geocode_meta.get("geocode_outcome"),
                    "trusted_match_status": geocode_meta.get("trusted_match_status"),
                    "trusted_match_failure_reason": geocode_meta.get("trusted_match_failure_reason"),
                    "fallback_eligibility": geocode_meta.get("fallback_eligibility"),
                    "submitted_address": geocode_meta.get("submitted_address"),
                    "normalized_address": geocode_meta.get("normalized_address"),
                    "resolved_latitude": geocode_meta.get("resolved_latitude"),
                    "resolved_longitude": geocode_meta.get("resolved_longitude"),
                    "coverage_available": False,
                    "resolved_region_id": None,
                    "reason": "no_prepared_region_for_location",
                    "prep_job_id": status.job_id,
                    "prep_job_status": status.status,
                    "requested_bbox": requested_bbox,
                    "message": (
                        "No prepared region currently covers this address. A region prep job has been queued; "
                        "retry assessment after the job completes."
                    ),
                    "recommended_action": "Retry assessment after prep job completion.",
                    "diagnostics": coverage.get("diagnostics", []),
                    "region_distance_to_boundary_m": coverage.get("region_distance_to_boundary_m"),
                    "nearest_region_id": coverage.get("nearest_region_id"),
                },
            )
        _log_region_resolution_event(
            address=payload.address,
            latitude=lat,
            longitude=lon,
            region_resolution={
                "coverage_available": False,
                "resolved_region_id": None,
                "reason": "no_prepared_region_for_location",
                "diagnostics": coverage.get("diagnostics", []),
            },
            manifest_path=None,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "region_not_ready": True,
                "geocode_status": geocode_meta.get("geocode_status"),
                "geocode_outcome": geocode_meta.get("geocode_outcome"),
                "trusted_match_status": geocode_meta.get("trusted_match_status"),
                "trusted_match_failure_reason": geocode_meta.get("trusted_match_failure_reason"),
                "fallback_eligibility": geocode_meta.get("fallback_eligibility"),
                "submitted_address": geocode_meta.get("submitted_address"),
                "normalized_address": geocode_meta.get("normalized_address"),
                "resolved_latitude": geocode_meta.get("resolved_latitude"),
                "resolved_longitude": geocode_meta.get("resolved_longitude"),
                "coverage_available": False,
                "resolved_region_id": None,
                "reason": "no_prepared_region_for_location",
                "prep_job_id": None,
                "prep_job_status": None,
                "requested_bbox": _derive_region_bbox_from_point(lat=lat, lon=lon),
                "message": (
                    "Prepared region coverage is required for assessment, but this address is outside "
                    "current prepared regions. Run the offline region-prep command, validate, then retry."
                ),
                "recommended_action": (
                    "Run scripts/prepare_region_from_catalog_or_sources.py for this bbox, validate, then retry."
                ),
                "diagnostics": coverage.get("diagnostics", []),
                "region_distance_to_boundary_m": coverage.get("region_distance_to_boundary_m"),
                "nearest_region_id": coverage.get("nearest_region_id"),
            },
        )

    result = _compute_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
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

    def _delta(before_score: float | None, after_score: float | None) -> float | None:
        if before_score is None or after_score is None:
            return None
        return round(after_score - before_score, 1)

    wildfire_delta = _delta(baseline.wildfire_risk_score, simulated.wildfire_risk_score)
    readiness_delta = _delta(baseline.insurance_readiness_score, simulated.insurance_readiness_score)

    sim_result = SimulationResult(
        scenario_name=payload.scenario_name,
        baseline=baseline,
        simulated=simulated,
        delta=SimulationDelta(
            wildfire_risk_score_delta=wildfire_delta,
            insurance_readiness_score_delta=readiness_delta,
        ),
        changed_inputs=changed_inputs,
        next_best_actions=simulated.mitigation_plan,
        base_assessment_id=payload.assessment_id or baseline.assessment_id,
        simulated_assessment_id=simulated.assessment_id,
        base_scores=baseline.risk_scores,
        simulated_scores=simulated.risk_scores,
        score_delta=SimulationDelta(
            wildfire_risk_score_delta=wildfire_delta,
            insurance_readiness_score_delta=readiness_delta,
        ),
        base_confidence=baseline.confidence,
        simulated_confidence=simulated.confidence,
        base_assumptions=baseline.assumptions,
        simulated_assumptions=simulated.assumptions,
        summary=(
            f"Wildfire risk changed by {wildfire_delta if wildfire_delta is not None else 'not computed'} "
            f"and insurance readiness changed by "
            f"{readiness_delta if readiness_delta is not None else 'not computed'}."
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
def debug_risk(
    payload: AddressRequest,
    include_benchmark_hints: bool = Query(default=False),
    ctx: ActorContext = Depends(get_actor_context),
) -> dict:
    if _is_dev_mode():
        LOGGER.info(
            "debug_assessment_request %s",
            json.dumps(
                {
                    "event": "debug_assessment_request",
                    "route_name": "/risk/debug",
                    "address_raw": payload.address,
                    "address_normalized": normalize_address(payload.address or ""),
                    "payload_shape": sorted(list(payload.model_dump(exclude_none=True).keys())),
                },
                sort_keys=True,
            ),
        )
    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)
    geocode_resolution = _resolve_trusted_geocode(
        address_input=payload.address,
        purpose="assessment",
        route_name="/risk/debug",
    )
    coverage_resolution = _resolve_prepared_region(
        latitude=geocode_resolution.latitude,
        longitude=geocode_resolution.longitude,
        route_name="/risk/debug",
        address_input=payload.address,
    )
    result, debug_payload = _run_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
    )
    if include_benchmark_hints:
        debug_payload["benchmark_hints"] = build_benchmark_hints_for_assessment(result)
    return debug_payload


@app.post("/risk/layer-diagnostics", dependencies=[Depends(require_api_key)])
def layer_diagnostics(payload: AddressRequest, ctx: ActorContext = Depends(get_actor_context)) -> dict:
    organization_id = _resolve_org_id(payload.organization_id, ctx)
    _enforce_org_scope(ctx, organization_id)
    ruleset = _get_ruleset_or_default(payload.ruleset_id)
    geocode_resolution = _resolve_trusted_geocode(
        address_input=payload.address,
        purpose="assessment",
        route_name="/risk/layer-diagnostics",
    )
    coverage_resolution = _resolve_prepared_region(
        latitude=geocode_resolution.latitude,
        longitude=geocode_resolution.longitude,
        route_name="/risk/layer-diagnostics",
        address_input=payload.address,
    )
    _, debug_payload = _run_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
    )
    return {
        "address": debug_payload.get("address"),
        "organization_id": debug_payload.get("organization_id"),
        "ruleset_id": debug_payload.get("ruleset_id"),
        "coordinates": debug_payload.get("coordinates"),
        "geocoding": debug_payload.get("geocoding", {}),
        "region": {
            "region_id": (debug_payload.get("property_level_context") or {}).get("region_id"),
            "region_status": (debug_payload.get("property_level_context") or {}).get("region_status"),
            "manifest_path": (debug_payload.get("property_level_context") or {}).get("region_manifest_path"),
            "region_resolution": debug_payload.get("region_resolution", {}),
        },
        "structure_footprint": {
            "footprint_used": (debug_payload.get("property_level_context") or {}).get("footprint_used"),
            "footprint_status": (debug_payload.get("property_level_context") or {}).get("footprint_status"),
            "footprint_source": (debug_payload.get("property_level_context") or {}).get("footprint_source"),
            "footprint_source_name": (debug_payload.get("property_level_context") or {}).get("footprint_source_name"),
            "footprint_source_vintage": (debug_payload.get("property_level_context") or {}).get("footprint_source_vintage"),
            "property_anchor_point": (debug_payload.get("property_level_context") or {}).get("property_anchor_point"),
            "property_anchor_source": (debug_payload.get("property_level_context") or {}).get("property_anchor_source"),
            "property_anchor_precision": (debug_payload.get("property_level_context") or {}).get("property_anchor_precision"),
            "parcel_id": (debug_payload.get("property_level_context") or {}).get("parcel_id"),
            "parcel_lookup_method": (debug_payload.get("property_level_context") or {}).get("parcel_lookup_method"),
            "parcel_lookup_distance_m": (debug_payload.get("property_level_context") or {}).get("parcel_lookup_distance_m"),
            "parcel_source_name": (debug_payload.get("property_level_context") or {}).get("parcel_source_name"),
            "parcel_source_vintage": (debug_payload.get("property_level_context") or {}).get("parcel_source_vintage"),
            "structure_match_status": (debug_payload.get("property_level_context") or {}).get("structure_match_status"),
            "structure_match_method": (debug_payload.get("property_level_context") or {}).get("structure_match_method"),
            "matched_structure_id": (debug_payload.get("property_level_context") or {}).get("matched_structure_id"),
            "structure_match_confidence": (debug_payload.get("property_level_context") or {}).get("structure_match_confidence"),
            "building_source": (debug_payload.get("property_level_context") or {}).get("building_source"),
            "building_source_version": (debug_payload.get("property_level_context") or {}).get("building_source_version"),
            "building_source_confidence": (debug_payload.get("property_level_context") or {}).get("building_source_confidence"),
            "structure_match_distance_m": (debug_payload.get("property_level_context") or {}).get("structure_match_distance_m"),
            "candidate_structure_count": (debug_payload.get("property_level_context") or {}).get("candidate_structure_count"),
            "structure_match_candidates": (debug_payload.get("property_level_context") or {}).get("structure_match_candidates"),
            "structure_geometry_source": (debug_payload.get("property_level_context") or {}).get("structure_geometry_source"),
            "selected_structure_id": (debug_payload.get("property_level_context") or {}).get("selected_structure_id"),
            "selected_structure_geometry": (debug_payload.get("property_level_context") or {}).get("selected_structure_geometry"),
            "display_point_source": (debug_payload.get("property_level_context") or {}).get("display_point_source"),
            "assessed_property_display_point": (debug_payload.get("property_level_context") or {}).get("assessed_property_display_point"),
            "source_conflict_flag": (debug_payload.get("property_level_context") or {}).get("source_conflict_flag"),
            "alignment_notes": (debug_payload.get("property_level_context") or {}).get("alignment_notes"),
            "fallback_mode": (debug_payload.get("property_level_context") or {}).get("fallback_mode"),
            "defensible_space_analysis": (debug_payload.get("property_level_context") or {}).get("defensible_space_analysis"),
        },
        "top_near_structure_risk_drivers": debug_payload.get("top_near_structure_risk_drivers", []),
        "prioritized_vegetation_actions": debug_payload.get("prioritized_vegetation_actions", []),
        "defensible_space_limitations_summary": debug_payload.get("defensible_space_limitations_summary", []),
        "layer_coverage_audit": debug_payload.get("layer_coverage_audit", []),
        "coverage_summary": debug_payload.get("coverage_summary", {}),
        "fallback_decisions": {
            "assumptions_used": debug_payload.get("assumptions_used", []),
            "assessment_blockers": ((debug_payload.get("eligibility") or {}).get("assessment_blockers") or []),
            "confidence_use_restriction": ((debug_payload.get("confidence_gating") or {}).get("use_restriction")),
        },
        "warnings": debug_payload.get("coverage_summary", {}).get("recommended_actions", []),
    }


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
    include_benchmark_hints: bool = Query(default=False),
    ctx: ActorContext = Depends(get_actor_context),
) -> ReportExport:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    export = _build_report_export(
        result,
        actor=ctx,
        audience=audience,
        audience_mode=audience_mode,
        include_benchmark_hints=include_benchmark_hints,
    )
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


@app.get("/report/{assessment_id}/map", response_model=AssessmentMapPayload, dependencies=[Depends(require_api_key)])
def get_assessment_map(
    assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentMapPayload:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    return build_assessment_map_payload(_refresh_result_governance(result), wildfire_data=wildfire_data)


@app.get("/report/{assessment_id}/homeowner", response_model=HomeownerReport, dependencies=[Depends(require_api_key)])
def get_homeowner_report(
    assessment_id: str,
    include_professional_debug_metadata: bool = Query(default=False),
    ctx: ActorContext = Depends(get_actor_context),
) -> HomeownerReport:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    return build_homeowner_report(
        _refresh_result_governance(result),
        include_professional_debug_metadata=include_professional_debug_metadata,
    )


@app.get("/report/{assessment_id}/homeowner/pdf", dependencies=[Depends(require_api_key)])
def download_homeowner_report_pdf(
    assessment_id: str,
    include_professional_debug_metadata: bool = Query(default=False),
    ctx: ActorContext = Depends(get_actor_context),
) -> Response:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    report = build_homeowner_report(
        _refresh_result_governance(result),
        include_professional_debug_metadata=include_professional_debug_metadata,
    )
    pdf_bytes = render_homeowner_report_pdf(report)
    filename = f"wildfire_homeowner_report_{assessment_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
