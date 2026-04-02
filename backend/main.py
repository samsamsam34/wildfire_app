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
from pathlib import Path
from typing import Any, Dict, Iterable
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from backend.auth import require_api_key
from backend.address_resolution import infer_localities_for_zip, resolve_local_address_candidate
from backend.assessment_map import build_assessment_map_payload
from backend.benchmarking import (
    build_benchmark_hints_for_assessment,
)
from backend.calibration import resolve_public_calibration
from backend.differentiation import (
    build_differentiation_snapshot,
    should_trigger_nearby_home_comparison_safeguard,
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
from backend.homeowner_advisor import (
    build_confidence_summary,
    build_ranked_risk_drivers,
    build_simulator_explanations,
    prioritize_mitigation_actions,
)
from backend.homeowner_improvement import (
    build_improvement_why_it_matters,
    build_improve_your_result_block,
    build_homeowner_improvement_options,
    build_improvement_change_set,
    defensible_space_ft_from_condition,
    summarize_assessment_for_improvement,
)
from backend.homeowner_report import build_homeowner_report, render_homeowner_report_pdf
from backend.internal_diagnostics_artifacts import (
    SECTION_FILES,
    build_no_ground_truth_health_summary,
    compare_no_ground_truth_runs,
    list_no_ground_truth_runs,
    load_no_ground_truth_run_bundle,
)
from backend.public_outcome_artifacts import load_public_outcome_governance_snapshot
from backend.layer_diagnostics import LAYER_SPECS
from backend.mitigation import build_mitigation_plan
from backend.property_linkage import build_property_linkage_summary
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
    AddressCandidateSearchRequest,
    AddressCandidateSearchResponse,
    ManualAddressCandidate,
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
    GeometryResolutionSummary,
    InputSourceMetadata,
    GeocodeDebugRequest,
    GeocodingDetails,
    HomeownerReport,
    HomeownerImprovementOptions,
    HomeownerImprovementRunRequest,
    HomeownerImprovementRunResponse,
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
    AssessmentWithDiagnosticsResponse,
    CalibratedPublicOutcomeMetadata,
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
from backend.trust_metadata import build_trust_diagnostics, load_trust_reference_artifacts
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
    WildfireContext,
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
secondary_geocoder = Geocoder(
    user_agent=os.getenv("WF_GEOCODE_SECONDARY_USER_AGENT", geocoder.user_agent),
    provider_name=os.getenv("WF_GEOCODE_SECONDARY_PROVIDER_NAME", "Secondary Geocoder"),
    search_url=os.getenv("WF_GEOCODE_SECONDARY_SEARCH_URL", ""),
)
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
DIRECT_SOURCE_TYPES = {"observed", "footprint_derived", "user_provided"}
INFERRED_SOURCE_TYPES = {"public_record_inferred"}
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
    "near_structure_vegetation_0_5_pct",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
}

NEAR_STRUCTURE_PROXY_CRITICAL_FIELDS = {
    "near_structure_vegetation_0_5_pct",
    "canopy_adjacency_proxy_pct",
    "vegetation_continuity_proxy_pct",
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
        "near_structure_vegetation_0_5_pct",
        "canopy_adjacency_proxy_pct",
        "vegetation_continuity_proxy_pct",
        "nearest_high_fuel_patch_distance_ft",
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
        "near_structure_vegetation_0_5_pct",
        "canopy_adjacency_proxy_pct",
        "vegetation_continuity_proxy_pct",
        "fuel_model",
        "wildland_distance",
        "historic_fire_distance",
    ],
}

FEATURE_FLAG_LABELS: dict[str, str] = {
    "parcel_polygon_available": "Parcel boundary geometry",
    "building_footprint_available": "Building footprint geometry",
    "hazard_severity_available": "Wildfire hazard severity layer",
    "burn_probability_available": "Burn probability layer",
    "dryness_available": "Climate dryness context",
    "road_network_available": "Road/access network context",
    "near_structure_vegetation_available": "Near-structure vegetation context",
}

LAYER_FRIENDLY_NAMES: dict[str, str] = {
    "dem": "Terrain elevation (DEM)",
    "fuel": "LANDFIRE fuel",
    "canopy": "LANDFIRE canopy",
    "fire_perimeters": "Fire perimeter history",
    "building_footprints": "Building footprints",
    "building_footprints_overture": "Overture building footprints",
    "roads": "Road/access network",
    "whp": "Wildfire Hazard Potential (WHP)",
    "mtbs_severity": "MTBS burn severity",
    "gridmet_dryness": "GRIDMET dryness",
    "parcel_polygons": "Parcel polygons",
    "parcel_address_points": "Parcel address points",
    "naip_imagery": "NAIP imagery",
}

LAYER_CONFIG_ACTIONS: dict[str, str] = {
    "gridmet_dryness": (
        "Configure GRIDMET dryness source overrides (WF_DEFAULT_GRIDMET_DRYNESS_ENDPOINT "
        "or WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL), then rerun prep."
    ),
    "building_footprints_overture": (
        "Configure Overture buildings source (WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT, "
        "WF_DEFAULT_OVERTURE_BUILDINGS_URL, or WF_DEFAULT_OVERTURE_BUILDINGS_PATH), then rerun prep."
    ),
    "parcel_polygons": (
        "Configure parcel polygon source (WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT "
        "or WF_DEFAULT_PARCEL_POLYGONS_PATH), then rerun prep."
    ),
    "parcel_address_points": (
        "Configure parcel address points source (WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT "
        "or WF_DEFAULT_PARCEL_ADDRESS_POINTS_PATH), then rerun prep."
    ),
    "whp": "Validate WHP service access/credentials (403s are common) and rerun prep.",
    "mtbs_severity": "Validate MTBS service access/credentials (403s are common) and rerun prep.",
}

OUTPUT_STATE_LIMITATION_TEXT: dict[str, str] = {
    "property_specific_assessment": "Property-specific geometry and environmental coverage were sufficient for detailed scoring.",
    "address_level_estimate": (
        "This result is an address-level estimate. Some property-specific geometry was unavailable, "
        "so parts of the analysis rely on nearby proxy context."
    ),
    "limited_regional_estimate": (
        "This result is based mostly on regional conditions because property-specific data coverage was limited."
    ),
    "insufficient_data": (
        "Available data was too limited for a credible property-level estimate. "
        "Select your home on the map and/or prepare additional regional layers."
    ),
}

OUTPUT_STATE_TO_HOMEOWNER_MODE: dict[str, str] = {
    "property_specific_assessment": "property_specific",
    "address_level_estimate": "address_level",
    "limited_regional_estimate": "limited_regional_estimate",
    "insufficient_data": "insufficient_data",
}

REGION_PROPERTY_SPECIFIC_READINESS_ORDER: dict[str, int] = {
    "limited_regional_ready": 0,
    "address_level_only": 1,
    "property_specific_ready": 2,
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
        fallback_decisions.append(
            {
                "fallback_type": "missing_factor_omitted",
                "missing_input": "defensible_space_ft",
                "substitute_input": "none",
                "confidence_penalty_hint": 4.0,
                "quality_label": "missing",
                "note": (
                    "Defensible space input is missing. Numeric defaults are no longer injected; "
                    "this factor is omitted from direct weighting and confidence is reduced."
                ),
            }
        )

    if updated.roof_type is None:
        fallback_decisions.append(
            {
                "fallback_type": "missing_factor_omitted",
                "missing_input": "roof_type",
                "substitute_input": "none",
                "confidence_penalty_hint": 2.0,
                "quality_label": "missing",
                "note": (
                    "Roof type is missing. Numeric defaults are no longer injected; "
                    "the model uses observed factors only and lowers specificity/confidence."
                ),
            }
        )

    if updated.vent_type is None:
        fallback_decisions.append(
            {
                "fallback_type": "missing_factor_omitted",
                "missing_input": "vent_type",
                "substitute_input": "none",
                "confidence_penalty_hint": 2.0,
                "quality_label": "missing",
                "note": (
                    "Vent type is missing. Numeric defaults are no longer injected; "
                    "the model uses observed factors only and lowers specificity/confidence."
                ),
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
        missing_inputs.append("roof_type")
        assumptions_used.append("Roof type missing; factor omitted from direct structure weighting.")
    if attrs.vent_type is None:
        missing_inputs.append("vent_type")
        assumptions_used.append("Vent type missing; factor omitted from direct structure weighting.")
    if attrs.defensible_space_ft is None:
        missing_inputs.append("defensible_space_ft")
        assumptions_used.append("Defensible space missing; factor omitted from direct structure weighting.")
    if attrs.construction_year is None:
        missing_inputs.append("construction_year")
        assumptions_used.append("Construction year missing; age modifier omitted from direct structure weighting.")

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
    preflight: dict[str, Any] | None = None,
    assessment_output_state: str | None = None,
    observed_weight_fraction: float = 0.0,
    fallback_dominance_ratio: float = 0.0,
) -> ConfidenceBlock:
    missing_inputs_set = set(assumptions.missing_inputs)
    important_missing = len([m for m in assumptions.missing_inputs if m in CORE_FACT_FIELDS or m.endswith("_layer")])
    missing_layer_count = len([m for m in assumptions.missing_inputs if m.endswith("_layer")])
    inferred_count = max(
        len(assumptions.inferred_inputs),
        len((data_provenance.inferred_inputs_used if data_provenance else []) or []),
    )
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
    provider_error_count = sum(1 for status in (environmental_layer_status or {}).values() if status == "error")
    provider_missing_count = sum(1 for status in (environmental_layer_status or {}).values() if status == "missing")
    external_fail_count = provider_error_count + provider_missing_count

    data_completeness_score = round(max(0.0, min(100.0, 100.0 - (len(assumptions.missing_inputs) * 3.0))), 1)

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

    preflight = dict(preflight or {})
    has_preflight = bool(preflight)
    output_state = str(
        assessment_output_state
        or preflight.get("assessment_output_state")
        or "property_specific_assessment"
    )
    coverage_percent = float(preflight.get("feature_coverage_percent") or 0.0)
    missing_core_layer_count = int(preflight.get("missing_core_layer_count") or 0)
    major_env_missing_count = int(preflight.get("major_environmental_missing_count") or 0)
    geometry_basis = str(preflight.get("geometry_basis") or "geocode_point")
    fallback_weight_fraction = float(
        preflight.get("fallback_weight_fraction")
        if preflight.get("fallback_weight_fraction") is not None
        else fallback_dominance_ratio
    )
    effective_observed_weight_fraction = float(observed_weight_fraction)
    if effective_observed_weight_fraction <= 0.0 and not has_preflight:
        observed_signals = float(confirmed_core_count + observed_core_count + max(0, 6 - missing_layer_count))
        inferred_or_missing_signals = float(
            inferred_count + max(0, len(assumptions.missing_inputs) - confirmed_core_count)
        )
        total_signals = max(1.0, observed_signals + inferred_or_missing_signals)
        effective_observed_weight_fraction = max(0.0, min(1.0, observed_signals / total_signals))
    if (
        has_ring_metrics
        and confirmed_core_count >= 3
        and environmental_data_completeness >= 90.0
    ):
        # Confidence should follow evidence quality, not only contribution-weight internals.
        effective_observed_weight_fraction = max(effective_observed_weight_fraction, 0.70)
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    region_required_missing_count = int(preflight.get("region_required_missing_count") or 0)
    region_optional_missing_count = int(preflight.get("region_optional_missing_count") or 0)
    region_enrichment_missing_count = int(preflight.get("region_enrichment_missing_count") or 0)
    observed_feature_count = int(preflight.get("observed_feature_count") or 0)
    inferred_feature_count = int(preflight.get("inferred_feature_count") or 0)
    fallback_feature_count = int(preflight.get("fallback_feature_count") or 0)
    geometry_quality_score = float(preflight.get("geometry_quality_score") or preflight.get("structure_geometry_quality_score") or 0.0)
    regional_context_coverage_score = float(
        preflight.get("regional_context_coverage_score")
        or preflight.get("environmental_layer_coverage_score")
        or 0.0
    )
    regional_enrichment_consumption_score = float(
        preflight.get("regional_enrichment_consumption_score")
        if preflight.get("regional_enrichment_consumption_score") is not None
        else regional_context_coverage_score
    )
    property_specificity_score = float(preflight.get("property_specificity_score") or 0.0)
    missing_critical_fields_set: set[str] = {
        missing
        for missing in assumptions.missing_inputs
        if missing in CORE_FACT_FIELDS
        or missing.endswith("_layer")
        or missing in {"geocode_verification", "building_footprint"}
    }
    inferred_critical_fields_set: set[str] = set()
    if data_provenance:
        for meta in data_provenance.inputs:
            if meta.field_name not in CRITICAL_PROVENANCE_FIELDS:
                continue
            if has_ring_metrics and meta.field_name in NEAR_STRUCTURE_PROXY_CRITICAL_FIELDS:
                # Footprint/parcel ring metrics already provide direct near-structure evidence.
                # Do not double-penalize missing imagery-derived proxy fields in this case.
                continue
            if meta.source_type in LOW_QUALITY_SOURCE_TYPES:
                missing_critical_fields_set.add(meta.field_name)
            elif meta.source_type in INFERRED_SOURCE_TYPES:
                inferred_critical_fields_set.add(meta.field_name)
    if has_ring_metrics:
        missing_critical_fields_set.difference_update(NEAR_STRUCTURE_PROXY_CRITICAL_FIELDS)
        inferred_critical_fields_set.difference_update(NEAR_STRUCTURE_PROXY_CRITICAL_FIELDS)
    missing_critical_fields = sorted(missing_critical_fields_set)
    inferred_critical_fields = sorted(inferred_critical_fields_set)

    # Confidence is an evidence-quality score. It should rise with observed evidence
    # and drop as fallback/inferred/missing signals become dominant.
    core_quality_by_field: list[float] = []
    for field in sorted(CORE_FACT_FIELDS):
        if field in assumptions.confirmed_inputs and field not in missing_inputs_set:
            core_quality_by_field.append(100.0)
        elif assumptions.observed_inputs.get(field) is not None and field not in missing_inputs_set:
            core_quality_by_field.append(82.0)
        elif field in assumptions.inferred_inputs:
            core_quality_by_field.append(52.0)
        else:
            core_quality_by_field.append(12.0)
    structural_signal_score = (
        round(sum(core_quality_by_field) / len(core_quality_by_field), 1)
        if core_quality_by_field
        else 0.0
    )
    geometry_signal_score = max(
        0.0,
        min(
            100.0,
            (geometry_quality_score * 100.0)
            if geometry_quality_score > 0.0
            else (88.0 if has_ring_metrics else (60.0 if property_context_present else 28.0)),
        ),
    )
    regional_signal_score = max(
        0.0,
        min(
            100.0,
            max(
                regional_context_coverage_score,
                regional_enrichment_consumption_score,
                environmental_data_completeness,
            ),
        ),
    )
    property_context_score = 92.0 if has_ring_metrics else (64.0 if property_context_present else 24.0)
    observed_weight_pct = max(0.0, min(100.0, float(effective_observed_weight_fraction) * 100.0))
    confidence = (
        0.32 * observed_weight_pct
        + 0.20 * max(0.0, min(100.0, environmental_data_completeness))
        + 0.18 * structural_signal_score
        + 0.12 * geometry_signal_score
        + 0.10 * regional_signal_score
        + 0.08 * property_context_score
    )

    inferred_feature_count_effective = max(inferred_count, inferred_feature_count)
    confidence -= min(36.0, len(missing_critical_fields) * 5.5)
    confidence -= min(
        45.0,
        (fallback_weight_fraction * 42.0)
        + (max(0, fallback_feature_count - observed_feature_count) * 0.8),
    )
    confidence -= min(
        16.0,
        (float(inferred_feature_count_effective) * 1.0)
        + (float(len(inferred_critical_fields)) * 1.5),
    )
    confidence -= max(0.0, (missing_layer_count - 1) * 3.5)
    confidence -= min(14.0, stale_share * 0.16)
    confidence -= critical_unknown_or_stale * 2.5
    confidence -= min(8.0, heuristic_count * 1.2)
    confidence -= provider_error_count * 12.0
    confidence -= provider_missing_count * 4.0
    confidence += min(8.0, confirmed_core_count * 1.8)
    if has_ring_metrics:
        confidence += 4.0
    if geocode_verified:
        confidence += 2.0

    has_meaningful_environment = environmental_data_completeness >= 25.0 or missing_layer_count <= 2
    has_meaningful_property = has_ring_metrics or observed_core_count > 0 or confirmed_core_count > 0
    if not geocode_verified or (not has_meaningful_environment and not has_meaningful_property):
        confidence = 0.0

    critical_near_structure_missing = (
        not has_ring_metrics
        and all(field in missing_inputs_set for field in {"roof_type", "vent_type", "defensible_space_ft"})
    )
    if critical_near_structure_missing:
        confidence = min(confidence, 50.0)

    # Fallback confidence cap is strictly decreasing with fallback share.
    fallback_confidence_cap = max(18.0, 96.0 - (72.0 * fallback_weight_fraction))
    confidence = min(confidence, fallback_confidence_cap)
    if fallback_feature_count > observed_feature_count:
        confidence = min(confidence, 45.0)

    if has_preflight:
        if coverage_percent <= 15.0:
            confidence = min(confidence, 30.0)
        if missing_core_layer_count >= 4:
            confidence = min(confidence, 38.0)
        if major_env_missing_count >= 2:
            confidence = min(confidence, 44.0)
        if geometry_basis == "geocode_point":
            confidence = min(confidence, 54.0)
        if geometry_quality_score < 0.50:
            confidence = min(confidence, 38.0)
        elif geometry_quality_score < 0.62:
            confidence = min(confidence, 46.0)
        if regional_context_coverage_score < 50.0:
            confidence = min(confidence, 42.0)
        elif regional_context_coverage_score < 65.0:
            confidence = min(confidence, 50.0)
        if regional_enrichment_consumption_score < 50.0:
            confidence = min(confidence, 42.0)
        elif regional_enrichment_consumption_score < 65.0:
            confidence = min(confidence, 52.0)
        if property_specificity_score < 45.0:
            confidence = min(confidence, 40.0)
        elif property_specificity_score < 60.0:
            confidence = min(confidence, 50.0)
        if float(fallback_dominance_ratio) >= 0.70:
            confidence = min(confidence, 40.0)
        if fallback_weight_fraction >= 0.72:
            confidence = min(confidence, 35.0)
        elif fallback_weight_fraction >= 0.60:
            confidence = min(confidence, 42.0)
        elif fallback_weight_fraction >= 0.45:
            confidence = min(confidence, 56.0)
        if float(effective_observed_weight_fraction) < 0.30:
            confidence = min(confidence, 48.0)
        elif float(effective_observed_weight_fraction) < 0.40:
            confidence = min(confidence, 60.0)
        if region_readiness == "address_level_only":
            confidence = min(confidence, 56.0)
        elif region_readiness == "limited_regional_ready":
            confidence = min(confidence, 34.0)
        if region_required_missing_count > 0:
            confidence = min(confidence, 20.0)
        if (region_optional_missing_count + region_enrichment_missing_count) >= 6:
            confidence = min(confidence, 48.0)
        if output_state == "address_level_estimate":
            confidence = min(confidence, 66.0)
        elif output_state == "limited_regional_estimate":
            confidence = min(confidence, 42.0)
        elif output_state == "insufficient_data":
            confidence = 0.0

    if len(missing_critical_fields) >= 5:
        confidence = min(confidence, 30.0)
    elif len(missing_critical_fields) >= 3:
        confidence = min(confidence, 44.0)
    elif len(missing_critical_fields) >= 1:
        confidence = min(confidence, 62.0)

    strong_observed_evidence = bool(
        has_ring_metrics
        and confirmed_core_count >= 3
        and environmental_data_completeness >= 90.0
        and provider_missing_count == 0
        and provider_error_count == 0
        and len(missing_critical_fields) == 0
        and fallback_weight_fraction < 0.25
    )
    if strong_observed_evidence:
        confidence = max(confidence, 86.0)

    critical_missing_pair = {
        "burn_probability_layer",
        "defensible_space_ft",
    }
    if critical_missing_pair.issubset(missing_inputs_set):
        confidence = min(confidence - 14.0, 18.0)
    elif "burn_probability_layer" in missing_inputs_set or "defensible_space_ft" in missing_inputs_set:
        confidence -= 8.0

    # Final directional guardrail: higher fallback share always reduces confidence.
    confidence -= float(fallback_weight_fraction) * 6.0

    confidence = max(0.0, min(100.0, round(confidence, 1)))

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
    if geometry_quality_score >= 0.80:
        confidence_drivers.append("Structure geometry quality is high for this run.")
    if regional_context_coverage_score >= 80.0:
        confidence_drivers.append("Regional hazard/burn/dryness coverage is strong.")
    if regional_enrichment_consumption_score >= 80.0:
        confidence_drivers.append("Regional enrichment layers are present and consumed by runtime scoring.")
    if observed_feature_count > 0 and observed_feature_count >= fallback_feature_count:
        confidence_drivers.append("Observed feature signals outweigh fallback feature signals.")
    if confirmed_core_count > 0:
        confidence_drivers.append(f"{confirmed_core_count} core home fact(s) were confirmed by the user.")
    elif observed_core_count > 0:
        confidence_drivers.append(f"{observed_core_count} core home fact(s) were provided without confirmation.")

    if missing_layer_count > 0:
        confidence_limiters.append(f"{missing_layer_count} environmental layer(s) missing or unavailable.")
    if inferred_feature_count_effective > 0:
        confidence_limiters.append(
            f"{inferred_feature_count_effective} property/scoring input(s) are inferred or proxy-derived."
        )
    if inferred_critical_fields:
        confidence_limiters.append(
            f"{len(inferred_critical_fields)} critical input(s) are inferred/proxy-derived."
        )
    if not has_ring_metrics:
        confidence_limiters.append("Building footprint rings unavailable; using point-based property context.")
    if geometry_quality_score < 0.62:
        confidence_limiters.append("Structure geometry quality is limited; property-specific factor weighting was reduced.")
    if regional_context_coverage_score < 65.0:
        confidence_limiters.append("Regional hazard/burn/dryness coverage is incomplete.")
    if regional_enrichment_consumption_score < 65.0:
        confidence_limiters.append("Some regional enrichment layers are configured but not fully consumed for this property.")
    if fallback_feature_count > observed_feature_count and has_preflight:
        confidence_limiters.append("Fallback feature count exceeds observed feature count.")
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
    if inferred_feature_count_effective >= 3:
        low_confidence_flags.append("Several property/scoring inputs were inferred or proxy-derived")
    if len(inferred_critical_fields) >= 2:
        low_confidence_flags.append("Critical inputs include inferred/proxy evidence")
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
    if fallback_weight_fraction >= 0.60:
        low_confidence_flags.append("Fallback-weighted factors dominate active score composition")
    if geometry_quality_score < 0.62:
        low_confidence_flags.append("Structure geometry quality is limited; structure-specific factors are suppressed or downgraded")
    if regional_context_coverage_score < 65.0:
        low_confidence_flags.append("Regional hazard/burn/dryness coverage is limited; regional context confidence is capped")
    if regional_enrichment_consumption_score < 65.0:
        low_confidence_flags.append("Regional enrichment layers are partially consumed or missing for this location")
    if property_specificity_score < 60.0:
        low_confidence_flags.append("Property-specific evidence strength is limited for this run")
    if fallback_feature_count > observed_feature_count and has_preflight:
        low_confidence_flags.append("Fallback feature count exceeds observed feature count")
    if any("provisional" in note.lower() for note in assumptions.assumptions_used):
        low_confidence_flags.append("Access scoring is provisional and not yet parcel/egress-based")
    if confidence < 70:
        low_confidence_flags.append("Overall confidence below recommended underwriting threshold")
    if output_state == "address_level_estimate":
        low_confidence_flags.append("Assessment is capped at address-level specificity.")
    elif output_state == "limited_regional_estimate":
        low_confidence_flags.append("Assessment is limited to regional-estimate specificity due to low coverage.")
    elif output_state == "insufficient_data":
        low_confidence_flags.append("Data is insufficient for a credible automatic estimate.")
    if region_readiness != "property_specific_ready":
        low_confidence_flags.append(
            f"Prepared region readiness is {region_readiness}; property-specific confidence is capped."
        )
    if region_required_missing_count > 0:
        low_confidence_flags.append("Prepared region is missing required layers for this bbox.")

    severe_layer_failure = external_fail_count >= 2 or missing_layer_count >= 4 or provider_error_count >= 1
    major_layer_failure = external_fail_count >= 1 or missing_layer_count >= 1 or provider_error_count >= 1
    multiple_critical_missing = important_missing >= 4

    if (
        confidence < 35
        or not geocode_verified
        or severe_layer_failure
        or multiple_critical_missing
        or critical_provider_errors >= 1
        or (not has_meaningful_environment and not has_meaningful_property)
    ):
        confidence_tier = "preliminary"
    elif confidence < 62 or stale_share >= 25.0 or critical_unknown_or_stale >= 3:
        confidence_tier = "low"
    elif (
        confidence < 82
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
        or fallback_weight_fraction >= 0.45
        or len(missing_critical_fields) > 0
        or len(inferred_critical_fields) > 0
    ):
        confidence_tier = "moderate"

    if fallback_weight_fraction >= 0.60:
        if confidence_tier in {"high", "moderate"}:
            confidence_tier = "low"
    elif fallback_weight_fraction >= 0.45 and confidence_tier == "high":
        confidence_tier = "moderate"
    if len(missing_critical_fields) >= 4:
        confidence_tier = "preliminary"
    elif len(missing_critical_fields) >= 3 and confidence_tier == "high":
        confidence_tier = "low"
    elif len(missing_critical_fields) >= 1 and confidence_tier == "high":
        confidence_tier = "moderate"

    # Output-state caps prevent low-coverage runs from appearing overconfident.
    if output_state == "address_level_estimate" and confidence_tier == "high":
        confidence_tier = "moderate"
    elif output_state == "limited_regional_estimate":
        if confidence_tier in {"high", "moderate"}:
            confidence_tier = "low"
    elif output_state == "insufficient_data":
        confidence_tier = "preliminary"

    if region_readiness == "address_level_only" and confidence_tier == "high":
        confidence_tier = "moderate"
    if region_readiness == "limited_regional_ready":
        if confidence_tier in {"high", "moderate"}:
            confidence_tier = "low"
    if region_required_missing_count > 0:
        confidence_tier = "preliminary"

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

    if region_readiness == "address_level_only" and use_restriction == "shareable":
        use_restriction = "homeowner_review_recommended"
    if region_readiness == "limited_regional_ready" or region_required_missing_count > 0:
        use_restriction = "not_for_underwriting_or_binding"

    if output_state in {"limited_regional_estimate", "insufficient_data"}:
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
        inferred_fields_count=inferred_feature_count_effective,
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
    if "ring_100_300" in k or "zone_100_300" in k:
        return "zone_100_300_ft"
    if "near_structure_vegetation_0_5" in k:
        return "zone_0_5_ft"
    if "canopy_adjacency_proxy" in k:
        return "zone_0_5_ft"
    if "vegetation_continuity_proxy" in k:
        return "zone_30_100_ft"
    if "nearby_structure_count" in k or "structure_to_structure" in k:
        return "zone_30_100_ft"
    if "nearest_structure_distance" in k or "nearest_structure_proximity" in k:
        return "zone_30_100_ft"
    if "building_age_proxy" in k:
        return "construction_year"
    if "nearest_high_fuel_patch" in k:
        return "wildland_distance"
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
    risk: RiskComputation | None,
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
        factor_evidence_status = str(weight.factor_evidence_status or "")
        if factor_evidence_status == "suppressed":
            evidence_status = "fallback"
        elif factor_evidence_status in {"observed", "inferred", "fallback"}:
            evidence_status = factor_evidence_status
        factor_notes = [score.explanation] + score.assumptions[:2]
        if factor_evidence_status == "suppressed":
            factor_notes.append("Factor contribution was suppressed due to low evidence quality.")
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
            notes=factor_notes,
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

    env_weight, struct_weight_total, readiness_weight = risk_engine.resolve_blend_weights(
        insurance_readiness_score=readiness_score,
        risk=risk,
    )
    if env_weight <= 0 or struct_weight_total <= 0:
        env_weight = sum(
            weighted_contributions[key].weight
            for key in weighted_contributions
            if key in ENVIRONMENTAL_SUBMODELS
        )
        struct_weight_total = sum(
            weighted_contributions[key].weight
            for key in weighted_contributions
            if key in STRUCTURAL_SUBMODELS
        )

    denom = env_weight + struct_weight_total + max(0.0, readiness_weight)
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
        if site_hazard_score is not None and home_ignition_vulnerability_score is not None:
            hazard_norm = max(0.0, min(1.0, float(site_hazard_score) / 100.0))
            vulnerability_norm = max(0.0, min(1.0, float(home_ignition_vulnerability_score) / 100.0))
            harmonic_core = (
                (2.0 * hazard_norm * vulnerability_norm) / (hazard_norm + vulnerability_norm)
                if (hazard_norm + vulnerability_norm) > 0.0
                else 0.0
            )
            product_core = hazard_norm * vulnerability_norm
            hazard_vulnerability_core = 100.0 * ((0.55 * harmonic_core) + (0.45 * product_core))
            wildfire_entries.append(
                ScoreEvidenceFactor(
                    factor_key="hazard_vulnerability_interaction_core",
                    display_name="Hazard-Structure Interaction Core",
                    category="composite",
                    raw_value=round(hazard_vulnerability_core, 2),
                    normalized_value=round(hazard_vulnerability_core, 2),
                    weight=0.42,
                    contribution=round(hazard_vulnerability_core * 0.42, 2),
                    direction="composes_score",
                    evidence_status=_combine_evidence_status(
                        [f.evidence_status for f in site_entries + home_entries]
                    ),
                    source_field="site_hazard_score+home_ignition_vulnerability_score",
                    source_layer="derived_interaction",
                    notes=[
                        "Captures multiplicative overlap between landscape hazard and structure vulnerability."
                    ],
                )
            )
        if readiness_score is not None and readiness_weight > 0:
            readiness_ratio = readiness_weight / denom
            readiness_risk_equivalent = max(0.0, min(100.0, 100.0 - float(readiness_score)))
            readiness_status = _combine_evidence_status([f.evidence_status for f in readiness_entries])
            wildfire_entries.append(
                ScoreEvidenceFactor(
                    factor_key="readiness_risk_component",
                    display_name="Insurance Readiness Risk Equivalent",
                    category="structural",
                    raw_value=readiness_risk_equivalent,
                    normalized_value=readiness_risk_equivalent,
                    weight=round(readiness_ratio, 4),
                    contribution=round(readiness_risk_equivalent * readiness_ratio, 2),
                    direction="composes_score",
                    evidence_status=readiness_status,
                    source_field="insurance_readiness_score",
                    source_layer="insurance_readiness_score",
                    notes=["Bounded conversion of readiness score into blended wildfire risk composition."],
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
            "geometry_basis": "point",
            "structure_match_status": "none",
            "structure_match_method": None,
            "structure_selection_method": None,
            "structure_match_confidence": 0.0,
            "building_source": None,
            "building_source_version": None,
            "building_source_confidence": 0.0,
            "structure_match_distance_m": None,
            "candidate_structure_count": 0,
            "structure_match_candidates": [],
            "footprint_resolution": {
                "selected_source": None,
                "confidence_score": 0.0,
                "candidates_considered": 0,
                "fallback_used": True,
                "match_status": "none",
                "match_method": None,
                "match_distance_m": None,
            },
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
            "property_anchor_selection_method": "geocode_fallback",
            "property_anchor_quality": "low",
            "property_anchor_quality_score": 0.0,
            "anchor_quality": "low",
            "anchor_quality_score": 0.0,
            "address_point_lookup_distance_m": None,
            "geocoded_address_point": None,
            "assessed_property_display_point": None,
            "matched_structure_centroid": None,
            "matched_structure_footprint": None,
            "parcel_id": None,
            "parcel_source": None,
            "parcel_lookup_method": None,
            "parcel_lookup_distance_m": None,
            "parcel_geometry": None,
            "parcel_address_point": None,
            "parcel_resolution": {
                "status": "not_found",
                "confidence": 0.0,
                "source": None,
                "geometry_used": "none",
                "overlap_score": 0.0,
                "candidates_considered": 0,
                "lookup_method": "none",
                "lookup_distance_m": None,
            },
            "property_linkage": {
                "geocode_confidence": 0.0,
                "parcel_confidence": 0.0,
                "footprint_confidence": 0.0,
                "overall_property_confidence": 0.0,
                "parcel_status": "not_found",
                "footprint_status": "none",
                "footprint_match_method": None,
                "multiple_footprints_on_parcel": False,
                "footprint_outside_parcel": False,
                "structure_candidate_count": 0,
                "selection_notes": [],
            },
            "parcel_bounding_approximation": None,
            "alignment_notes": [],
            "source_conflict_flag": False,
            "fallback_mode": "point_based",
            "ring_metrics": None,
            "near_structure_features": {},
            "directional_risk": {},
            "structure_relative_slope": {
                "local_slope": None,
                "slope_within_30_ft": None,
                "uphill_gradient_deg": None,
                "downhill_gradient_deg": None,
                "uphill_exposure": None,
                "downhill_buffer": None,
                "precision_flag": "fallback_point_proxy",
                "confidence_flag": "low",
                "source": "unavailable",
            },
            "structure_attributes": {
                "area": {"sqft": None, "source": None},
                "density_context": {
                    "index": None,
                    "tier": "unknown",
                    "nearby_structure_count_100_ft": None,
                    "nearby_structure_count_300_ft": None,
                    "nearest_structure_distance_ft": None,
                    "source": None,
                },
                "estimated_age_proxy": None,
                "shape_complexity": {"index": None, "source": None},
                "provenance": {
                    "area": "unavailable",
                    "density_context": "unavailable",
                    "estimated_age_proxy": "unavailable",
                    "shape_complexity": "unavailable",
                },
            },
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
    normalized.setdefault("structure_selection_method", normalized.get("structure_match_method"))
    normalized.setdefault("geometry_basis", "footprint" if footprint_used else ("parcel" if normalized.get("parcel_id") else "point"))
    normalized.setdefault("structure_match_confidence", float(normalized.get("footprint_confidence") or (0.9 if footprint_used else 0.0)))
    normalized.setdefault("building_source", str(normalized.get("footprint_source_name") or "") or None)
    normalized.setdefault("building_source_version", str(normalized.get("footprint_source_vintage") or "") or None)
    normalized.setdefault("building_source_confidence", float(normalized.get("structure_match_confidence") or 0.0))
    normalized.setdefault("structure_match_distance_m", 0.0 if footprint_used else None)
    normalized.setdefault("candidate_structure_count", 1 if footprint_used else 0)
    candidates = normalized.get("structure_match_candidates")
    normalized["structure_match_candidates"] = candidates if isinstance(candidates, list) else []
    raw_footprint_resolution = normalized.get("footprint_resolution")
    if isinstance(raw_footprint_resolution, dict):
        selected_source = str(raw_footprint_resolution.get("selected_source") or "").strip() or None
        try:
            confidence_score = float(raw_footprint_resolution.get("confidence_score") or 0.0)
        except (TypeError, ValueError):
            confidence_score = 0.0
        try:
            candidates_considered = int(raw_footprint_resolution.get("candidates_considered") or 0)
        except (TypeError, ValueError):
            candidates_considered = 0
        fallback_used = bool(raw_footprint_resolution.get("fallback_used"))
        match_status = str(raw_footprint_resolution.get("match_status") or "").strip().lower() or "none"
        match_method = str(raw_footprint_resolution.get("match_method") or "").strip() or None
        try:
            match_distance_m = (
                float(raw_footprint_resolution.get("match_distance_m"))
                if raw_footprint_resolution.get("match_distance_m") is not None
                else None
            )
        except (TypeError, ValueError):
            match_distance_m = None
        sources_considered = raw_footprint_resolution.get("sources_considered")
        if isinstance(sources_considered, list):
            normalized_sources_considered = [str(v).strip() for v in sources_considered if str(v).strip()]
        else:
            normalized_sources_considered = []
    else:
        selected_source = str(normalized.get("footprint_source_name") or normalized.get("building_source") or "").strip() or None
        try:
            confidence_score = float(normalized.get("structure_match_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence_score = 0.0
        try:
            candidates_considered = int(normalized.get("candidate_structure_count") or 0)
        except (TypeError, ValueError):
            candidates_considered = 0
        fallback_used = not footprint_used
        match_status = str(normalized.get("structure_match_status") or "none").strip().lower() or "none"
        match_method = str(normalized.get("structure_match_method") or "").strip() or None
        try:
            match_distance_m = (
                float(normalized.get("structure_match_distance_m"))
                if normalized.get("structure_match_distance_m") is not None
                else None
            )
        except (TypeError, ValueError):
            match_distance_m = None
        normalized_sources_considered = []
    normalized["footprint_resolution"] = {
        "selected_source": selected_source,
        "confidence_score": round(max(0.0, min(1.0, float(confidence_score))), 3),
        "candidates_considered": max(0, int(candidates_considered)),
        "fallback_used": bool(fallback_used),
        "match_status": match_status,
        "match_method": match_method,
        "match_distance_m": match_distance_m,
        "sources_considered": normalized_sources_considered,
    }
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
        "parcel_inferred_home_location",
        "raw_geocode_point",
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
    normalized.setdefault("property_anchor_selection_method", "geocode_fallback")
    normalized.setdefault("property_anchor_quality", "low")
    try:
        normalized["property_anchor_quality_score"] = float(normalized.get("property_anchor_quality_score") or 0.0)
    except (TypeError, ValueError):
        normalized["property_anchor_quality_score"] = 0.0
    normalized.setdefault("anchor_quality", normalized.get("property_anchor_quality"))
    try:
        normalized["anchor_quality_score"] = float(
            normalized.get("anchor_quality_score")
            if normalized.get("anchor_quality_score") is not None
            else normalized.get("property_anchor_quality_score")
            or 0.0
        )
    except (TypeError, ValueError):
        normalized["anchor_quality_score"] = 0.0
    try:
        lookup_distance = normalized.get("address_point_lookup_distance_m")
        normalized["address_point_lookup_distance_m"] = (
            float(lookup_distance) if lookup_distance is not None else None
        )
    except (TypeError, ValueError):
        normalized["address_point_lookup_distance_m"] = None
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
    normalized.setdefault("parcel_source", normalized.get("parcel_source_name"))
    normalized.setdefault("parcel_lookup_method", None)
    normalized.setdefault("parcel_lookup_distance_m", None)
    normalized.setdefault("parcel_geometry", None)
    normalized.setdefault("parcel_address_point", None)
    raw_parcel_resolution = normalized.get("parcel_resolution")
    if isinstance(raw_parcel_resolution, dict):
        status = str(raw_parcel_resolution.get("status") or "").strip().lower() or "not_found"
        if status not in {"matched", "multiple_candidates", "not_found"}:
            status = "not_found"
        try:
            parcel_confidence = float(raw_parcel_resolution.get("confidence") or 0.0)
        except (TypeError, ValueError):
            parcel_confidence = 0.0
        try:
            overlap_score = float(raw_parcel_resolution.get("overlap_score") or 0.0)
        except (TypeError, ValueError):
            overlap_score = 0.0
        source = str(raw_parcel_resolution.get("source") or "").strip() or None
        geometry_used = str(raw_parcel_resolution.get("geometry_used") or "").strip().lower() or "none"
        if geometry_used not in {"parcel_polygon", "bounding_approximation", "none"}:
            geometry_used = "none"
        try:
            candidates_considered = int(raw_parcel_resolution.get("candidates_considered") or 0)
        except (TypeError, ValueError):
            candidates_considered = 0
        lookup_method = str(raw_parcel_resolution.get("lookup_method") or "").strip().lower() or "none"
        try:
            lookup_distance_m = (
                float(raw_parcel_resolution.get("lookup_distance_m"))
                if raw_parcel_resolution.get("lookup_distance_m") is not None
                else None
            )
        except (TypeError, ValueError):
            lookup_distance_m = None
        bounding_geometry = (
            raw_parcel_resolution.get("bounding_geometry")
            if isinstance(raw_parcel_resolution.get("bounding_geometry"), dict)
            else None
        )
    else:
        has_parcel_geom = isinstance(normalized.get("parcel_geometry"), dict)
        parcel_lookup_method = str(normalized.get("parcel_lookup_method") or "").strip().lower()
        status = "matched" if (normalized.get("parcel_id") or has_parcel_geom) else "not_found"
        if parcel_lookup_method == "multiple_candidates":
            status = "multiple_candidates"
        parcel_confidence = 92.0 if status == "matched" else (62.0 if status == "multiple_candidates" else 0.0)
        if parcel_lookup_method == "nearest_within_tolerance":
            try:
                lookup_distance = float(normalized.get("parcel_lookup_distance_m") or 0.0)
            except (TypeError, ValueError):
                lookup_distance = 0.0
            parcel_confidence = max(35.0, min(90.0, 80.0 - (lookup_distance * 0.6)))
        overlap_score = 100.0 if status in {"matched", "multiple_candidates"} else 0.0
        source = str(normalized.get("parcel_source") or normalized.get("parcel_source_name") or "").strip() or None
        geometry_used = "parcel_polygon" if status in {"matched", "multiple_candidates"} else "none"
        candidates_considered = 1 if status in {"matched", "multiple_candidates"} else 0
        lookup_method = parcel_lookup_method or ("contains_point" if status == "matched" else "none")
        try:
            lookup_distance_m = (
                float(normalized.get("parcel_lookup_distance_m"))
                if normalized.get("parcel_lookup_distance_m") is not None
                else None
            )
        except (TypeError, ValueError):
            lookup_distance_m = None
        bounding_geometry = (
            normalized.get("parcel_bounding_approximation")
            if isinstance(normalized.get("parcel_bounding_approximation"), dict)
            else None
        )
    normalized["parcel_resolution"] = {
        "status": status,
        "confidence": round(max(0.0, min(100.0, parcel_confidence)), 1),
        "source": source,
        "geometry_used": geometry_used,
        "overlap_score": round(max(0.0, min(100.0, overlap_score)), 1),
        "candidates_considered": max(0, int(candidates_considered)),
        "lookup_method": lookup_method,
        "lookup_distance_m": lookup_distance_m,
        "bounding_geometry": bounding_geometry,
    }
    normalized["parcel_bounding_approximation"] = (
        bounding_geometry if isinstance(bounding_geometry, dict) else None
    )
    raw_property_linkage = normalized.get("property_linkage")
    if isinstance(raw_property_linkage, dict):
        normalized["property_linkage"] = {
            "anchor_status": (
                str(raw_property_linkage.get("anchor_status") or "").strip().lower() or "unresolved"
            ),
            "anchor_confidence": round(
                max(0.0, min(100.0, _safe_float(raw_property_linkage.get("anchor_confidence")) or 0.0)),
                1,
            ),
            "anchor_source": (
                str(raw_property_linkage.get("anchor_source")).strip()
                if raw_property_linkage.get("anchor_source") is not None
                else None
            ),
            "selected_structure_id": (
                str(raw_property_linkage.get("selected_structure_id")).strip()
                if raw_property_linkage.get("selected_structure_id") is not None
                else None
            ),
            "parcel_source": (
                str(raw_property_linkage.get("parcel_source")).strip()
                if raw_property_linkage.get("parcel_source") is not None
                else None
            ),
            "footprint_source": (
                str(raw_property_linkage.get("footprint_source")).strip()
                if raw_property_linkage.get("footprint_source") is not None
                else None
            ),
            "parcel_candidate_count": max(0, int(raw_property_linkage.get("parcel_candidate_count") or 0)),
            "footprint_candidate_count": max(0, int(raw_property_linkage.get("footprint_candidate_count") or 0)),
            "mismatch_flags": [
                str(flag).strip()
                for flag in (raw_property_linkage.get("mismatch_flags") or [])
                if str(flag).strip()
            ][:8],
            "geocode_confidence": round(
                max(0.0, min(100.0, _safe_float(raw_property_linkage.get("geocode_confidence")) or 0.0)),
                1,
            ),
            "parcel_confidence": round(
                max(0.0, min(100.0, _safe_float(raw_property_linkage.get("parcel_confidence")) or 0.0)),
                1,
            ),
            "footprint_confidence": round(
                max(0.0, min(100.0, _safe_float(raw_property_linkage.get("footprint_confidence")) or 0.0)),
                1,
            ),
            "overall_property_confidence": round(
                max(0.0, min(100.0, _safe_float(raw_property_linkage.get("overall_property_confidence")) or 0.0)),
                1,
            ),
            "parcel_status": str(raw_property_linkage.get("parcel_status") or "not_found").strip().lower() or "not_found",
            "footprint_status": str(raw_property_linkage.get("footprint_status") or "none").strip().lower() or "none",
            "footprint_match_method": (
                str(raw_property_linkage.get("footprint_match_method"))
                if raw_property_linkage.get("footprint_match_method")
                else None
            ),
            "multiple_footprints_on_parcel": bool(raw_property_linkage.get("multiple_footprints_on_parcel")),
            "footprint_outside_parcel": bool(raw_property_linkage.get("footprint_outside_parcel")),
            "structure_candidate_count": max(0, int(raw_property_linkage.get("structure_candidate_count") or 0)),
            "selection_notes": [
                str(note) for note in (raw_property_linkage.get("selection_notes") or []) if str(note).strip()
            ][:4],
        }
    else:
        normalized["property_linkage"] = {
            "anchor_status": "unresolved",
            "anchor_confidence": round(
                max(0.0, min(100.0, _safe_float(normalized.get("property_anchor_quality_score")) or 0.0)),
                1,
            ),
            "anchor_source": (
                str(normalized.get("property_anchor_source")).strip()
                if normalized.get("property_anchor_source") is not None
                else None
            ),
            "selected_structure_id": (
                str(normalized.get("selected_structure_id")).strip()
                if normalized.get("selected_structure_id") is not None
                else None
            ),
            "parcel_source": (
                str((normalized.get("parcel_resolution") or {}).get("source")).strip()
                if (normalized.get("parcel_resolution") or {}).get("source") is not None
                else None
            ),
            "footprint_source": (
                str((normalized.get("footprint_resolution") or {}).get("selected_source")).strip()
                if (normalized.get("footprint_resolution") or {}).get("selected_source") is not None
                else None
            ),
            "parcel_candidate_count": max(
                0,
                int((normalized.get("parcel_resolution") or {}).get("candidates_considered") or 0),
            ),
            "footprint_candidate_count": max(
                0,
                int((normalized.get("footprint_resolution") or {}).get("candidates_considered") or 0),
            ),
            "mismatch_flags": [],
            "geocode_confidence": 0.0,
            "parcel_confidence": round(
                max(0.0, min(100.0, _safe_float((normalized.get("parcel_resolution") or {}).get("confidence")) or 0.0)),
                1,
            ),
            "footprint_confidence": round(
                max(
                    0.0,
                    min(
                        100.0,
                        (_safe_float((normalized.get("footprint_resolution") or {}).get("confidence_score")) or 0.0)
                        * 100.0,
                    ),
                ),
                1,
            ),
            "overall_property_confidence": 0.0,
            "parcel_status": str((normalized.get("parcel_resolution") or {}).get("status") or "not_found"),
            "footprint_status": str((normalized.get("footprint_resolution") or {}).get("match_status") or "none"),
            "footprint_match_method": (normalized.get("footprint_resolution") or {}).get("match_method"),
            "multiple_footprints_on_parcel": False,
            "footprint_outside_parcel": False,
            "structure_candidate_count": max(0, int(normalized.get("candidate_structure_count") or 0)),
            "selection_notes": [],
        }
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
    near_structure_features = normalized.get("near_structure_features")
    if isinstance(near_structure_features, dict) and near_structure_features:
        normalized["near_structure_features"] = dict(near_structure_features)
    else:
        ring0 = {}
        ring5 = {}
        if isinstance(normalized.get("ring_metrics"), dict):
            ring0 = (
                normalized["ring_metrics"].get("ring_0_5_ft")
                or normalized["ring_metrics"].get("zone_0_5_ft")
                or {}
            )
            ring5 = (
                normalized["ring_metrics"].get("ring_5_30_ft")
                or normalized["ring_metrics"].get("zone_5_30_ft")
                or {}
            )
        veg_density_0_5 = _safe_float(normalized.get("near_structure_vegetation_0_5_pct"))
        if veg_density_0_5 is None:
            veg_density_0_5 = _safe_float((ring0 or {}).get("vegetation_density"))
        veg_density_5_30 = _safe_float(normalized.get("near_structure_vegetation_5_30_pct"))
        if veg_density_5_30 is None:
            veg_density_5_30 = _safe_float((ring5 or {}).get("vegetation_density"))
        canopy_overlap = _safe_float(normalized.get("canopy_adjacency_proxy_pct"))
        if canopy_overlap is None:
            canopy_overlap = _safe_float((ring0 or {}).get("imagery_canopy_proxy_pct"))
        if canopy_overlap is None:
            canopy_overlap = _safe_float((ring0 or {}).get("coverage_pct"))
        hardscape_ratio = _safe_float((ring0 or {}).get("imagery_impervious_low_fuel_pct"))
        if hardscape_ratio is None and veg_density_0_5 is not None:
            hardscape_ratio = round(max(0.0, min(100.0, 100.0 - veg_density_0_5)), 1)
        geometry_type = (
            str(((normalized.get("ring_metrics") or {}).get("geometry_type") or "")).strip().lower()
            if isinstance(normalized.get("ring_metrics"), dict)
            else ""
        )
        if geometry_type not in {"footprint", "point"}:
            geometry_type = "footprint" if bool(normalized.get("footprint_used")) else "point"
        precision_flag = (
            str(((normalized.get("ring_metrics") or {}).get("precision_flag") or "")).strip().lower()
            if isinstance(normalized.get("ring_metrics"), dict)
            else ""
        )
        if not precision_flag:
            precision_flag = "footprint_relative" if geometry_type == "footprint" else "fallback_point_proxy"
        imagery_available = bool(normalized.get("naip_feature_source"))
        confidence_flag = (
            "high"
            if (imagery_available and geometry_type == "footprint")
            else ("moderate" if imagery_available else "low")
        )
        normalized["near_structure_features"] = {
            "veg_density_0_5": veg_density_0_5,
            "veg_density_5_30": veg_density_5_30,
            "canopy_overlap": canopy_overlap,
            "hardscape_ratio": hardscape_ratio,
            "geometry_type": geometry_type,
            "precision_flag": precision_flag,
            "imagery_available": imagery_available,
            "confidence_flag": confidence_flag,
            "source": "naip_imagery" if imagery_available else "fallback_layers",
        }
    directional_risk = normalized.get("directional_risk")
    if isinstance(directional_risk, dict):
        normalized["directional_risk"] = dict(directional_risk)
    else:
        normalized["directional_risk"] = {}
    structure_relative_slope = normalized.get("structure_relative_slope")
    default_precision_flag = "footprint_relative" if bool(normalized.get("footprint_used")) else "fallback_point_proxy"
    default_confidence_flag = "high" if default_precision_flag == "footprint_relative" else "low"
    if isinstance(structure_relative_slope, dict):
        normalized_slope = dict(structure_relative_slope)
    else:
        normalized_slope = {}
    normalized_slope.setdefault("local_slope", None)
    normalized_slope.setdefault("slope_within_30_ft", None)
    normalized_slope.setdefault("uphill_gradient_deg", None)
    normalized_slope.setdefault("downhill_gradient_deg", None)
    normalized_slope.setdefault("uphill_exposure", None)
    normalized_slope.setdefault("downhill_buffer", None)
    normalized_slope.setdefault("precision_flag", default_precision_flag)
    normalized_slope.setdefault("confidence_flag", default_confidence_flag)
    normalized_slope.setdefault("source", "unavailable")
    normalized["structure_relative_slope"] = normalized_slope
    raw_structure_attributes = normalized.get("structure_attributes")
    if isinstance(raw_structure_attributes, dict):
        area_row = raw_structure_attributes.get("area")
        density_row = raw_structure_attributes.get("density_context")
        shape_row = raw_structure_attributes.get("shape_complexity")
        provenance = raw_structure_attributes.get("provenance")
        normalized["structure_attributes"] = {
            "area": {
                "sqft": _safe_float((area_row or {}).get("sqft")) if isinstance(area_row, dict) else None,
                "source": (
                    str((area_row or {}).get("source")).strip() if isinstance(area_row, dict) and str((area_row or {}).get("source") or "").strip() else None
                ),
            },
            "density_context": {
                "index": _safe_float((density_row or {}).get("index")) if isinstance(density_row, dict) else None,
                "tier": (
                    str((density_row or {}).get("tier") or "unknown").strip().lower()
                    if isinstance(density_row, dict)
                    else "unknown"
                ),
                "nearby_structure_count_100_ft": (
                    _safe_float((density_row or {}).get("nearby_structure_count_100_ft"))
                    if isinstance(density_row, dict)
                    else None
                ),
                "nearby_structure_count_300_ft": (
                    _safe_float((density_row or {}).get("nearby_structure_count_300_ft"))
                    if isinstance(density_row, dict)
                    else None
                ),
                "nearest_structure_distance_ft": (
                    _safe_float((density_row or {}).get("nearest_structure_distance_ft"))
                    if isinstance(density_row, dict)
                    else None
                ),
                "source": (
                    str((density_row or {}).get("source")).strip()
                    if isinstance(density_row, dict) and str((density_row or {}).get("source") or "").strip()
                    else None
                ),
            },
            "estimated_age_proxy": (
                {
                    "proxy_year": _safe_float((raw_structure_attributes.get("estimated_age_proxy") or {}).get("proxy_year")),
                    "era_bucket": (
                        str((raw_structure_attributes.get("estimated_age_proxy") or {}).get("era_bucket"))
                        if str((raw_structure_attributes.get("estimated_age_proxy") or {}).get("era_bucket") or "").strip()
                        else None
                    ),
                }
                if isinstance(raw_structure_attributes.get("estimated_age_proxy"), dict)
                else None
            ),
            "shape_complexity": {
                "index": _safe_float((shape_row or {}).get("index")) if isinstance(shape_row, dict) else None,
                "source": (
                    str((shape_row or {}).get("source")).strip()
                    if isinstance(shape_row, dict) and str((shape_row or {}).get("source") or "").strip()
                    else None
                ),
            },
            "provenance": {
                "area": (
                    str((provenance or {}).get("area") or "unavailable")
                    if isinstance(provenance, dict)
                    else "unavailable"
                ),
                "density_context": (
                    str((provenance or {}).get("density_context") or "unavailable")
                    if isinstance(provenance, dict)
                    else "unavailable"
                ),
                "estimated_age_proxy": (
                    str((provenance or {}).get("estimated_age_proxy") or "unavailable")
                    if isinstance(provenance, dict)
                    else "unavailable"
                ),
                "shape_complexity": (
                    str((provenance or {}).get("shape_complexity") or "unavailable")
                    if isinstance(provenance, dict)
                    else "unavailable"
                ),
            },
        }
    else:
        normalized["structure_attributes"] = {
            "area": {"sqft": None, "source": None},
            "density_context": {
                "index": None,
                "tier": "unknown",
                "nearby_structure_count_100_ft": None,
                "nearby_structure_count_300_ft": None,
                "nearest_structure_distance_ft": None,
                "source": None,
            },
            "estimated_age_proxy": None,
            "shape_complexity": {"index": None, "source": None},
            "provenance": {
                "area": "unavailable",
                "density_context": "unavailable",
                "estimated_age_proxy": "unavailable",
                "shape_complexity": "unavailable",
            },
        }
    return normalized


def _build_geometry_resolution_summary(
    property_level_context: dict[str, Any],
) -> GeometryResolutionSummary:
    ctx = property_level_context if isinstance(property_level_context, dict) else {}

    anchor_source = str(ctx.get("property_anchor_source") or "geocoded_address_point").strip().lower()
    if not anchor_source:
        anchor_source = "geocoded_address_point"

    raw_anchor_quality = (
        ctx.get("property_anchor_quality_score")
        if ctx.get("property_anchor_quality_score") is not None
        else ctx.get("anchor_quality_score")
    )
    try:
        anchor_quality_score = float(raw_anchor_quality if raw_anchor_quality is not None else 0.0)
    except (TypeError, ValueError):
        anchor_quality_score = 0.0
    if anchor_quality_score > 1.0 and anchor_quality_score <= 100.0:
        anchor_quality_score = anchor_quality_score / 100.0
    anchor_quality_score = round(max(0.0, min(1.0, anchor_quality_score)), 3)

    parcel_match_status = "not_found"
    parcel_resolution = ctx.get("parcel_resolution")
    if isinstance(parcel_resolution, dict):
        status = str(parcel_resolution.get("status") or "").strip().lower()
        if status in {"matched", "multiple_candidates", "not_found"}:
            parcel_match_status = status
        elif status in {"provider_unavailable", "lookup_unavailable"}:
            parcel_match_status = "provider_unavailable"
    if parcel_match_status == "not_found":
        parcel_lookup_method = str(ctx.get("parcel_lookup_method") or "").strip().lower()
        if ctx.get("parcel_id"):
            parcel_match_status = "matched"
        elif isinstance(ctx.get("parcel_geometry"), dict):
            parcel_match_status = "matched"
        elif parcel_lookup_method in {"contains_point", "nearest_within_tolerance"}:
            parcel_match_status = "matched"
        elif parcel_lookup_method in {"multiple_candidates"}:
            parcel_match_status = "multiple_candidates"
        elif parcel_lookup_method in {"provider_unavailable", "lookup_unavailable"}:
            parcel_match_status = "provider_unavailable"
        elif parcel_lookup_method:
            parcel_match_status = "not_found"
        elif ctx.get("parcel_source") or ctx.get("parcel_source_name"):
            parcel_match_status = "not_found"

    footprint_match_status = str(ctx.get("structure_match_status") or "none").strip().lower()
    if not footprint_match_status:
        footprint_match_status = "none"
    if footprint_match_status not in {"matched", "none", "ambiguous", "provider_unavailable", "error"}:
        footprint_match_status = "matched" if bool(ctx.get("footprint_used")) else "none"

    footprint_source = (
        str(
            ctx.get("footprint_source_name")
            or ctx.get("building_source")
            or ctx.get("footprint_source")
            or ""
        ).strip()
        or None
    )

    ring_generation_mode = str(ctx.get("ring_generation_mode") or "").strip().lower()
    if ring_generation_mode not in {"footprint_aware_rings", "point_annulus_fallback", "point_annulus_parcel_clipped"}:
        ring_generation_mode = "footprint_aware_rings" if bool(ctx.get("footprint_used")) else "point_annulus_fallback"

    has_near_structure_values = any(
        ctx.get(key) is not None
        for key in (
            "near_structure_vegetation_0_5_pct",
            "near_structure_vegetation_5_30_pct",
            "canopy_adjacency_proxy_pct",
            "vegetation_continuity_proxy_pct",
            "nearest_high_fuel_patch_distance_ft",
        )
    )
    naip_status = "missing"
    naip_feature_source = str(ctx.get("naip_feature_source") or "").strip()
    if has_near_structure_values and naip_feature_source:
        naip_status = "observed"
    elif has_near_structure_values:
        naip_status = "fallback_or_proxy"
    layer_rows = ctx.get("layer_coverage_audit")
    if isinstance(layer_rows, list):
        for row in layer_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("layer_key") or "").strip().lower() != "naip_structure_features":
                continue
            coverage_status = str(row.get("coverage_status") or "").strip().lower()
            if coverage_status in {"observed", "partial"} and naip_status != "observed":
                naip_status = "fallback_or_proxy" if has_near_structure_values else "present_but_not_consumed"
            elif coverage_status in {"missing", "not_configured", "not_covered"}:
                naip_status = "missing"
            elif coverage_status in {"provider_unavailable", "fetch_failed", "error"}:
                naip_status = "provider_unavailable"
            break

    property_linkage = ctx.get("property_linkage") if isinstance(ctx.get("property_linkage"), dict) else {}
    footprint_outside_parcel = bool(property_linkage.get("footprint_outside_parcel"))
    multiple_footprints_on_parcel = bool(property_linkage.get("multiple_footprints_on_parcel"))
    geocode_to_anchor_distance_m = _safe_float(ctx.get("geocode_to_anchor_distance_m"))
    structure_match_distance_m = _safe_float(ctx.get("structure_match_distance_m"))
    source_conflict_flag = bool(ctx.get("source_conflict_flag"))

    mismatch_reasons: list[str] = []
    if geocode_to_anchor_distance_m is not None and geocode_to_anchor_distance_m >= 35.0:
        mismatch_reasons.append(
            f"Geocoded address point is {geocode_to_anchor_distance_m:.1f} m from the analyzed property anchor."
        )
    if (
        footprint_match_status == "matched"
        and structure_match_distance_m is not None
        and structure_match_distance_m >= 30.0
    ):
        mismatch_reasons.append(
            f"Matched footprint is {structure_match_distance_m:.1f} m from the selected anchor point."
        )
    if footprint_outside_parcel:
        mismatch_reasons.append("Matched footprint does not align with the matched parcel boundary.")
    if parcel_match_status == "multiple_candidates":
        mismatch_reasons.append("Parcel resolution returned multiple plausible parcels.")
    if parcel_match_status in {"not_found", "provider_unavailable"} and footprint_match_status == "matched":
        mismatch_reasons.append("Footprint matched but parcel linkage is unresolved.")
    if multiple_footprints_on_parcel and footprint_match_status in {"matched", "ambiguous"}:
        mismatch_reasons.append("Multiple structures exist on the parcel and structure selection may be ambiguous.")
    if source_conflict_flag:
        mismatch_reasons.append("Anchor and geometry sources are materially misaligned.")
    property_mismatch_flag = bool(mismatch_reasons)
    mismatch_reason = mismatch_reasons[0] if mismatch_reasons else None

    geometry_limitations: list[str] = []
    if anchor_quality_score < 0.60:
        geometry_limitations.append(
            "Anchor quality is low; property location may be approximate."
        )
    if parcel_match_status != "matched":
        if parcel_match_status == "provider_unavailable":
            geometry_limitations.append(
                "Parcel geometry provider was unavailable."
            )
        elif parcel_match_status == "multiple_candidates":
            geometry_limitations.append(
                "Parcel lookup returned multiple plausible candidates."
            )
        else:
            geometry_limitations.append(
                "Parcel geometry was not matched."
            )
    if footprint_match_status != "matched":
        if footprint_match_status == "ambiguous":
            geometry_limitations.append(
                "Multiple nearby footprint candidates were similarly plausible."
            )
        elif footprint_match_status == "provider_unavailable":
            geometry_limitations.append(
                "Building-footprint provider was unavailable."
            )
        elif footprint_match_status == "error":
            geometry_limitations.append(
                "Building-footprint lookup returned an error."
            )
        else:
            geometry_limitations.append(
                "Building footprint was not matched."
            )
    if ring_generation_mode in {"point_annulus_fallback", "point_annulus_parcel_clipped"}:
        if ring_generation_mode == "point_annulus_parcel_clipped":
            geometry_limitations.append(
                "Near-structure rings were generated from point-based annulus sampling clipped to parcel boundaries."
            )
        else:
            geometry_limitations.append(
                "Near-structure rings were generated from point-based annulus fallback."
            )
    if naip_status in {"missing", "provider_unavailable", "present_but_not_consumed"}:
        geometry_limitations.append(
            "NAIP-derived near-structure vegetation features were unavailable."
        )
    elif naip_status == "fallback_or_proxy":
        geometry_limitations.append(
            "Near-structure vegetation features rely on proxy/fallback signals."
        )
    if source_conflict_flag:
        geometry_limitations.append(
            "Anchor and geometry sources were partially conflicting."
        )
    if property_mismatch_flag and mismatch_reason:
        geometry_limitations.append(
            f"Potential property mismatch detected: {mismatch_reason}"
        )
    geometry_limitations = list(dict.fromkeys(geometry_limitations))[:8]

    return GeometryResolutionSummary(
        anchor_source=anchor_source,
        anchor_quality_score=anchor_quality_score,
        parcel_match_status=parcel_match_status,
        footprint_match_status=footprint_match_status,
        footprint_source=footprint_source,
        ring_generation_mode=ring_generation_mode,
        naip_structure_feature_status=naip_status,
        property_mismatch_flag=property_mismatch_flag,
        mismatch_reason=mismatch_reason,
        geometry_limitations=geometry_limitations,
    )


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


def _coerce_region_readiness(value: object) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        # Legacy contexts may not include explicit region readiness metadata.
        return "property_specific_ready"
    if candidate in REGION_PROPERTY_SPECIFIC_READINESS_ORDER:
        return candidate
    return "limited_regional_ready"


def _region_readiness_penalty_summary(property_level_context: dict[str, Any]) -> dict[str, Any]:
    readiness = _coerce_region_readiness(property_level_context.get("region_property_specific_readiness"))
    required_missing = list(property_level_context.get("region_required_layers_missing") or [])
    optional_missing = list(property_level_context.get("region_optional_layers_missing") or [])
    enrichment_missing = list(property_level_context.get("region_enrichment_layers_missing") or [])
    missing_reason = property_level_context.get("region_missing_reason_by_layer")
    return {
        "region_property_specific_readiness": readiness,
        "region_required_missing_count": len(required_missing),
        "region_optional_missing_count": len(optional_missing),
        "region_enrichment_missing_count": len(enrichment_missing),
        "region_required_layers_missing": [str(v) for v in required_missing if str(v).strip()],
        "region_optional_layers_missing": [str(v) for v in optional_missing if str(v).strip()],
        "region_enrichment_layers_missing": [str(v) for v in enrichment_missing if str(v).strip()],
        "region_missing_reason_by_layer": dict(missing_reason) if isinstance(missing_reason, dict) else {},
    }


def _property_data_confidence_level(score: float) -> str:
    if score >= 70.0:
        return "high"
    if score >= 45.0:
        return "medium"
    return "low"


def _build_property_confidence_summary(
    *,
    payload: AddressRequest | None,
    property_level_context: dict[str, Any],
    fallback_evidence_fraction: float,
    fallback_dominance_ratio: float,
) -> dict[str, Any]:
    parcel_resolution = (
        property_level_context.get("parcel_resolution")
        if isinstance(property_level_context.get("parcel_resolution"), dict)
        else {}
    )
    footprint_resolution = (
        property_level_context.get("footprint_resolution")
        if isinstance(property_level_context.get("footprint_resolution"), dict)
        else {}
    )
    structure_attributes = (
        property_level_context.get("structure_attributes")
        if isinstance(property_level_context.get("structure_attributes"), dict)
        else {}
    )
    parcel_status = str(parcel_resolution.get("status") or "").strip().lower()
    footprint_status = str(footprint_resolution.get("match_status") or "").strip().lower()
    fallback_mode = str(property_level_context.get("fallback_mode") or "").strip().lower()

    try:
        raw_parcel_confidence = float(parcel_resolution.get("confidence") or 0.0)
    except (TypeError, ValueError):
        raw_parcel_confidence = 0.0
    if raw_parcel_confidence > 0.0:
        parcel_confidence = raw_parcel_confidence
    elif parcel_status == "matched" or property_level_context.get("parcel_id"):
        parcel_confidence = 74.0
    elif parcel_status == "multiple_candidates":
        parcel_confidence = 46.0
    elif parcel_status in {"not_found", "none"}:
        parcel_confidence = 25.0
    else:
        # Unknown parcel status is treated as neutral, not hard-failed.
        parcel_confidence = 60.0

    try:
        raw_footprint_confidence = float(footprint_resolution.get("confidence_score") or 0.0) * 100.0
    except (TypeError, ValueError):
        raw_footprint_confidence = 0.0
    if raw_footprint_confidence > 0.0:
        footprint_confidence = raw_footprint_confidence
    elif footprint_status in {"matched", "selected", "trusted"} or bool(property_level_context.get("footprint_used")):
        footprint_confidence = 82.0
    elif footprint_status in {"none", "not_found", "ambiguous", "provider_unavailable", "error"}:
        footprint_confidence = 24.0
    elif fallback_mode == "point_based":
        footprint_confidence = 36.0
    else:
        # Unknown footprint status is treated as neutral, not hard-failed.
        footprint_confidence = 58.0

    area_sqft = _safe_float(((structure_attributes.get("area") or {}).get("sqft")))
    density_index = _safe_float(((structure_attributes.get("density_context") or {}).get("index")))
    age_proxy_year = _safe_float(((structure_attributes.get("estimated_age_proxy") or {}).get("proxy_year")))
    shape_index = _safe_float(((structure_attributes.get("shape_complexity") or {}).get("index")))
    structure_available_count = sum(
        1
        for value in [area_sqft, density_index, age_proxy_year, shape_index]
        if value is not None
    )
    structure_attributes_score = (float(structure_available_count) / 4.0) * 100.0

    user_inputs_score = 0.0
    user_inputs_provided = 0
    if payload is not None:
        attrs = payload.attributes if isinstance(payload.attributes, PropertyAttributes) else PropertyAttributes()
        provided = [
            attrs.roof_type,
            attrs.vent_type,
            attrs.window_type,
            attrs.defensible_space_ft,
            attrs.construction_year,
            attrs.siding_type,
        ]
        user_inputs_provided = sum(1 for value in provided if value not in {None, ""})
        geometry_user_input = bool(
            payload.property_anchor_point is not None
            or payload.user_selected_point is not None
            or payload.selected_structure_geometry is not None
            or payload.selected_structure_id is not None
        )
        user_inputs_score = ((float(user_inputs_provided) + (1.0 if geometry_user_input else 0.0)) / 7.0) * 100.0

    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context.get("ring_metrics"), dict) else {}
    near_structure_ring_count = sum(
        1
        for key in ("ring_0_5_ft", "zone_0_5_ft", "ring_5_30_ft", "zone_5_30_ft", "ring_30_100_ft", "zone_30_100_ft")
        if isinstance(rings.get(key), dict) and _safe_float((rings.get(key) or {}).get("vegetation_density")) is not None
    )
    if near_structure_ring_count >= 2:
        near_structure_evidence_score = 90.0
    elif near_structure_ring_count == 1:
        near_structure_evidence_score = 72.0
    elif fallback_mode == "point_based":
        near_structure_evidence_score = 30.0
    else:
        near_structure_evidence_score = 55.0

    combined_structure_score = min(
        100.0,
        0.45 * structure_attributes_score + 0.55 * user_inputs_score,
    )
    if combined_structure_score <= 0.0:
        combined_structure_score = 32.0 if fallback_mode == "point_based" else 48.0

    score = (
        0.22 * max(0.0, min(100.0, parcel_confidence))
        + 0.28 * max(0.0, min(100.0, footprint_confidence))
        + 0.25 * max(0.0, min(100.0, combined_structure_score))
        + 0.25 * max(0.0, min(100.0, near_structure_evidence_score))
    )
    if fallback_mode == "point_based" and near_structure_ring_count == 0:
        score -= 8.0
    if fallback_evidence_fraction >= 0.65 or fallback_dominance_ratio >= 0.75:
        score = min(score, 42.0)
    elif fallback_evidence_fraction >= 0.50 or fallback_dominance_ratio >= 0.60:
        score = min(score, 55.0)
    if not bool(property_level_context.get("footprint_used")) and parcel_confidence < 45.0 and near_structure_ring_count == 0:
        score = min(score, 35.0)

    key_gaps: list[str] = []
    if parcel_status in {"not_found", "multiple_candidates"} or parcel_confidence < 60.0:
        key_gaps.append("Parcel match is missing or low-confidence.")
    if footprint_status in {"none", "ambiguous", "provider_unavailable", "error"} or footprint_confidence < 65.0:
        key_gaps.append("Building footprint match is missing or low-confidence.")
    if structure_available_count < 2:
        key_gaps.append("Structure attributes are limited; more property-specific structure detail is needed.")
    if payload is not None and user_inputs_provided < 2:
        key_gaps.append("Few homeowner-provided structure details were supplied.")
    if fallback_evidence_fraction >= 0.50 or fallback_dominance_ratio >= 0.60:
        key_gaps.append("Fallback-heavy evidence reduces property-level confidence.")

    score = round(max(0.0, min(100.0, score)), 1)
    return {
        "score": score,
        "level": _property_data_confidence_level(score),
        "key_gaps": key_gaps[:4],
    }


def _build_feature_coverage_preflight(
    *,
    context: WildfireContext,
    property_level_context: dict[str, Any],
    coverage_summary: LayerCoverageSummary,
    payload: AddressRequest | None = None,
) -> dict[str, Any]:
    rings = property_level_context.get("ring_metrics") if isinstance(property_level_context, dict) else None
    rings = rings if isinstance(rings, dict) else {}
    near_structure_ring_available = any(
        isinstance(rings.get(key), dict)
        and (rings.get(key) or {}).get("vegetation_density") is not None
        for key in ["ring_0_5_ft", "zone_0_5_ft", "ring_5_30_ft", "zone_5_30_ft", "ring_30_100_ft", "zone_30_100_ft"]
    )
    near_structure_proxy_available = any(
        _safe_float(property_level_context.get(field)) is not None
        for field in [
            "near_structure_vegetation_0_5_pct",
            "canopy_adjacency_proxy_pct",
            "vegetation_continuity_proxy_pct",
        ]
    )
    parcel_polygon_available = isinstance(property_level_context.get("parcel_geometry"), dict)
    footprint_available = bool(property_level_context.get("footprint_used"))
    hazard_available = getattr(context, "hazard_severity_index", None) is not None
    burn_prob_available = getattr(context, "burn_probability_index", None) is not None
    dryness_available = getattr(context, "moisture_index", None) is not None
    access_context = getattr(context, "access_context", {}) or {}
    road_network_available = (
        getattr(context, "access_exposure_index", None) is not None
        or str(access_context.get("status") or "") in {"ok", "partial"}
    )
    near_structure_available = near_structure_ring_available or near_structure_proxy_available
    region_readiness = _region_readiness_penalty_summary(property_level_context)
    region_readiness_state = str(region_readiness.get("region_property_specific_readiness") or "limited_regional_ready")
    feature_bundle_summary = (
        property_level_context.get("feature_bundle_summary")
        if isinstance(property_level_context.get("feature_bundle_summary"), dict)
        else {}
    )
    bundle_metrics = (
        feature_bundle_summary.get("coverage_metrics")
        if isinstance(feature_bundle_summary.get("coverage_metrics"), dict)
        else {}
    )

    feature_coverage_summary = {
        "parcel_polygon_available": parcel_polygon_available,
        "building_footprint_available": footprint_available,
        "hazard_severity_available": hazard_available,
        "burn_probability_available": burn_prob_available,
        "dryness_available": dryness_available,
        "road_network_available": road_network_available,
        "near_structure_vegetation_available": near_structure_available,
    }
    observed_count = sum(1 for available in feature_coverage_summary.values() if available)
    total_count = max(1, len(feature_coverage_summary))
    feature_coverage_percent = round((observed_count / float(total_count)) * 100.0, 1)
    missing_core_layer_count = total_count - observed_count
    observed_feature_count = int(bundle_metrics.get("observed_feature_count") or 0)
    inferred_feature_count = int(bundle_metrics.get("inferred_feature_count") or 0)
    fallback_feature_count = int(bundle_metrics.get("fallback_feature_count") or 0)
    missing_feature_count = int(bundle_metrics.get("missing_feature_count") or 0)
    feature_count_total = max(
        1,
        observed_feature_count + inferred_feature_count + fallback_feature_count + missing_feature_count,
    )
    fallback_evidence_fraction = float(fallback_feature_count) / float(feature_count_total)
    observed_evidence_fraction = float(observed_feature_count) / float(feature_count_total)
    bundle_metrics_present = bool(bundle_metrics)
    observed_weight_fraction = float(bundle_metrics.get("observed_weight_fraction") or 0.0)
    fallback_dominance_ratio = float(bundle_metrics.get("fallback_dominance_ratio") or 0.0)
    geometry_quality_score = float(
        bundle_metrics.get("structure_geometry_quality_score")
        if bundle_metrics.get("structure_geometry_quality_score") is not None
        else (0.92 if footprint_available else (0.72 if parcel_polygon_available else 0.46))
    )
    environmental_layer_coverage_score = float(
        bundle_metrics.get("environmental_layer_coverage_score")
        if bundle_metrics.get("environmental_layer_coverage_score") is not None
        else (
            (
                sum(1 for available in [hazard_available, burn_prob_available, dryness_available] if available)
                / 3.0
            )
            * 100.0
        )
    )
    regional_enrichment_consumption_score = float(
        bundle_metrics.get("regional_enrichment_consumption_score")
        if bundle_metrics.get("regional_enrichment_consumption_score") is not None
        else environmental_layer_coverage_score
    )
    property_specificity_score = float(
        bundle_metrics.get("property_specificity_score")
        if bundle_metrics.get("property_specificity_score") is not None
        else (
            88.0
            if (footprint_available and near_structure_available)
            else (72.0 if footprint_available else (62.0 if parcel_polygon_available else 42.0))
        )
    )
    geometry_basis = (
        "footprint"
        if footprint_available
        else ("parcel" if parcel_polygon_available else "geocode_point")
    )
    property_confidence_summary = _build_property_confidence_summary(
        payload=payload,
        property_level_context=property_level_context,
        fallback_evidence_fraction=fallback_evidence_fraction,
        fallback_dominance_ratio=fallback_dominance_ratio,
    )
    property_data_confidence = float(property_confidence_summary.get("score") or 0.0)
    property_mismatch_flag = bool(property_level_context.get("property_mismatch_flag"))
    mismatch_reason = str(property_level_context.get("mismatch_reason") or "").strip() or None
    major_environmental_missing_count = sum(
        1
        for available in [hazard_available, burn_prob_available, dryness_available]
        if not available
    )

    if (
        feature_coverage_percent >= 70.0
        and footprint_available
        and near_structure_available
        and major_environmental_missing_count == 0
        and coverage_summary.failed_count <= 1
    ):
        assessment_specificity_tier = "property_specific"
    elif (
        feature_coverage_percent >= 38.0
        and major_environmental_missing_count <= 1
        and (not bundle_metrics_present or fallback_dominance_ratio < 0.72)
    ):
        assessment_specificity_tier = "address_level"
    else:
        assessment_specificity_tier = "regional_estimate"

    if region_readiness_state == "limited_regional_ready":
        assessment_specificity_tier = "regional_estimate"
    elif region_readiness_state == "address_level_only" and assessment_specificity_tier == "property_specific":
        assessment_specificity_tier = "address_level"

    # If both parcel and footprint are missing, never allow property-specific semantics.
    if not footprint_available and not parcel_polygon_available and assessment_specificity_tier == "property_specific":
        assessment_specificity_tier = "address_level"
    if bundle_metrics_present and geometry_quality_score and geometry_quality_score < 0.62 and assessment_specificity_tier == "property_specific":
        assessment_specificity_tier = "address_level"
    if bundle_metrics_present and regional_enrichment_consumption_score < 60.0 and assessment_specificity_tier == "property_specific":
        assessment_specificity_tier = "address_level"
    if bundle_metrics_present and regional_enrichment_consumption_score < 45.0:
        assessment_specificity_tier = "regional_estimate"
    if bundle_metrics_present and fallback_dominance_ratio >= 0.80:
        assessment_specificity_tier = "regional_estimate"
    if property_data_confidence < 30.0:
        assessment_specificity_tier = "regional_estimate"
    elif property_data_confidence < 50.0 and assessment_specificity_tier == "property_specific":
        assessment_specificity_tier = "address_level"
    if property_mismatch_flag:
        if assessment_specificity_tier == "property_specific":
            assessment_specificity_tier = "address_level"
        else:
            assessment_specificity_tier = "regional_estimate"

    limited_assessment_flag = (
        assessment_specificity_tier != "property_specific"
        or missing_core_layer_count >= 3
        or bool(coverage_summary.critical_missing_layers)
        or major_environmental_missing_count >= 2
        or (bundle_metrics_present and fallback_dominance_ratio >= 0.70)
        or (bundle_metrics_present and observed_weight_fraction <= 0.35)
        or (bundle_metrics_present and regional_enrichment_consumption_score < 60.0)
        or region_readiness_state != "property_specific_ready"
        or int(region_readiness.get("region_required_missing_count") or 0) > 0
        or property_data_confidence < 40.0
        or property_mismatch_flag
    )
    specificity_warning = None
    if limited_assessment_flag:
        specificity_warning = (
            "Assessment uses partial/fallback evidence. Missing data reduces specificity and confidence "
            "instead of being treated as a property-specific observation."
        )
        if region_readiness_state != "property_specific_ready":
            specificity_warning = (
                "Prepared region data is not property-specific-ready for this location. "
                "Assessment is intentionally downgraded to lower-specificity guidance."
            )
        if property_mismatch_flag:
            specificity_warning = (
                "Geometry inputs may refer to the wrong property; specificity is downgraded until location/building "
                "alignment is corrected."
            )

    preflight = {
        "feature_coverage_summary": feature_coverage_summary,
        "feature_coverage_percent": feature_coverage_percent,
        "assessment_specificity_tier": assessment_specificity_tier,
        "limited_assessment_flag": limited_assessment_flag,
        "missing_core_layer_count": missing_core_layer_count,
        "geometry_basis": geometry_basis,
        "major_environmental_missing_count": major_environmental_missing_count,
        "observed_feature_count": observed_feature_count,
        "inferred_feature_count": inferred_feature_count,
        "fallback_feature_count": fallback_feature_count,
        "missing_feature_count": missing_feature_count,
        "feature_count_total": feature_count_total,
        "observed_evidence_fraction": round(observed_evidence_fraction, 3),
        "fallback_evidence_fraction": round(fallback_evidence_fraction, 3),
        # Confidence should use fallback share derived from evidence availability,
        # not contribution-weighted fallback shares from risk composition.
        "fallback_weight_fraction": round(fallback_evidence_fraction, 3),
        "observed_weight_fraction": round(observed_weight_fraction, 3),
        "fallback_dominance_ratio": round(fallback_dominance_ratio, 3),
        "structure_geometry_quality_score": round(geometry_quality_score, 3),
        "geometry_quality_score": round(geometry_quality_score, 3),
        "environmental_layer_coverage_score": round(environmental_layer_coverage_score, 1),
        "regional_context_coverage_score": round(environmental_layer_coverage_score, 1),
        "regional_enrichment_consumption_score": round(regional_enrichment_consumption_score, 1),
        "property_specificity_score": round(property_specificity_score, 1),
        "property_data_confidence": round(property_data_confidence, 1),
        "property_confidence_summary": dict(property_confidence_summary),
        "property_mismatch_flag": property_mismatch_flag,
        "mismatch_reason": mismatch_reason,
        "score_specificity_warning": specificity_warning,
    }
    preflight.update(region_readiness)
    return preflight


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

    imagery_source = str(property_level_context.get("naip_feature_source") or "").strip()
    imagery_fields = [
        "near_structure_vegetation_0_5_pct",
        "canopy_adjacency_proxy_pct",
        "vegetation_continuity_proxy_pct",
        "nearest_high_fuel_patch_distance_ft",
    ]
    for field_name in imagery_fields:
        raw_value = property_level_context.get(field_name) if isinstance(property_level_context, dict) else None
        if raw_value is not None and imagery_source:
            src_type: SourceType = "observed"
            provider_status: ProviderStatus = "ok"
            details = "Observed from precomputed NAIP ring-feature artifact."
            source_name = "naip_structure_features"
        elif raw_value is not None:
            src_type = "heuristic"
            provider_status = "ok"
            details = "Derived from fallback ring context without dedicated imagery artifact."
            source_name = "ring_context_fallback"
        else:
            src_type = "missing"
            provider_status = "missing"
            details = "Imagery-derived near-structure field unavailable."
            source_name = "naip_structure_features"
        property_sources[field_name] = _metadata(
            field_name=field_name,
            source_type=src_type,
            source_name=source_name,
            provider_status=provider_status,
            source_class="open_data_derived",
            dataset_version=os.getenv("WF_LAYER_NAIP_VERSION"),
            observed_at=os.getenv("WF_LAYER_NAIP_DATE") or (now_iso if raw_value is not None else None),
            used_in_scoring=True,
            spatial_resolution_m=1.0,
            details=details,
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
    heuristic_count = sum(1 for meta in inputs if meta.source_type == "heuristic")
    inferred_equivalent_count = max(inferred_count + heuristic_count, total - direct_count)
    missing_count = sum(1 for meta in inputs if meta.source_type in LOW_QUALITY_SOURCE_TYPES)
    stale_count = sum(1 for meta in inputs if meta.freshness_status == "stale")
    current_count = sum(1 for meta in inputs if meta.freshness_status == "current")

    direct_score = round((direct_count / total) * 100.0, 1)
    inferred_score = round((inferred_equivalent_count / total) * 100.0, 1)
    missing_share = round((missing_count / total) * 100.0, 1)
    stale_share = round((stale_count / total) * 100.0, 1)

    summary = DataProvenanceSummary(
        direct_data_coverage_score=direct_score,
        inferred_data_coverage_score=inferred_score,
        missing_data_share=missing_share,
        stale_data_share=stale_share,
        heuristic_input_count=heuristic_count,
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
    geometry_basis = str(
        property_level_context.get("geometry_basis")
        or ("footprint" if footprint_used else ("parcel" if property_level_context.get("parcel_id") else "point"))
    )
    weak_structure_readiness_evidence = (
        geometry_basis == "point"
        and not footprint_used
        and not ring_signal
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
    if weak_structure_readiness_evidence:
        readiness_caveats.append("Structure geometry evidence is weak; readiness is downgraded to avoid false precision.")
        if known_structure_count < 2:
            readiness_blockers.append("Weak structure geometry evidence with limited confirmed structure facts")

    readiness_status: EligibilityStatus
    if (
        site_status == "full"
        and home_status == "full"
        and known_structure_count >= 2
        and not weak_structure_readiness_evidence
    ):
        readiness_status = "full"
    elif (
        site_status != "insufficient"
        and home_status != "insufficient"
        and known_structure_count >= 1
        and not (weak_structure_readiness_evidence and known_structure_count < 2)
    ):
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


def _apply_preflight_specificity_gate(
    *,
    site_eligibility: ScoreEligibility,
    home_eligibility: ScoreEligibility,
    readiness_eligibility: ScoreEligibility,
    assessment_status: AssessmentStatus,
    assessment_blockers: list[str],
    preflight: dict[str, Any],
    fallback_dominance_ratio: float,
    observed_weight_fraction: float,
) -> tuple[ScoreEligibility, ScoreEligibility, ScoreEligibility, AssessmentStatus, list[str], str]:
    tier = str(preflight.get("assessment_specificity_tier") or "regional_estimate")
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    limited = bool(preflight.get("limited_assessment_flag"))
    output_state = _derive_assessment_output_state(
        preflight=preflight,
        fallback_dominance_ratio=fallback_dominance_ratio,
        observed_weight_fraction=observed_weight_fraction,
    )
    if not limited:
        return (
            site_eligibility,
            home_eligibility,
            readiness_eligibility,
            assessment_status,
            assessment_blockers,
            output_state,
        )

    site_updated = site_eligibility.model_copy(deep=True)
    home_updated = home_eligibility.model_copy(deep=True)
    readiness_updated = readiness_eligibility.model_copy(deep=True)
    blockers = list(assessment_blockers)
    pct = float(preflight.get("feature_coverage_percent") or 0.0)
    core_missing = int(preflight.get("missing_core_layer_count") or 0)
    warning = (
        f"Limited assessment specificity ({tier}, {pct:.1f}% feature coverage, {core_missing} core features missing)."
    )

    if warning not in home_updated.caveats:
        home_updated.caveats.append(warning)
    if warning not in readiness_updated.caveats:
        readiness_updated.caveats.append(warning)
    if region_readiness != "property_specific_ready":
        region_warning = (
            "Prepared region data is not property-specific-ready; specificity and confidence are intentionally capped."
        )
        if region_warning not in home_updated.caveats:
            home_updated.caveats.append(region_warning)
        if region_warning not in readiness_updated.caveats:
            readiness_updated.caveats.append(region_warning)

    if output_state == "insufficient_data":
        any_component_still_usable = bool(
            site_updated.eligible
            or home_updated.eligible
            or readiness_updated.eligible
        )
        if any_component_still_usable:
            # Preserve trustworthy partial scoring when at least one component still has defensible evidence.
            output_state = "limited_regional_estimate"
            if assessment_status == "insufficient_data":
                assessment_status = "partially_scored"
            blockers.append(
                "Coverage is too limited for full scoring; returning a constrained homeowner estimate from available components."
            )
        else:
            site_updated.eligible = False
            site_updated.eligibility_status = "insufficient"
            home_updated.eligible = False
            home_updated.eligibility_status = "insufficient"
            readiness_updated.eligible = False
            readiness_updated.eligibility_status = "insufficient"
            assessment_status = "insufficient_data"
            blockers.append("Insufficient data for a credible property or regional estimate.")
    elif tier == "regional_estimate":
        if home_updated.eligibility_status == "full":
            home_updated.eligibility_status = "partial"
            home_updated.eligible = True
        if readiness_updated.eligibility_status == "full":
            readiness_updated.eligibility_status = "partial"
            readiness_updated.eligible = True
        if assessment_status == "fully_scored":
            assessment_status = "partially_scored"
        blockers.append("Property-level specificity is limited to regional estimate due to missing core coverage.")
    elif tier == "address_level" and assessment_status == "fully_scored":
        assessment_status = "partially_scored"
        blockers.append("Assessment downgraded to address-level specificity due to partial feature coverage.")

    if region_readiness == "limited_regional_ready":
        if assessment_status == "fully_scored":
            assessment_status = "partially_scored"
        blockers.append("Prepared region is classified as limited regional readiness for property-specific scoring.")
    elif region_readiness == "address_level_only" and assessment_status == "fully_scored":
        assessment_status = "partially_scored"
        blockers.append("Prepared region supports address-level specificity only.")

    return (
        site_updated,
        home_updated,
        readiness_updated,
        assessment_status,
        sorted(set(blockers)),
        output_state,
    )


def _apply_hard_trust_guardrails(
    confidence: ConfidenceBlock,
    *,
    site_eligibility: ScoreEligibility,
    home_eligibility: ScoreEligibility,
    readiness_eligibility: ScoreEligibility,
    assessment_status: AssessmentStatus,
    coverage_summary: LayerCoverageSummary | None = None,
    preflight: dict[str, Any] | None = None,
    assessment_output_state: str | None = None,
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

    if preflight and bool(preflight.get("limited_assessment_flag")):
        tier = str(preflight.get("assessment_specificity_tier") or "regional_estimate")
        pct = float(preflight.get("feature_coverage_percent") or 0.0)
        region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
        region_required_missing_count = int(preflight.get("region_required_missing_count") or 0)
        reason = f"Coverage preflight indicates {tier} specificity ({pct:.1f}% feature coverage)."
        downgrade_reasons.append(reason)
        trust_tier_blockers.append(reason)
        if region_readiness != "property_specific_ready":
            readiness_reason = (
                f"Prepared region readiness is {region_readiness}; property-level confidence is capped."
            )
            downgrade_reasons.append(readiness_reason)
            trust_tier_blockers.append(readiness_reason)
        if tier == "regional_estimate":
            updated.confidence_tier = "low" if updated.confidence_tier != "preliminary" else "preliminary"
            updated.use_restriction = "not_for_underwriting_or_binding"
        elif updated.confidence_tier == "high":
            updated.confidence_tier = "moderate"
            if updated.use_restriction == "shareable":
                updated.use_restriction = "homeowner_review_recommended"
        if region_readiness == "limited_regional_ready":
            if updated.confidence_tier in {"high", "moderate"}:
                updated.confidence_tier = "low"
            updated.use_restriction = "not_for_underwriting_or_binding"
        if region_required_missing_count > 0:
            updated.confidence_tier = "preliminary"
            updated.use_restriction = "not_for_underwriting_or_binding"
            trust_tier_blockers.append("Prepared region reports required-layer gaps for this bbox.")

    preflight_state = str((preflight or {}).get("assessment_output_state") or "").strip()
    output_state = str(assessment_output_state or preflight_state or "").strip()
    if output_state == "address_level_estimate":
        if updated.confidence_tier == "high":
            updated.confidence_tier = "moderate"
        updated.confidence_score = min(float(updated.confidence_score), 68.0)
        updated.use_restriction = (
            "homeowner_review_recommended"
            if updated.use_restriction == "shareable"
            else updated.use_restriction
        )
    elif output_state == "limited_regional_estimate":
        if updated.confidence_tier in {"high", "moderate"}:
            updated.confidence_tier = "low"
        updated.confidence_score = min(float(updated.confidence_score), 45.0)
        updated.use_restriction = "not_for_underwriting_or_binding"
        downgrade_reasons.append("Limited regional estimate state caps confidence and sharing.")
        trust_tier_blockers.append("Limited regional estimate state.")
    elif output_state == "insufficient_data":
        updated.confidence_tier = "preliminary"
        updated.confidence_score = 0.0
        updated.use_restriction = "not_for_underwriting_or_binding"
        downgrade_reasons.append("Insufficient-data state suppresses confidence for homeowner-facing use.")
        trust_tier_blockers.append("Insufficient-data state.")

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
    property_level_context: dict[str, Any] | None = None,
    fallback_decisions: list[dict[str, object]] | None = None,
) -> AssessmentDiagnostics:
    critical_present: list[str] = []
    critical_missing: list[str] = []
    stale_inputs: list[str] = []
    inferred_inputs: list[str] = []
    heuristic_inputs: list[str] = []

    rings = (
        property_level_context.get("ring_metrics")
        if isinstance(property_level_context, dict) and isinstance(property_level_context.get("ring_metrics"), dict)
        else {}
    )
    has_ring_metrics = any(
        isinstance(rings.get(key), dict) and (rings.get(key) or {}).get("vegetation_density") is not None
        for key in ("ring_0_5_ft", "zone_0_5_ft", "ring_5_30_ft", "zone_5_30_ft", "ring_30_100_ft", "zone_30_100_ft")
    )

    for meta in data_provenance.inputs:
        if meta.field_name in CRITICAL_PROVENANCE_FIELDS:
            if has_ring_metrics and meta.field_name in NEAR_STRUCTURE_PROXY_CRITICAL_FIELDS:
                critical_present.append(meta.field_name)
                continue
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


def _derive_assessment_output_state(
    *,
    preflight: dict[str, Any],
    fallback_dominance_ratio: float,
    observed_weight_fraction: float,
) -> str:
    tier = str(preflight.get("assessment_specificity_tier") or "regional_estimate")
    feature_coverage_percent = float(preflight.get("feature_coverage_percent") or 0.0)
    missing_core_layer_count = int(preflight.get("missing_core_layer_count") or 0)
    major_environmental_missing_count = int(preflight.get("major_environmental_missing_count") or 0)
    geometry_basis = str(preflight.get("geometry_basis") or "geocode_point")
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    region_required_missing_count = int(preflight.get("region_required_missing_count") or 0)
    region_optional_missing_count = int(preflight.get("region_optional_missing_count") or 0)
    region_enrichment_missing_count = int(preflight.get("region_enrichment_missing_count") or 0)
    no_property_geometry = geometry_basis == "geocode_point"

    extreme_low_coverage = feature_coverage_percent <= 15.0
    low_coverage = feature_coverage_percent <= 30.0
    high_fallback = float(fallback_dominance_ratio) >= 0.70 and float(observed_weight_fraction) < 0.35
    severe_fallback = float(fallback_dominance_ratio) >= 0.85 and float(observed_weight_fraction) < 0.25
    low_observed_weight = float(observed_weight_fraction) < 0.45
    severe_observed_weight_loss = float(observed_weight_fraction) < 0.30

    if (
        (extreme_low_coverage and no_property_geometry and major_environmental_missing_count >= 2)
        or (missing_core_layer_count >= 6 and major_environmental_missing_count >= 2)
        or (severe_fallback and severe_observed_weight_loss and major_environmental_missing_count >= 2)
        or (region_required_missing_count > 0 and extreme_low_coverage)
    ):
        return "insufficient_data"

    if region_readiness == "limited_regional_ready":
        if severe_fallback and severe_observed_weight_loss:
            return "insufficient_data"
        return "limited_regional_estimate"

    if (
        tier == "regional_estimate"
        or low_coverage
        or high_fallback
        or major_environmental_missing_count >= 2
        or (no_property_geometry and high_fallback and major_environmental_missing_count >= 1)
    ):
        return "limited_regional_estimate"

    if region_readiness == "address_level_only":
        return "address_level_estimate"

    if tier == "address_level" or no_property_geometry:
        return "address_level_estimate"

    return "property_specific_assessment"


def _to_homeowner_assessment_mode(output_state: str) -> str:
    return OUTPUT_STATE_TO_HOMEOWNER_MODE.get(
        str(output_state or "").strip(),
        "limited_regional_estimate",
    )


def _confidence_label_for_score(score: float | None) -> str:
    if score is None:
        return "Confidence unavailable"
    value = float(score)
    if value >= 75.0:
        return "High confidence"
    if value >= 50.0:
        return "Medium confidence"
    if value > 0.0:
        return "Low confidence"
    return "Confidence unavailable"


def _friendly_layer_name(layer_key: str) -> str:
    token = str(layer_key or "").strip()
    if not token:
        return "Unknown layer"
    if token in LAYER_FRIENDLY_NAMES:
        return LAYER_FRIENDLY_NAMES[token]
    return token.replace("_", " ")


def _build_confidence_improvement_actions(
    *,
    preflight: dict[str, Any],
    property_level_context: dict[str, Any],
    geocode_meta: dict[str, Any],
    missing_inputs: list[str],
    recommended_data_improvements: list[str],
) -> list[str]:
    coverage = dict(preflight.get("feature_coverage_summary") or {})
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    geometry_basis = str(preflight.get("geometry_basis") or "geocode_point")
    geocode_tier = str(
        geocode_meta.get("final_location_confidence")
        or geocode_meta.get("confidence_tier")
        or geocode_meta.get("match_confidence")
        or ""
    ).strip().lower()
    geocode_status = str(geocode_meta.get("geocode_status") or "").strip().lower()
    region_id = str(property_level_context.get("region_id") or "").strip()
    required_missing = [str(v).strip() for v in (preflight.get("region_required_layers_missing") or []) if str(v).strip()]
    optional_missing = [str(v).strip() for v in (preflight.get("region_optional_layers_missing") or []) if str(v).strip()]
    enrichment_missing = [str(v).strip() for v in (preflight.get("region_enrichment_layers_missing") or []) if str(v).strip()]
    feature_bundle_summary = (
        property_level_context.get("feature_bundle_summary")
        if isinstance(property_level_context.get("feature_bundle_summary"), dict)
        else {}
    )
    enrichment_runtime_status = (
        feature_bundle_summary.get("enrichment_runtime_status")
        if isinstance(feature_bundle_summary.get("enrichment_runtime_status"), dict)
        else {}
    )

    actions: list[str] = []
    seen: set[str] = set()

    def _add(action: str) -> None:
        text = str(action or "").strip()
        token = text.lower()
        if not text or token in seen:
            return
        seen.add(token)
        actions.append(text)

    if geocode_status in {"low_confidence", "ambiguous_match"} or geocode_tier in {"low", "unknown"}:
        _add("Verify the address match in the map modal and place the pin on the home before running another assessment.")

    if geometry_basis == "geocode_point":
        _add("Use map point selection to anchor the property directly on the home when building polygons are missing or misaligned.")

    if not bool(coverage.get("building_footprint_available")):
        _add("Ingest building footprints for this region to unlock structure-level ring metrics and defensible-space scoring.")
    if not bool(coverage.get("near_structure_vegetation_available")):
        _add("Improve near-structure vegetation inputs (building footprint or parcel geometry) so near-home risk can be measured directly.")
    if not bool(coverage.get("hazard_severity_available")) or not bool(coverage.get("burn_probability_available")):
        _add("Prepare both hazard and burn-probability layers for this region so regional fire context is observed instead of proxied.")
    if not bool(coverage.get("dryness_available")):
        _add("Add a configured dryness source (GRIDMET or equivalent) so climate stress is sampled directly.")
    if not bool(coverage.get("road_network_available")):
        _add("Add roads/access network coverage to improve evacuation and responder access context.")

    if required_missing:
        required_names = ", ".join(_friendly_layer_name(layer) for layer in required_missing[:4])
        suffix = ", ..." if len(required_missing) > 4 else ""
        if region_id:
            _add(
                f"Prepared region '{region_id}' is missing required layers ({required_names}{suffix}); "
                "rerun region prep after sources are available."
            )
        else:
            _add(f"Required regional layers are missing ({required_names}{suffix}); rerun region prep after sources are available.")

    remaining = optional_missing + enrichment_missing
    if remaining:
        readable = ", ".join(_friendly_layer_name(layer) for layer in remaining[:4])
        suffix = ", ..." if len(remaining) > 4 else ""
        _add(
            f"Add optional/enrichment coverage for this region ({readable}{suffix}) to reduce fallback-heavy scoring."
        )

    present_not_consumed = sorted(
        [
            layer
            for layer, status in enrichment_runtime_status.items()
            if str(status or "") == "present_but_not_consumed"
        ]
    )
    if present_not_consumed:
        readable = ", ".join(_friendly_layer_name(layer) for layer in present_not_consumed[:4])
        suffix = ", ..." if len(present_not_consumed) > 4 else ""
        _add(
            f"Some enrichment layers are present but not consumed here ({readable}{suffix}); validate layer extent/CRS and runtime mappings."
        )

    for layer in remaining:
        hint = LAYER_CONFIG_ACTIONS.get(layer)
        if hint:
            _add(hint)
        if len(actions) >= 8:
            break

    missing_fact_hints = {
        "roof_type": "Provide roof material (for example, Class A vs combustible) to improve structure vulnerability accuracy.",
        "vent_type": "Provide vent protection details to improve ember intrusion scoring.",
        "defensible_space_ft": "Provide defensible-space distance to strengthen near-home ignition scoring.",
        "construction_year": "Provide construction year to improve home hardening readiness calibration.",
        "siding_type": "Provide siding material to improve flame-contact susceptibility scoring.",
        "window_type": "Provide window type to improve ember/glass vulnerability scoring.",
    }
    for field in missing_inputs:
        hint = missing_fact_hints.get(str(field))
        if hint:
            _add(hint)

    for raw in recommended_data_improvements:
        text = str(raw or "").strip()
        if not text:
            continue
        _add(f"Improve data coverage for: {text}.")
        if len(actions) >= 10:
            break

    if region_readiness == "limited_regional_ready":
        _add("Current prepared data supports only regional guidance; add optional/enrichment layers to enable stronger property-specific confidence.")
    elif region_readiness == "address_level_only":
        _add("Prepared data currently supports address-level estimates; add structure and parcel coverage for full property-specific analysis.")

    if not actions:
        _add("Data coverage is already strong; confidence gains are most likely from confirming roof, vents, and defensible-space details.")

    return actions[:10]


def _build_homeowner_confidence_summary(
    *,
    confidence_score: float,
    assessment_output_state: str,
    grouped_limitations: list[dict[str, str]],
    why_limited: str,
    feature_coverage_percent: float,
    missing_core_layer_count: int,
    geometry_basis: str,
    improvement_actions: list[str] | None = None,
) -> dict[str, Any]:
    mode = _to_homeowner_assessment_mode(assessment_output_state)
    score_value: float | None = round(float(confidence_score), 1) if confidence_score > 0.0 else None
    label = _confidence_label_for_score(score_value)
    if mode in {"limited_regional_estimate", "insufficient_data"}:
        label = "Low confidence" if mode == "limited_regional_estimate" else "Confidence unavailable"

    headline = OUTPUT_STATE_LIMITATION_TEXT.get(
        assessment_output_state,
        OUTPUT_STATE_LIMITATION_TEXT["limited_regional_estimate"],
    )
    reasons: list[str] = []
    for row in grouped_limitations:
        summary = str((row or {}).get("summary") or "").strip()
        if summary:
            reasons.append(summary)
    if not reasons and why_limited:
        reasons.append(str(why_limited))
    if feature_coverage_percent <= 35.0:
        reasons.append(
            f"Only {feature_coverage_percent:.1f}% of core assessment signals were observed directly for this run."
        )
    if missing_core_layer_count >= 4:
        reasons.append("Several core data layers were unavailable and had to be treated as missing.")
    if geometry_basis == "geocode_point":
        reasons.append("The analysis used a point anchor instead of full property geometry.")

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        token = reason.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(reason.strip())
        if len(deduped) >= 4:
            break

    return {
        "score": score_value,
        "label": label,
        "assessment_type": mode.replace("_", " "),
        "headline": headline,
        "why_confidence_is_limited": deduped,
        "how_to_improve_confidence": [str(item).strip() for item in (improvement_actions or []) if str(item).strip()][:6],
    }


def _to_homeowner_confidence_phrase(confidence_tier: str) -> str:
    normalized = str(confidence_tier or "").strip().lower()
    if normalized == "high":
        return "high confidence"
    if normalized in {"moderate", "medium"}:
        return "moderate confidence"
    return "limited confidence"


def _build_specificity_summary(
    *,
    assessment_specificity_tier: str,
    assessment_mode: str,
    limited_assessment_flag: bool,
    confidence_summary: dict[str, Any],
    trust_summary: dict[str, Any],
    property_confidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tier = str(assessment_specificity_tier or "regional_estimate").strip().lower()
    mode = str(assessment_mode or "insufficient_data").strip().lower()

    if mode == "insufficient_data":
        specificity_tier = "insufficient_data"
    elif tier in {"property_specific", "address_level", "regional_estimate"}:
        specificity_tier = tier
    else:
        specificity_tier = "regional_estimate"

    differentiation_mode = str(
        trust_summary.get("differentiation_mode") or "mostly_regional"
    ).strip().lower()
    local_differentiation_score = _safe_float(
        trust_summary.get("local_differentiation_score")
    )
    if local_differentiation_score is None:
        local_differentiation_score = _safe_float(
            trust_summary.get("neighborhood_differentiation_confidence")
        )
    local_differentiation_score = float(local_differentiation_score or 0.0)
    property_mismatch_flag = bool(trust_summary.get("property_mismatch_flag"))
    resolved_property_confidence = (
        dict(property_confidence_summary)
        if isinstance(property_confidence_summary, dict)
        else (
            dict(trust_summary.get("property_confidence_summary"))
            if isinstance(trust_summary.get("property_confidence_summary"), dict)
            else {}
        )
    )
    raw_property_confidence_score = (
        _safe_float(resolved_property_confidence.get("score"))
        if resolved_property_confidence
        else _safe_float(trust_summary.get("property_data_confidence"))
    )
    if raw_property_confidence_score is None:
        raw_property_confidence_score = 100.0
    property_data_confidence = float(raw_property_confidence_score)
    property_confidence_level = str(
        resolved_property_confidence.get("level")
        if resolved_property_confidence.get("level")
        else _property_data_confidence_level(property_data_confidence)
    ).strip().lower()
    if property_confidence_level not in {"high", "medium", "low"}:
        property_confidence_level = _property_data_confidence_level(property_data_confidence)

    # Property-specificity output should degrade when local evidence is weak.
    if (
        specificity_tier == "property_specific"
        and differentiation_mode == "mostly_regional"
    ):
        specificity_tier = "regional_estimate"
    elif (
        specificity_tier == "property_specific"
        and local_differentiation_score < 65.0
    ):
        specificity_tier = "address_level"
    if (
        specificity_tier == "property_specific"
        and local_differentiation_score < 35.0
    ):
        specificity_tier = "regional_estimate"
    if property_data_confidence < 30.0:
        specificity_tier = "regional_estimate"
    elif property_data_confidence < 50.0 and specificity_tier == "property_specific":
        specificity_tier = "address_level"
    if property_mismatch_flag:
        if specificity_tier == "property_specific":
            specificity_tier = "address_level"
        else:
            specificity_tier = "regional_estimate"

    nearby_home_guardrail = bool(trust_summary.get("nearby_home_comparison_safeguard_triggered"))
    comparison_allowed = bool(trust_summary.get("parcel_level_comparison_allowed"))
    if (
        specificity_tier == "property_specific"
        and differentiation_mode != "mostly_regional"
        and local_differentiation_score >= 60.0
        and not nearby_home_guardrail
        and property_data_confidence >= 55.0
        and property_confidence_level != "low"
    ):
        comparison_allowed = True
    if specificity_tier in {"regional_estimate", "insufficient_data"}:
        comparison_allowed = False
    if specificity_tier != "property_specific":
        comparison_allowed = False
    if nearby_home_guardrail:
        comparison_allowed = False
    if property_mismatch_flag:
        comparison_allowed = False

    confidence_level = str(trust_summary.get("confidence_level") or "").strip().lower()
    if not confidence_level:
        confidence_level = str(confidence_summary.get("assessment_type") or "").strip().lower()

    if specificity_tier == "property_specific":
        headline = "Property-specific estimate"
        what_this_means = (
            "This result uses home-specific geometry and nearby conditions. "
            "It is generally suitable for comparing nearby homes when local conditions differ."
        )
    elif specificity_tier == "address_level":
        headline = "Address-level estimate"
        what_this_means = (
            "This result uses address-level and nearby context, but some home-specific details were estimated."
        )
        if confidence_level in {"limited confidence", "low confidence"} or limited_assessment_flag:
            what_this_means += " Nearby-home comparisons should be treated as directional, not precise."
    elif specificity_tier == "regional_estimate":
        headline = "Regional estimate"
        what_this_means = (
            "This result relies more on shared neighborhood and regional conditions than on home-specific measurements, "
            "so nearby homes may appear similar."
        )
    else:
        headline = "Insufficient data for property estimate"
        what_this_means = (
            "This location does not have enough reliable home-level and regional inputs yet. "
            "Nearby homes may appear similar because the estimate relies on limited regional context."
        )

    safeguard_message = str(trust_summary.get("nearby_home_comparison_safeguard_message") or "").strip()
    if (
        specificity_tier in {"regional_estimate", "insufficient_data"}
        and safeguard_message
        and "nearby homes may appear similar" not in what_this_means.lower()
    ):
        what_this_means = f"{what_this_means} {safeguard_message}"

    return {
        "specificity_tier": specificity_tier,
        "headline": headline,
        "what_this_means": what_this_means.strip(),
        "comparison_allowed": bool(comparison_allowed),
    }


def _friendly_missing_input_name(field_name: str) -> str:
    token = str(field_name or "").strip().lower()
    mapping = {
        "roof_type": "roof type",
        "vent_type": "vent type",
        "defensible_space_ft": "defensible space condition",
        "construction_year": "construction year",
        "siding_type": "siding type",
        "window_type": "window type",
    }
    if token in mapping:
        return mapping[token]
    return token.replace("_", " ")


def _build_homeowner_trust_summary(
    *,
    confidence_tier: str,
    fallback_decisions: list[dict[str, object]],
    missing_inputs: list[str],
    preflight: dict[str, Any],
    differentiation_snapshot: dict[str, Any] | None = None,
    geometry_resolution: dict[str, Any] | None = None,
    property_confidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confidence_level = _to_homeowner_confidence_phrase(confidence_tier)

    fallback_drivers: list[str] = []
    seen_fallback: set[str] = set()
    for row in list(fallback_decisions or []):
        note = str((row or {}).get("note") or "").strip()
        if note:
            key = note.lower()
            if key not in seen_fallback:
                seen_fallback.add(key)
                fallback_drivers.append(note.rstrip("."))
        else:
            missing_input = str((row or {}).get("missing_input") or "").strip()
            if missing_input:
                fallback_label = _friendly_layer_name(missing_input.replace("_layer", ""))
                phrase = f"{fallback_label} data used fallback assumptions."
                key = phrase.lower()
                if key not in seen_fallback:
                    seen_fallback.add(key)
                    fallback_drivers.append(phrase)
        if len(fallback_drivers) >= 3:
            break

    missing_field_labels = [
        _friendly_missing_input_name(field)
        for field in list(missing_inputs or [])
        if str(field).strip()
    ]
    missing_field_labels = list(dict.fromkeys(missing_field_labels))[:4]

    coverage_gaps: list[str] = []
    missing_core_layer_count = int(preflight.get("missing_core_layer_count") or 0)
    feature_coverage_percent = float(preflight.get("feature_coverage_percent") or 0.0)
    required_missing = [
        _friendly_layer_name(str(layer))
        for layer in list(preflight.get("region_required_layers_missing") or [])
        if str(layer).strip()
    ]
    optional_missing = [
        _friendly_layer_name(str(layer))
        for layer in list(preflight.get("region_optional_layers_missing") or [])
        if str(layer).strip()
    ]
    enrichment_missing = [
        _friendly_layer_name(str(layer))
        for layer in list(preflight.get("region_enrichment_layers_missing") or [])
        if str(layer).strip()
    ]

    if missing_core_layer_count > 0:
        coverage_gaps.append(
            f"{missing_core_layer_count} core data layer(s) were unavailable in this region."
        )
    if required_missing:
        coverage_gaps.append(
            "Missing required regional layers: " + ", ".join(required_missing[:3]) + ("..." if len(required_missing) > 3 else "")
        )
    if optional_missing or enrichment_missing:
        combined = (optional_missing + enrichment_missing)[:3]
        coverage_gaps.append(
            "Additional data layers were unavailable: " + ", ".join(combined) + ("..." if (len(optional_missing) + len(enrichment_missing)) > 3 else "")
        )
    if feature_coverage_percent < 60.0:
        coverage_gaps.append(
            f"Only {feature_coverage_percent:.1f}% of assessment signals were observed directly."
        )

    uncertainty_drivers: list[str] = []
    if fallback_drivers:
        uncertainty_drivers.append("Fallback assumptions were used for one or more inputs.")
    if missing_field_labels:
        uncertainty_drivers.append("Missing property details: " + ", ".join(missing_field_labels[:3]) + ("..." if len(missing_field_labels) > 3 else ""))
    if coverage_gaps:
        uncertainty_drivers.append(coverage_gaps[0])
    uncertainty_drivers = uncertainty_drivers[:4]

    differentiation_snapshot = dict(differentiation_snapshot or {})
    differentiation_mode = str(
        differentiation_snapshot.get("differentiation_mode") or "mostly_regional"
    )
    local_differentiation_score = float(
        differentiation_snapshot.get("local_differentiation_score")
        or differentiation_snapshot.get("neighborhood_differentiation_confidence")
        or 0.0
    )
    differentiation_confidence = local_differentiation_score
    differentiation_notes = [
        str(item).strip()
        for item in list(differentiation_snapshot.get("notes") or [])
        if str(item).strip()
    ][:4]
    nearby_home_comparison_safeguard_triggered = should_trigger_nearby_home_comparison_safeguard(
        differentiation_mode,
        differentiation_confidence,
    )
    nearby_home_comparison_safeguard_message = (
        "This estimate is not precise enough to compare adjacent homes."
        if nearby_home_comparison_safeguard_triggered
        else None
    )
    if uncertainty_drivers:
        summary = f"This assessment is presented with {confidence_level}. Main uncertainty drivers are listed below."
    else:
        summary = f"This assessment is presented with {confidence_level} using mostly observed property and regional data."

    if nearby_home_comparison_safeguard_triggered:
        differentiation_summary = (
            "This estimate is not precise enough to compare adjacent homes."
        )
    elif differentiation_mode == "mostly_regional":
        differentiation_summary = "Nearby homes may receive similar results because this run relies mostly on regional context."
    elif differentiation_mode == "mixed":
        differentiation_summary = "This run combines property-specific and regional signals; nearby homes may still partially converge."
    else:
        differentiation_summary = "This run is primarily property-specific and should differentiate nearby homes when local conditions differ."
    if nearby_home_comparison_safeguard_triggered:
        uncertainty_drivers = list(
            dict.fromkeys(
                [nearby_home_comparison_safeguard_message] + list(uncertainty_drivers)
            )
        )[:4]
        differentiation_notes = list(
            dict.fromkeys(
                [nearby_home_comparison_safeguard_message] + list(differentiation_notes)
            )
        )[:4]

    low_differentiation_explanation = _build_low_differentiation_explanation_block(
        preflight=preflight,
        differentiation_mode=differentiation_mode,
        differentiation_confidence=differentiation_confidence,
        missing_field_labels=missing_field_labels,
        fallback_decisions=fallback_decisions,
        nearby_home_comparison_safeguard_triggered=nearby_home_comparison_safeguard_triggered,
    )
    geometry_resolution = geometry_resolution if isinstance(geometry_resolution, dict) else {}
    ring_mode = str(geometry_resolution.get("ring_generation_mode") or "").strip().lower()
    footprint_match_status = str(geometry_resolution.get("footprint_match_status") or "").strip().lower()
    parcel_match_status = str(geometry_resolution.get("parcel_match_status") or "").strip().lower()
    naip_status = str(geometry_resolution.get("naip_structure_feature_status") or "").strip().lower()
    property_mismatch_flag = bool(geometry_resolution.get("property_mismatch_flag"))
    mismatch_reason = str(geometry_resolution.get("mismatch_reason") or "").strip() or None
    try:
        anchor_quality_score = float(geometry_resolution.get("anchor_quality_score") or 0.0)
    except (TypeError, ValueError):
        anchor_quality_score = 0.0
    geometry_specificity_limited = bool(
        ring_mode in {"point_annulus_fallback", "point_annulus_parcel_clipped"}
        or footprint_match_status in {"none", "ambiguous", "provider_unavailable", "error"}
        or parcel_match_status in {"not_found", "provider_unavailable"}
        or anchor_quality_score < 0.60
        or naip_status in {"missing", "provider_unavailable", "present_but_not_consumed", "fallback_or_proxy"}
        or property_mismatch_flag
    )
    geometry_resolution_summary = None
    if geometry_specificity_limited:
        geometry_resolution_summary = {
            "anchor_source": str(geometry_resolution.get("anchor_source") or ""),
            "anchor_quality_score": round(max(0.0, min(1.0, anchor_quality_score)), 3),
            "parcel_match_status": parcel_match_status or "not_found",
            "footprint_match_status": footprint_match_status or "none",
            "ring_generation_mode": ring_mode or "point_annulus_fallback",
            "naip_structure_feature_status": naip_status or "missing",
            "property_mismatch_flag": property_mismatch_flag,
            "mismatch_reason": mismatch_reason,
        }
        uncertainty_drivers = list(
            dict.fromkeys(
                ["Structure geometry detail is limited, so nearby-home comparisons are less precise."]
                + list(uncertainty_drivers)
            )
        )[:4]
    if property_mismatch_flag:
        mismatch_note = (
            f"Potential property mismatch: {mismatch_reason}"
            if mismatch_reason
            else "Potential property mismatch was detected from geometry alignment checks."
        )
        uncertainty_drivers = list(dict.fromkeys([mismatch_note] + list(uncertainty_drivers)))[:4]

    resolved_property_confidence = (
        dict(property_confidence_summary)
        if isinstance(property_confidence_summary, dict)
        else (
            dict(preflight.get("property_confidence_summary"))
            if isinstance(preflight.get("property_confidence_summary"), dict)
            else {}
        )
    )
    raw_property_confidence_score = (
        _safe_float(resolved_property_confidence.get("score"))
        if resolved_property_confidence
        else _safe_float(preflight.get("property_data_confidence"))
    )
    if raw_property_confidence_score is None:
        raw_property_confidence_score = 70.0 if differentiation_mode != "mostly_regional" else 40.0
    property_data_confidence = float(raw_property_confidence_score)
    property_confidence_level = str(
        resolved_property_confidence.get("level")
        if resolved_property_confidence.get("level")
        else _property_data_confidence_level(property_data_confidence)
    ).strip().lower()
    if property_confidence_level not in {"high", "medium", "low"}:
        property_confidence_level = _property_data_confidence_level(property_data_confidence)
    property_confidence_gaps = [
        str(item).strip()
        for item in list(resolved_property_confidence.get("key_gaps") or [])
        if str(item).strip()
    ][:4]

    return {
        "confidence_level": confidence_level,
        "summary": summary,
        "uncertainty_drivers": uncertainty_drivers,
        "fallback_drivers": fallback_drivers,
        "missing_inputs": missing_field_labels,
        "coverage_gaps": coverage_gaps[:4],
        "differentiation_mode": differentiation_mode,
        "local_differentiation_score": round(local_differentiation_score, 1),
        "neighborhood_differentiation_confidence": round(differentiation_confidence, 1),
        "property_specific_feature_count": int(
            differentiation_snapshot.get("property_specific_feature_count") or 0
        ),
        "proxy_feature_count": int(differentiation_snapshot.get("proxy_feature_count") or 0),
        "defaulted_feature_count": int(
            differentiation_snapshot.get("defaulted_feature_count") or 0
        ),
        "regional_feature_count": int(
            differentiation_snapshot.get("regional_feature_count") or 0
        ),
        "differentiation_summary": differentiation_summary,
        "differentiation_notes": differentiation_notes,
        "low_differentiation_explanation": low_differentiation_explanation,
        "nearby_home_comparison_safeguard_triggered": nearby_home_comparison_safeguard_triggered,
        "nearby_home_comparison_safeguard_message": nearby_home_comparison_safeguard_message,
        "parcel_level_comparison_allowed": (
            not nearby_home_comparison_safeguard_triggered
            and differentiation_mode != "mostly_regional"
            and local_differentiation_score >= 60.0
            and property_data_confidence >= 55.0
            and property_confidence_level != "low"
        ),
        "property_data_confidence": round(property_data_confidence, 1),
        "property_confidence_summary": {
            "score": round(property_data_confidence, 1),
            "level": property_confidence_level,
            "key_gaps": property_confidence_gaps,
        },
        "property_mismatch_flag": property_mismatch_flag,
        "mismatch_reason": mismatch_reason,
        "geometry_specificity_limited": geometry_specificity_limited,
        "geometry_resolution_summary": geometry_resolution_summary,
    }


def _build_low_differentiation_explanation_block(
    *,
    preflight: dict[str, Any],
    differentiation_mode: str,
    differentiation_confidence: float,
    missing_field_labels: list[str],
    fallback_decisions: list[dict[str, object]],
    nearby_home_comparison_safeguard_triggered: bool,
) -> dict[str, Any] | None:
    mode = str(differentiation_mode or "").strip().lower()
    trigger = (
        nearby_home_comparison_safeguard_triggered
        or mode == "mostly_regional"
        or (mode == "mixed" and float(differentiation_confidence) <= 45.0)
    )
    if not trigger:
        return None

    coverage = dict(preflight.get("feature_coverage_summary") or {})
    geometry_basis = str(preflight.get("geometry_basis") or "geocode_point").strip().lower()

    footprint_available = bool(coverage.get("building_footprint_available"))
    parcel_available = bool(coverage.get("parcel_polygon_available"))
    near_structure_available = bool(coverage.get("near_structure_vegetation_available"))
    hazard_available = bool(coverage.get("hazard_severity_available"))
    burn_available = bool(coverage.get("burn_probability_available"))
    dryness_available = bool(coverage.get("dryness_available"))

    measured_directly: list[str] = []
    estimated_from_regional: list[str] = []
    make_more_property_specific: list[str] = []

    if footprint_available:
        measured_directly.append("Home shape was measured from a building footprint.")
    elif parcel_available:
        measured_directly.append("A parcel boundary was available for location context.")
    else:
        estimated_from_regional.append("Building footprint was missing, so location detail was limited.")
        make_more_property_specific.append("Add or confirm a building footprint for this home.")

    if near_structure_available and geometry_basis != "geocode_point":
        measured_directly.append("Vegetation close to the home was measured around the structure.")
    elif near_structure_available and geometry_basis == "geocode_point":
        estimated_from_regional.append("Near-home vegetation was sampled from a map point estimate.")
        make_more_property_specific.append("Pin the exact home location or add structure geometry for better near-home vegetation detail.")
    else:
        estimated_from_regional.append("Near-home vegetation detail was estimated from broader area signals.")
        make_more_property_specific.append("Add near-home vegetation detail, especially in the 0–5 ft and 5–30 ft zones.")

    if hazard_available and burn_available and dryness_available:
        measured_directly.append("Regional hazard and dryness layers were available.")
    else:
        estimated_from_regional.append("Some wildfire hazard or dryness layers were estimated from regional context.")
        make_more_property_specific.append("Use a prepared region with fuller hazard, burn history, and dryness coverage.")

    structure_fields = {"roof type", "vent type", "defensible space condition"}
    missing_structure_fields = [
        field for field in missing_field_labels if str(field).strip().lower() in structure_fields
    ]
    if missing_structure_fields:
        estimated_from_regional.append(
            "Some structure details were unknown: "
            + ", ".join(missing_structure_fields[:3])
            + ("..." if len(missing_structure_fields) > 3 else "")
            + "."
        )
        make_more_property_specific.append("Add roof type, vent type, and defensible space details.")
    else:
        structure_fallback_detected = any(
            str((row or {}).get("missing_input") or "").strip().lower()
            in {"roof_type", "vent_type", "defensible_space_ft"}
            for row in list(fallback_decisions or [])
            if isinstance(row, dict)
        )
        if structure_fallback_detected:
            estimated_from_regional.append("Some structure details were estimated from default assumptions.")
            make_more_property_specific.append("Add roof type, vent type, and defensible space details when unknown.")

    if geometry_basis == "geocode_point":
        make_more_property_specific.append("Verify the map pin is on the correct structure, not just the street address point.")

    measured_directly = list(dict.fromkeys([row.strip() for row in measured_directly if row.strip()]))[:3]
    estimated_from_regional = list(
        dict.fromkeys([row.strip() for row in estimated_from_regional if row.strip()])
    )[:4]
    make_more_property_specific = list(
        dict.fromkeys([row.strip() for row in make_more_property_specific if row.strip()])
    )[:6]

    why_similar = (
        "Nearby homes may look similar because this estimate relies more on shared neighborhood and regional conditions than on home-specific measurements."
    )
    if nearby_home_comparison_safeguard_triggered:
        why_similar = "This estimate is not precise enough to compare adjacent homes because key home-level details are still estimated."

    return {
        "applies": True,
        "what_was_measured_directly": measured_directly,
        "what_was_estimated_from_regional_context": estimated_from_regional,
        "why_nearby_properties_may_appear_similar": why_similar,
        "what_would_make_this_more_property_specific": make_more_property_specific,
    }


def _build_data_quality_summary(
    *,
    preflight: dict[str, Any],
    property_level_context: dict[str, Any],
) -> dict[str, str]:
    coverage = dict(preflight.get("feature_coverage_summary") or {})
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    geometry_basis = str(preflight.get("geometry_basis") or "geocode_point")
    parcel_available = bool(coverage.get("parcel_polygon_available"))
    footprint_available = bool(coverage.get("building_footprint_available"))
    near_structure_available = bool(coverage.get("near_structure_vegetation_available"))
    hazard_available = bool(coverage.get("hazard_severity_available"))
    burn_available = bool(coverage.get("burn_probability_available"))
    dryness_available = bool(coverage.get("dryness_available"))
    roads_available = bool(coverage.get("road_network_available"))

    property_geometry = "observed" if footprint_available else ("partial" if parcel_available else "missing")
    structure_features = "observed" if (footprint_available and near_structure_available) else (
        "partial" if near_structure_available else "missing"
    )
    vegetation_context = "observed" if near_structure_available else "partial"
    if not near_structure_available and geometry_basis == "geocode_point":
        vegetation_context = "missing"
    regional_hazard_layers = "observed" if (hazard_available and burn_available) else (
        "partial" if (hazard_available or burn_available) else "missing"
    )
    road_access_context = "observed" if roads_available else "missing"
    climate_dryness_context = "observed" if dryness_available else "missing"

    return {
        "property_geometry": property_geometry,
        "structure_features": structure_features,
        "vegetation_context": vegetation_context,
        "regional_hazard_layers": regional_hazard_layers,
        "road_access_context": road_access_context,
        "climate_dryness_context": climate_dryness_context,
        "prepared_region_readiness": region_readiness,
    }


def _build_homeowner_limitation_groups(
    *,
    preflight: dict[str, Any],
    data_quality_summary: dict[str, str],
    fallback_decisions: list[dict[str, object]],
    score_availability_notes: list[str],
    assumptions: list[str],
    assessment_output_state: str,
    fallback_dominance_ratio: float,
) -> tuple[list[dict[str, str]], list[str], list[str], list[str], str]:
    coverage = dict(preflight.get("feature_coverage_summary") or {})
    region_readiness = _coerce_region_readiness(preflight.get("region_property_specific_readiness"))
    region_optional_missing_count = int(preflight.get("region_optional_missing_count") or 0)
    region_enrichment_missing_count = int(preflight.get("region_enrichment_missing_count") or 0)
    limitations: list[dict[str, str]] = []

    def _add(category: str, summary: str) -> None:
        item = {"category": category, "summary": summary}
        if item not in limitations:
            limitations.append(item)

    if data_quality_summary.get("property_geometry") == "missing":
        _add(
            "property_shape_and_structure",
            "We could not identify the building outline or parcel boundary for this property, so near-home vegetation and defensible-space analysis was limited.",
        )
    elif data_quality_summary.get("property_geometry") == "partial":
        _add(
            "property_shape_and_structure",
            "Only partial property geometry was available, so some near-structure findings rely on approximations.",
        )

    if not bool(coverage.get("hazard_severity_available")) or not bool(coverage.get("burn_probability_available")):
        _add(
            "regional_fire_context",
            "Key wildfire hazard and burn-probability layers were unavailable for this location, so the result relies more on general regional context than property-specific fire history.",
        )

    if not bool(coverage.get("road_network_available")):
        _add(
            "access_and_response_context",
            "Road and access-network context was unavailable, so emergency access and suppression-related exposure could not be evaluated.",
        )

    if (not bool(coverage.get("near_structure_vegetation_available"))) or (not bool(coverage.get("dryness_available"))):
        _add(
            "vegetation_and_dryness",
            "Detailed vegetation continuity and dryness inputs were unavailable, reducing confidence in vegetation-related risk estimates.",
        )

    if assessment_output_state in {"limited_regional_estimate", "insufficient_data"}:
        _add(
            "overall_coverage",
            OUTPUT_STATE_LIMITATION_TEXT.get(assessment_output_state, OUTPUT_STATE_LIMITATION_TEXT["limited_regional_estimate"]),
        )
    if region_readiness == "address_level_only":
        _add(
            "prepared_region_readiness",
            "Prepared regional data supports address-level analysis, but not full property-specific structure context here.",
        )
    elif region_readiness == "limited_regional_ready":
        _add(
            "prepared_region_readiness",
            "Prepared regional data for this location is limited; this run should be treated as regional guidance only.",
        )
    if (region_optional_missing_count + region_enrichment_missing_count) > 0:
        _add(
            "enrichment_coverage",
            (
                f"{region_optional_missing_count + region_enrichment_missing_count} optional/enrichment layer(s) "
                "are still missing in the prepared region, which increases fallback usage."
            ),
        )
    if float(fallback_dominance_ratio) >= 0.70:
        _add(
            "fallback_dominance",
            "Most model factors used fallback or proxy behavior for this run, so treat this output as planning guidance.",
        )

    observed: list[str] = []
    missing: list[str] = []
    for key, label in FEATURE_FLAG_LABELS.items():
        if bool(coverage.get(key)):
            observed.append(label)
        else:
            missing.append(label)

    estimated: list[str] = []
    fallback_types = {
        str(row.get("fallback_type") or "").strip().lower()
        for row in fallback_decisions
        if isinstance(row, dict)
    }
    has_point_proxy = bool(
        "point_based_context" in fallback_types
        or "point" in str(preflight.get("geometry_basis") or "").lower()
    )
    if has_point_proxy:
        estimated.append("Near-home vegetation context was approximated from a map point because structure geometry was unavailable.")
    if not bool(coverage.get("hazard_severity_available")) or not bool(coverage.get("burn_probability_available")):
        estimated.append("Regional wildfire hazard context used partial proxy coverage due to missing hazard/burn layers.")
    if not bool(coverage.get("dryness_available")):
        estimated.append("Dryness context was estimated from available regional signals.")
    if not bool(coverage.get("road_network_available")):
        estimated.append("Road and access context was estimated conservatively because direct network data was unavailable.")
    if (region_optional_missing_count + region_enrichment_missing_count) > 0:
        estimated.append("Optional enrichment layers were missing, so this run relied more on core regional context.")
    if len(estimated) < 3:
        for note in score_availability_notes:
            lowered_note = str(note).lower()
            if any(token in lowered_note for token in {"component only", "proxy", "fallback", "insufficient"}):
                estimated.append(str(note))
    if len(estimated) < 3:
        for note in assumptions:
            lowered = str(note).lower()
            if any(token in lowered for token in ["fallback", "proxy", "approxim", "inferred", "estimated"]):
                estimated.append(str(note))

    def _dedupe(items: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in items:
            item = str(raw).strip()
            if not item:
                continue
            token = item.lower()
            if token in seen:
                continue
            seen.add(token)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    observed = _dedupe(observed, 8)
    missing = _dedupe(missing, 8)
    estimated = _dedupe(estimated, 5)
    why_limited = OUTPUT_STATE_LIMITATION_TEXT.get(
        assessment_output_state,
        OUTPUT_STATE_LIMITATION_TEXT["limited_regional_estimate"],
    )
    return limitations[:5], observed, estimated, missing, why_limited


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


def _build_scoring_component_status(
    *,
    score_outputs: dict[str, float | None | bool],
    site_hazard_eligibility: ScoreEligibility,
    home_vulnerability_eligibility: ScoreEligibility,
    insurance_readiness_eligibility: ScoreEligibility,
    assessment_output_state: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    site_available = bool(score_outputs.get("site_hazard_score_available"))
    home_available = bool(score_outputs.get("home_ignition_vulnerability_score_available"))
    readiness_available = bool(score_outputs.get("insurance_readiness_score_available"))

    computed_components: list[str] = []
    blocked_components: list[str] = []
    minimum_missing_requirements: list[str] = []
    if site_available:
        computed_components.append("site_hazard")
    else:
        blocked_components.append("site_hazard")
        minimum_missing_requirements.extend(site_hazard_eligibility.blocking_reasons or [])
    if home_available:
        computed_components.append("home_ignition_vulnerability")
    else:
        blocked_components.append("home_ignition_vulnerability")
        minimum_missing_requirements.extend(home_vulnerability_eligibility.blocking_reasons or [])
    if readiness_available:
        computed_components.append("home_hardening_readiness")
    else:
        blocked_components.append("home_hardening_readiness")
        minimum_missing_requirements.extend(insurance_readiness_eligibility.blocking_reasons or [])

    if site_available and home_available and readiness_available and assessment_output_state == "property_specific_assessment":
        scoring_status = "full_property_assessment"
    elif computed_components:
        scoring_status = "limited_homeowner_estimate"
    else:
        scoring_status = "insufficient_data_to_score"

    coverage = dict(preflight.get("feature_coverage_summary") or {})
    recommended_data_improvements: list[str] = []
    for key, label in FEATURE_FLAG_LABELS.items():
        if not bool(coverage.get(key)):
            recommended_data_improvements.append(label)
    if str(preflight.get("geometry_basis") or "geocode_point") == "geocode_point":
        recommended_data_improvements.append("Building footprint or parcel geometry for structure-level analysis")
    if str(preflight.get("assessment_specificity_tier") or "regional_estimate") != "property_specific":
        recommended_data_improvements.append("Prepared region data with stronger property-specific coverage")

    def _dedupe(values: list[str], limit: int = 6) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = str(raw).strip()
            if not value:
                continue
            token = value.lower()
            if token in seen:
                continue
            seen.add(token)
            out.append(value)
            if len(out) >= limit:
                break
        return out

    return {
        "scoring_status": scoring_status,
        "computed_components": computed_components,
        "blocked_components": blocked_components,
        "minimum_missing_requirements": _dedupe(minimum_missing_requirements, limit=8),
        "recommended_data_improvements": _dedupe(recommended_data_improvements, limit=8),
    }


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
    readiness_provisional: bool = False,
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
        label="Home Hardening Readiness",
        score=insurance_readiness_score,
        summary=(
            f"Home Hardening Readiness {_score_text(insurance_readiness_score)} summarizes practical mitigation readiness for this property."
        ),
        explanation="How prepared the home appears for practical hardening actions.",
        top_drivers=(readiness_blockers[:3] or ["No major readiness blockers detected"]),
        key_drivers=(readiness_blockers[:3] or ["No major readiness blockers detected"]),
        protective_factors=top_protective_factors[:3],
        top_next_actions=readiness_actions,
        next_actions=readiness_actions,
    )

    if confidence_block.use_restriction == "not_for_underwriting_or_binding":
        readiness_section.summary += " Current confidence gating: not for underwriting or binding decisions."
    if readiness_provisional:
        readiness_section.summary += " Readiness estimate is provisional because structure/evidence quality is limited."

    return site_section, home_section, readiness_section


def _build_factor_breakdown(submodels: dict[str, SubmodelScore], risk: RiskComputation) -> FactorBreakdown:
    canonical = {name: round(submodels[name].score, 1) for name in CANONICAL_SUBMODELS if name in submodels}
    environmental = {name: canonical[name] for name in ENVIRONMENTAL_SUBMODELS if name in canonical}
    structural = {name: canonical[name] for name in STRUCTURAL_SUBMODELS if name in canonical}

    return FactorBreakdown(
        submodels=canonical,
        environmental=environmental,
        structural=structural,
        component_scores={
            "regional_context_score": round(float(risk.regional_context_score), 1),
            "property_surroundings_score": round(float(risk.property_surroundings_score), 1),
            "structure_specific_score": round(float(risk.structure_specific_score), 1),
        },
        component_weight_fractions={
            key: round(float(value), 4)
            for key, value in (risk.component_weight_fractions or {}).items()
        },
        environmental_risk=risk.drivers.environmental,
        structural_risk=risk.drivers.structural,
        access_risk=risk.drivers.access_exposure,
        access_risk_provisional=risk.access_provisional,
        access_included_in_total=False,
        access_risk_note=risk.access_note,
    )


def _numeric_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / float(len(values))
    variance = sum((v - mean) ** 2 for v in values) / float(len(values))
    return variance ** 0.5


def _factor_scope_from_fields(source_fields: list[str], input_source_metadata: dict[str, InputSourceMetadata]) -> str:
    if not source_fields:
        return "fallback"
    source_types = {str((input_source_metadata.get(field).source_type if input_source_metadata.get(field) else "missing")) for field in source_fields}
    if source_types & {"missing", "heuristic"}:
        return "fallback"
    if source_fields and any(
        field.startswith("zone_")
        or field
        in {
            "roof_type",
            "vent_type",
            "defensible_space_ft",
            "near_structure_vegetation_0_5_pct",
            "canopy_adjacency_proxy_pct",
            "vegetation_continuity_proxy_pct",
            "nearest_high_fuel_patch_distance_ft",
        }
        for field in source_fields
    ):
        return "property_specific"
    if source_fields and any(field in {"fuel_model", "canopy_cover", "wildland_distance"} for field in source_fields):
        return "neighborhood_level"
    return "region_level"


def _build_score_variance_diagnostics(
    *,
    context: WildfireContext,
    risk: RiskComputation,
    submodel_scores: dict[str, SubmodelScore],
    weighted_contributions: dict[str, WeightedContribution],
    property_level_context: dict[str, Any],
    input_source_metadata: dict[str, InputSourceMetadata],
    resolved_region_id: str | None,
    coverage_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ring_metrics = (property_level_context or {}).get("ring_metrics") if isinstance(property_level_context, dict) else {}
    ring_metrics = ring_metrics if isinstance(ring_metrics, dict) else {}
    feature_sampling = (property_level_context or {}).get("feature_sampling") if isinstance(property_level_context, dict) else {}
    feature_sampling = feature_sampling if isinstance(feature_sampling, dict) else {}

    raw_feature_vector = {
        "burn_probability": context.burn_probability,
        "wildfire_hazard": context.wildfire_hazard,
        "slope": context.slope,
        "fuel_model": context.fuel_model,
        "canopy_cover": context.canopy_cover,
        "historic_fire_distance_km": context.historic_fire_distance,
        "wildland_distance_m": context.wildland_distance,
        "ring_0_5_ft_vegetation_density": _safe_float((ring_metrics.get("ring_0_5_ft") or {}).get("vegetation_density")),
        "ring_5_30_ft_vegetation_density": _safe_float((ring_metrics.get("ring_5_30_ft") or {}).get("vegetation_density")),
        "ring_30_100_ft_vegetation_density": _safe_float((ring_metrics.get("ring_30_100_ft") or {}).get("vegetation_density")),
        "ring_100_300_ft_vegetation_density": _safe_float((ring_metrics.get("ring_100_300_ft") or {}).get("vegetation_density")),
        "nearest_vegetation_distance_ft": _safe_float((property_level_context or {}).get("nearest_vegetation_distance_ft")),
        "near_structure_vegetation_0_5_pct": _safe_float((property_level_context or {}).get("near_structure_vegetation_0_5_pct")),
        "near_structure_connectivity_index": _safe_float((property_level_context or {}).get("near_structure_connectivity_index")),
        "canopy_adjacency_proxy_pct": _safe_float((property_level_context or {}).get("canopy_adjacency_proxy_pct")),
        "vegetation_continuity_proxy_pct": _safe_float((property_level_context or {}).get("vegetation_continuity_proxy_pct")),
        "nearest_high_fuel_patch_distance_ft": _safe_float((property_level_context or {}).get("nearest_high_fuel_patch_distance_ft")),
        "nearby_structure_count_100_ft": _safe_float(
            ((property_level_context or {}).get("neighboring_structure_metrics") or {}).get("nearby_structure_count_100_ft")
        ),
        "nearby_structure_count_300_ft": _safe_float(
            ((property_level_context or {}).get("neighboring_structure_metrics") or {}).get("nearby_structure_count_300_ft")
        ),
        "nearest_structure_distance_ft": _safe_float(
            ((property_level_context or {}).get("neighboring_structure_metrics") or {}).get("nearest_structure_distance_ft")
        ),
        "distance_to_nearest_structure_ft": _safe_float(
            (property_level_context or {}).get("distance_to_nearest_structure_ft")
            or ((property_level_context or {}).get("neighboring_structure_metrics") or {}).get("distance_to_nearest_structure_ft")
        ),
        "structure_density": _safe_float((property_level_context or {}).get("structure_density")),
        "clustering_index": _safe_float((property_level_context or {}).get("clustering_index")),
        "building_age_proxy_year": _safe_float((property_level_context or {}).get("building_age_proxy_year")),
        "building_age_material_proxy_risk": _safe_float((property_level_context or {}).get("building_age_material_proxy_risk")),
    }

    transformed_feature_vector = {
        "burn_probability_index": context.burn_probability_index,
        "hazard_severity_index": context.hazard_severity_index,
        "slope_index": context.slope_index,
        "aspect_index": context.aspect_index,
        "fuel_index": context.fuel_index,
        "moisture_index": context.moisture_index,
        "canopy_index": context.canopy_index,
        "wildland_distance_index": context.wildland_distance_index,
        "historic_fire_index": context.historic_fire_index,
        "imagery_local_percentiles": (
            property_level_context.get("imagery_local_percentiles")
            if isinstance(property_level_context.get("imagery_local_percentiles"), dict)
            else {}
        ),
        "feature_sampling": feature_sampling,
    }

    factor_contribution_breakdown: dict[str, Any] = {}
    fallback_count = 0
    property_specific_count = 0
    for name, sm in submodel_scores.items():
        canonical = name if name in CANONICAL_SUBMODELS else SUBMODEL_ALIASES.get(name)
        if canonical not in CANONICAL_SUBMODELS:
            continue
        wc = weighted_contributions.get(canonical)
        if wc is None:
            continue
        result_meta = risk.submodel_scores.get(canonical)
        source_fields = [
            field for field in (_map_key_input_to_source_field(k) for k in (sm.key_inputs or {}).keys()) if field
        ]
        scope = _factor_scope_from_fields(source_fields, input_source_metadata)
        fallback_flag = bool(
            scope == "fallback"
            or any(v is None for v in (sm.key_inputs or {}).values())
            or any(tok in " ".join(sm.assumptions).lower() for tok in ("fallback", "missing", "unavailable"))
        )
        if fallback_flag:
            fallback_count += 1
        if scope == "property_specific":
            property_specific_count += 1
        factor_contribution_breakdown[canonical] = {
            "weight": wc.weight,
            "base_weight": wc.base_weight,
            "effective_weight": wc.effective_weight,
            "observed_fraction": wc.observed_fraction,
            "availability_multiplier": wc.availability_multiplier,
            "omitted_due_to_missing": wc.omitted_due_to_missing,
            "score": wc.score,
            "contribution": wc.contribution,
            "basis": wc.basis,
            "factor_evidence_status": wc.factor_evidence_status or ("suppressed" if wc.omitted_due_to_missing else wc.basis),
            "support_level": wc.support_level,
            "component": wc.component,
            "raw_submodel_score_before_clamp": getattr(result_meta, "raw_score", None),
            "clamped_submodel_score": getattr(result_meta, "clamped_score", wc.score),
            "scope": scope,
            "fallback_or_default_used": fallback_flag,
            "source_fields": source_fields,
            "source_metadata": {
                field: {
                    "source_name": (input_source_metadata.get(field).source_name if input_source_metadata.get(field) else None),
                    "source_type": (input_source_metadata.get(field).source_type if input_source_metadata.get(field) else None),
                    "spatial_resolution_m": (
                        input_source_metadata.get(field).spatial_resolution_m if input_source_metadata.get(field) else None
                    ),
                }
                for field in source_fields
            },
            "key_inputs": sm.key_inputs,
        }

    contributions = [float(v.get("contribution") or 0.0) for v in factor_contribution_breakdown.values()]
    contribution_stddev = _numeric_stddev(contributions)
    absolute_contributions = [abs(v) for v in contributions]
    top_two_share = 0.0
    if absolute_contributions:
        total_abs = max(1e-6, sum(absolute_contributions))
        top_two_share = sum(sorted(absolute_contributions, reverse=True)[:2]) / total_abs

    compression_flags: list[str] = []
    if fallback_count >= 3:
        compression_flags.append("fallback_heavy_factor_inputs")
    if contribution_stddev < 2.5:
        compression_flags.append("low_submodel_contribution_spread")
    if top_two_share >= 0.72:
        compression_flags.append("top_components_dominate_total")
    if property_specific_count <= 2:
        compression_flags.append("limited_property_specific_signal")
    if raw_feature_vector["ring_0_5_ft_vegetation_density"] is None and raw_feature_vector["ring_5_30_ft_vegetation_density"] is None:
        compression_flags.append("near_structure_ring_metrics_missing")
    if raw_feature_vector["near_structure_vegetation_0_5_pct"] is None and raw_feature_vector["vegetation_continuity_proxy_pct"] is None:
        compression_flags.append("imagery_feature_signal_missing")
    if raw_feature_vector["near_structure_connectivity_index"] is None:
        compression_flags.append("near_structure_connectivity_missing")
    if risk.observed_weight_fraction < 0.70:
        compression_flags.append("low_observed_weight_fraction")
    if risk.fallback_dominance_ratio >= 0.60:
        compression_flags.append("fallback_dominance_high")
    if coverage_preflight and bool(coverage_preflight.get("limited_assessment_flag")):
        compression_flags.append("limited_assessment_specificity")

    compression_analysis_summary: list[str] = []
    if "fallback_heavy_factor_inputs" in compression_flags:
        compression_analysis_summary.append("Multiple submodels used fallback/default inputs, which narrows property-to-property separation.")
    if "low_submodel_contribution_spread" in compression_flags:
        compression_analysis_summary.append("Submodel contributions cluster tightly, indicating compressed score spread.")
    if "top_components_dominate_total" in compression_flags:
        compression_analysis_summary.append("A small number of components dominate total risk, reducing sensitivity to other property signals.")
    if "limited_property_specific_signal" in compression_flags:
        compression_analysis_summary.append("Few property-specific factors are active; neighborhood/regional context is dominating.")
    if "imagery_feature_signal_missing" in compression_flags:
        compression_analysis_summary.append("Imagery-derived near-structure features were unavailable, reducing parcel-level discrimination.")
    if "near_structure_connectivity_missing" in compression_flags:
        compression_analysis_summary.append("Near-structure vegetation connectivity was unavailable, reducing structure-level ignition sensitivity.")
    if "low_observed_weight_fraction" in compression_flags:
        compression_analysis_summary.append("A low fraction of score weight is backed by observed evidence; omitted factors reduced specificity.")
    if "fallback_dominance_high" in compression_flags:
        compression_analysis_summary.append("Fallback-driven factors dominate active inputs; score precision is limited.")
    if "limited_assessment_specificity" in compression_flags:
        compression_analysis_summary.append("Coverage preflight downgraded this run to limited specificity.")
    if not compression_analysis_summary:
        compression_analysis_summary.append("No major compression pattern detected for this assessment.")

    return {
        "resolved_region_id": resolved_region_id,
        "raw_feature_vector": raw_feature_vector,
        "transformed_feature_vector": transformed_feature_vector,
        "factor_contribution_breakdown": factor_contribution_breakdown,
        "compression_flags": compression_flags,
        "compression_analysis_summary": compression_analysis_summary,
        "factor_fallback_count": fallback_count,
        "factor_count": len(factor_contribution_breakdown),
        "contribution_stddev": round(contribution_stddev, 4),
        "top_two_component_share": round(top_two_share, 4),
        "observed_factor_count": risk.observed_factor_count,
        "missing_factor_count": risk.missing_factor_count,
        "fallback_factor_count": risk.fallback_factor_count,
        "observed_weight_fraction": risk.observed_weight_fraction,
        "fallback_dominance_ratio": risk.fallback_dominance_ratio,
        "fallback_weight_fraction": risk.fallback_weight_fraction,
        "uncertainty_penalty": risk.uncertainty_penalty,
        "component_scores": {
            "regional_context_score": risk.regional_context_score,
            "property_surroundings_score": risk.property_surroundings_score,
            "structure_specific_score": risk.structure_specific_score,
        },
        "component_weight_fractions": risk.component_weight_fractions,
        "geometry_basis": risk.geometry_basis,
        "missing_core_layer_count": int((coverage_preflight or {}).get("missing_core_layer_count") or 0),
        "observed_weight_fraction_from_weights": round(
            sum(float(v.get("effective_weight") or 0.0) for v in factor_contribution_breakdown.values()),
            4,
        ),
    }


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
            result.risk_scores.home_hardening_readiness = result.insurance_readiness_score
        if result.insurance_readiness_section:
            result.insurance_readiness_section.score = result.insurance_readiness_score
            result.insurance_readiness_section.summary = (
                f"Home Hardening Readiness {result.insurance_readiness_score:.1f}/100 summarizes practical mitigation readiness for this property."
            )
        if result.score_summaries:
            result.score_summaries.insurance_readiness.score = result.insurance_readiness_score
            result.score_summaries.insurance_readiness.summary = (
                f"Home Hardening Readiness {result.insurance_readiness_score:.1f}/100 summarizes practical mitigation readiness for this property."
            )

    result.overall_wildfire_risk = result.wildfire_risk_score
    result.home_hardening_readiness = result.insurance_readiness_score
    result.home_hardening_readiness_score_available = bool(result.insurance_readiness_score_available)
    if result.risk_scores:
        result.risk_scores.overall_wildfire_risk = result.wildfire_risk_score
        result.risk_scores.home_hardening_readiness = result.insurance_readiness_score
        result.risk_scores.home_hardening_readiness_score_available = bool(result.insurance_readiness_score_available)

    if required_priority_boost > 0:
        for rec in result.mitigation_plan:
            if rec.insurer_relevance == "required":
                rec.priority = max(1, rec.priority - required_priority_boost)
        result.mitigation_plan.sort(key=lambda x: x.priority)
        result.mitigation_recommendations = list(result.mitigation_plan)

    result.prioritized_mitigation_actions = prioritize_mitigation_actions(result.mitigation_plan, limit=5)
    result.top_recommended_actions = [
        row.action
        for row in result.prioritized_mitigation_actions[:3]
        if str(row.action or "").strip()
    ]

    result.readiness_blockers = sorted(set(result.readiness_blockers))
    result.readiness_summary = f"{result.readiness_summary} Ruleset: {ruleset.ruleset_name} ({ruleset.ruleset_id})."
    result.scoring_notes.append(
        f"Underwriting ruleset {ruleset.ruleset_id}@{ruleset.ruleset_version} applied (multiplier={penalty_multiplier})."
    )
    if result.confidence:
        carried_penalties = list(result.evidence_quality_summary.confidence_penalties or [])
        result.score_evidence_ledger = _build_score_evidence_ledger(
            risk=None,
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


def _calibration_coverage_tier(calibration_payload: dict[str, Any]) -> tuple[str, str | None]:
    dataset_meta = calibration_payload.get("outcome_dataset")
    if not isinstance(dataset_meta, dict):
        return "unknown", "Calibration artifact dataset coverage metadata is unavailable."
    try:
        row_count = int(float(dataset_meta.get("row_count") or 0))
    except (TypeError, ValueError):
        row_count = 0
    if row_count >= 1000:
        tier = "high"
    elif row_count >= 250:
        tier = "moderate"
    elif row_count > 0:
        tier = "low"
    else:
        tier = "unknown"
    if row_count <= 0:
        note = "Calibration artifact did not expose usable row-count metadata."
    else:
        note = f"Calibration artifact was fit using {row_count} labeled public-outcome rows."
    return tier, note


def _calibration_availability_state(calibration_payload: dict[str, Any]) -> str:
    status = str(calibration_payload.get("calibration_status") or "disabled").strip().lower()
    mapping = {
        "applied": "available_applied",
        "disabled_no_artifact": "unavailable_no_artifact",
        "invalid_artifact": "unavailable_incompatible_artifact",
        "incompatible_version": "unavailable_incompatible_artifact",
        "invalid_method_or_parameters": "unavailable_incompatible_artifact",
        "out_of_scope": "unavailable_out_of_scope",
        "score_unavailable": "unavailable_raw_score_missing",
    }
    return mapping.get(status, f"unavailable_{status}" if status else "unavailable_unknown")


def _build_calibrated_public_outcome_metadata(
    *,
    requested: bool,
    calibration_payload: dict[str, Any],
    calibration_version: str,
    raw_wildfire_risk_score: float | None,
) -> CalibratedPublicOutcomeMetadata | None:
    if not requested:
        return None
    available = bool(calibration_payload.get("calibration_applied"))
    coverage_tier, coverage_note = _calibration_coverage_tier(calibration_payload)
    caveat = (
        "This calibrated value is based on public observed wildfire damage outcomes and should not be interpreted "
        "as carrier underwriting probability or claims likelihood. "
        "Availability depends on calibration artifact coverage and model version compatibility."
    )
    notes: list[str] = []
    notes.extend([str(x) for x in (calibration_payload.get("calibration_limitations") or []) if str(x).strip()])
    if calibration_payload.get("scope_warning"):
        notes.append(str(calibration_payload.get("scope_warning")))
    calibrated_prob = calibration_payload.get("calibrated_damage_likelihood")
    metadata = CalibratedPublicOutcomeMetadata(
        requested=True,
        available=available,
        availability_status=_calibration_availability_state(calibration_payload),
        calibration_version=str(calibration_version or CALIBRATION_VERSION),
        calibrated_public_outcome_probability=(
            float(calibrated_prob) if calibrated_prob is not None else None
        ),
        calibration_basis_summary=(
            "Public observed wildfire structure-damage outcomes were used to fit an optional calibration layer "
            "on top of raw deterministic wildfire risk scores."
        ),
        calibration_caveat=caveat,
        calibration_data_coverage_tier=coverage_tier,
        calibration_data_coverage_note=coverage_note,
        raw_score_reference={
            "raw_wildfire_risk_score": raw_wildfire_risk_score,
            "calibration_raw_score_input": calibration_payload.get("raw_wildfire_risk_score"),
            "raw_score_units": "0-100 wildfire_risk_score",
        },
        fallback_state=(
            "calibration_unavailable_using_raw_scores_only"
            if not available
            else "calibration_applied"
        ),
        notes=notes[:8],
    )
    return metadata


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
    include_calibrated_outputs: bool = False,
) -> tuple[AssessmentResult, dict]:
    requested_anchor = _coerce_point_payload(payload.property_anchor_point or payload.user_selected_point)
    geocode_resolution = geocode_resolution or _resolve_trusted_geocode(
        address_input=payload.address,
        purpose="assessment",
        route_name="assessment_core",
        property_anchor_point=requested_anchor,
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
    if not property_level_context.get("geometry_source"):
        final_geometry_source = str(property_level_context.get("final_structure_geometry_source") or "").strip().lower()
        if final_geometry_source == "user_selected_point_snapped":
            property_level_context["geometry_source"] = "user_selected_map_point_snapped_structure"
        elif final_geometry_source == "user_selected_point_unsnapped":
            property_level_context["geometry_source"] = "user_selected_map_point_unsnapped"
        elif str(property_level_context.get("geometry_basis") or "").strip().lower() == "parcel":
            property_level_context["geometry_source"] = "parcel_geometry_inferred_home_location"
        elif bool(property_level_context.get("footprint_used")):
            property_level_context["geometry_source"] = "trusted_building_footprint"
        else:
            property_level_context["geometry_source"] = "raw_geocode_point"
    if property_level_context.get("geometry_confidence") is None:
        try:
            property_level_context["geometry_confidence"] = float(
                property_level_context.get("structure_geometry_confidence") or 0.0
            )
        except (TypeError, ValueError):
            property_level_context["geometry_confidence"] = 0.0
    if not property_level_context.get("ring_generation_mode"):
        property_level_context["ring_generation_mode"] = (
            "footprint_aware_rings" if bool(property_level_context.get("footprint_used")) else "point_annulus_fallback"
        )
    property_level_context["property_linkage"] = build_property_linkage_summary(
        geocode_meta=geocode_meta,
        property_level_context=property_level_context,
    )
    geometry_resolution = _build_geometry_resolution_summary(property_level_context)
    property_level_context["geometry_resolution"] = geometry_resolution.model_dump()
    property_level_context["property_mismatch_flag"] = bool(geometry_resolution.property_mismatch_flag)
    property_level_context["mismatch_reason"] = (
        str(geometry_resolution.mismatch_reason).strip()
        if geometry_resolution.mismatch_reason
        else None
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
            base_weight=risk.weighted_contributions[name].get("base_weight"),
            effective_weight=risk.weighted_contributions[name].get("effective_weight"),
            observed_fraction=risk.weighted_contributions[name].get("observed_fraction"),
            availability_multiplier=risk.weighted_contributions[name].get("availability_multiplier"),
            basis=risk.weighted_contributions[name].get("basis"),
            factor_evidence_status=risk.weighted_contributions[name].get("factor_evidence_status"),
            support_level=risk.weighted_contributions[name].get("support_level"),
            component=risk.weighted_contributions[name].get("component"),
            omitted_due_to_missing=bool(risk.weighted_contributions[name].get("omitted_due_to_missing", False)),
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
        geocode_meta=geocode_meta,
    )
    coverage_lookup = dict(coverage_resolution.coverage or {})
    geocode_meta["trusted_match_subchecks"] = _build_trusted_match_subchecks(
        submitted_address=payload.address,
        geocode_meta=geocode_meta,
        coverage=coverage_lookup,
    )
    if geocode_meta.get("geocode_outcome") == "geocode_succeeded_untrusted":
        geocode_meta["geocode_decision"] = (
            "geocode_candidate_inside_region_but_low_precision"
            if bool(coverage_lookup.get("coverage_available"))
            else "geocode_candidate_outside_region"
        )
    elif geocode_meta.get("geocode_outcome") == "geocode_succeeded_trusted":
        geocode_meta["geocode_decision"] = "trusted_geocode_success"
    geocode_meta["selected_region_id"] = coverage_lookup.get("resolved_region_id")
    geocode_meta["selected_region_display_name"] = (
        coverage_lookup.get("resolved_region_display_name")
        or coverage_lookup.get("display_name")
    )
    geocode_meta["region_check_result"] = coverage_lookup.get("region_check_result")
    geocode_meta["region_distance_to_boundary_m"] = coverage_lookup.get("region_distance_to_boundary_m")
    geocode_meta["candidate_regions_containing_point"] = coverage_lookup.get("candidate_regions_containing_point")
    geocode_meta["unsupported_location_reason"] = (
        coverage_lookup.get("reason")
        if not bool(coverage_lookup.get("coverage_available"))
        else None
    )
    geocode_meta["within_downloaded_zone_check"] = bool(coverage_lookup.get("coverage_available"))
    layer_coverage_audit, coverage_summary = _normalize_layer_coverage(
        property_level_context,
        environmental_layer_status=context.environmental_layer_status,
    )
    coverage_preflight = _build_feature_coverage_preflight(
        context=context,
        property_level_context=property_level_context,
        coverage_summary=coverage_summary,
        payload=payload,
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
        site_hazard_eligibility,
        home_vulnerability_eligibility,
        insurance_readiness_eligibility,
        assessment_status,
        assessment_blockers,
        assessment_output_state,
    ) = _apply_preflight_specificity_gate(
        site_eligibility=site_hazard_eligibility,
        home_eligibility=home_vulnerability_eligibility,
        readiness_eligibility=insurance_readiness_eligibility,
        assessment_status=assessment_status,
        assessment_blockers=assessment_blockers,
        preflight=coverage_preflight,
        fallback_dominance_ratio=float(risk.fallback_dominance_ratio),
        observed_weight_fraction=float(risk.observed_weight_fraction),
    )
    coverage_preflight["assessment_output_state"] = assessment_output_state
    coverage_preflight["scoring_fallback_weight_fraction"] = float(risk.fallback_weight_fraction)
    coverage_preflight["adaptive_component_weights"] = dict(risk.component_weight_fractions or {})
    coverage_preflight["adaptive_component_scores"] = {
        "regional_context_score": float(risk.regional_context_score),
        "property_surroundings_score": float(risk.property_surroundings_score),
        "structure_specific_score": float(risk.structure_specific_score),
    }
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
        preflight=coverage_preflight,
        assessment_output_state=assessment_output_state,
        observed_weight_fraction=float(risk.observed_weight_fraction),
        fallback_dominance_ratio=float(risk.fallback_dominance_ratio),
    )
    if risk.uncertainty_penalty > 0:
        confidence_adjustment = round(min(8.0, float(risk.uncertainty_penalty) * 0.35), 1)
        confidence_block = confidence_block.model_copy(deep=True)
        confidence_block.confidence_score = round(
            max(0.0, confidence_block.confidence_score - confidence_adjustment),
            1,
        )
        confidence_block.low_confidence_flags = sorted(
            set(
                list(confidence_block.low_confidence_flags)
                + [
                    (
                        "Missing-factor omission reduced observed scoring coverage; "
                        "confidence is downgraded by an explicit uncertainty penalty."
                    )
                ]
            )
        )
    confidence_block, confidence_downgrade_reasons, trust_tier_blockers = _apply_hard_trust_guardrails(
        confidence_block,
        site_eligibility=site_hazard_eligibility,
        home_eligibility=home_vulnerability_eligibility,
        readiness_eligibility=insurance_readiness_eligibility,
        assessment_status=assessment_status,
        coverage_summary=coverage_summary,
        preflight=coverage_preflight,
        assessment_output_state=assessment_output_state,
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
    if risk.uncertainty_penalty > 0:
        confidence_penalties.append(
            ConfidencePenalty(
                penalty_key="missing_factor_uncertainty",
                reason=(
                    "Missing factors were omitted from numeric weighting and converted into an explicit "
                    "uncertainty penalty."
                ),
                amount=round(float(risk.uncertainty_penalty), 1),
            )
        )
    ranked_driver_titles, ranked_driver_details = build_ranked_risk_drivers(
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        limit=5,
    )
    property_findings = _build_property_findings(property_level_context)
    if ranked_driver_titles:
        driver_seed = ranked_driver_titles
    elif float(risk.observed_weight_fraction) >= 0.50 and float(risk.fallback_weight_fraction) < 0.55:
        driver_seed = _build_top_risk_drivers(submodel_scores)
    else:
        driver_seed = ["Limited direct evidence for a stable factor-level ranking in this run."]
    top_risk_drivers = _merge_property_drivers(
        driver_seed,
        property_findings,
    )
    for driver in top_near_structure_risk_drivers:
        if driver not in top_risk_drivers:
            top_risk_drivers.insert(0, driver)
    top_risk_drivers = top_risk_drivers[:3]
    prioritized_mitigation_actions = prioritize_mitigation_actions(mitigation_plan, limit=5)
    top_recommended_actions = [row.action for row in prioritized_mitigation_actions[:3]]
    confidence_summary = build_confidence_summary(
        confidence_tier=confidence_block.confidence_tier,
        observed_inputs=assumptions_block.observed_inputs,
        inferred_inputs=assumptions_block.inferred_inputs,
        missing_inputs=assumptions_block.missing_inputs,
        assumptions_used=assumptions_block.assumptions_used,
    )
    top_protective_factors = _build_top_protective_factors(scoring_attrs, submodel_scores)

    raw_site_hazard_score, raw_home_ignition_vulnerability_score = _build_score_decomposition(risk=risk)
    raw_legacy_weighted_wildfire_risk_score = risk.total_score
    raw_blended_wildfire_risk_score = risk_engine.compute_blended_wildfire_score(
        site_hazard_score=raw_site_hazard_score,
        home_ignition_vulnerability_score=raw_home_ignition_vulnerability_score,
        insurance_readiness_score=readiness.insurance_readiness_score,
        risk=risk,
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
    component_status = _build_scoring_component_status(
        score_outputs=score_outputs,
        site_hazard_eligibility=site_hazard_eligibility,
        home_vulnerability_eligibility=home_vulnerability_eligibility,
        insurance_readiness_eligibility=insurance_readiness_eligibility,
        assessment_output_state=assessment_output_state,
        preflight=coverage_preflight,
    )
    scoring_status = str(component_status.get("scoring_status") or "insufficient_data_to_score")
    computed_components = list(component_status.get("computed_components") or [])
    blocked_components = list(component_status.get("blocked_components") or [])
    minimum_missing_requirements = list(component_status.get("minimum_missing_requirements") or [])
    recommended_data_improvements = list(component_status.get("recommended_data_improvements") or [])

    if "home_ignition_vulnerability" not in computed_components:
        top_near_structure_risk_drivers = []
        prioritized_vegetation_actions = []
        if defensible_space_limitations_summary:
            defensible_space_limitations_summary = list(
                dict.fromkeys(
                    [
                        "Near-structure observations are preliminary and were not used for a computed Home Ignition Vulnerability score."
                    ]
                    + defensible_space_limitations_summary
                )
            )[:4]
        defensible_space_analysis = dict(defensible_space_analysis or {})
        existing_summary = str(defensible_space_analysis.get("summary") or "").strip()
        defensible_space_analysis["summary"] = (
            "Preliminary near-structure observation only; not enough verified structure evidence to compute Home Ignition Vulnerability."
            + (f" {existing_summary}" if existing_summary else "")
        ).strip()

    if scoring_status == "insufficient_data_to_score":
        ranked_driver_details = []
        top_risk_drivers = [
            "We found some relevant signals, but not enough verified inputs to compute a reliable score."
        ]
        prioritized_mitigation_actions = []
        top_recommended_actions = []
    else:
        if "site_hazard" not in computed_components:
            ranked_driver_details = [
                row
                for row in ranked_driver_details
                if str(row.factor) not in {
                    "vegetation_proximity",
                    "nearby_fuel_load",
                    "slope_and_terrain",
                    "ember_exposure",
                    "flame_contact_potential",
                    "historical_fire_exposure",
                }
            ]
            top_risk_drivers = top_risk_drivers[:3]
        if "home_ignition_vulnerability" not in computed_components:
            ranked_driver_details = [
                row
                for row in ranked_driver_details
                if str(row.factor) not in {"home_hardening_gaps", "defensible_space"}
            ]
    calibration_payload = resolve_public_calibration(
        raw_wildfire_score=blended_wildfire_risk_score,
        resolved_region_id=region_resolution.resolved_region_id,
    )
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
        property_level_context=property_level_context,
        fallback_decisions=fallback_decisions,
    )
    assessment_limitations_summary = _build_assessment_limitations_summary(
        fallback_decisions=fallback_decisions,
        score_availability_notes=score_availability_notes,
        coverage_summary=coverage_summary,
    )
    data_quality_summary = _build_data_quality_summary(
        preflight=coverage_preflight,
        property_level_context=property_level_context,
    )
    grouped_limitations, what_was_observed, what_was_estimated, what_was_missing, why_limited = (
        _build_homeowner_limitation_groups(
            preflight=coverage_preflight,
            data_quality_summary=data_quality_summary,
            fallback_decisions=fallback_decisions,
            score_availability_notes=score_availability_notes,
            assumptions=all_assumptions,
            assessment_output_state=assessment_output_state,
            fallback_dominance_ratio=float(risk.fallback_dominance_ratio),
        )
    )
    confidence_improvement_actions = _build_confidence_improvement_actions(
        preflight=coverage_preflight,
        property_level_context=property_level_context,
        geocode_meta=(geocode_meta if isinstance(geocode_meta, dict) else {}),
        missing_inputs=list(assumptions_block.missing_inputs or []),
        recommended_data_improvements=list(recommended_data_improvements or []),
    )
    homeowner_confidence_summary = _build_homeowner_confidence_summary(
        confidence_score=float(confidence_block.confidence_score),
        assessment_output_state=assessment_output_state,
        grouped_limitations=grouped_limitations,
        why_limited=why_limited,
        feature_coverage_percent=float(coverage_preflight.get("feature_coverage_percent") or 0.0),
        missing_core_layer_count=int(coverage_preflight.get("missing_core_layer_count") or 0),
        geometry_basis=str(coverage_preflight.get("geometry_basis") or "geocode_point"),
        improvement_actions=confidence_improvement_actions,
    )
    differentiation_snapshot = build_differentiation_snapshot(
        feature_coverage_summary=dict(coverage_preflight.get("feature_coverage_summary") or {}),
        preflight=coverage_preflight,
        property_level_context=property_level_context,
        environmental_layer_status=dict(context.environmental_layer_status or {}),
        fallback_weight_fraction=float(risk.fallback_weight_fraction),
        missing_inputs=list(assumptions_block.missing_inputs or []),
        inferred_inputs=list(assumptions_block.inferred_inputs or []),
        input_source_metadata=input_source_metadata,
        fallback_decisions=fallback_decisions,
    )
    trust_summary = _build_homeowner_trust_summary(
        confidence_tier=str(confidence_block.confidence_tier),
        fallback_decisions=fallback_decisions,
        missing_inputs=list(assumptions_block.missing_inputs or []),
        preflight=coverage_preflight,
        differentiation_snapshot=differentiation_snapshot,
        geometry_resolution=geometry_resolution.model_dump(),
        property_confidence_summary=(
            dict(coverage_preflight.get("property_confidence_summary"))
            if isinstance(coverage_preflight.get("property_confidence_summary"), dict)
            else {}
        ),
    )
    homeowner_assessment_mode = _to_homeowner_assessment_mode(assessment_output_state)
    specificity_summary = _build_specificity_summary(
        assessment_specificity_tier=str(coverage_preflight.get("assessment_specificity_tier") or "regional_estimate"),
        assessment_mode=homeowner_assessment_mode,
        limited_assessment_flag=bool(coverage_preflight.get("limited_assessment_flag")),
        confidence_summary=homeowner_confidence_summary,
        trust_summary=trust_summary,
        property_confidence_summary=(
            dict(coverage_preflight.get("property_confidence_summary"))
            if isinstance(coverage_preflight.get("property_confidence_summary"), dict)
            else {}
        ),
    )
    force_low_specificity_safeguard = specificity_summary.get("specificity_tier") in {
        "regional_estimate",
        "insufficient_data",
    }
    nearby_home_comparison_safeguard_triggered = bool(
        trust_summary.get("nearby_home_comparison_safeguard_triggered")
    ) or bool(force_low_specificity_safeguard)
    nearby_home_comparison_safeguard_message = str(
        trust_summary.get("nearby_home_comparison_safeguard_message")
        or "This estimate is not precise enough to compare adjacent homes."
    ).strip()
    if nearby_home_comparison_safeguard_triggered:
        trust_summary["nearby_home_comparison_safeguard_triggered"] = True
        trust_summary["nearby_home_comparison_safeguard_message"] = nearby_home_comparison_safeguard_message
        trust_summary["parcel_level_comparison_allowed"] = False
        trust_summary["differentiation_summary"] = nearby_home_comparison_safeguard_message
    specificity_summary = _build_specificity_summary(
        assessment_specificity_tier=str(coverage_preflight.get("assessment_specificity_tier") or "regional_estimate"),
        assessment_mode=homeowner_assessment_mode,
        limited_assessment_flag=bool(coverage_preflight.get("limited_assessment_flag")),
        confidence_summary=homeowner_confidence_summary,
        trust_summary=trust_summary,
        property_confidence_summary=(
            dict(coverage_preflight.get("property_confidence_summary"))
            if isinstance(coverage_preflight.get("property_confidence_summary"), dict)
            else {}
        ),
    )
    if nearby_home_comparison_safeguard_triggered and assessment_status != "insufficient_data":
        # Suppress parcel-level differentiation claims when the evidence is
        # mostly regional and neighborhood differentiation confidence is low.
        top_near_structure_risk_drivers = []
        prioritized_vegetation_actions = []
        ranked_driver_details = [
            row
            for row in list(ranked_driver_details or [])
            if str(getattr(row, "factor", "") or "").strip().lower()
            not in {
                "defensible_space",
                "flame_contact",
                "ember_exposure",
                "vegetation_proximity",
                "nearby_fuel_load",
            }
        ]
        property_findings = [
            row
            for row in list(property_findings or [])
            if not any(
                token in str(row).lower()
                for token in (
                    "near-structure",
                    "0-5 ft",
                    "5-30 ft",
                    "defensible space",
                    "adjacent to the home",
                )
            )
        ]
        top_risk_drivers = [
            row
            for row in list(top_risk_drivers or [])
            if not any(
                token in str(row).lower()
                for token in (
                    "near-structure",
                    "0-5 ft",
                    "5-30 ft",
                    "defensible space",
                )
            )
        ]
        top_risk_drivers = [nearby_home_comparison_safeguard_message] + top_risk_drivers
        top_risk_drivers = list(dict.fromkeys(top_risk_drivers))[:3]
        defensible_space_limitations_summary = list(
            dict.fromkeys([nearby_home_comparison_safeguard_message] + list(defensible_space_limitations_summary or []))
        )[:4]
        defensible_space_analysis = dict(defensible_space_analysis or {})
        defensible_space_analysis["summary"] = nearby_home_comparison_safeguard_message
    readiness_provisional = bool(
        float(risk.geometry_quality_score) < 0.62
        or float(risk.fallback_weight_fraction) >= 0.60
        or float(coverage_preflight.get("regional_enrichment_consumption_score") or 100.0) < 60.0
    )
    structure_data_completeness = float(risk.structure_data_completeness or 0.0)
    structure_assumption_mode = str(risk.structure_assumption_mode or "unknown")
    if structure_assumption_mode not in {"observed", "mixed", "default_assumed"}:
        structure_assumption_mode = "unknown"
    structure_score_confidence = float(risk.structure_score_confidence or 0.0)
    developer_diagnostics = {
        "fallback_decisions": fallback_decisions,
        "score_availability_notes": score_availability_notes,
        "coverage_summary": coverage_summary.model_dump(),
        "layer_coverage_audit": [row.model_dump() for row in layer_coverage_audit],
        "assessment_diagnostics": assessment_diagnostics.model_dump(),
        "preflight": dict(coverage_preflight),
        "adaptive_component_scores": {
            "regional_context_score": float(risk.regional_context_score),
            "property_surroundings_score": float(risk.property_surroundings_score),
            "structure_specific_score": float(risk.structure_specific_score),
        },
        "adaptive_component_weights": dict(risk.component_weight_fractions or {}),
        "geometry_basis_used_for_weighting": str(risk.geometry_basis),
        "geometry_quality_score": float(risk.geometry_quality_score),
        "regional_context_coverage_score": float(risk.regional_context_coverage_score),
        "regional_enrichment_consumption_score": float(
            coverage_preflight.get("regional_enrichment_consumption_score")
            if coverage_preflight.get("regional_enrichment_consumption_score") is not None
            else coverage_preflight.get("regional_context_coverage_score") or risk.regional_context_coverage_score
        ),
        "property_specificity_score": float(risk.property_specificity_score),
        "observed_feature_count": int(coverage_preflight.get("observed_feature_count") or risk.observed_feature_count),
        "inferred_feature_count": int(coverage_preflight.get("inferred_feature_count") or risk.inferred_feature_count),
        "fallback_feature_count": int(coverage_preflight.get("fallback_feature_count") or risk.fallback_feature_count),
        "missing_feature_count": int(coverage_preflight.get("missing_feature_count") or 0),
        "fallback_weight_fraction": float(risk.fallback_weight_fraction),
        "structure_data_completeness": structure_data_completeness,
        "structure_assumption_mode": structure_assumption_mode,
        "structure_score_confidence": structure_score_confidence,
        "scoring_status": scoring_status,
        "computed_components": computed_components,
        "blocked_components": blocked_components,
        "minimum_missing_requirements": minimum_missing_requirements,
        "recommended_data_improvements": recommended_data_improvements,
        "confidence_improvement_actions": confidence_improvement_actions,
        "differentiation_diagnostics": differentiation_snapshot,
        "property_data_confidence": float(coverage_preflight.get("property_data_confidence") or 0.0),
        "property_confidence_summary": dict(coverage_preflight.get("property_confidence_summary") or {}),
        "specificity_summary": dict(specificity_summary),
        "geometry_resolution": geometry_resolution.model_dump(),
        "nearby_home_comparison_safeguard_triggered": nearby_home_comparison_safeguard_triggered,
        "nearby_home_comparison_safeguard_message": (
            nearby_home_comparison_safeguard_message if nearby_home_comparison_safeguard_triggered else None
        ),
    }
    homeowner_summary = {
        "assessment_mode": homeowner_assessment_mode,
        "assessment_output_state": assessment_output_state,
        "scoring_status": scoring_status,
        "computed_components": computed_components,
        "blocked_components": blocked_components,
        "minimum_missing_requirements": minimum_missing_requirements,
        "recommended_data_improvements": recommended_data_improvements,
        "confidence_improvement_actions": confidence_improvement_actions,
        "risk_label": (
            "Wildfire Risk Estimate"
            if assessment_output_state in {"address_level_estimate", "limited_regional_estimate", "insufficient_data"}
            else "Wildfire Risk Score"
        ),
        "evidence_snapshot": {
            "observed_feature_count": int(coverage_preflight.get("observed_feature_count") or risk.observed_feature_count),
            "estimated_feature_count": int(
                (coverage_preflight.get("inferred_feature_count") or risk.inferred_feature_count)
                + (coverage_preflight.get("fallback_feature_count") or risk.fallback_feature_count)
            ),
            "missing_feature_count": int(coverage_preflight.get("missing_feature_count") or 0),
            "fallback_weight_fraction": round(float(risk.fallback_weight_fraction), 3),
            "geometry_quality_score": round(float(coverage_preflight.get("geometry_quality_score") or risk.geometry_quality_score), 3),
            "regional_enrichment_consumption_score": round(
                float(
                    coverage_preflight.get("regional_enrichment_consumption_score")
                    if coverage_preflight.get("regional_enrichment_consumption_score") is not None
                    else coverage_preflight.get("regional_context_coverage_score") or risk.regional_context_coverage_score
                ),
                1,
            ),
            "structure_data_completeness": round(structure_data_completeness, 1),
            "structure_assumption_mode": structure_assumption_mode,
            "structure_score_confidence": round(structure_score_confidence, 1),
        },
        "specificity_summary": dict(specificity_summary),
        "property_data_confidence": float(coverage_preflight.get("property_data_confidence") or 0.0),
        "property_confidence_summary": dict(coverage_preflight.get("property_confidence_summary") or {}),
        "geometry_resolution": geometry_resolution.model_dump(),
        "differentiation": differentiation_snapshot,
        "nearby_home_comparison_safeguard_triggered": nearby_home_comparison_safeguard_triggered,
        "nearby_home_comparison_safeguard_message": (
            nearby_home_comparison_safeguard_message if nearby_home_comparison_safeguard_triggered else None
        ),
        "home_hardening_readiness_precision": "provisional" if readiness_provisional else "stable",
        "confidence_summary": homeowner_confidence_summary,
        "trust_summary": trust_summary,
        "assessment_limitations": grouped_limitations,
        "what_was_observed": what_was_observed,
        "what_was_estimated": what_was_estimated,
        "what_was_missing": what_was_missing,
        "why_this_result_is_limited": why_limited,
        "data_quality_summary": data_quality_summary,
    }
    readiness_factors = [
        ReadinessFactor(name=f["name"], status=f["status"], score_impact=f["score_impact"], detail=f["detail"])
        for f in readiness.readiness_factors
    ]

    score_evidence_ledger = _build_score_evidence_ledger(
        risk=risk,
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
        readiness_provisional=readiness_provisional,
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
    if nearby_home_comparison_safeguard_triggered:
        scoring_notes.append(nearby_home_comparison_safeguard_message)
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
    region_readiness = _coerce_region_readiness(property_level_context.get("region_property_specific_readiness"))
    region_required_missing_layers = list(property_level_context.get("region_required_layers_missing") or [])
    region_optional_missing_layers = list(property_level_context.get("region_optional_layers_missing") or [])
    region_enrichment_missing_layers = list(property_level_context.get("region_enrichment_layers_missing") or [])
    if region_readiness != "property_specific_ready":
        scoring_notes.append(
            f"Prepared region readiness is {region_readiness}; output confidence/specificity is capped."
        )
    if region_required_missing_layers:
        scoring_notes.append(
            "Prepared region still reports required-layer gaps: "
            + ", ".join(str(v) for v in region_required_missing_layers[:6])
            + "."
        )
    if region_optional_missing_layers or region_enrichment_missing_layers:
        scoring_notes.append(
            "Prepared region optional/enrichment gaps: "
            + ", ".join(
                sorted(
                    set(
                        [str(v) for v in region_optional_missing_layers[:6]]
                        + [str(v) for v in region_enrichment_missing_layers[:6]]
                    )
                )[:8]
            )
            + "."
        )
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
    if bool(coverage_preflight.get("limited_assessment_flag")):
        scoring_notes.append(
            f"Coverage preflight downgraded specificity to {coverage_preflight.get('assessment_specificity_tier')} "
            f"({float(coverage_preflight.get('feature_coverage_percent') or 0.0):.1f}% feature coverage)."
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
    if calibration_payload.get("calibration_applied"):
        scoring_notes.append(
            "Public-outcome calibration applied to produce calibrated damage-likelihood guidance."
        )
        property_level_context["calibration"] = calibration_payload
    elif calibration_payload.get("calibration_status") == "out_of_scope":
        warning = str(calibration_payload.get("scope_warning") or "").strip()
        if warning:
            scoring_notes.append(f"Calibration out of scope: {warning}")

    def _score_phrase(label: str, value: float | None) -> str:
        return f"{label}: {value:.1f}/100" if value is not None else f"{label}: not computed"

    assumptions_and_unknowns = list(
        dict.fromkeys(
            list(confidence_summary.missing_data[:6])
            + list(confidence_summary.estimated_data[:6])
            + list(confidence_summary.fallback_assumptions[:6])
            + list(confidence_summary.accuracy_improvements[:4])
        )
    )[:12]

    if scoring_status == "insufficient_data_to_score":
        explanation_summary = (
            f"{_score_phrase('Site Hazard', site_hazard_score)}. "
            f"{_score_phrase('Home Ignition Vulnerability', home_ignition_vulnerability_score)}. "
            f"{_score_phrase('Home Hardening Readiness', insurance_readiness_score)}. "
            "We found some relevant signals, but not enough verified inputs to compute a reliable score."
        )
    else:
        drivers_phrase = ", ".join(top_risk_drivers[:2]) if top_risk_drivers else "Limited signal set"
        near_structure_phrase = str(defensible_space_analysis.get("summary") or "Not available")
        if "home_ignition_vulnerability" not in computed_components:
            near_structure_phrase = (
                "Preliminary near-structure observation only; this component was not used in a computed vulnerability score."
            )
        explanation_summary = (
            f"{_score_phrase('Site Hazard', site_hazard_score)}. "
            f"{_score_phrase('Home Ignition Vulnerability', home_ignition_vulnerability_score)}. "
            f"{_score_phrase('Home Hardening Readiness', insurance_readiness_score)}. "
            f"Primary drivers: {drivers_phrase}. "
            f"Near-structure summary: {near_structure_phrase}."
        )

    risk_scores = RiskScores(
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        wildfire_risk_score=blended_wildfire_risk_score,
        insurance_readiness_score=insurance_readiness_score,
        overall_wildfire_risk=blended_wildfire_risk_score,
        home_hardening_readiness=insurance_readiness_score,
        site_hazard_score_available=bool(score_outputs["site_hazard_score_available"]),
        home_ignition_vulnerability_score_available=bool(score_outputs["home_ignition_vulnerability_score_available"]),
        wildfire_risk_score_available=bool(score_outputs["wildfire_risk_score_available"]),
        insurance_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
        home_hardening_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
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
    calibrated_public_outcome_metadata = _build_calibrated_public_outcome_metadata(
        requested=bool(include_calibrated_outputs or payload.include_calibrated_outputs),
        calibration_payload=calibration_payload,
        calibration_version=str(governance["calibration_version"] or CALIBRATION_VERSION),
        raw_wildfire_risk_score=blended_wildfire_risk_score,
    )
    if calibrated_public_outcome_metadata is not None:
        property_level_context["calibrated_public_outcome_metadata"] = (
            calibrated_public_outcome_metadata.model_dump()
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
        overall_wildfire_risk=blended_wildfire_risk_score,
        legacy_weighted_wildfire_risk_score=legacy_weighted_wildfire_risk_score,
        site_hazard_score=site_hazard_score,
        home_ignition_vulnerability_score=home_ignition_vulnerability_score,
        insurance_readiness_score=insurance_readiness_score,
        home_hardening_readiness=insurance_readiness_score,
        calibrated_damage_likelihood=(
            float(calibration_payload.get("calibrated_damage_likelihood"))
            if calibration_payload.get("calibrated_damage_likelihood") is not None
            else None
        ),
        empirical_damage_likelihood_proxy=(
            float(calibration_payload.get("empirical_damage_likelihood_proxy"))
            if calibration_payload.get("empirical_damage_likelihood_proxy") is not None
            else None
        ),
        empirical_loss_likelihood_proxy=(
            float(calibration_payload.get("empirical_loss_likelihood_proxy"))
            if calibration_payload.get("empirical_loss_likelihood_proxy") is not None
            else None
        ),
        calibration_applied=bool(calibration_payload.get("calibration_applied")),
        calibration_method=(
            str(calibration_payload.get("calibration_method"))
            if calibration_payload.get("calibration_method")
            else None
        ),
        calibration_status=str(calibration_payload.get("calibration_status") or "disabled"),
        calibration_limitations=[
            str(item) for item in (calibration_payload.get("calibration_limitations") or []) if str(item).strip()
        ],
        calibration_scope_warning=(
            str(calibration_payload.get("scope_warning"))
            if calibration_payload.get("scope_warning")
            else None
        ),
        calibrated_public_outcome_metadata=calibrated_public_outcome_metadata,
        wildfire_risk_score_available=bool(score_outputs["wildfire_risk_score_available"]),
        site_hazard_score_available=bool(score_outputs["site_hazard_score_available"]),
        home_ignition_vulnerability_score_available=bool(score_outputs["home_ignition_vulnerability_score_available"]),
        insurance_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
        home_hardening_readiness_score_available=bool(score_outputs["insurance_readiness_score_available"]),
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
        near_structure_features=dict(property_level_context.get("near_structure_features") or {}),
        parcel_based_metrics=dict(property_level_context.get("parcel_based_metrics") or {}),
        directional_risk=dict(property_level_context.get("directional_risk") or {}),
        structure_relative_slope=dict(property_level_context.get("structure_relative_slope") or {}),
        structure_attributes=dict(property_level_context.get("structure_attributes") or {}),
        top_risk_drivers=top_risk_drivers,
        top_risk_drivers_detailed=ranked_driver_details[:3],
        prioritized_mitigation_actions=prioritized_mitigation_actions,
        confidence_summary=confidence_summary,
        top_recommended_actions=top_recommended_actions,
        top_protective_factors=top_protective_factors,
        explanation_summary=explanation_summary,
        confirmed_inputs=assumptions_block.confirmed_inputs,
        observed_inputs=assumptions_block.observed_inputs,
        inferred_inputs=assumptions_block.inferred_inputs,
        missing_inputs=assumptions_block.missing_inputs,
        assumptions_used=all_assumptions,
        assumptions_and_unknowns=assumptions_and_unknowns,
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
        feature_coverage_summary=dict(coverage_preflight.get("feature_coverage_summary") or {}),
        feature_coverage_percent=float(coverage_preflight.get("feature_coverage_percent") or 0.0),
        assessment_specificity_tier=str(coverage_preflight.get("assessment_specificity_tier") or "regional_estimate"),
        specificity_summary=dict(specificity_summary),
        geometry_resolution=geometry_resolution,
        footprint_resolution=dict(property_level_context.get("footprint_resolution") or {}),
        parcel_resolution=dict(property_level_context.get("parcel_resolution") or {}),
        property_linkage=dict(property_level_context.get("property_linkage") or {}),
        assessment_output_state=str(assessment_output_state or "insufficient_data"),
        assessment_mode=homeowner_assessment_mode,
        scoring_status=scoring_status,
        computed_components=computed_components,
        blocked_components=blocked_components,
        minimum_missing_requirements=minimum_missing_requirements,
        recommended_data_improvements=recommended_data_improvements,
        limited_assessment_flag=bool(coverage_preflight.get("limited_assessment_flag")),
        confidence_not_meaningful=bool(assessment_output_state == "insufficient_data"),
        observed_factor_count=int(risk.observed_factor_count),
        missing_factor_count=int(risk.missing_factor_count),
        fallback_factor_count=int(risk.fallback_factor_count),
        observed_feature_count=int(coverage_preflight.get("observed_feature_count") or risk.observed_feature_count),
        inferred_feature_count=int(coverage_preflight.get("inferred_feature_count") or risk.inferred_feature_count),
        fallback_feature_count=int(coverage_preflight.get("fallback_feature_count") or risk.fallback_feature_count),
        missing_feature_count=int(coverage_preflight.get("missing_feature_count") or 0),
        observed_weight_fraction=float(risk.observed_weight_fraction),
        fallback_dominance_ratio=float(risk.fallback_dominance_ratio),
        fallback_weight_fraction=float(risk.fallback_weight_fraction),
        structure_data_completeness=structure_data_completeness,
        structure_assumption_mode=structure_assumption_mode,
        structure_score_confidence=structure_score_confidence,
        geometry_quality_score=float(coverage_preflight.get("geometry_quality_score") or risk.geometry_quality_score),
        regional_context_coverage_score=float(
            coverage_preflight.get("regional_context_coverage_score") or risk.regional_context_coverage_score
        ),
        property_specificity_score=float(coverage_preflight.get("property_specificity_score") or risk.property_specificity_score),
        property_data_confidence=float(coverage_preflight.get("property_data_confidence") or 0.0),
        property_confidence_summary=dict(coverage_preflight.get("property_confidence_summary") or {}),
        score_specificity_warning=(
            str(coverage_preflight.get("score_specificity_warning"))
            if coverage_preflight.get("score_specificity_warning")
            else None
        ),
        data_quality_summary=data_quality_summary,
        assessment_limitations=grouped_limitations,
        what_was_observed=what_was_observed,
        what_was_estimated=what_was_estimated,
        what_was_missing=what_was_missing,
        why_this_result_is_limited=why_limited,
        developer_diagnostics=developer_diagnostics,
        homeowner_summary=homeowner_summary,
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
        parcel_source=(
            str(property_level_context.get("parcel_source"))
            if property_level_context.get("parcel_source")
            else (
                str(property_level_context.get("parcel_source_name"))
                if property_level_context.get("parcel_source_name")
                else None
            )
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
        geometry_source=(
            str(property_level_context.get("geometry_source"))
            if property_level_context.get("geometry_source")
            else None
        ),
        geometry_confidence=(
            float(property_level_context.get("geometry_confidence"))
            if property_level_context.get("geometry_confidence") is not None
            else None
        ),
        ring_generation_mode=(
            str(property_level_context.get("ring_generation_mode"))
            if property_level_context.get("ring_generation_mode")
            else None
        ),
        property_mismatch_flag=bool(
            geometry_resolution.property_mismatch_flag
            if hasattr(geometry_resolution, "property_mismatch_flag")
            else property_level_context.get("property_mismatch_flag")
        ),
        mismatch_reason=(
            str(
                geometry_resolution.mismatch_reason
                if hasattr(geometry_resolution, "mismatch_reason")
                else (property_level_context.get("mismatch_reason") or "")
            ).strip()
            or None
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
    improve_your_result = build_improve_your_result_block(result)
    homeowner_summary_with_improvement = dict(result.homeowner_summary or {})
    homeowner_summary_with_improvement["improve_your_result"] = improve_your_result
    result.homeowner_summary = homeowner_summary_with_improvement
    score_variance_diagnostics = _build_score_variance_diagnostics(
        context=context,
        risk=risk,
        submodel_scores=submodel_scores,
        weighted_contributions=weighted_contributions,
        property_level_context=property_level_context,
        input_source_metadata=input_source_metadata,
        resolved_region_id=result.resolved_region_id,
        coverage_preflight=coverage_preflight,
    )

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
        "geometry_resolution": result.geometry_resolution.model_dump(),
        "footprint_resolution": result.footprint_resolution.model_dump(),
        "parcel_resolution": result.parcel_resolution.model_dump(),
        "property_linkage": result.property_linkage.model_dump(),
        "submodel_scores": {
            name: {
                "score": sm.score,
                "weighted_contribution": sm.weighted_contribution,
                "explanation": sm.explanation,
                "key_inputs": sm.key_inputs,
                "assumptions": sm.assumptions,
                "raw_score_before_clamp": (
                    risk.submodel_scores.get(name).raw_score
                    if risk.submodel_scores.get(name) is not None
                    else None
                ),
                "clamped_score": (
                    risk.submodel_scores.get(name).clamped_score
                    if risk.submodel_scores.get(name) is not None
                    else sm.score
                ),
            }
            for name, sm in submodel_scores.items()
        },
        "weighted_contributions": {
            name: {
                "weight": wc.weight,
                "base_weight": wc.base_weight,
                "effective_weight": wc.effective_weight,
                "observed_fraction": wc.observed_fraction,
                "omitted_due_to_missing": wc.omitted_due_to_missing,
                "factor_evidence_status": wc.factor_evidence_status,
                "score": wc.score,
                "contribution": wc.contribution,
            }
            for name, wc in weighted_contributions.items()
        },
        "readiness": {
            "score": result.insurance_readiness_score,
            "home_hardening_readiness": result.home_hardening_readiness,
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
            "overall_wildfire_risk": result.overall_wildfire_risk,
            "wildfire_risk_score_available": result.wildfire_risk_score_available,
            "legacy_weighted_wildfire_risk_score": result.legacy_weighted_wildfire_risk_score,
            "insurance_readiness_score": result.insurance_readiness_score,
            "home_hardening_readiness": result.home_hardening_readiness,
            "insurance_readiness_score_available": result.insurance_readiness_score_available,
            "home_hardening_readiness_score_available": result.home_hardening_readiness_score_available,
            "calibrated_damage_likelihood": result.calibrated_damage_likelihood,
            "empirical_damage_likelihood_proxy": result.empirical_damage_likelihood_proxy,
            "empirical_loss_likelihood_proxy": result.empirical_loss_likelihood_proxy,
            "calibration_applied": result.calibration_applied,
            "calibration_method": result.calibration_method,
            "calibration_status": result.calibration_status,
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
            "assessment_output_state": result.assessment_output_state,
        },
        "assessment_limitations_summary": list(result.assessment_limitations_summary),
        "assessment_limitations": list(result.assessment_limitations),
        "assessment_mode": result.assessment_mode,
        "what_was_observed": list(result.what_was_observed),
        "what_was_estimated": list(result.what_was_estimated),
        "what_was_missing": list(result.what_was_missing),
        "why_this_result_is_limited": result.why_this_result_is_limited,
        "coverage": {
            "direct_data_coverage_score": result.direct_data_coverage_score,
            "inferred_data_coverage_score": result.inferred_data_coverage_score,
            "missing_data_share": result.missing_data_share,
            "stale_data_share": result.data_provenance.summary.stale_data_share,
            "heuristic_input_count": result.data_provenance.summary.heuristic_input_count,
            "feature_coverage_summary": result.feature_coverage_summary,
            "feature_coverage_percent": result.feature_coverage_percent,
            "assessment_specificity_tier": result.assessment_specificity_tier,
            "assessment_output_state": result.assessment_output_state,
            "limited_assessment_flag": result.limited_assessment_flag,
            "score_specificity_warning": result.score_specificity_warning,
            "confidence_not_meaningful": result.confidence_not_meaningful,
            "data_quality_summary": result.data_quality_summary,
            "observed_feature_count": result.observed_feature_count,
            "inferred_feature_count": result.inferred_feature_count,
            "fallback_feature_count": result.fallback_feature_count,
            "missing_feature_count": result.missing_feature_count,
            "structure_data_completeness": result.structure_data_completeness,
            "structure_assumption_mode": result.structure_assumption_mode,
            "structure_score_confidence": result.structure_score_confidence,
            "fallback_evidence_fraction": round(
                float(result.fallback_feature_count)
                / float(
                    max(
                        1,
                        int(result.observed_feature_count)
                        + int(result.inferred_feature_count)
                        + int(result.fallback_feature_count)
                        + int(result.missing_feature_count),
                    )
                ),
                4,
            ),
            "scoring_fallback_weight_fraction": result.fallback_weight_fraction,
            "geometry_quality_score": result.geometry_quality_score,
            "regional_context_coverage_score": result.regional_context_coverage_score,
            "property_specificity_score": result.property_specificity_score,
        },
        "feature_bundle_data_sources": (
            (property_level_context.get("feature_bundle_data_sources") or {})
            if isinstance(property_level_context, dict)
            else {}
        ),
        "feature_bundle_summary": (
            (property_level_context.get("feature_bundle_summary") or {})
            if isinstance(property_level_context, dict)
            else {}
        ),
        "layer_coverage_audit": [row.model_dump() for row in result.layer_coverage_audit],
        "coverage_summary": result.coverage_summary.model_dump(),
        "calibration": {
            **calibration_payload,
            "top_calibration_drivers": top_risk_drivers[:3],
            "calibration_warning": (
                calibration_payload.get("scope_warning")
                if calibration_payload.get("calibration_status") == "out_of_scope"
                else None
            ),
        },
        "raw_feature_vector": score_variance_diagnostics.get("raw_feature_vector", {}),
        "transformed_feature_vector": score_variance_diagnostics.get("transformed_feature_vector", {}),
        "factor_contribution_breakdown": score_variance_diagnostics.get("factor_contribution_breakdown", {}),
        "compression_flags": score_variance_diagnostics.get("compression_flags", []),
        "score_variance_diagnostics": score_variance_diagnostics,
        "fallback_dominance_ratio": result.fallback_dominance_ratio,
        "scoring_status": result.scoring_status,
        "computed_components": list(result.computed_components),
        "blocked_components": list(result.blocked_components),
        "minimum_missing_requirements": list(result.minimum_missing_requirements),
        "recommended_data_improvements": list(result.recommended_data_improvements),
        "missing_core_layer_count": int(coverage_preflight.get("missing_core_layer_count") or 0),
        "observed_feature_count": result.observed_feature_count,
        "inferred_feature_count": result.inferred_feature_count,
        "fallback_feature_count": result.fallback_feature_count,
        "missing_feature_count": result.missing_feature_count,
        "observed_weight_fraction": result.observed_weight_fraction,
        "fallback_weight_fraction": result.fallback_weight_fraction,
        "geometry_quality_score": result.geometry_quality_score,
        "regional_context_coverage_score": result.regional_context_coverage_score,
        "property_specificity_score": result.property_specificity_score,
        "score_specificity_warning": result.score_specificity_warning,
        "defensible_space_analysis": result.defensible_space_analysis,
        "top_risk_drivers": result.top_risk_drivers,
        "top_risk_drivers_detailed": [row.model_dump() for row in result.top_risk_drivers_detailed],
        "top_near_structure_risk_drivers": result.top_near_structure_risk_drivers,
        "prioritized_vegetation_actions": [a.model_dump() for a in result.prioritized_vegetation_actions],
        "prioritized_mitigation_actions": [row.model_dump() for row in result.prioritized_mitigation_actions],
        "defensible_space_limitations_summary": result.defensible_space_limitations_summary,
        "top_recommended_actions": result.top_recommended_actions,
        "confidence_summary": result.confidence_summary.model_dump(),
        "homeowner_summary": result.homeowner_summary,
        "developer_diagnostics": result.developer_diagnostics,
        "assumptions_and_unknowns": result.assumptions_and_unknowns,
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
    include_calibrated_outputs: bool = False,
) -> AssessmentResult:
    result, _ = _run_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        portfolio_name=portfolio_name,
        tags=tags,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
        include_calibrated_outputs=include_calibrated_outputs,
    )
    return result


def _risk_band_for_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    thresholds = scoring_config.risk_bucket_thresholds or {}
    try:
        low_max = float(thresholds.get("low_max", 40.0))
    except (TypeError, ValueError):
        low_max = 40.0
    try:
        medium_max = float(thresholds.get("medium_max", 60.0))
    except (TypeError, ValueError):
        medium_max = 60.0
    if score < low_max:
        return "low"
    if score < medium_max:
        return "moderate"
    return "high"


def _clone_payload_with_attribute_overrides(
    payload: AddressRequest,
    *,
    overrides: dict[str, object],
    remove_confirmed_fields: list[str] | None = None,
) -> AddressRequest:
    cloned = payload.model_copy(deep=True)
    attrs = cloned.attributes.model_dump()
    for key, value in overrides.items():
        attrs[key] = value
    cloned.attributes = PropertyAttributes.model_validate(attrs)
    confirmed = [str(token) for token in (cloned.confirmed_fields or [])]
    if remove_confirmed_fields:
        removal = {str(token) for token in remove_confirmed_fields}
        confirmed = [token for token in confirmed if token not in removal]
    for key, value in overrides.items():
        if value is not None and str(key) not in confirmed:
            confirmed.append(str(key))
    cloned.confirmed_fields = sorted(set(confirmed))
    return cloned


def _jitter_geocode_resolution(
    base: GeocodeResolution,
    *,
    lat_offset: float,
    lon_offset: float,
) -> GeocodeResolution:
    lat = float(base.latitude) + float(lat_offset)
    lon = float(base.longitude) + float(lon_offset)
    meta = dict(base.geocode_meta or {})
    meta["resolved_latitude"] = lat
    meta["resolved_longitude"] = lon
    meta["final_coordinates_used"] = {"latitude": lat, "longitude": lon}
    meta["geocoded_point"] = {"latitude": lat, "longitude": lon}
    return GeocodeResolution(
        raw_input=base.raw_input,
        normalized_address=base.normalized_address,
        geocode_status=base.geocode_status,
        candidate_count=base.candidate_count,
        selected_candidate=base.selected_candidate,
        confidence_score=base.confidence_score,
        latitude=lat,
        longitude=lon,
        geocode_source=base.geocode_source,
        geocode_meta=meta,
        geocode_outcome=base.geocode_outcome,
        trusted_match_status=base.trusted_match_status,
        rejection_reason=base.rejection_reason,
    )


def _build_assessment_trust_metadata(
    *,
    result: AssessmentResult,
    payload: AddressRequest,
    organization_id: str,
    ruleset: UnderwritingRuleset,
    geocode_resolution: GeocodeResolution,
    coverage_resolution: RegionCoverageResolution,
) -> Any:
    base_risk = float(result.wildfire_risk_score) if result.wildfire_risk_score is not None else None
    base_readiness = (
        float(result.insurance_readiness_score) if result.insurance_readiness_score is not None else None
    )
    base_conf_tier = str(result.confidence_tier or "unknown")
    base_band = _risk_band_for_score(result.wildfire_risk_score)
    stability_samples: list[dict[str, object]] = []
    mitigation_samples: list[dict[str, object]] = []

    def _risk_delta(variant: AssessmentResult) -> float | None:
        if base_risk is None or variant.wildfire_risk_score is None:
            return None
        return float(variant.wildfire_risk_score) - base_risk

    def _readiness_delta(variant: AssessmentResult) -> float | None:
        if base_readiness is None or variant.insurance_readiness_score is None:
            return None
        return float(variant.insurance_readiness_score) - base_readiness

    def _run_variant(
        variant_payload: AddressRequest,
        *,
        geocode_override: GeocodeResolution | None = None,
    ) -> AssessmentResult | None:
        try:
            return _compute_assessment(
                variant_payload,
                organization_id=organization_id,
                ruleset=ruleset,
                geocode_resolution=geocode_override or geocode_resolution,
                coverage_resolution=coverage_resolution,
            )
        except Exception:
            return None

    # Local stability checks: tiny geocode jitter + fallback-assumption perturbation.
    jitter_offsets = [("jitter_north", 0.00010, 0.0), ("jitter_west", 0.0, -0.00010)]
    for name, dlat, dlon in jitter_offsets:
        jitter_geo = _jitter_geocode_resolution(geocode_resolution, lat_offset=dlat, lon_offset=dlon)
        jitter_payload = payload.model_copy(deep=True)
        jitter_payload.selection_mode = "point"
        jitter_payload.property_anchor_point = Coordinates(latitude=jitter_geo.latitude, longitude=jitter_geo.longitude)
        jitter_payload.user_selected_point = Coordinates(latitude=jitter_geo.latitude, longitude=jitter_geo.longitude)
        variant = _run_variant(jitter_payload, geocode_override=jitter_geo)
        if variant is None:
            continue
        stability_samples.append(
            {
                "name": name,
                "sample_type": "geocode_jitter",
                "risk_delta": _risk_delta(variant),
                "tier_changed": str(variant.confidence_tier or "unknown") != base_conf_tier,
                "band_changed": _risk_band_for_score(variant.wildfire_risk_score) != base_band,
            }
        )

    fallback_payload = _clone_payload_with_attribute_overrides(
        payload,
        overrides={"roof_type": None, "vent_type": None, "defensible_space_ft": None},
        remove_confirmed_fields=["roof_type", "vent_type", "defensible_space_ft"],
    )
    fallback_variant = _run_variant(fallback_payload)
    if fallback_variant is not None:
        stability_samples.append(
            {
                "name": "fallback_assumption_toggle",
                "sample_type": "fallback_assumption",
                "risk_delta": _risk_delta(fallback_variant),
                "tier_changed": str(fallback_variant.confidence_tier or "unknown") != base_conf_tier,
                "band_changed": _risk_band_for_score(fallback_variant.wildfire_risk_score) != base_band,
            }
        )

    # Mitigation sensitivity checks (bounded counterfactual variants).
    current_defensible = payload.attributes.defensible_space_ft
    target_defensible = max(30.0, float(current_defensible or 0.0))
    mitigation_variants: list[tuple[str, AddressRequest, str, list[str]]] = [
        (
            "clear_0_5ft_zone",
            _clone_payload_with_attribute_overrides(
                payload,
                overrides={"defensible_space_ft": target_defensible},
            ),
            "down",
            ["Approximation from defensible-space input rather than imagery-derived ring editing."],
        ),
        (
            "upgrade_roof_class_a",
            _clone_payload_with_attribute_overrides(payload, overrides={"roof_type": "class a"}),
            "down",
            [],
        ),
        (
            "add_ember_resistant_vents",
            _clone_payload_with_attribute_overrides(payload, overrides={"vent_type": "ember-resistant"}),
            "down",
            [],
        ),
        (
            "degrade_hardening_control",
            _clone_payload_with_attribute_overrides(
                payload,
                overrides={"roof_type": "wood", "vent_type": "standard", "defensible_space_ft": 5.0},
            ),
            "up",
            [],
        ),
    ]
    for name, variant_payload, expected_direction, notes in mitigation_variants:
        variant = _run_variant(variant_payload)
        if variant is None:
            continue
        mitigation_samples.append(
            {
                "name": name,
                "expected_direction": expected_direction,
                "risk_delta": _risk_delta(variant),
                "readiness_delta": _readiness_delta(variant),
                "notes": notes,
            }
        )

    reference_artifacts = load_trust_reference_artifacts()
    return build_trust_diagnostics(
        result=result,
        stability_samples=stability_samples,
        mitigation_samples=mitigation_samples,
        reference_artifacts=reference_artifacts,
    )


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


def _geocode_resolution_from_assessment(existing: AssessmentResult) -> GeocodeResolution:
    geocode = existing.geocoding if isinstance(existing.geocoding, GeocodingDetails) else GeocodingDetails()
    geocode_meta = geocode.model_dump(mode="json")
    lat = float(existing.latitude)
    lon = float(existing.longitude)
    if not isinstance(geocode_meta.get("resolved_latitude"), (int, float)):
        geocode_meta["resolved_latitude"] = lat
    if not isinstance(geocode_meta.get("resolved_longitude"), (int, float)):
        geocode_meta["resolved_longitude"] = lon
    if not isinstance(geocode_meta.get("geocoded_point"), dict):
        geocode_meta["geocoded_point"] = {"latitude": lat, "longitude": lon}
    if not isinstance(geocode_meta.get("final_coordinates_used"), dict):
        geocode_meta["final_coordinates_used"] = {"latitude": lat, "longitude": lon}
    return GeocodeResolution(
        raw_input=existing.address,
        normalized_address=str(geocode.normalized_address or normalize_address(existing.address)),
        geocode_status=str(geocode.geocode_status or "accepted"),
        candidate_count=int(geocode.candidate_count or 0),
        selected_candidate=(
            geocode.final_candidate_selected
            if isinstance(geocode.final_candidate_selected, dict)
            else None
        ),
        confidence_score=(
            float(geocode.confidence_score)
            if geocode.confidence_score is not None
            else None
        ),
        latitude=lat,
        longitude=lon,
        geocode_source=str(geocode.geocode_source or geocode.provider or "stored_assessment"),
        geocode_meta=geocode_meta,
        geocode_outcome=str(geocode.geocode_outcome or "geocode_succeeded_trusted"),
        trusted_match_status=str(geocode.trusted_match_status or "trusted"),
        rejection_reason=geocode.rejection_reason,
    )


def _coverage_resolution_from_assessment(existing: AssessmentResult) -> RegionCoverageResolution:
    region_resolution = (
        existing.region_resolution
        if isinstance(existing.region_resolution, RegionResolution)
        else RegionResolution()
    )
    coverage_payload = {
        "covered": bool(existing.coverage_available or region_resolution.coverage_available),
        "coverage_available": bool(existing.coverage_available or region_resolution.coverage_available),
        "resolved_region_id": existing.resolved_region_id or region_resolution.resolved_region_id,
        "selected_region_id": existing.resolved_region_id or region_resolution.resolved_region_id,
        "selected_region_display_name": region_resolution.resolved_region_display_name,
        "reason": region_resolution.reason,
        "diagnostics": list(region_resolution.diagnostics or []),
    }
    return RegionCoverageResolution(
        coverage_available=bool(coverage_payload.get("coverage_available")),
        resolved_region_id=(
            str(coverage_payload.get("resolved_region_id"))
            if coverage_payload.get("resolved_region_id")
            else None
        ),
        reason=str(coverage_payload.get("reason") or "unknown"),
        diagnostics=list(coverage_payload.get("diagnostics") or []),
        coverage=coverage_payload,
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
            "prioritized_mitigation_actions": [m.model_dump() for m in result.prioritized_mitigation_actions[:5]],
            "plain_language_summary": result.explanation_summary,
            "near_structure_summary": result.defensible_space_analysis.get("summary"),
            "top_risk_drivers_detailed": [d.model_dump() for d in result.top_risk_drivers_detailed[:3]],
            "confidence_summary": result.confidence_summary.model_dump(),
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
        "assumptions_and_unknowns": result.assumptions_and_unknowns,
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
        "feature_coverage_summary": result.feature_coverage_summary,
        "feature_coverage_percent": result.feature_coverage_percent,
        "assessment_specificity_tier": result.assessment_specificity_tier,
        "assessment_output_state": result.assessment_output_state,
        "limited_assessment_flag": result.limited_assessment_flag,
        "confidence_not_meaningful": result.confidence_not_meaningful,
        "observed_factor_count": result.observed_factor_count,
        "missing_factor_count": result.missing_factor_count,
        "fallback_factor_count": result.fallback_factor_count,
        "observed_weight_fraction": result.observed_weight_fraction,
        "fallback_dominance_ratio": result.fallback_dominance_ratio,
        "score_specificity_warning": result.score_specificity_warning,
        "data_quality_summary": result.data_quality_summary,
        "assessment_limitations": result.assessment_limitations,
        "what_was_observed": result.what_was_observed,
        "what_was_estimated": result.what_was_estimated,
        "what_was_missing": result.what_was_missing,
        "why_this_result_is_limited": result.why_this_result_is_limited,
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
            "overall_wildfire_risk": result.overall_wildfire_risk,
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
        home_hardening_readiness_summary={
            "home_hardening_readiness": result.home_hardening_readiness,
            "home_hardening_readiness_score_available": result.home_hardening_readiness_score_available,
            "home_hardening_section": result.insurance_readiness_section.model_dump(),
            "home_hardening_eligibility": result.insurance_readiness_eligibility.model_dump(),
            "readiness_factors": [r.model_dump() for r in result.readiness_factors],
            "readiness_blockers": result.readiness_blockers,
            "readiness_penalties": result.readiness_penalties,
            "readiness_summary": result.readiness_summary,
            "top_recommended_actions": result.top_recommended_actions,
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
        top_risk_drivers_detailed=result.top_risk_drivers_detailed,
        prioritized_mitigation_actions=result.prioritized_mitigation_actions,
        confidence_summary=result.confidence_summary,
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
<div class=\"card\"><h3>Home Hardening Readiness</h3><p>{_score_html(result.home_hardening_readiness)}</p></div>
</div>
<div class=\"card\"><h3>Top Risk Drivers</h3><ul>{drivers}</ul></div>
<div class=\"card\"><h3>Near-Structure Drivers</h3><ul>{near_structure_drivers}</ul></div>
<div class=\"card\"><h3>Top Protective Factors</h3><ul>{protective}</ul></div>
<div class=\"card\"><h3>Home Hardening Blockers</h3><ul>{blockers}</ul></div>
<div class=\"card\"><h3>Mitigation Recommendations</h3><ul>{mitigations}</ul></div>
<div class=\"card\"><h3>Prioritized Vegetation Actions</h3><ul>{vegetation_actions}</ul></div>
<div class=\"card\"><h3>Insurance Readiness (Optional/Future-facing)</h3><p>{_score_html(result.insurance_readiness_score)}</p></div>
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


def _comparison_safeguard_from_result(result: AssessmentResult) -> tuple[bool, str]:
    homeowner_summary = result.homeowner_summary if isinstance(result.homeowner_summary, dict) else {}
    trust_summary = homeowner_summary.get("trust_summary") if isinstance(homeowner_summary.get("trust_summary"), dict) else {}
    trust_summary = trust_summary if isinstance(trust_summary, dict) else {}
    mode = str(trust_summary.get("differentiation_mode") or "mostly_regional")
    try:
        confidence = float(trust_summary.get("neighborhood_differentiation_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    triggered = should_trigger_nearby_home_comparison_safeguard(mode, confidence)
    message = str(
        trust_summary.get("nearby_home_comparison_safeguard_message")
        or "This estimate is not precise enough to compare adjacent homes."
    ).strip()
    return triggered, message


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
    base_guardrail, base_message = _comparison_safeguard_from_result(base)
    other_guardrail, other_message = _comparison_safeguard_from_result(other)
    comparison_safeguard_triggered = bool(base_guardrail or other_guardrail)
    comparison_safeguard_message = base_message if base_guardrail else other_message
    if comparison_safeguard_triggered:
        version_comparison = dict(version_comparison or {})
        version_comparison["comparison_precision_safeguard"] = {
            "triggered": True,
            "message": comparison_safeguard_message,
            "base_triggered": bool(base_guardrail),
            "other_triggered": bool(other_guardrail),
            "reason": "low_neighborhood_differentiation_confidence",
        }

    return AssessmentComparisonResult(
        base=_to_comparison_item(base),
        other=_to_comparison_item(other),
        wildfire_risk_delta=(
            None if comparison_safeguard_triggered else _delta(base.wildfire_risk_score, other.wildfire_risk_score)
        ),
        insurance_readiness_delta=(
            None if comparison_safeguard_triggered else _delta(base.insurance_readiness_score, other.insurance_readiness_score)
        ),
        driver_differences=(
            {"added": [], "removed": []}
            if comparison_safeguard_triggered
            else {
                "added": sorted(other_drivers - base_drivers),
                "removed": sorted(base_drivers - other_drivers),
            }
        ),
        blocker_differences=(
            {"added": [], "removed": []}
            if comparison_safeguard_triggered
            else {
                "added": sorted(other_blockers - base_blockers),
                "removed": sorted(base_blockers - other_blockers),
            }
        ),
        mitigation_differences=(
            {"added": [], "removed": []}
            if comparison_safeguard_triggered
            else {
                "added": sorted(other_mitigations - base_mitigations),
                "removed": sorted(base_mitigations - other_mitigations),
            }
        ),
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


def _secondary_geocoder_enabled() -> bool:
    if not _env_flag("WF_GEOCODE_SECONDARY_ENABLED", False):
        return False
    secondary_url = str(os.getenv("WF_GEOCODE_SECONDARY_SEARCH_URL") or "").strip()
    if not secondary_url:
        return False
    return True


def _coerce_point_payload(raw_point: Any) -> dict[str, float] | None:
    if raw_point is None:
        return None
    try:
        if isinstance(raw_point, dict):
            lat = float(raw_point.get("latitude"))
            lon = float(raw_point.get("longitude"))
        else:
            lat = float(getattr(raw_point, "latitude"))
            lon = float(getattr(raw_point, "longitude"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"latitude": lat, "longitude": lon}


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
        "no_match": (
            "No geocoding candidates were returned by the configured providers. "
            "Verify street/city/state/ZIP, or continue with a user-selected map point when available."
        ),
        "ambiguous_match": "Address matched multiple possible locations. Add city/state or ZIP and try again.",
        "low_confidence": "Address match confidence was below policy threshold. Add more address detail and retry.",
        "trust_filter_rejected": "Address candidates were found but rejected by trust filters. Add city/state or ZIP and try again.",
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
    coverage = (len(overlap) / max(1, len(submitted_set))) if submitted_set else 0.0
    submitted_house = (
        re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\b", submitted_norm).group(1)
        if re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\b", submitted_norm)
        else ""
    )
    candidate_house = (
        re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\b", candidate_norm).group(1)
        if re.match(r"^\s*(\d+[a-zA-Z0-9-]*)\b", candidate_norm)
        else ""
    )
    return {
        "submitted_normalized": submitted_norm,
        "candidate_normalized": candidate_norm,
        "exact_normalized_match": submitted_norm == candidate_norm if submitted_norm and candidate_norm else False,
        "token_similarity_ratio": round(similarity, 3),
        "token_coverage_ratio": round(coverage, 3),
        "house_number_match": bool(submitted_house and candidate_house and submitted_house == candidate_house),
        "matched_tokens": overlap[:12],
        "missing_from_candidate": sorted(submitted_set - candidate_set)[:8],
        "extra_in_candidate": sorted(candidate_set - submitted_set)[:8],
    }


def _derive_geocode_trust_tier(
    *,
    geocode_precision: str | None,
    confidence_score: float | None,
    trusted_match_status: str | None,
) -> str:
    status = str(trusted_match_status or "").strip().lower()
    precision = str(geocode_precision or "").strip().lower()
    if status == "trusted" and precision in {"rooftop", "parcel_or_address_point"}:
        return "high"
    if status == "trusted":
        return "medium"
    if status == "untrusted_fallback":
        if precision in {"rooftop", "parcel_or_address_point", "interpolated"}:
            return "medium"
        return "low"
    if confidence_score is not None and float(confidence_score) >= 0.2:
        return "medium"
    return "low"


def _build_trusted_match_subchecks(
    *,
    submitted_address: str,
    geocode_meta: dict[str, Any],
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matched_address = geocode_meta.get("matched_address") or geocode_meta.get("geocoded_address")
    component_cmp = _build_address_component_comparison(submitted_address, str(matched_address or ""))
    token_similarity = float(component_cmp.get("token_similarity_ratio") or 0.0)
    location_type = str(geocode_meta.get("geocode_location_type") or "").lower()
    match_count = int(geocode_meta.get("candidate_count") or 0)
    in_region = bool((coverage or {}).get("coverage_available", False))
    region_distance = (coverage or {}).get("region_distance_to_boundary_m")
    top_candidate = None
    raw_preview = geocode_meta.get("raw_response_preview")
    if isinstance(raw_preview, dict):
        top_candidate = raw_preview.get("top_candidate")
    top_address = top_candidate.get("address") if isinstance(top_candidate, dict) else None
    locality = (
        str((top_address or {}).get("city") or "").strip().lower()
        if isinstance(top_address, dict)
        else ""
    )
    submitted_norm = _normalize_address_for_component_compare(submitted_address)
    locality_match = bool(locality and locality in submitted_norm)
    state = (
        str((top_address or {}).get("state") or "").strip().lower()
        if isinstance(top_address, dict)
        else ""
    )
    state_match = bool(state and state in submitted_norm)
    postcode = (
        str((top_address or {}).get("postcode") or "").strip()
        if isinstance(top_address, dict)
        else ""
    )
    postal_match = bool(postcode and postcode[:5] in submitted_address)
    return {
        "address_string_match": bool(component_cmp.get("exact_normalized_match") or token_similarity >= 0.72),
        "locality_match": locality_match,
        "state_match": state_match,
        "postal_match": postal_match,
        "candidate_count": match_count,
        "in_region_check": in_region,
        "distance_to_region_m": region_distance,
        "within_downloaded_zone_check": in_region,
        "address_similarity_ratio": round(token_similarity, 3),
        "location_type": location_type or None,
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


def _geocode_address_or_raise(
    *,
    address: str,
    purpose: str,
    geocoder_client: Geocoder | None = None,
    provider_override: str | None = None,
) -> tuple[float, float, str, dict[str, Any]]:
    submitted_address = str(address or "")
    normalized_address = normalize_address(submitted_address)
    geocoder_client = geocoder_client or geocoder
    provider = str(provider_override or getattr(geocoder_client, "provider_name", "") or "OpenStreetMap Nominatim")

    try:
        lat, lon, geocode_source = geocoder_client.geocode(submitted_address)
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
                "geocode_trust_tier": "medium",
                "geocode_decision": "geocode_candidates_returned_but_untrusted",
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
            geocode_meta["geocode_trust_tier"] = _derive_geocode_trust_tier(
                geocode_precision=geocode_meta.get("geocode_precision"),
                confidence_score=(
                    float(geocode_meta.get("confidence_score"))
                    if geocode_meta.get("confidence_score") is not None
                    else None
                ),
                trusted_match_status=geocode_meta.get("trusted_match_status"),
            )
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
                "geocode_trust_tier": "low",
                "geocode_decision": (
                    "geocode_candidates_returned_but_untrusted"
                    if fallback_candidate is not None
                    else "no_geocode_candidates"
                ),
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
            "trusted_match_subchecks": _build_trusted_match_subchecks(
                submitted_address=submitted_address,
                geocode_meta={
                    "matched_address": (fallback_candidate or {}).get("display_name") if fallback_candidate else None,
                    "candidate_count": raw_preview.get("candidate_count") or 0,
                    "geocode_location_type": (
                        f"{str((fallback_candidate or {}).get('class') or '').strip().lower()}:"
                        f"{str((fallback_candidate or {}).get('type') or '').strip().lower()}"
                    )
                    if fallback_candidate
                    else None,
                    "raw_response_preview": raw_preview,
                },
                coverage=None,
            ),
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
            "geocode_trust_tier": "low",
            "geocode_decision": "no_geocode_candidates",
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

    geocoder_meta = getattr(geocoder_client, "last_result", None)
    geocode_meta: dict[str, Any] = {
        "geocode_status": "accepted",
        "geocode_outcome": "geocode_succeeded_trusted",
        "trusted_match_status": "trusted",
        "geocode_trust_tier": "high",
        "geocode_decision": "trusted_geocode_success",
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
        meta_normalized_address = str(geocoder_meta.get("normalized_address") or "").strip()
        if meta_normalized_address:
            if normalize_address(meta_normalized_address) != normalize_address(submitted_address):
                # Ignore stale metadata from a previous geocode invocation.
                geocoder_meta = {}
        meta_submitted_address = str(geocoder_meta.get("submitted_address") or "").strip()
        if meta_submitted_address and normalize_address(meta_submitted_address) != normalize_address(submitted_address):
            geocoder_meta = {}
    if isinstance(geocoder_meta, dict) and geocoder_meta:
        matched_address = geocoder_meta.get("matched_address")
        geocode_meta.update(
            {
                "geocode_status": _normalize_geocode_status(str(geocoder_meta.get("geocode_status") or "accepted")),
                "geocode_outcome": "geocode_succeeded_trusted",
                "trusted_match_status": "trusted",
                "geocode_decision": "trusted_geocode_success",
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

    geocode_meta["geocode_trust_tier"] = _derive_geocode_trust_tier(
        geocode_precision=geocode_meta.get("geocode_precision"),
        confidence_score=(
            float(geocode_meta.get("confidence_score"))
            if geocode_meta.get("confidence_score") is not None
            else None
        ),
        trusted_match_status=geocode_meta.get("trusted_match_status"),
    )

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


def _build_provider_attempt(
    *,
    stage: str,
    provider_name: str,
    query: str,
    accepted: bool,
    geocode_status: str,
    geocode_outcome: str | None,
    rejection_reason: str | None,
    candidate_count: int | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "provider": provider_name,
        "query": query,
        "accepted": accepted,
        "geocode_status": geocode_status,
        "geocode_outcome": geocode_outcome,
        "rejection_reason": rejection_reason,
        "candidate_count": candidate_count,
    }


def _build_provider_backoff_queries(address_input: str) -> list[str]:
    base = normalize_address(address_input)
    variants: list[str] = []
    if base:
        variants.append(base)
    no_house = re.sub(r"^\s*\d+[a-zA-Z0-9-]*\s+", "", base, count=1).strip()
    if no_house and no_house not in variants:
        variants.append(no_house)
    parts = [part.strip() for part in base.split(",") if part.strip()]
    if len(parts) >= 2:
        street_and_locality = ", ".join(parts[:2])
        if street_and_locality not in variants:
            variants.append(street_and_locality)
        street_only = parts[0]
        if street_only not in variants:
            variants.append(street_only)
    return [v for v in variants if v][:4]


def _secondary_provider_name() -> str:
    return str(
        os.getenv("WF_GEOCODE_SECONDARY_PROVIDER_NAME")
        or getattr(secondary_geocoder, "provider_name", None)
        or "Secondary Geocoder"
    ).strip()


def _resolve_local_fallback_coordinates(address_input: str) -> dict[str, Any]:
    return resolve_local_address_candidate(
        address=address_input,
        regions_root=_region_data_root(),
        alias_path=str(os.getenv("WF_LOCAL_ADDRESS_FALLBACK_PATH", "")).strip() or None,
        include_authoritative_sources=False,
        include_alias_sources=True,
    )


def _resolve_local_authoritative_coordinates(address_input: str) -> dict[str, Any]:
    return resolve_local_address_candidate(
        address=address_input,
        regions_root=_region_data_root(),
        alias_path=str(os.getenv("WF_LOCAL_ADDRESS_FALLBACK_PATH", "")).strip() or None,
        include_authoritative_sources=True,
        include_alias_sources=False,
        allowed_source_types={
            "prepared_region_address_dataset",
            "prepared_region_parcel_address_dataset",
            "county_address_dataset",
        },
    )


def _resolve_statewide_parcel_coordinates(address_input: str) -> dict[str, Any]:
    return resolve_local_address_candidate(
        address=address_input,
        regions_root=_region_data_root(),
        alias_path=str(os.getenv("WF_LOCAL_ADDRESS_FALLBACK_PATH", "")).strip() or None,
        include_authoritative_sources=True,
        include_alias_sources=False,
        allowed_source_types={
            "statewide_parcel_dataset",
            "prepared_region_parcel_dataset",
        },
    )


def _extract_zip5(value: str | None) -> str:
    match = re.search(r"\b(\d{5})(?:-\d{4})?\b", str(value or ""))
    return match.group(1) if match else ""


def _infer_locality_from_address_text(address: str) -> str | None:
    parts = [part.strip() for part in str(address or "").split(",") if part.strip()]
    if len(parts) >= 2:
        locality = parts[1]
        if locality:
            return locality
    return None


def _manual_candidate_rank_score(candidate: dict[str, Any]) -> float:
    confidence = str(candidate.get("confidence") or "low").lower()
    confidence_rank = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(confidence, 0.0)
    source_type = str(candidate.get("source_type") or "")
    source_bonus = 0.0
    if source_type in {"county_address_dataset", "prepared_region_address_dataset", "prepared_region_parcel_address_dataset"}:
        source_bonus = 1.4
    elif source_type in {"statewide_parcel_dataset", "prepared_region_parcel_dataset"}:
        source_bonus = 0.9
    elif source_type in {"primary_geocoder", "secondary_geocoder", "provider_backoff_query"}:
        source_bonus = 0.5
    elif source_type == "explicit_fallback_record":
        source_bonus = 0.35
    if bool(candidate.get("coverage_available")):
        source_bonus += 0.8
    return float(confidence_rank + source_bonus)


def _normalize_manual_candidate(
    *,
    row: dict[str, Any],
    fallback_id: str,
    source_kind: str,
) -> dict[str, Any] | None:
    try:
        lat = float(row.get("latitude"))
        lon = float(row.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    components = row.get("candidate_components") if isinstance(row.get("candidate_components"), dict) else {}
    locality = str(
        row.get("locality")
        or row.get("city")
        or (components or {}).get("city")
        or ""
    ).strip()
    postal = _extract_zip5(
        str(
            row.get("postal")
            or row.get("postal_code")
            or (components or {}).get("postal")
            or ""
        )
    ) or None
    state = str(
        row.get("state")
        or (components or {}).get("state")
        or "WA"
    ).strip() or None
    formatted = str(
        row.get("formatted_address")
        or row.get("matched_address")
        or row.get("display_name")
        or ""
    ).strip()
    if not formatted:
        return None

    source_stage = str(row.get("source_stage") or source_kind).strip() or source_kind
    source_type = str(row.get("source_type") or source_stage).strip() or source_stage
    candidate_id = str(row.get("candidate_id") or row.get("source_record_id") or fallback_id)
    confidence = str(
        row.get("confidence")
        or row.get("confidence_tier")
        or row.get("match_confidence")
        or "low"
    ).lower()

    return {
        "candidate_id": candidate_id,
        "formatted_address": formatted,
        "locality": locality or None,
        "postal_code": postal,
        "state": state,
        "source": source_stage,
        "source_type": source_type,
        "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
        "match_method": str(row.get("match_method") or row.get("match_type") or "").strip() or None,
        "latitude": lat,
        "longitude": lon,
        "diagnostics": list(row.get("diagnostics") or []),
        "source_record_id": row.get("source_record_id"),
    }


def _build_manual_address_candidates(
    *,
    address: str,
    zip_code: str | None,
    locality: str | None,
    state: str | None,
    limit: int,
) -> dict[str, Any]:
    normalized_input = normalize_address(address)
    zip5 = _extract_zip5(zip_code or address)
    inferred_locality_raw = _infer_locality_from_address_text(address)
    inferred = infer_localities_for_zip(
        zip_code=zip5,
        regions_root=_region_data_root(),
        state_hint=state or "WA",
        max_localities=10,
    )
    inferred_localities = list(inferred.get("localities") or [])
    selected_locality = str(locality or "").strip() or inferred_locality_raw or (inferred_localities[0] if inferred_localities else None)

    preferred_localities = []
    if selected_locality:
        preferred_localities.append(selected_locality)
    for row in inferred_localities:
        if row and row not in preferred_localities:
            preferred_localities.append(row)

    local_result = resolve_local_address_candidate(
        address=address,
        regions_root=_region_data_root(),
        include_authoritative_sources=True,
        include_alias_sources=True,
        min_auto_confidence_tier="low",
        preferred_localities=preferred_localities or None,
        preferred_postal=zip5 or None,
        required_state=state or "WA",
        top_candidate_limit=max(limit * 3, 12),
    )

    geocode_debug = _build_geocode_debug_payload(address)
    candidates_by_key: dict[str, dict[str, Any]] = {}

    def _upsert(candidate: dict[str, Any]) -> None:
        key = (
            normalize_address(candidate.get("formatted_address") or ""),
            round(float(candidate.get("latitude") or 0.0), 6),
            round(float(candidate.get("longitude") or 0.0), 6),
        )
        existing = candidates_by_key.get(str(key))
        if existing is None:
            candidates_by_key[str(key)] = candidate
            return
        if _manual_candidate_rank_score(candidate) > _manual_candidate_rank_score(existing):
            candidates_by_key[str(key)] = candidate

    for idx, row in enumerate(local_result.get("top_candidates") or []):
        normalized = _normalize_manual_candidate(
            row=dict(row or {}),
            fallback_id=f"local_{idx + 1}",
            source_kind="manual_local_search",
        )
        if normalized is not None:
            _upsert(normalized)

    for idx, row in enumerate((geocode_debug.get("resolver_candidates") or [])):
        normalized = _normalize_manual_candidate(
            row=dict(row or {}),
            fallback_id=f"resolver_{idx + 1}",
            source_kind="resolver_candidate",
        )
        if normalized is not None:
            _upsert(normalized)

    candidates = list(candidates_by_key.values())
    for row in candidates:
        coverage = _region_coverage_for_coordinates(float(row["latitude"]), float(row["longitude"]))
        row["coverage_available"] = bool(coverage.get("coverage_available"))
        row["resolved_region_id"] = coverage.get("resolved_region_id")
        row["resolved_region_display_name"] = coverage.get("resolved_region_display_name")
        row["region_reason"] = coverage.get("reason")
        row["coverage_rank_score"] = _manual_candidate_rank_score(row)

    candidates.sort(
        key=lambda row: (
            -float(row.get("coverage_rank_score") or 0.0),
            -int(bool(row.get("coverage_available"))),
            str(row.get("formatted_address") or ""),
        )
    )

    trimmed = candidates[: max(1, min(25, int(limit or 8)))]
    for row in trimmed:
        row.pop("coverage_rank_score", None)
        row.pop("source_record_id", None)

    status = "ready_for_map_click_fallback"
    if trimmed:
        status = "address_unresolved_needs_manual_selection"
        if all(not bool(row.get("coverage_available")) for row in trimmed):
            status = "outside_prepared_region"

    diagnostics = list(local_result.get("diagnostics") or [])[:8]
    diagnostics.extend(list(inferred.get("diagnostics") or [])[:8])
    if not trimmed:
        diagnostics.append("No usable manual address candidates were found. Map-click fallback is recommended.")

    return {
        "status": status,
        "input_address": address,
        "normalized_address": normalized_input,
        "zip_code": zip5 or None,
        "inferred_localities": inferred_localities[:10],
        "selected_locality": selected_locality,
        "candidates": trimmed,
        "map_click_fallback_recommended": not bool(trimmed),
        "diagnostics": diagnostics[:20],
        "final_status": geocode_debug.get("final_status") or geocode_debug.get("resolution_status"),
    }


def _resolve_trusted_geocode(
    *,
    address_input: str,
    purpose: str,
    route_name: str,
    property_anchor_point: dict[str, float] | None = None,
) -> GeocodeResolution:
    submitted_address = str(address_input or "")
    normalized_address = normalize_address(submitted_address or "")
    provider_attempts: list[dict[str, Any]] = []
    provider_statuses: dict[str, str] = {}
    resolver_candidates: list[dict[str, Any]] = []
    local_fallback_attempted = False
    local_fallback_result: dict[str, Any] | None = None
    authoritative_fallback_result: dict[str, Any] | None = None
    statewide_parcel_result: dict[str, Any] | None = None
    last_failure: HTTPException | None = None

    try:
        conflict_distance_m = max(50.0, float(os.getenv("WF_RESOLVER_CONFLICT_DISTANCE_M", "1500")))
    except ValueError:
        conflict_distance_m = 1500.0
    try:
        conflict_score_margin = max(1.0, float(os.getenv("WF_RESOLVER_CONFLICT_SCORE_MARGIN", "18")))
    except ValueError:
        conflict_score_margin = 18.0
    try:
        in_region_boost = max(0.0, float(os.getenv("WF_RESOLVER_IN_REGION_BOOST", "35")))
    except ValueError:
        in_region_boost = 35.0
    try:
        authoritative_source_bonus = max(0.0, float(os.getenv("WF_RESOLVER_AUTHORITATIVE_SOURCE_BONUS", "18")))
    except ValueError:
        authoritative_source_bonus = 18.0
    try:
        clear_winner_min_margin = max(0.0, float(os.getenv("WF_RESOLVER_CLEAR_WINNER_MIN_MARGIN", "12")))
    except ValueError:
        clear_winner_min_margin = 12.0
    try:
        clear_winner_min_score = max(0.0, float(os.getenv("WF_RESOLVER_CLEAR_WINNER_MIN_SCORE", "230")))
    except ValueError:
        clear_winner_min_score = 230.0
    try:
        in_region_preference_margin = max(0.0, float(os.getenv("WF_RESOLVER_IN_REGION_PREFERENCE_MARGIN", "18")))
    except ValueError:
        in_region_preference_margin = 18.0
    try:
        conflict_min_authority_gap = max(
            0.0, float(os.getenv("WF_RESOLVER_CONFLICT_MIN_AUTHORITY_GAP", "8"))
        )
    except ValueError:
        conflict_min_authority_gap = 8.0
    try:
        conflict_dominant_score_margin = max(
            0.0, float(os.getenv("WF_RESOLVER_CONFLICT_DOMINANT_SCORE_MARGIN", "20"))
        )
    except ValueError:
        conflict_dominant_score_margin = 20.0
    try:
        centroid_tolerance_multiplier = max(
            1.0, float(os.getenv("WF_RESOLVER_CENTROID_TOLERANCE_MULTIPLIER", "2.5"))
        )
    except ValueError:
        centroid_tolerance_multiplier = 2.5
    try:
        interpolated_tolerance_multiplier = max(
            1.0, float(os.getenv("WF_RESOLVER_INTERPOLATED_TOLERANCE_MULTIPLIER", "1.8"))
        )
    except ValueError:
        interpolated_tolerance_multiplier = 1.8
    allow_interpolated_auto = _env_flag("WF_RESOLVER_ALLOW_INTERPOLATED_AUTO", True)
    try:
        min_geocoder_token_similarity = max(
            0.0, min(1.0, float(os.getenv("WF_RESOLVER_MIN_GEOCODER_TOKEN_SIMILARITY", "0.55")))
        )
    except ValueError:
        min_geocoder_token_similarity = 0.55
    try:
        min_geocoder_token_coverage = max(
            0.0, min(1.0, float(os.getenv("WF_RESOLVER_MIN_GEOCODER_TOKEN_COVERAGE", "0.72")))
        )
    except ValueError:
        min_geocoder_token_coverage = 0.72
    try:
        min_auto_candidate_score = max(0.0, float(os.getenv("WF_RESOLVER_MIN_AUTO_CANDIDATE_SCORE", "150")))
    except ValueError:
        min_auto_candidate_score = 150.0
    emergency_in_region_guardrail = _env_flag("WF_RESOLVER_EMERGENCY_IN_REGION_MEDIUM_AUTORESOLVE", True)
    try:
        emergency_min_score = max(0.0, float(os.getenv("WF_RESOLVER_EMERGENCY_MIN_SCORE", "155")))
    except ValueError:
        emergency_min_score = 155.0
    try:
        emergency_min_margin = max(0.0, float(os.getenv("WF_RESOLVER_EMERGENCY_MIN_MARGIN", "8")))
    except ValueError:
        emergency_min_margin = 8.0

    submitted_token_count = len([tok for tok in normalize_address(submitted_address).split() if tok])
    submitted_has_street_number = bool(re.match(r"^\s*\d+[a-zA-Z0-9-]*\b", submitted_address or ""))

    def _confidence_rank(tier: str | None) -> int:
        mapping = {"high": 3, "medium": 2, "low": 1}
        return mapping.get(str(tier or "").strip().lower(), 0)

    def _confidence_allows_auto_use(tier: str | None) -> bool:
        return _confidence_rank(tier) >= 2

    def _normalize_precision(precision: str | None) -> str:
        raw = str(precision or "").strip().lower()
        if raw in {"rooftop", "parcel_or_address_point", "interpolated", "user_selected_point"}:
            return raw
        if raw in {"parcel", "address_point", "road", "locality", "zip_centroid"}:
            return raw
        if raw.startswith("parcel"):
            return "parcel"
        if "road" in raw:
            return "road"
        if "locality" in raw or "city" in raw:
            return "locality"
        return "unknown"

    def _precision_rank(precision: str) -> float:
        mapping = {
            "rooftop": 42.0,
            "parcel_or_address_point": 38.0,
            "address_point": 36.0,
            "parcel": 30.0,
            "user_selected_point": 34.0,
            "interpolated": 18.0,
            "road": 12.0,
            "locality": 4.0,
            "zip_centroid": 0.0,
            "unknown": 0.0,
        }
        return mapping.get(precision, 0.0)

    def _source_rank(stage: str, source_type: str | None) -> float:
        st = str(stage or "")
        source_kind = str(source_type or "")
        if st == "user_selected_point":
            return 95.0
        if source_kind in {"county_address_dataset", "prepared_region_address_dataset"}:
            return 90.0
        if source_kind in {"prepared_region_parcel_address_dataset", "local_authoritative_dataset"}:
            return 86.0
        if source_kind in {"statewide_parcel_dataset", "prepared_region_parcel_dataset"}:
            return 80.0
        if st == "primary_geocoder":
            return 72.0
        if st == "secondary_geocoder":
            return 68.0
        if st == "explicit_fallback_record":
            return 35.0
        if st == "provider_backoff_query":
            return 45.0
        return 30.0

    def _candidate_distance_m(first: dict[str, Any], second: dict[str, Any]) -> float:
        a_lat = float(first["latitude"])
        a_lon = float(first["longitude"])
        b_lat = float(second["latitude"])
        b_lon = float(second["longitude"])
        lat_mid = math.radians((a_lat + b_lat) / 2.0)
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = max(1.0, 111_320.0 * math.cos(lat_mid))
        return float(math.hypot((a_lat - b_lat) * meters_per_deg_lat, (a_lon - b_lon) * meters_per_deg_lon))

    def _parse_candidate_lat_lon(detail: dict[str, Any]) -> tuple[float, float] | None:
        try:
            lat_raw = detail.get("top_candidate_lat")
            lon_raw = detail.get("top_candidate_lng")
            if lat_raw is not None and lon_raw is not None:
                lat = float(lat_raw)
                lon = float(lon_raw)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
        candidate = _extract_candidate_from_preview(detail.get("raw_response_preview"))
        fallback = _extract_lat_lon_from_candidate(candidate)
        if fallback is None:
            return None
        return float(fallback[0]), float(fallback[1])

    def _candidate_record_id(best_match: dict[str, Any]) -> str | None:
        props = best_match.get("feature_properties")
        if not isinstance(props, dict):
            return None
        for key in ("address_id", "site_id", "parcel_id", "objectid", "OBJECTID", "id"):
            value = props.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _candidate_is_street_only(match_method: str | None) -> bool:
        return str(match_method or "") in {
            "street_only_match",
            "locality_or_postal_only",
            "house_and_street_partial",
            "no_viable_match",
        }

    def _candidate_auto_gate_reason(candidate: dict[str, Any]) -> str:
        if candidate.get("source") == "user_selected_point":
            return "eligible"
        if _confidence_rank(str(candidate.get("confidence_tier") or "low")) < 2:
            return "confidence_below_medium"
        precision = str(candidate.get("precision_type") or "unknown")
        if precision in {"locality", "zip_centroid", "unknown"}:
            return "precision_too_coarse"
        if not allow_interpolated_auto and precision in {"interpolated", "road"}:
            return "interpolated_auto_disabled"
        if _candidate_is_street_only(candidate.get("match_method")):
            return "street_only_or_partial_match"
        if float(candidate.get("rank_score") or 0.0) < min_auto_candidate_score:
            return "score_below_min_auto_threshold"
        return "eligible"

    def _candidate_is_auto_eligible(candidate: dict[str, Any]) -> bool:
        return _candidate_auto_gate_reason(candidate) == "eligible"

    def _is_authoritative_source(candidate: dict[str, Any]) -> bool:
        source_type = str(candidate.get("source_type") or "")
        return source_type in {
            "county_address_dataset",
            "prepared_region_address_dataset",
            "prepared_region_parcel_address_dataset",
            "statewide_parcel_dataset",
            "prepared_region_parcel_dataset",
        }

    def _is_parcel_backed(candidate: dict[str, Any]) -> bool:
        source_type = str(candidate.get("source_type") or "")
        precision = str(candidate.get("precision_type") or "")
        match_method = str(candidate.get("match_method") or "").lower()
        return (
            source_type in {"prepared_region_parcel_address_dataset", "statewide_parcel_dataset", "prepared_region_parcel_dataset"}
            or precision in {"parcel", "parcel_or_address_point"}
            or "parcel" in match_method
            or "situs" in match_method
        )

    def _is_centroid_or_interpolated(candidate: dict[str, Any]) -> bool:
        precision = str(candidate.get("precision_type") or "")
        match_method = str(candidate.get("match_method") or "").lower()
        return (
            precision in {"interpolated", "road", "locality", "zip_centroid", "parcel"}
            or "centroid" in match_method
            or "interpolated" in match_method
        )

    def _candidate_authority_rank(candidate: dict[str, Any]) -> float:
        authority = _source_rank(
            str(candidate.get("source_stage") or ""),
            str(candidate.get("source_type") or ""),
        )
        if bool(candidate.get("coverage_available")):
            authority += 6.0
        if _is_authoritative_source(candidate):
            authority += 6.0
        if str(candidate.get("match_method") or "") == "exact_normalized_address":
            authority += 4.0
        if str(candidate.get("source_stage") or "") == "user_selected_point":
            authority += 20.0
        return round(authority, 4)

    def _pair_distance_tolerance_m(first: dict[str, Any], second: dict[str, Any]) -> float:
        tolerance = conflict_distance_m
        if _is_centroid_or_interpolated(first) or _is_centroid_or_interpolated(second):
            tolerance *= interpolated_tolerance_multiplier
        if _is_parcel_backed(first) != _is_parcel_backed(second):
            tolerance *= centroid_tolerance_multiplier
        return float(tolerance)

    def _candidate_allows_medium_auto(candidate: dict[str, Any]) -> bool:
        confidence_tier = str(candidate.get("confidence_tier") or "low")
        if _confidence_rank(confidence_tier) < 2:
            return False
        if _candidate_is_street_only(candidate.get("match_method")):
            return False
        precision = str(candidate.get("precision_type") or "unknown")
        if precision in {"locality", "zip_centroid", "unknown"}:
            return False
        if not allow_interpolated_auto and precision in {"interpolated", "road"}:
            return False
        return True

    def _is_clearly_best_candidate(candidate: dict[str, Any], pool: list[dict[str, Any]]) -> bool:
        if not pool:
            return False
        top_score = float(candidate.get("rank_score") or 0.0)
        if top_score < clear_winner_min_score:
            return False
        ordered = sorted(pool, key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)
        if len(ordered) == 1:
            return True
        second_score = float(ordered[1].get("rank_score") or 0.0)
        return (top_score - second_score) >= clear_winner_min_margin

    def _upsert_stage_status(stage: str, status: str) -> None:
        rank = {"accepted": 3, "low_confidence": 2, "no_match": 1, "provider_error": 0, "parser_error": 0}
        existing = provider_statuses.get(stage)
        if existing is None or rank.get(status, -1) >= rank.get(existing, -1):
            provider_statuses[stage] = status

    def _register_candidate(
        *,
        stage: str,
        latitude: float,
        longitude: float,
        geocode_meta: dict[str, Any],
        source: str,
        source_type: str | None = None,
        source_record_id: str | None = None,
    ) -> None:
        coverage_lookup = _region_coverage_for_coordinates(float(latitude), float(longitude))
        precision_type = _normalize_precision(str(geocode_meta.get("geocode_precision") or geocode_meta.get("precision_type") or "unknown"))
        if precision_type == "unknown" and stage in {"primary_geocoder", "secondary_geocoder", "provider_backoff_query"}:
            # Legacy geocoder stubs often return coordinates without precision metadata.
            precision_type = "interpolated"
        confidence_tier = str(
            geocode_meta.get("geocode_trust_tier")
            or geocode_meta.get("final_location_confidence")
            or geocode_meta.get("confidence_tier")
            or "low"
        ).lower()
        confidence_score_raw = geocode_meta.get("confidence_score")
        try:
            confidence_score = float(confidence_score_raw) if confidence_score_raw is not None else None
        except (TypeError, ValueError):
            confidence_score = None
        diagnostics = list(geocode_meta.get("diagnostics") or [])
        address_cmp = (
            dict(geocode_meta.get("address_component_comparison"))
            if isinstance(geocode_meta.get("address_component_comparison"), dict)
            else {}
        )
        token_similarity = None
        try:
            token_similarity = (
                float(address_cmp.get("token_similarity_ratio"))
                if address_cmp.get("token_similarity_ratio") is not None
                else None
            )
        except (TypeError, ValueError):
            token_similarity = None
        token_coverage = None
        try:
            token_coverage = (
                float(address_cmp.get("token_coverage_ratio"))
                if address_cmp.get("token_coverage_ratio") is not None
                else None
            )
        except (TypeError, ValueError):
            token_coverage = None
        house_number_match = bool(address_cmp.get("house_number_match"))
        exact_normalized_match = bool(address_cmp.get("exact_normalized_match"))
        geocoder_stage = stage in {"primary_geocoder", "secondary_geocoder", "provider_backoff_query"}
        if (
            geocoder_stage
            and submitted_has_street_number
            and submitted_token_count >= 3
            and bool(str(address_cmp.get("candidate_normalized") or "").strip())
            and token_similarity is not None
            and token_coverage is not None
            and token_similarity < min_geocoder_token_similarity
            and token_coverage < min_geocoder_token_coverage
            and not house_number_match
            and not exact_normalized_match
        ):
            # Guardrail: geocoder string similarity is too weak for property-level coordinate auto-use.
            confidence_tier = "low"
            diagnostics.append(
                "Geocoder candidate failed address-similarity validation "
                "(token_similarity="
                f"{token_similarity:.2f} < {min_geocoder_token_similarity:.2f}, "
                "token_coverage="
                f"{token_coverage:.2f} < {min_geocoder_token_coverage:.2f})."
            )
        candidate = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "source": str(source or stage),
            "source_stage": stage,
            "source_type": str(source_type or geocode_meta.get("source_type") or ""),
            "source_record_id": source_record_id or geocode_meta.get("source_record_id"),
            "provider": str(geocode_meta.get("provider") or source or stage),
            "formatted_address": geocode_meta.get("matched_address") or geocode_meta.get("geocoded_address"),
            "normalized_address": geocode_meta.get("normalized_address") or normalized_address,
            "match_method": str(geocode_meta.get("match_method") or geocode_meta.get("geocode_decision") or "unknown"),
            "confidence_score": confidence_score,
            "confidence_tier": confidence_tier,
            "precision_type": precision_type,
            "trusted_match_status": str(geocode_meta.get("trusted_match_status") or "untrusted_fallback"),
            "geocode_status": str(geocode_meta.get("geocode_status") or "accepted"),
            "geocode_outcome": str(geocode_meta.get("geocode_outcome") or "geocode_succeeded_untrusted"),
            "diagnostics": diagnostics,
            "coverage_available": bool(coverage_lookup.get("coverage_available")),
            "resolved_region_id": coverage_lookup.get("resolved_region_id"),
            "candidate_regions_containing_point": list(coverage_lookup.get("candidate_regions_containing_point") or []),
            "region_distance_to_boundary_m": coverage_lookup.get("region_distance_to_boundary_m"),
            "nearest_region_id": coverage_lookup.get("nearest_region_id"),
            "unsupported_location_reason": coverage_lookup.get("reason"),
            "geocode_meta": dict(geocode_meta or {}),
            "address_component_comparison": address_cmp or None,
            "address_similarity_token_ratio": token_similarity,
            "address_similarity_token_coverage": token_coverage,
            "address_similarity_exact_match": exact_normalized_match,
            "house_number_match": house_number_match,
        }
        confidence_rank = _confidence_rank(candidate["confidence_tier"])
        base_score = (
            float(confidence_rank * 100)
            + _precision_rank(candidate["precision_type"])
            + _source_rank(stage, candidate.get("source_type"))
            + (in_region_boost if candidate["coverage_available"] else 0.0)
            + (
                max(0.0, min(1.0, float(candidate["confidence_score"]))) * 20.0
                if candidate["confidence_score"] is not None
                else 0.0
            )
        )
        source_bonus = authoritative_source_bonus if _is_authoritative_source(candidate) else 0.0
        exact_address_bonus = 0.0
        if str(candidate.get("match_method") or "") == "exact_normalized_address":
            exact_address_bonus = 12.0
        candidate["raw_score"] = round(base_score, 4)
        candidate["source_bonus"] = round(source_bonus, 4)
        candidate["rank_score"] = round(base_score + source_bonus + exact_address_bonus, 4)
        gate_reason = _candidate_auto_gate_reason(candidate)
        candidate["auto_gate_reason"] = gate_reason
        candidate["auto_eligible"] = gate_reason == "eligible"
        resolver_candidates.append(candidate)

    def _record_failure(stage: str, query: str, provider_name: str, exc: HTTPException) -> dict[str, Any]:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        status = str(detail.get("geocode_status") or "provider_error")
        _upsert_stage_status(stage, status)
        provider_attempts.append(
            _build_provider_attempt(
                stage=stage,
                provider_name=provider_name,
                query=query,
                accepted=False,
                geocode_status=status,
                geocode_outcome=str(detail.get("geocode_outcome") or "geocode_failed"),
                rejection_reason=str(detail.get("rejection_reason") or detail.get("message") or ""),
                candidate_count=(int(detail.get("candidate_count")) if detail.get("candidate_count") is not None else None),
            )
        )
        fallback_coords = _parse_candidate_lat_lon(detail)
        if fallback_coords is not None:
            fallback_lat, fallback_lon = fallback_coords
            fallback_meta = {
                "geocode_status": status,
                "geocode_outcome": str(detail.get("geocode_outcome") or "geocode_failed"),
                "trusted_match_status": str(detail.get("trusted_match_status") or "rejected"),
                "geocode_trust_tier": str(detail.get("geocode_trust_tier") or "low"),
                "geocode_decision": str(detail.get("geocode_decision") or "candidate_from_error_detail"),
                "submitted_address": submitted_address,
                "normalized_address": detail.get("normalized_address") or normalized_address,
                "provider": detail.get("provider") or provider_name,
                "geocoded_address": detail.get("top_candidate_formatted_address"),
                "matched_address": detail.get("top_candidate_formatted_address"),
                "confidence_score": None,
                "candidate_count": int(detail.get("candidate_count") or 1),
                "geocode_precision": str(detail.get("geocode_precision") or "interpolated"),
                "raw_response_preview": detail.get("raw_response_preview"),
                "rejection_reason": detail.get("rejection_reason"),
                "trusted_match_failure_reason": detail.get("trusted_match_failure_reason") or detail.get("rejection_reason"),
                "fallback_eligibility": bool(detail.get("fallback_eligibility")),
                "match_method": str(detail.get("match_method") or detail.get("geocode_decision") or "candidate_from_error_detail"),
                "diagnostics": [str(detail.get("message") or "")] if detail.get("message") else [],
            }
            _register_candidate(
                stage=stage,
                latitude=float(fallback_lat),
                longitude=float(fallback_lon),
                geocode_meta=fallback_meta,
                source=str(fallback_meta.get("provider") or provider_name),
            )
        return detail

    # Stage A: primary geocoder
    try:
        lat, lon, source, geocode_meta = _geocode_address_or_raise(
            address=submitted_address,
            purpose=purpose,
            geocoder_client=geocoder,
            provider_override=getattr(geocoder, "provider_name", None),
        )
        trust_tier = str(geocode_meta.get("geocode_trust_tier") or "low")
        accepted = _confidence_allows_auto_use(trust_tier)
        _upsert_stage_status("primary_geocoder", "accepted" if accepted else "low_confidence")
        provider_attempts.append(
            _build_provider_attempt(
                stage="primary_geocoder",
                provider_name=str(geocode_meta.get("provider") or getattr(geocoder, "provider_name", "primary")),
                query=submitted_address,
                accepted=accepted,
                geocode_status=str(geocode_meta.get("geocode_status") or ("accepted" if accepted else "low_confidence")),
                geocode_outcome=str(geocode_meta.get("geocode_outcome") or "geocode_succeeded_trusted"),
                rejection_reason=None if accepted else "candidate_below_auto_use_confidence",
                candidate_count=(int(geocode_meta.get("candidate_count")) if geocode_meta.get("candidate_count") is not None else None),
            )
        )
        _register_candidate(
            stage="primary_geocoder",
            latitude=float(lat),
            longitude=float(lon),
            geocode_meta=dict(geocode_meta or {}),
            source=str(source),
        )
    except HTTPException as exc:
        last_failure = exc
        _record_failure("primary_geocoder", submitted_address, str(getattr(geocoder, "provider_name", "OpenStreetMap Nominatim")), exc)

    # Stage B: optional secondary geocoder
    if _secondary_geocoder_enabled():
        try:
            lat, lon, source, geocode_meta = _geocode_address_or_raise(
                address=submitted_address,
                purpose=purpose,
                geocoder_client=secondary_geocoder,
                provider_override=_secondary_provider_name(),
            )
            trust_tier = str(geocode_meta.get("geocode_trust_tier") or "low")
            accepted = _confidence_allows_auto_use(trust_tier)
            _upsert_stage_status("secondary_geocoder", "accepted" if accepted else "low_confidence")
            provider_attempts.append(
                _build_provider_attempt(
                    stage="secondary_geocoder",
                    provider_name=str(geocode_meta.get("provider") or _secondary_provider_name()),
                    query=submitted_address,
                    accepted=accepted,
                    geocode_status=str(geocode_meta.get("geocode_status") or ("accepted" if accepted else "low_confidence")),
                    geocode_outcome=str(geocode_meta.get("geocode_outcome") or "geocode_succeeded_untrusted"),
                    rejection_reason=None if accepted else "candidate_below_auto_use_confidence",
                    candidate_count=(int(geocode_meta.get("candidate_count")) if geocode_meta.get("candidate_count") is not None else None),
                )
            )
            _register_candidate(
                stage="secondary_geocoder",
                latitude=float(lat),
                longitude=float(lon),
                geocode_meta=dict(geocode_meta or {}),
                source=str(source),
            )
        except HTTPException as exc:
            last_failure = exc
            _record_failure("secondary_geocoder", submitted_address, _secondary_provider_name(), exc)

    # Stage C: county/prepared local authoritative address points
    local_fallback_attempted = True
    try:
        authoritative_fallback_result = _resolve_local_authoritative_coordinates(submitted_address)
    except Exception as exc:  # pragma: no cover
        authoritative_fallback_result = {
            "matched": False,
            "candidate_count": 0,
            "failure_reason": "local_authoritative_resolver_error",
            "diagnostics": [f"Local authoritative fallback resolver error: {exc}"],
            "top_candidates": [],
        }
    local_best = (
        dict((authoritative_fallback_result or {}).get("best_match") or {})
        if bool((authoritative_fallback_result or {}).get("matched"))
        else dict((authoritative_fallback_result or {}).get("best_candidate") or {})
    )
    if local_best:
        try:
            lat = float(local_best["latitude"])
            lon = float(local_best["longitude"])
        except (TypeError, ValueError, KeyError):
            lat = None  # type: ignore[assignment]
            lon = None  # type: ignore[assignment]
        if lat is not None and lon is not None:
            source_type = str(local_best.get("source_type") or "")
            stage_label = "county_address_points" if source_type == "county_address_dataset" else "local_authoritative_fallback"
            confidence = str((authoritative_fallback_result or {}).get("confidence") or local_best.get("confidence_tier") or "low")
            accepted = _confidence_allows_auto_use(confidence)
            _upsert_stage_status(stage_label, "accepted" if accepted else "low_confidence")
            provider_attempts.append(
                _build_provider_attempt(
                    stage=stage_label,
                    provider_name=stage_label,
                    query=submitted_address,
                    accepted=accepted,
                    geocode_status="accepted" if accepted else "low_confidence",
                    geocode_outcome="geocode_succeeded_untrusted" if accepted else "geocode_failed",
                    rejection_reason=None if accepted else str((authoritative_fallback_result or {}).get("failure_reason") or "candidate_below_auto_use_confidence"),
                    candidate_count=int((authoritative_fallback_result or {}).get("candidate_count") or 1),
                )
            )
            local_meta = {
                "geocode_status": "accepted",
                "geocode_outcome": "geocode_succeeded_untrusted",
                "trusted_match_status": "trusted" if confidence == "high" else "untrusted_fallback",
                "geocode_trust_tier": confidence,
                "geocode_decision": (
                    "resolved_via_county_address_points"
                    if stage_label == "county_address_points"
                    else "resolved_via_local_authoritative_fallback"
                ),
                "submitted_address": submitted_address,
                "normalized_address": (authoritative_fallback_result or {}).get("normalized_address") or normalized_address,
                "provider": stage_label,
                "geocode_provider": stage_label,
                "geocode_source": stage_label,
                "matched_address": local_best.get("matched_address"),
                "geocoded_address": local_best.get("matched_address"),
                "confidence_score": local_best.get("match_score"),
                "candidate_count": int((authoritative_fallback_result or {}).get("candidate_count") or 1),
                "geocode_precision": "parcel_or_address_point",
                "match_method": (authoritative_fallback_result or {}).get("match_method") or local_best.get("match_type"),
                "diagnostics": list((authoritative_fallback_result or {}).get("diagnostics") or []),
            }
            _register_candidate(
                stage=stage_label,
                latitude=float(lat),
                longitude=float(lon),
                geocode_meta=local_meta,
                source=stage_label,
                source_type=source_type,
                source_record_id=_candidate_record_id(local_best),
            )
    else:
        _upsert_stage_status("county_address_points", "no_match")
        provider_attempts.append(
            _build_provider_attempt(
                stage="county_address_points",
                provider_name="county_address_points",
                query=submitted_address,
                accepted=False,
                geocode_status="no_match",
                geocode_outcome="geocode_failed",
                rejection_reason=str((authoritative_fallback_result or {}).get("failure_reason") or "no_local_authoritative_match"),
                candidate_count=int((authoritative_fallback_result or {}).get("candidate_count") or 0),
            )
        )

    # Stage D: statewide parcels / parcel-centric fallback
    try:
        statewide_parcel_result = _resolve_statewide_parcel_coordinates(submitted_address)
    except Exception as exc:  # pragma: no cover
        statewide_parcel_result = {
            "matched": False,
            "candidate_count": 0,
            "failure_reason": "statewide_parcel_resolver_error",
            "diagnostics": [f"Statewide parcel resolver error: {exc}"],
            "top_candidates": [],
        }
    parcel_best = (
        dict((statewide_parcel_result or {}).get("best_match") or {})
        if bool((statewide_parcel_result or {}).get("matched"))
        else dict((statewide_parcel_result or {}).get("best_candidate") or {})
    )
    if parcel_best:
        try:
            lat = float(parcel_best["latitude"])
            lon = float(parcel_best["longitude"])
        except (TypeError, ValueError, KeyError):
            lat = None  # type: ignore[assignment]
            lon = None  # type: ignore[assignment]
        if lat is not None and lon is not None:
            confidence = str((statewide_parcel_result or {}).get("confidence") or parcel_best.get("confidence_tier") or "low")
            accepted = _confidence_allows_auto_use(confidence)
            _upsert_stage_status("statewide_parcel_lookup", "accepted" if accepted else "low_confidence")
            provider_attempts.append(
                _build_provider_attempt(
                    stage="statewide_parcel_lookup",
                    provider_name="statewide_parcel_lookup",
                    query=submitted_address,
                    accepted=accepted,
                    geocode_status="accepted" if accepted else "low_confidence",
                    geocode_outcome="geocode_succeeded_untrusted" if accepted else "geocode_failed",
                    rejection_reason=None if accepted else str((statewide_parcel_result or {}).get("failure_reason") or "candidate_below_auto_use_confidence"),
                    candidate_count=int((statewide_parcel_result or {}).get("candidate_count") or 1),
                )
            )
            parcel_meta = {
                "geocode_status": "accepted",
                "geocode_outcome": "geocode_succeeded_untrusted",
                "trusted_match_status": "trusted" if confidence == "high" else "untrusted_fallback",
                "geocode_trust_tier": confidence,
                "geocode_decision": "resolved_via_statewide_parcel_lookup",
                "submitted_address": submitted_address,
                "normalized_address": (statewide_parcel_result or {}).get("normalized_address") or normalized_address,
                "provider": "statewide_parcel_lookup",
                "geocode_provider": "statewide_parcel_lookup",
                "geocode_source": "statewide_parcel_lookup",
                "matched_address": parcel_best.get("matched_address"),
                "geocoded_address": parcel_best.get("matched_address"),
                "confidence_score": parcel_best.get("match_score"),
                "candidate_count": int((statewide_parcel_result or {}).get("candidate_count") or 1),
                "geocode_precision": "parcel",
                "match_method": (statewide_parcel_result or {}).get("match_method") or parcel_best.get("match_type"),
                "diagnostics": list((statewide_parcel_result or {}).get("diagnostics") or []),
            }
            _register_candidate(
                stage="statewide_parcel_lookup",
                latitude=float(lat),
                longitude=float(lon),
                geocode_meta=parcel_meta,
                source="statewide_parcel_lookup",
                source_type=str(parcel_best.get("source_type") or "statewide_parcel_dataset"),
                source_record_id=_candidate_record_id(parcel_best),
            )
    else:
        _upsert_stage_status("statewide_parcel_lookup", "no_match")
        provider_attempts.append(
            _build_provider_attempt(
                stage="statewide_parcel_lookup",
                provider_name="statewide_parcel_lookup",
                query=submitted_address,
                accepted=False,
                geocode_status="no_match",
                geocode_outcome="geocode_failed",
                rejection_reason=str((statewide_parcel_result or {}).get("failure_reason") or "no_statewide_parcel_match"),
                candidate_count=int((statewide_parcel_result or {}).get("candidate_count") or 0),
            )
        )

    # Stage E: explicit fallback records
    local_fallback_attempted = True
    try:
        local_fallback_result = _resolve_local_fallback_coordinates(submitted_address)
    except Exception as exc:  # pragma: no cover
        local_fallback_result = {
            "matched": False,
            "candidate_count": 0,
            "failure_reason": "explicit_fallback_resolver_error",
            "diagnostics": [f"Explicit fallback resolver error: {exc}"],
            "top_candidates": [],
        }
    fallback_best = (
        dict((local_fallback_result or {}).get("best_match") or {})
        if bool((local_fallback_result or {}).get("matched"))
        else dict((local_fallback_result or {}).get("best_candidate") or {})
    )
    if fallback_best:
        try:
            lat = float(fallback_best["latitude"])
            lon = float(fallback_best["longitude"])
        except (TypeError, ValueError, KeyError):
            lat = None  # type: ignore[assignment]
            lon = None  # type: ignore[assignment]
        if lat is not None and lon is not None:
            confidence = str((local_fallback_result or {}).get("confidence") or fallback_best.get("confidence_tier") or "low")
            accepted = _confidence_allows_auto_use(confidence)
            _upsert_stage_status("explicit_fallback_record", "accepted" if accepted else "low_confidence")
            provider_attempts.append(
                _build_provider_attempt(
                    stage="explicit_fallback_record",
                    provider_name="explicit_fallback_record",
                    query=submitted_address,
                    accepted=accepted,
                    geocode_status="accepted" if accepted else "low_confidence",
                    geocode_outcome="geocode_succeeded_untrusted" if accepted else "geocode_failed",
                    rejection_reason=None if accepted else str((local_fallback_result or {}).get("failure_reason") or "candidate_below_auto_use_confidence"),
                    candidate_count=int((local_fallback_result or {}).get("candidate_count") or 1),
                )
            )
            fallback_meta = {
                "geocode_status": "accepted",
                "geocode_outcome": "geocode_succeeded_untrusted",
                "trusted_match_status": "trusted" if confidence == "high" else "untrusted_fallback",
                "geocode_trust_tier": confidence,
                "geocode_decision": "resolved_via_explicit_fallback_record",
                "submitted_address": submitted_address,
                "normalized_address": (local_fallback_result or {}).get("normalized_address") or normalized_address,
                "provider": "explicit_fallback_record",
                "geocode_provider": "explicit_fallback_record",
                "geocode_source": "explicit_fallback_record",
                "matched_address": fallback_best.get("matched_address"),
                "geocoded_address": fallback_best.get("matched_address"),
                "confidence_score": fallback_best.get("match_score"),
                "candidate_count": int((local_fallback_result or {}).get("candidate_count") or 1),
                "geocode_precision": "parcel_or_address_point" if confidence == "high" else "interpolated",
                "match_method": (local_fallback_result or {}).get("match_method") or fallback_best.get("match_type"),
                "diagnostics": list((local_fallback_result or {}).get("diagnostics") or []),
            }
            _register_candidate(
                stage="explicit_fallback_record",
                latitude=float(lat),
                longitude=float(lon),
                geocode_meta=fallback_meta,
                source="explicit_fallback_record",
                source_type="explicit_fallback_record",
                source_record_id=_candidate_record_id(fallback_best),
            )
    else:
        _upsert_stage_status("explicit_fallback_record", "no_match")
        provider_attempts.append(
            _build_provider_attempt(
                stage="explicit_fallback_record",
                provider_name="explicit_fallback_record",
                query=submitted_address,
                accepted=False,
                geocode_status="no_match",
                geocode_outcome="geocode_failed",
                rejection_reason=str((local_fallback_result or {}).get("failure_reason") or "no_explicit_fallback_match"),
                candidate_count=int((local_fallback_result or {}).get("candidate_count") or 0),
            )
        )

    # Stage F: provider backoff query variants
    if _env_flag("WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY", True):
        for query in _build_provider_backoff_queries(submitted_address):
            if query == submitted_address:
                continue
            try:
                lat, lon, source, geocode_meta = _geocode_address_or_raise(
                    address=query,
                    purpose=purpose,
                    geocoder_client=geocoder,
                    provider_override=getattr(geocoder, "provider_name", None),
                )
                trust_tier = str(geocode_meta.get("geocode_trust_tier") or "low")
                accepted = _confidence_allows_auto_use(trust_tier)
                _upsert_stage_status("provider_backoff_query", "accepted" if accepted else "low_confidence")
                provider_attempts.append(
                    _build_provider_attempt(
                        stage="provider_backoff_query",
                        provider_name=str(geocode_meta.get("provider") or getattr(geocoder, "provider_name", "primary")),
                        query=query,
                        accepted=accepted,
                        geocode_status=str(geocode_meta.get("geocode_status") or ("accepted" if accepted else "low_confidence")),
                        geocode_outcome=str(geocode_meta.get("geocode_outcome") or "geocode_succeeded_untrusted"),
                        rejection_reason=None if accepted else "candidate_below_auto_use_confidence",
                        candidate_count=(int(geocode_meta.get("candidate_count")) if geocode_meta.get("candidate_count") is not None else None),
                    )
                )
                _register_candidate(
                    stage="provider_backoff_query",
                    latitude=float(lat),
                    longitude=float(lon),
                    geocode_meta=dict(geocode_meta or {}),
                    source=str(source),
                )
            except HTTPException as exc:
                last_failure = exc
                _record_failure(
                    "provider_backoff_query",
                    query,
                    str(getattr(geocoder, "provider_name", "OpenStreetMap Nominatim")),
                    exc,
                )

    # Stage G: user selected property anchor
    if property_anchor_point is not None:
        lat = float(property_anchor_point["latitude"])
        lon = float(property_anchor_point["longitude"])
        user_meta = {
            "geocode_status": "accepted",
            "geocode_outcome": "geocode_succeeded_untrusted",
            "trusted_match_status": "trusted",
            "geocode_trust_tier": "high",
            "geocode_decision": "resolved_via_user_selected_point",
            "submitted_address": submitted_address,
            "normalized_address": normalized_address,
            "provider": "user_selected_point",
            "geocode_provider": "user_selected_point",
            "geocode_source": "user_selected_point",
            "matched_address": None,
            "geocoded_address": submitted_address,
            "candidate_count": 0,
            "geocode_precision": "user_selected_point",
            "match_method": "user_selected_point",
            "diagnostics": ["User-selected property anchor point supplied by request payload."],
        }
        _upsert_stage_status("user_selected_point", "accepted")
        provider_attempts.append(
            _build_provider_attempt(
                stage="user_selected_point",
                provider_name="user_selected_point",
                query=submitted_address,
                accepted=True,
                geocode_status="accepted",
                geocode_outcome="geocode_succeeded_untrusted",
                rejection_reason=None,
                candidate_count=0,
            )
        )
        _register_candidate(
            stage="user_selected_point",
            latitude=lat,
            longitude=lon,
            geocode_meta=user_meta,
            source="user_selected_point",
        )

    # Rank candidates using score + source quality + in-region preference.
    resolver_candidates.sort(
        key=lambda row: (
            -float(row.get("rank_score") or 0.0),
            -int(bool(row.get("coverage_available"))),
            -_confidence_rank(str(row.get("confidence_tier") or "low")),
            str(row.get("source_stage") or ""),
        )
    )
    for idx, row in enumerate(resolver_candidates, start=1):
        row["candidate_id"] = str(row.get("candidate_id") or f"cand_{idx}")
    top_rank_score = float(resolver_candidates[0].get("rank_score") or 0.0) if resolver_candidates else 0.0
    for idx, row in enumerate(resolver_candidates):
        rank_score = float(row.get("rank_score") or 0.0)
        row["normalized_score"] = round((rank_score / top_rank_score), 4) if top_rank_score > 0 else 0.0
        next_score = (
            float(resolver_candidates[idx + 1].get("rank_score") or 0.0)
            if idx + 1 < len(resolver_candidates)
            else None
        )
        row["score_margin_vs_next_candidate"] = (
            round(rank_score - next_score, 4) if next_score is not None else None
        )
        row["authoritative_source_flag"] = _is_authoritative_source(row)
        row["authority_rank"] = _candidate_authority_rank(row)
        row["in_region_boost_applied"] = bool(row.get("coverage_available"))
        row["exact_match_flag"] = str(row.get("match_method") or "") == "exact_normalized_address"
        row["parcel_backed_flag"] = _is_parcel_backed(row)
        row["centroid_or_interpolated_flag"] = _is_centroid_or_interpolated(row)
        row["conflict_penalty_applied"] = False

    strong_candidates = [row for row in resolver_candidates if _confidence_rank(str(row.get("confidence_tier"))) >= 2]
    disagreement_rows: list[dict[str, Any]] = []
    for idx, first in enumerate(strong_candidates):
        nearest = None
        for second in strong_candidates[idx + 1 :]:
            distance = _candidate_distance_m(first, second)
            score_gap = abs(float(first.get("rank_score") or 0.0) - float(second.get("rank_score") or 0.0))
            tolerance_m = _pair_distance_tolerance_m(first, second)
            threshold_exceeded = distance > tolerance_m
            first_authority = float(first.get("authority_rank") or _candidate_authority_rank(first))
            second_authority = float(second.get("authority_rank") or _candidate_authority_rank(second))
            authority_gap = abs(first_authority - second_authority)
            precision_mismatch_explainable = (
                threshold_exceeded
                and (
                    (_is_centroid_or_interpolated(first) != _is_centroid_or_interpolated(second))
                    or (_is_parcel_backed(first) != _is_parcel_backed(second))
                )
            )
            first_dominates = (
                (first_authority - second_authority) >= conflict_min_authority_gap
                or (
                    bool(first.get("coverage_available"))
                    and not bool(second.get("coverage_available"))
                    and score_gap >= 0.0
                )
                or score_gap >= conflict_dominant_score_margin
            )
            second_dominates = (
                (second_authority - first_authority) >= conflict_min_authority_gap
                or (
                    bool(second.get("coverage_available"))
                    and not bool(first.get("coverage_available"))
                    and score_gap >= 0.0
                )
                or score_gap >= conflict_dominant_score_margin
            )
            disagreement_rows.append(
                {
                    "first_candidate_id": first.get("candidate_id"),
                    "second_candidate_id": second.get("candidate_id"),
                    "first_source": first.get("source_stage"),
                    "second_source": second.get("source_stage"),
                    "distance_m": round(distance, 2),
                    "distance_threshold_m": round(tolerance_m, 2),
                    "disagreement_threshold_exceeded": bool(threshold_exceeded),
                    "precision_mismatch_explainable": bool(precision_mismatch_explainable),
                    "first_authority_rank": round(first_authority, 4),
                    "second_authority_rank": round(second_authority, 4),
                    "authority_gap": round(authority_gap, 4),
                    "first_dominates": bool(first_dominates),
                    "second_dominates": bool(second_dominates),
                    "score_gap": round(score_gap, 2),
                }
            )
            if nearest is None or distance < nearest:
                nearest = distance
        first["nearest_strong_candidate_distance_m"] = round(nearest, 2) if nearest is not None else None

    in_region_medium_candidates = [
        row for row in resolver_candidates if bool(row.get("coverage_available")) and _candidate_allows_medium_auto(row)
    ]

    auto_candidates = [row for row in resolver_candidates if bool(row.get("auto_eligible"))]
    in_region_auto = [row for row in auto_candidates if bool(row.get("coverage_available"))]

    # Permit medium-confidence in-region auto-resolution when clearly best.
    if not in_region_auto and in_region_medium_candidates:
        top_in_region_medium = in_region_medium_candidates[0]
        if _is_clearly_best_candidate(top_in_region_medium, resolver_candidates):
            top_in_region_medium["auto_eligible"] = True
            top_in_region_medium["auto_gate_reason"] = "medium_in_region_clearly_best"
            auto_candidates = [row for row in resolver_candidates if bool(row.get("auto_eligible"))]
            in_region_auto = [row for row in auto_candidates if bool(row.get("coverage_available"))]

    candidate_needs_confirmation = resolver_candidates[0] if resolver_candidates else None
    address_exists = bool(resolver_candidates)
    if not resolver_candidates:
        address_confidence = "low"
    else:
        highest_candidate_rank = max(
            _confidence_rank(str(row.get("confidence_tier") or "low")) for row in resolver_candidates
        )
        if highest_candidate_rank >= 3:
            address_confidence = "high"
        elif highest_candidate_rank >= 2:
            address_confidence = "medium"
        else:
            address_confidence = "low"
    address_validation_sources = sorted(
        {
            str(row.get("source_stage"))
            for row in resolver_candidates
            if row.get("source_stage")
        }
        | {
            str(row.get("stage"))
            for row in provider_attempts
            if row.get("stage")
        }
    )
    top_candidate_score = float(resolver_candidates[0].get("rank_score") or 0.0) if resolver_candidates else None
    second_candidate_score = (
        float(resolver_candidates[1].get("rank_score") or 0.0) if len(resolver_candidates) > 1 else None
    )

    selected_candidate: dict[str, Any] | None = None
    if in_region_auto:
        selected_candidate = in_region_auto[0]
    elif auto_candidates:
        selected_candidate = auto_candidates[0]

    # If a non-covered candidate currently wins, prefer a strong in-region medium candidate when close in score.
    if (
        selected_candidate is not None
        and not bool(selected_candidate.get("coverage_available"))
        and in_region_medium_candidates
    ):
        in_region_best = in_region_medium_candidates[0]
        score_gap = float(selected_candidate.get("rank_score") or 0.0) - float(in_region_best.get("rank_score") or 0.0)
        if score_gap <= in_region_preference_margin:
            in_region_best["auto_eligible"] = True
            in_region_best["auto_gate_reason"] = "in_region_preference_override"
            selected_candidate = in_region_best

    # Emergency guardrail: prevent blanket "needs map confirmation" regressions for clearly best in-region medium candidates.
    if selected_candidate is None and emergency_in_region_guardrail and in_region_medium_candidates:
        fallback_candidate = in_region_medium_candidates[0]
        top_score = float(fallback_candidate.get("rank_score") or 0.0)
        margin = float(fallback_candidate.get("score_margin_vs_next_candidate") or 0.0)
        if top_score >= emergency_min_score and margin >= emergency_min_margin:
            fallback_candidate["auto_eligible"] = True
            fallback_candidate["auto_gate_reason"] = "emergency_in_region_medium_autoresolve"
            selected_candidate = fallback_candidate

    ambiguous_conflict = False
    conflict_reason: str | None = None
    if selected_candidate is not None:
        selection_pool = [
            row
            for row in resolver_candidates
            if bool(row.get("auto_eligible"))
            and (
                bool(row.get("coverage_available"))
                if bool(selected_candidate.get("coverage_available"))
                else True
            )
        ]
        if len(selection_pool) >= 2:
            second = next((row for row in selection_pool if row is not selected_candidate), None)
            if second is None:
                second = selection_pool[1]
            distance = _candidate_distance_m(selected_candidate, second)
            score_gap = abs(float(selected_candidate.get("rank_score") or 0.0) - float(second.get("rank_score") or 0.0))
            tolerance_m = _pair_distance_tolerance_m(selected_candidate, second)
            threshold_exceeded = distance > tolerance_m
            selected_authority = float(selected_candidate.get("authority_rank") or _candidate_authority_rank(selected_candidate))
            second_authority = float(second.get("authority_rank") or _candidate_authority_rank(second))
            authority_gap = selected_authority - second_authority
            precision_mismatch_explainable = (
                threshold_exceeded
                and (
                    (_is_centroid_or_interpolated(selected_candidate) != _is_centroid_or_interpolated(second))
                    or (_is_parcel_backed(selected_candidate) != _is_parcel_backed(second))
                )
            )
            selected_dominates = (
                authority_gap >= conflict_min_authority_gap
                or score_gap >= conflict_dominant_score_margin
                or (
                    bool(selected_candidate.get("coverage_available"))
                    and not bool(second.get("coverage_available"))
                    and score_gap >= 0.0
                )
                or (
                    bool(selected_candidate.get("exact_match_flag"))
                    and not bool(second.get("exact_match_flag"))
                    and bool(selected_candidate.get("authoritative_source_flag"))
                )
            )
            if precision_mismatch_explainable and selected_dominates:
                conflict_reason = "precision_mismatch_explainable_dominant_candidate"
            elif (
                threshold_exceeded
                and score_gap <= conflict_score_margin
                and selected_candidate.get("source_stage") != "user_selected_point"
                and not selected_dominates
                and not _is_clearly_best_candidate(selected_candidate, selection_pool)
            ):
                ambiguous_conflict = True
                conflict_reason = "similar_authority_candidates_disagree_materially"
                selected_candidate["conflict_penalty_applied"] = True
                second["conflict_penalty_applied"] = True
            elif threshold_exceeded:
                conflict_reason = "distance_exceeded_but_dominant_candidate_selected"
            else:
                conflict_reason = "no_material_conflict"
            selected_candidate["conflict_diagnostics"] = {
                "distance_m": round(distance, 2),
                "distance_threshold_m": round(tolerance_m, 2),
                "threshold_exceeded": bool(threshold_exceeded),
                "score_gap": round(score_gap, 2),
                "selected_authority_rank": round(selected_authority, 4),
                "second_authority_rank": round(second_authority, 4),
                "authority_gap": round(authority_gap, 4),
                "selected_dominates": bool(selected_dominates),
                "precision_mismatch_explainable": bool(precision_mismatch_explainable),
                "reason": conflict_reason,
            }
            if (
                second is not None
                and "conflict_diagnostics" not in second
            ):
                second["conflict_diagnostics"] = {
                    "distance_m": round(distance, 2),
                    "distance_threshold_m": round(tolerance_m, 2),
                    "threshold_exceeded": bool(threshold_exceeded),
                    "score_gap": round(score_gap, 2),
                    "selected_authority_rank": round(second_authority, 4),
                    "second_authority_rank": round(selected_authority, 4),
                    "authority_gap": round(abs(authority_gap), 4),
                    "selected_dominates": bool(not selected_dominates),
                    "precision_mismatch_explainable": bool(precision_mismatch_explainable),
                    "reason": conflict_reason,
                }

    def _candidate_debug_payload(row: dict[str, Any]) -> dict[str, Any]:
        containing_regions = list(row.get("candidate_regions_containing_point") or [])
        return {
            "candidate_id": row.get("candidate_id"),
            "source": row.get("source"),
            "source_stage": row.get("source_stage"),
            "source_type": row.get("source_type"),
            "source_record_id": row.get("source_record_id"),
            "formatted_address": row.get("formatted_address"),
            "normalized_address": row.get("normalized_address"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "precision_type": row.get("precision_type"),
            "match_method": row.get("match_method"),
            "raw_score": row.get("raw_score"),
            "normalized_score": row.get("normalized_score"),
            "source_bonus": row.get("source_bonus"),
            "rank_score": row.get("rank_score"),
            "score_margin_vs_next_candidate": row.get("score_margin_vs_next_candidate"),
            "confidence_score": row.get("confidence_score"),
            "confidence_tier": row.get("confidence_tier"),
            "auto_eligible": bool(row.get("auto_eligible")),
            "auto_gate_reason": row.get("auto_gate_reason"),
            "authoritative_source_flag": bool(row.get("authoritative_source_flag")),
            "in_region_boost_applied": bool(row.get("in_region_boost_applied")),
            "exact_match_flag": bool(row.get("exact_match_flag")),
            "conflict_penalty_applied": bool(row.get("conflict_penalty_applied")),
            "in_region_result": (
                "inside_prepared_region"
                if bool(row.get("coverage_available"))
                else "outside_prepared_regions"
            ),
            "containing_regions": containing_regions,
            "resolved_region_id": row.get("resolved_region_id"),
            "distance_to_nearest_prepared_region_m": row.get("region_distance_to_boundary_m"),
            "nearest_region_id": row.get("nearest_region_id"),
            "disagreement_distance_to_other_strong_candidates_m": row.get("nearest_strong_candidate_distance_m"),
            "rejection_reason": row.get("rejection_reason"),
        }

    for row in resolver_candidates:
        if selected_candidate is not None and not ambiguous_conflict and row is selected_candidate:
            row["rejection_reason"] = None
        elif not bool(row.get("auto_eligible")):
            row["rejection_reason"] = str(row.get("auto_gate_reason") or "candidate_below_auto_use_confidence")
        else:
            row["rejection_reason"] = "not_selected_higher_rank_candidate_available"

    if selected_candidate is None or ambiguous_conflict:
        if last_failure is not None and isinstance(last_failure.detail, dict):
            detail = dict(last_failure.detail)
            status_code = int(last_failure.status_code)
        else:
            detail = {
                "error": "geocoding_failed",
                "geocode_status": "no_match",
                "geocode_outcome": "geocode_failed",
                "trusted_match_status": "rejected",
                "geocode_trust_tier": "low",
                "rejection_reason": "unresolved_no_safe_candidate",
                "message": _geocode_error_message("no_match", purpose),
                "submitted_address": submitted_address,
                "normalized_address": normalized_address,
            }
            status_code = 422

        detail["provider_attempts"] = provider_attempts
        detail["provider_statuses"] = provider_statuses
        detail["candidate_sources_attempted"] = [str(row.get("stage")) for row in provider_attempts]
        detail["candidates_found"] = len(resolver_candidates)
        detail["resolver_candidates"] = [_candidate_debug_payload(row) for row in resolver_candidates[:12]]
        detail["candidate_disagreement_distances"] = disagreement_rows
        detail["local_fallback_attempted"] = bool(local_fallback_attempted)
        detail["authoritative_fallback_result"] = authoritative_fallback_result
        detail["statewide_parcel_result"] = statewide_parcel_result
        detail["local_fallback_result"] = local_fallback_result
        detail["resolution_method"] = "none"
        detail["coordinate_source"] = None
        detail["final_coordinate_source"] = None
        detail["final_coordinates_used"] = None
        detail["match_confidence"] = "low"
        detail["coordinate_confidence"] = "none"
        detail["final_location_confidence"] = "low"
        detail["fallback_used"] = False
        detail["needs_user_confirmation"] = True
        detail["address_exists"] = bool(address_exists)
        detail["address_confidence"] = address_confidence
        detail["address_validation_sources"] = address_validation_sources
        detail["recommended_action"] = "Select your home location on the map to continue assessment."
        detail["final_candidate_selected"] = None
        detail["acceptance_threshold"] = min_auto_candidate_score
        detail["medium_confidence_threshold"] = "medium"
        detail["top_margin_threshold"] = clear_winner_min_margin
        detail["top_candidate_score"] = top_candidate_score
        detail["second_candidate_score"] = second_candidate_score
        detail["final_acceptance_decision"] = False
        detail["resolver_settings"] = {
            "conflict_distance_m": conflict_distance_m,
            "conflict_score_margin": conflict_score_margin,
            "in_region_boost": in_region_boost,
            "authoritative_source_bonus": authoritative_source_bonus,
            "clear_winner_min_margin": clear_winner_min_margin,
            "clear_winner_min_score": clear_winner_min_score,
            "in_region_preference_margin": in_region_preference_margin,
            "min_auto_candidate_score": min_auto_candidate_score,
            "emergency_in_region_guardrail": emergency_in_region_guardrail,
            "emergency_min_score": emergency_min_score,
            "emergency_min_margin": emergency_min_margin,
            "allow_interpolated_auto": allow_interpolated_auto,
            "min_geocoder_token_similarity": min_geocoder_token_similarity,
            "min_geocoder_token_coverage": min_geocoder_token_coverage,
        }

        if ambiguous_conflict:
            detail["resolution_status"] = "ambiguous_conflict"
            detail["geocode_status"] = "low_confidence"
            detail["geocode_outcome"] = "geocode_failed"
            detail["trusted_match_status"] = "rejected"
            detail["trusted_match_failure_reason"] = "candidate_conflict_across_sources"
            detail["rejection_reason"] = "candidate_conflict_across_sources"
            detail["rejection_category"] = "ambiguous_conflict"
            detail["error_class"] = "address_unresolved"
            detail["failure_reason"] = "candidate_conflict_across_sources"
            detail["message"] = (
                "Multiple plausible property coordinates from different sources disagree materially. "
                "Please confirm your home location on the map."
            )
            if selected_candidate is not None:
                detail["candidate_needs_confirmation"] = _candidate_debug_payload(selected_candidate)
                detail["final_candidate_selected"] = _candidate_debug_payload(selected_candidate)
            status_code = 422
        elif resolver_candidates:
            detail["resolution_status"] = "candidates_found_but_not_safe_enough"
            detail["geocode_status"] = "low_confidence"
            detail["geocode_outcome"] = "geocode_failed"
            detail["trusted_match_status"] = "rejected"
            detail["trusted_match_failure_reason"] = "candidate_below_auto_use_confidence"
            detail["rejection_reason"] = "candidate_below_auto_use_confidence"
            detail["rejection_category"] = "location_needs_confirmation"
            detail["error_class"] = "address_unresolved"
            detail["failure_reason"] = "candidate_below_auto_use_confidence"
            detail["message"] = (
                "Address candidates were found, but none were safe enough for automatic property coordinates. "
                "Please continue by selecting your home location on the map."
            )
            detail["candidate_needs_confirmation"] = (
                _candidate_debug_payload(candidate_needs_confirmation)
                if candidate_needs_confirmation is not None
                else None
            )
            detail["final_candidate_selected"] = detail["candidate_needs_confirmation"]
            status_code = 422
        else:
            detail["resolution_status"] = "unresolved"
            detail["error_class"] = "address_not_found"
            detail["failure_reason"] = "no_candidate_coordinates"
            if str(detail.get("geocode_status") or "") == "no_match":
                detail["rejection_category"] = "no_geocode_candidates"
                detail["message"] = (
                    "Geocoding failed for assessment. No safe candidate was found across configured providers "
                    "or local authoritative datasets."
                )
                detail["failure_reason"] = "no_geocode_candidates"
                status_code = 422
        detail["final_status"] = detail.get("resolution_status")
        if not address_exists:
            detail["error_class"] = "address_not_found"
            detail["rejection_category"] = detail.get("rejection_category") or "no_geocode_candidates"
        elif detail.get("error_class") is None:
            detail["error_class"] = "address_unresolved"

        if _is_dev_mode():
            LOGGER.warning(
                "route_geocode_resolution %s",
                json.dumps(
                    {
                        "event": "route_geocode_resolution",
                        "route_name": route_name,
                        "status_code": status_code,
                        "submitted_address": str(detail.get("submitted_address") or submitted_address),
                        "normalized_address": str(detail.get("normalized_address") or normalized_address),
                        "geocode_status": detail.get("geocode_status") or "provider_error",
                        "geocode_outcome": detail.get("geocode_outcome") or "geocode_failed",
                        "trusted_match_status": detail.get("trusted_match_status") or "rejected",
                        "resolution_status": detail.get("resolution_status"),
                        "final_acceptance_decision": detail.get("final_acceptance_decision"),
                        "failure_reason": detail.get("failure_reason"),
                        "rejection_reason": detail.get("rejection_reason"),
                        "provider_attempts": provider_attempts,
                        "candidates_found": len(resolver_candidates),
                    },
                    sort_keys=True,
                ),
            )
        raise HTTPException(status_code=status_code, detail=detail)

    # Finalize selected candidate.
    selected_meta = dict(selected_candidate.get("geocode_meta") or {})
    selected_lat = float(selected_candidate["latitude"])
    selected_lon = float(selected_candidate["longitude"])
    selected_source = str(selected_candidate.get("source") or selected_candidate.get("source_stage") or "resolver")
    selected_coverage = _region_coverage_for_coordinates(selected_lat, selected_lon)
    selected_confidence = str(selected_candidate.get("confidence_tier") or "low")
    selected_trust_status = "trusted" if _confidence_rank(selected_confidence) >= 3 else "untrusted_fallback"
    resolution_status = "resolved_high_confidence" if selected_confidence == "high" else "resolved_medium_confidence"

    selected_meta.update(
        {
            "submitted_address": submitted_address,
            "normalized_address": selected_meta.get("normalized_address") or normalized_address,
            "resolved_latitude": selected_lat,
            "resolved_longitude": selected_lon,
            "geocode_status": "accepted",
            "geocode_outcome": "geocode_succeeded_trusted" if selected_trust_status == "trusted" else "geocode_succeeded_untrusted",
            "trusted_match_status": selected_trust_status,
            "geocode_trust_tier": selected_confidence,
            "geocode_decision": resolution_status,
            "resolution_status": resolution_status,
            "resolution_method": str(selected_candidate.get("source_stage") or selected_source),
            "fallback_used": str(selected_candidate.get("source_stage") or "") not in {"primary_geocoder"},
            "final_location_confidence": selected_confidence,
            "coordinate_source": str(selected_candidate.get("source_stage") or selected_source),
            "match_confidence": selected_confidence,
            "coordinate_confidence": selected_confidence,
            "match_method": selected_candidate.get("match_method"),
            "provider_attempts": provider_attempts,
            "provider_statuses": provider_statuses,
            "local_fallback_attempted": bool(local_fallback_attempted),
            "authoritative_fallback_result": authoritative_fallback_result,
            "statewide_parcel_result": statewide_parcel_result,
            "local_fallback_result": local_fallback_result,
            "local_address_lookup_attempted": authoritative_fallback_result is not None,
            "local_address_lookup_result": authoritative_fallback_result,
            "address_point_match_found": bool((authoritative_fallback_result or {}).get("matched")),
            "parcel_lookup_attempted": statewide_parcel_result is not None,
            "parcel_lookup_result": statewide_parcel_result,
            "parcel_situs_match": bool((statewide_parcel_result or {}).get("matched")),
            "fallback_records_checked": bool(local_fallback_attempted),
            "fallback_match_method": (local_fallback_result or {}).get("match_method"),
            "candidate_sources_attempted": [str(row.get("stage")) for row in provider_attempts],
            "candidates_found": len(resolver_candidates),
            "address_exists": True,
            "address_confidence": address_confidence,
            "address_validation_sources": address_validation_sources,
            "final_coordinates_used": {"latitude": selected_lat, "longitude": selected_lon},
            "final_coordinate_source": str(selected_candidate.get("source_stage") or selected_source),
            "candidate_regions_containing_point": list(selected_coverage.get("candidate_regions_containing_point") or []),
            "region_lookup_result": (
                "inside_prepared_region"
                if bool(selected_coverage.get("coverage_available"))
                else "outside_prepared_regions"
            ),
            "region_distance_to_boundary_m": selected_coverage.get("region_distance_to_boundary_m"),
            "nearest_region_id": selected_coverage.get("nearest_region_id"),
            "unsupported_location_reason": (
                None
                if bool(selected_coverage.get("coverage_available"))
                else selected_coverage.get("reason")
            ),
            "selected_region_id": selected_coverage.get("resolved_region_id") if bool(selected_coverage.get("coverage_available")) else None,
            "selected_region_display_name": selected_coverage.get("resolved_region_display_name") if bool(selected_coverage.get("coverage_available")) else None,
            "error_class": (
                "ready_for_assessment"
                if bool(selected_coverage.get("coverage_available"))
                else "outside_prepared_region"
            ),
            "failure_reason": None,
            "acceptance_threshold": min_auto_candidate_score,
            "medium_confidence_threshold": "medium",
            "top_margin_threshold": clear_winner_min_margin,
            "top_candidate_score": top_candidate_score,
            "second_candidate_score": second_candidate_score,
            "final_acceptance_decision": True,
            "final_status": (
                "ready_for_assessment"
                if bool(selected_coverage.get("coverage_available"))
                else "outside_prepared_region"
            ),
            "needs_user_confirmation": False,
            "resolver_settings": {
                "conflict_distance_m": conflict_distance_m,
                "conflict_score_margin": conflict_score_margin,
                "in_region_boost": in_region_boost,
                "authoritative_source_bonus": authoritative_source_bonus,
                "clear_winner_min_margin": clear_winner_min_margin,
                "clear_winner_min_score": clear_winner_min_score,
                "in_region_preference_margin": in_region_preference_margin,
                "min_auto_candidate_score": min_auto_candidate_score,
                "emergency_in_region_guardrail": emergency_in_region_guardrail,
                "emergency_min_score": emergency_min_score,
                "emergency_min_margin": emergency_min_margin,
                "allow_interpolated_auto": allow_interpolated_auto,
                "min_geocoder_token_similarity": min_geocoder_token_similarity,
                "min_geocoder_token_coverage": min_geocoder_token_coverage,
            },
            "resolver_candidates": [_candidate_debug_payload(row) for row in resolver_candidates[:12]],
            "candidate_disagreement_distances": disagreement_rows,
            "final_candidate_selected": _candidate_debug_payload(selected_candidate),
        }
    )

    selected_candidate_payload = {
        "display_name": selected_meta.get("matched_address"),
        "confidence_score": selected_meta.get("confidence_score"),
        "provider": selected_meta.get("provider"),
    }
    if not selected_candidate_payload["display_name"] and selected_candidate_payload["confidence_score"] is None:
        selected_candidate_payload = None

    resolution = GeocodeResolution(
        raw_input=submitted_address,
        normalized_address=str(selected_meta.get("normalized_address") or normalized_address),
        geocode_status="accepted",
        candidate_count=int(selected_meta.get("candidate_count") or max(1, len(resolver_candidates))),
        selected_candidate=selected_candidate_payload,
        confidence_score=(
            float(selected_meta["confidence_score"])
            if selected_meta.get("confidence_score") is not None
            else None
        ),
        latitude=selected_lat,
        longitude=selected_lon,
        geocode_source=selected_source,
        geocode_meta=selected_meta,
        geocode_outcome=str(selected_meta.get("geocode_outcome") or "geocode_succeeded_untrusted"),
        trusted_match_status=str(selected_meta.get("trusted_match_status") or "untrusted_fallback"),
        rejection_reason=selected_meta.get("rejection_reason"),
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
                    "resolution_status": selected_meta.get("resolution_status"),
                    "final_acceptance_decision": selected_meta.get("final_acceptance_decision"),
                    "resolution_method": selected_meta.get("resolution_method"),
                    "coordinate_source": selected_meta.get("coordinate_source"),
                    "match_confidence": selected_meta.get("match_confidence"),
                    "match_method": selected_meta.get("match_method"),
                    "provider_attempts": provider_attempts,
                    "candidates_found": len(resolver_candidates),
                    "candidate_disagreement_distances": disagreement_rows[:5],
                    "latitude": round(resolution.latitude, 6),
                    "longitude": round(resolution.longitude, 6),
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
    geocode_meta: dict[str, Any] | None = None,
) -> RegionCoverageResolution:
    coverage = _region_coverage_for_coordinates(lat=float(latitude), lon=float(longitude))
    coverage["error_class"] = (
        "ready_for_assessment"
        if bool(coverage.get("coverage_available"))
        else "outside_prepared_region"
    )
    if geocode_meta is not None:
        coverage["trusted_match_subchecks"] = _build_trusted_match_subchecks(
            submitted_address=address_input,
            geocode_meta=geocode_meta,
            coverage=coverage,
        )
        coverage["region_check_result"] = (
            "inside_prepared_region"
            if bool(coverage.get("coverage_available"))
            else "outside_prepared_regions"
        )
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
                    "region_check_result": coverage.get("region_check_result"),
                    "trusted_match_subchecks": coverage.get("trusted_match_subchecks"),
                },
                sort_keys=True,
            ),
        )
    return resolution


def _resolve_location_for_route(
    *,
    route_name: str,
    purpose: str,
    address_input: str,
    latitude: float | None = None,
    longitude: float | None = None,
    property_anchor_point: dict[str, float] | None = None,
) -> tuple[GeocodeResolution | None, RegionCoverageResolution, float, float]:
    geocode_resolution: GeocodeResolution | None = None
    lat = latitude
    lon = longitude
    if lat is None or lon is None:
        geocode_resolution = _resolve_trusted_geocode(
            address_input=address_input,
            purpose=purpose,
            route_name=route_name,
            property_anchor_point=property_anchor_point,
        )
        lat = geocode_resolution.latitude
        lon = geocode_resolution.longitude
        final_coords = (
            dict(geocode_resolution.geocode_meta.get("final_coordinates_used") or {})
            if isinstance(geocode_resolution.geocode_meta, dict)
            else {}
        )
        if final_coords:
            try:
                final_lat = float(final_coords.get("latitude"))
                final_lon = float(final_coords.get("longitude"))
                if abs(float(lat) - final_lat) > 1e-9 or abs(float(lon) - final_lon) > 1e-9:
                    raise AssertionError(
                        "Resolver final candidate coordinates diverged from route geocode coordinates."
                    )
            except (TypeError, ValueError):
                pass

    coverage_resolution = _resolve_prepared_region(
        latitude=float(lat),
        longitude=float(lon),
        route_name=route_name,
        address_input=address_input,
        geocode_meta=geocode_resolution.geocode_meta if geocode_resolution else None,
    )
    return geocode_resolution, coverage_resolution, float(lat), float(lon)


def _build_geocode_debug_payload(address: str) -> dict[str, Any]:
    submitted_address = str(address or "")
    normalized = normalize_address(submitted_address)
    try:
        geocode_resolution, coverage_resolution, lat, lon = _resolve_location_for_route(
            address_input=submitted_address,
            purpose="assessment",
            route_name="/risk/geocode-debug",
            property_anchor_point=None,
        )
        assert geocode_resolution is not None
        meta = dict(geocode_resolution.geocode_meta or {})
        geocoder_last = dict(getattr(geocoder, "last_result", {}) or {})
        raw_preview = meta.get("raw_response_preview")
        if raw_preview is None and isinstance(geocoder_last.get("raw_response_preview"), dict):
            raw_preview = geocoder_last.get("raw_response_preview")
        coverage = dict(coverage_resolution.coverage or {})
        return {
            "original_input_address": submitted_address,
            "raw_input_address": submitted_address,
            "normalized_address": meta.get("normalized_address") or normalized,
            "geocode_status": meta.get("geocode_status") or "accepted",
            "geocode_outcome": meta.get("geocode_outcome") or geocode_resolution.geocode_outcome,
            "resolution_status": meta.get("resolution_status") or "accepted",
            "resolution_method": meta.get("resolution_method") or "primary_geocoder",
            "fallback_used": bool(meta.get("fallback_used")),
            "final_location_confidence": meta.get("final_location_confidence") or meta.get("geocode_trust_tier"),
            "address_exists": meta.get("address_exists"),
            "address_confidence": meta.get("address_confidence"),
            "address_validation_sources": meta.get("address_validation_sources"),
            "coordinate_confidence": meta.get("coordinate_confidence") or meta.get("match_confidence"),
            "error_class": meta.get("error_class") or (
                "ready_for_assessment"
                if bool((coverage or {}).get("coverage_available"))
                else "outside_prepared_region"
            ),
            "accepted": True,
            "geocode_provider": meta.get("provider") or geocode_resolution.geocode_source,
            "geocode_location_type": meta.get("geocode_location_type"),
            "geocode_precision": meta.get("geocode_precision") or "unknown",
            "match_count": int(meta.get("candidate_count") or geocode_resolution.candidate_count or 1),
            "parsed_candidates": meta.get("parsed_candidates")
            or (raw_preview or {}).get("parsed_candidates"),
            "candidate_summaries": (meta.get("parsed_candidates") or (raw_preview or {}).get("parsed_candidates") or [])[:3],
            "chosen_candidate_index": 0,
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
                "trusted_match_subchecks": meta.get("trusted_match_subchecks"),
            },
            "resolved_latitude": float(lat),
            "resolved_longitude": float(lon),
            "geocode_source": geocode_resolution.geocode_source,
            "selected_region_id": coverage.get("selected_region_id") or coverage.get("resolved_region_id"),
            "selected_region_display_name": coverage.get("selected_region_display_name") or coverage.get("display_name"),
            "rejection_reason": None,
            "rejection_category": None,
            "raw_response_preview": raw_preview,
            "provider_attempts": meta.get("provider_attempts"),
            "provider_statuses": meta.get("provider_statuses"),
            "candidate_sources_attempted": meta.get("candidate_sources_attempted"),
            "candidates_found": meta.get("candidates_found"),
            "coordinate_source": meta.get("coordinate_source"),
            "final_coordinate_source": meta.get("final_coordinate_source"),
            "final_coordinates_used": meta.get("final_coordinates_used"),
            "match_confidence": meta.get("match_confidence"),
            "match_method": meta.get("match_method"),
            "final_status": meta.get("final_status"),
            "unsupported_location_reason": meta.get("unsupported_location_reason"),
            "local_fallback_attempted": meta.get("local_fallback_attempted"),
            "authoritative_fallback_result": meta.get("authoritative_fallback_result"),
            "statewide_parcel_result": meta.get("statewide_parcel_result"),
            "local_fallback_result": meta.get("local_fallback_result"),
            "resolver_candidates": meta.get("resolver_candidates"),
            "candidate_disagreement_distances": meta.get("candidate_disagreement_distances"),
            "candidate_needs_confirmation": meta.get("candidate_needs_confirmation"),
            "final_candidate_selected": meta.get("final_candidate_selected"),
            "needs_user_confirmation": bool(meta.get("needs_user_confirmation", False)),
            "acceptance_threshold": meta.get("acceptance_threshold"),
            "medium_confidence_threshold": meta.get("medium_confidence_threshold"),
            "top_margin_threshold": meta.get("top_margin_threshold"),
            "top_candidate_score": meta.get("top_candidate_score"),
            "second_candidate_score": meta.get("second_candidate_score"),
            "final_acceptance_decision": meta.get("final_acceptance_decision"),
            "failure_reason": meta.get("failure_reason"),
            "resolver_settings": meta.get("resolver_settings"),
            "region_resolution": {
                "coverage_available": bool(coverage.get("coverage_available", False)),
                "resolved_region_id": coverage.get("resolved_region_id"),
                "selected_region_id": coverage.get("selected_region_id") or coverage.get("resolved_region_id"),
                "selected_region_display_name": coverage.get("selected_region_display_name") or coverage.get("display_name"),
                "error_class": coverage.get("error_class"),
                "reason": coverage.get("reason"),
                "diagnostics": list(coverage.get("diagnostics") or []),
                "region_distance_to_boundary_m": coverage.get("region_distance_to_boundary_m"),
                "nearest_region_id": coverage.get("nearest_region_id"),
                "candidate_regions_containing_point": coverage.get("candidate_regions_containing_point"),
                "in_region_check": bool(coverage.get("coverage_available", False)),
                "within_downloaded_zone_check": bool(coverage.get("coverage_available", False)),
            },
        }
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return {
            "original_input_address": submitted_address,
            "raw_input_address": submitted_address,
            "normalized_address": detail.get("normalized_address") or normalized,
            "geocode_status": detail.get("geocode_status") or "provider_error",
            "geocode_outcome": detail.get("geocode_outcome") or "geocode_failed",
            "resolution_status": detail.get("resolution_status") or "unresolved",
            "resolution_method": detail.get("resolution_method"),
            "fallback_used": bool(detail.get("fallback_used")),
            "final_location_confidence": detail.get("final_location_confidence"),
            "address_exists": detail.get("address_exists"),
            "address_confidence": detail.get("address_confidence"),
            "address_validation_sources": detail.get("address_validation_sources"),
            "coordinate_confidence": detail.get("coordinate_confidence"),
            "error_class": detail.get("error_class") or "address_unresolved",
            "accepted": False,
            "geocode_provider": detail.get("provider") or "OpenStreetMap Nominatim",
            "geocode_location_type": None,
            "geocode_precision": "unknown",
            "match_count": int(((detail.get("raw_response_preview") or {}).get("candidate_count")) or 0),
            "parsed_candidates": (detail.get("raw_response_preview") or {}).get("parsed_candidates"),
            "candidate_summaries": ((detail.get("raw_response_preview") or {}).get("parsed_candidates") or [])[:3],
            "chosen_candidate_index": None,
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
                "trusted_match_subchecks": detail.get("trusted_match_subchecks"),
            },
            "resolved_latitude": None,
            "resolved_longitude": None,
            "geocode_source": detail.get("provider") or "OpenStreetMap Nominatim",
            "selected_region_id": None,
            "selected_region_display_name": None,
            "rejection_reason": detail.get("rejection_reason"),
            "rejection_category": detail.get("rejection_category") or detail.get("geocode_status"),
            "raw_response_preview": detail.get("raw_response_preview"),
            "provider_attempts": detail.get("provider_attempts"),
            "provider_statuses": detail.get("provider_statuses"),
            "candidate_sources_attempted": detail.get("candidate_sources_attempted"),
            "candidates_found": detail.get("candidates_found"),
            "coordinate_source": detail.get("coordinate_source"),
            "final_coordinate_source": detail.get("final_coordinate_source"),
            "final_coordinates_used": detail.get("final_coordinates_used"),
            "match_confidence": detail.get("match_confidence"),
            "match_method": detail.get("match_method"),
            "final_status": detail.get("final_status"),
            "unsupported_location_reason": detail.get("unsupported_location_reason"),
            "local_fallback_attempted": detail.get("local_fallback_attempted"),
            "authoritative_fallback_result": detail.get("authoritative_fallback_result"),
            "statewide_parcel_result": detail.get("statewide_parcel_result"),
            "local_fallback_result": detail.get("local_fallback_result"),
            "resolver_candidates": detail.get("resolver_candidates"),
            "candidate_disagreement_distances": detail.get("candidate_disagreement_distances"),
            "candidate_needs_confirmation": detail.get("candidate_needs_confirmation"),
            "final_candidate_selected": detail.get("final_candidate_selected"),
            "needs_user_confirmation": bool(detail.get("needs_user_confirmation", False)),
            "acceptance_threshold": detail.get("acceptance_threshold"),
            "medium_confidence_threshold": detail.get("medium_confidence_threshold"),
            "top_margin_threshold": detail.get("top_margin_threshold"),
            "top_candidate_score": detail.get("top_candidate_score"),
            "second_candidate_score": detail.get("second_candidate_score"),
            "final_acceptance_decision": detail.get("final_acceptance_decision"),
            "failure_reason": detail.get("failure_reason"),
            "resolver_settings": detail.get("resolver_settings"),
            "region_resolution": None,
        }
    except Exception as exc:
        return {
            "original_input_address": submitted_address,
            "raw_input_address": submitted_address,
            "normalized_address": normalized,
            "geocode_status": "provider_error",
            "geocode_outcome": "geocode_failed",
            "resolution_status": "unresolved",
            "resolution_method": None,
            "fallback_used": False,
            "final_location_confidence": "low",
            "address_exists": False,
            "address_confidence": "low",
            "address_validation_sources": [],
            "coordinate_confidence": "none",
            "error_class": "address_not_found",
            "accepted": False,
            "geocode_provider": "OpenStreetMap Nominatim",
            "geocode_location_type": None,
            "geocode_precision": "unknown",
            "match_count": 0,
            "parsed_candidates": None,
            "candidate_summaries": [],
            "chosen_candidate_index": None,
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
                "trusted_match_subchecks": None,
            },
            "resolved_latitude": None,
            "resolved_longitude": None,
            "geocode_source": "OpenStreetMap Nominatim",
            "selected_region_id": None,
            "selected_region_display_name": None,
            "rejection_reason": str(exc),
            "rejection_category": "provider_error",
            "raw_response_preview": None,
            "provider_attempts": None,
            "provider_statuses": None,
            "candidate_sources_attempted": None,
            "candidates_found": None,
            "coordinate_source": None,
            "final_coordinate_source": None,
            "final_coordinates_used": None,
            "match_confidence": "low",
            "match_method": "unresolved",
            "final_status": "unresolved",
            "unsupported_location_reason": None,
            "local_fallback_attempted": False,
            "authoritative_fallback_result": None,
            "statewide_parcel_result": None,
            "local_fallback_result": None,
            "resolver_candidates": None,
            "candidate_disagreement_distances": None,
            "candidate_needs_confirmation": None,
            "final_candidate_selected": None,
            "needs_user_confirmation": False,
            "acceptance_threshold": None,
            "medium_confidence_threshold": None,
            "top_margin_threshold": None,
            "top_candidate_score": None,
            "second_candidate_score": None,
            "final_acceptance_decision": False,
            "failure_reason": "resolver_exception",
            "resolver_settings": None,
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


@app.post(
    "/risk/address-candidates",
    response_model=AddressCandidateSearchResponse,
    dependencies=[Depends(require_api_key)],
)
def search_manual_address_candidates(
    payload: AddressCandidateSearchRequest,
    _: ActorContext = Depends(get_actor_context),
) -> AddressCandidateSearchResponse:
    response_payload = _build_manual_address_candidates(
        address=str(payload.address or "").strip(),
        zip_code=payload.zip_code,
        locality=payload.locality,
        state=payload.state or "WA",
        limit=int(payload.limit or 8),
    )
    if _is_dev_mode():
        LOGGER.info(
            "manual_address_candidate_search %s",
            json.dumps(
                {
                    "event": "manual_address_candidate_search",
                    "input_address": response_payload.get("input_address"),
                    "normalized_address": response_payload.get("normalized_address"),
                    "zip_code": response_payload.get("zip_code"),
                    "selected_locality": response_payload.get("selected_locality"),
                    "candidate_count": len(response_payload.get("candidates") or []),
                    "status": response_payload.get("status"),
                    "map_click_fallback_recommended": bool(response_payload.get("map_click_fallback_recommended")),
                },
                sort_keys=True,
            ),
        )
    candidates = [ManualAddressCandidate.model_validate(row) for row in response_payload.get("candidates") or []]
    return AddressCandidateSearchResponse(
        status=str(response_payload.get("status") or "ready_for_map_click_fallback"),
        input_address=str(response_payload.get("input_address") or payload.address),
        normalized_address=str(response_payload.get("normalized_address") or normalize_address(payload.address)),
        zip_code=response_payload.get("zip_code"),
        inferred_localities=list(response_payload.get("inferred_localities") or []),
        selected_locality=response_payload.get("selected_locality"),
        candidates=candidates,
        map_click_fallback_recommended=bool(response_payload.get("map_click_fallback_recommended", False)),
        diagnostics=list(response_payload.get("diagnostics") or []),
        final_status=response_payload.get("final_status"),
    )


def _region_data_root() -> str:
    return os.getenv("WF_REGION_DATA_DIR", str(wildfire_data.region_data_dir))


def _region_coverage_for_coordinates(lat: float, lon: float) -> dict[str, Any]:
    lookup = lookup_region_for_point(lat=lat, lon=lon, regions_root=_region_data_root())
    containing_region_ids = list(lookup.get("containing_region_ids") or [])
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
            "resolved_region_display_name": lookup.get("display_name"),
            "selected_region_id": resolved_region_id,
            "selected_region_display_name": lookup.get("display_name"),
            "reason": "prepared_region_found",
            "recommended_action": None,
            "region_check_result": "inside_prepared_region",
            "region_distance_to_boundary_m": 0.0,
            "nearest_region_id": resolved_region_id,
            "edge_tolerance_m": lookup.get("edge_tolerance_m"),
            "candidate_regions_containing_point": containing_region_ids
            or ([str(resolved_region_id)] if resolved_region_id else []),
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
        "resolved_region_display_name": None,
        "selected_region_id": None,
        "selected_region_display_name": None,
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
        "candidate_regions_containing_point": containing_region_ids,
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
    if (lat is None or lon is None) and (not payload.address or len(payload.address.strip()) < 5):
        raise HTTPException(status_code=400, detail="Provide either latitude/longitude or a valid address.")

    geocode_resolution, coverage_resolution, lat, lon = _resolve_location_for_route(
        route_name="/regions/coverage-check",
        purpose="region_coverage_check",
        address_input=str(payload.address or ""),
        latitude=(float(lat) if lat is not None else None),
        longitude=(float(lon) if lon is not None else None),
    )
    coverage = dict(coverage_resolution.coverage)
    if geocode_resolution:
        geocode_meta = geocode_resolution.geocode_meta
        coverage["geocode_status"] = geocode_meta.get("geocode_status")
        coverage["geocode_outcome"] = geocode_meta.get("geocode_outcome")
        coverage["geocode_decision"] = geocode_meta.get("geocode_decision")
        coverage["geocode_trust_tier"] = geocode_meta.get("geocode_trust_tier")
        coverage["trusted_match_status"] = geocode_meta.get("trusted_match_status")
        coverage["trusted_match_failure_reason"] = geocode_meta.get("trusted_match_failure_reason")
        coverage["fallback_eligibility"] = geocode_meta.get("fallback_eligibility")
        coverage["normalized_address"] = geocode_meta.get("normalized_address")
        coverage["geocode_source"] = geocode_meta.get("geocode_source")
        coverage["geocode_precision"] = geocode_meta.get("geocode_precision")
        coverage["geocode_location_type"] = geocode_meta.get("geocode_location_type")
        coverage["resolution_status"] = geocode_meta.get("resolution_status")
        coverage["resolution_method"] = geocode_meta.get("resolution_method")
        coverage["fallback_used"] = geocode_meta.get("fallback_used")
        coverage["final_location_confidence"] = geocode_meta.get("final_location_confidence")
        coverage["provider_attempts"] = geocode_meta.get("provider_attempts")
        coverage["provider_statuses"] = geocode_meta.get("provider_statuses")
        coverage["candidate_sources_attempted"] = geocode_meta.get("candidate_sources_attempted")
        coverage["candidates_found"] = geocode_meta.get("candidates_found")
        coverage["coordinate_source"] = geocode_meta.get("coordinate_source")
        coverage["final_coordinate_source"] = geocode_meta.get("final_coordinate_source")
        coverage["final_coordinates_used"] = geocode_meta.get("final_coordinates_used")
        coverage["match_confidence"] = geocode_meta.get("match_confidence")
        coverage["match_method"] = geocode_meta.get("match_method")
        coverage["unsupported_location_reason"] = geocode_meta.get("unsupported_location_reason")
        coverage["local_fallback_attempted"] = geocode_meta.get("local_fallback_attempted")
        coverage["authoritative_fallback_result"] = geocode_meta.get("authoritative_fallback_result")
        coverage["local_fallback_result"] = geocode_meta.get("local_fallback_result")
        coverage["address_exists"] = geocode_meta.get("address_exists")
        coverage["address_confidence"] = geocode_meta.get("address_confidence")
        coverage["address_validation_sources"] = geocode_meta.get("address_validation_sources")
        coverage["coordinate_confidence"] = geocode_meta.get("coordinate_confidence")
        coverage["needs_user_confirmation"] = geocode_meta.get("needs_user_confirmation")
        coverage["final_status"] = geocode_meta.get("final_status")
        coverage["resolver_candidates"] = geocode_meta.get("resolver_candidates")
        coverage["candidate_disagreement_distances"] = geocode_meta.get("candidate_disagreement_distances")
        coverage["candidate_needs_confirmation"] = geocode_meta.get("candidate_needs_confirmation")
        coverage["final_candidate_selected"] = geocode_meta.get("final_candidate_selected")
        coverage["resolver_settings"] = geocode_meta.get("resolver_settings")
        coverage["acceptance_threshold"] = geocode_meta.get("acceptance_threshold")
        coverage["medium_confidence_threshold"] = geocode_meta.get("medium_confidence_threshold")
        coverage["top_margin_threshold"] = geocode_meta.get("top_margin_threshold")
        coverage["top_candidate_score"] = geocode_meta.get("top_candidate_score")
        coverage["second_candidate_score"] = geocode_meta.get("second_candidate_score")
        coverage["final_acceptance_decision"] = geocode_meta.get("final_acceptance_decision")
        coverage["failure_reason"] = geocode_meta.get("failure_reason")
        coverage["error_class"] = (
            geocode_meta.get("error_class")
            if bool(coverage.get("coverage_available"))
            else "outside_prepared_region"
        )
        coverage["candidate_regions_containing_point"] = coverage.get("candidate_regions_containing_point")
        coverage["trusted_match_subchecks"] = coverage.get("trusted_match_subchecks") or geocode_meta.get(
            "trusted_match_subchecks"
        )
    return RegionCoverageStatus.model_validate(coverage)


@app.post(
    "/risk/assess",
    response_model=AssessmentResult | AssessmentWithDiagnosticsResponse,
    dependencies=[Depends(require_api_key)],
)
def assess_risk(
    payload: AddressRequest,
    include_diagnostics: bool = Query(
        default=False,
        description=(
            "Include no-ground-truth trust diagnostics metadata. "
            "These diagnostics are coherence/evidence checks and are not ground-truth accuracy claims."
        ),
    ),
    include_calibrated_outputs: bool = Query(
        default=False,
        description=(
            "Include optional calibrated public-outcome metadata. "
            "This does not replace raw scores and is not carrier underwriting probability."
        ),
    ),
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult | AssessmentWithDiagnosticsResponse:
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
    requested_anchor = _coerce_point_payload(payload.property_anchor_point or payload.user_selected_point)
    geocode_resolution, coverage_resolution, _, _ = _resolve_location_for_route(
        route_name="/risk/assess",
        purpose="assessment",
        address_input=payload.address,
        property_anchor_point=requested_anchor,
    )
    assert geocode_resolution is not None
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
                    "geocode_decision": geocode_meta.get("geocode_decision"),
                    "geocode_trust_tier": geocode_meta.get("geocode_trust_tier"),
                    "trusted_match_status": geocode_meta.get("trusted_match_status"),
                    "trusted_match_failure_reason": geocode_meta.get("trusted_match_failure_reason"),
                    "trusted_match_subchecks": coverage.get("trusted_match_subchecks")
                    or geocode_meta.get("trusted_match_subchecks"),
                    "fallback_eligibility": geocode_meta.get("fallback_eligibility"),
                    "submitted_address": geocode_meta.get("submitted_address"),
                    "normalized_address": geocode_meta.get("normalized_address"),
                    "address_exists": geocode_meta.get("address_exists", True),
                    "address_confidence": geocode_meta.get("address_confidence"),
                    "address_validation_sources": geocode_meta.get("address_validation_sources"),
                    "resolved_latitude": geocode_meta.get("resolved_latitude"),
                    "resolved_longitude": geocode_meta.get("resolved_longitude"),
                    "coordinate_confidence": geocode_meta.get("coordinate_confidence"),
                    "coverage_available": False,
                    "resolved_region_id": None,
                    "selected_region_id": coverage.get("selected_region_id"),
                    "selected_region_display_name": coverage.get("selected_region_display_name"),
                    "reason": "no_prepared_region_for_location",
                    "error_class": "outside_prepared_region",
                    "unsupported_location_reason": coverage.get("reason"),
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
                    "candidate_regions_containing_point": coverage.get("candidate_regions_containing_point"),
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
                    "geocode_decision": geocode_meta.get("geocode_decision"),
                    "geocode_trust_tier": geocode_meta.get("geocode_trust_tier"),
                    "trusted_match_status": geocode_meta.get("trusted_match_status"),
                    "trusted_match_failure_reason": geocode_meta.get("trusted_match_failure_reason"),
                    "trusted_match_subchecks": coverage.get("trusted_match_subchecks")
                    or geocode_meta.get("trusted_match_subchecks"),
                    "fallback_eligibility": geocode_meta.get("fallback_eligibility"),
                "submitted_address": geocode_meta.get("submitted_address"),
                "normalized_address": geocode_meta.get("normalized_address"),
                "address_exists": geocode_meta.get("address_exists", True),
                "address_confidence": geocode_meta.get("address_confidence"),
                "address_validation_sources": geocode_meta.get("address_validation_sources"),
                "resolved_latitude": geocode_meta.get("resolved_latitude"),
                "resolved_longitude": geocode_meta.get("resolved_longitude"),
                "coordinate_confidence": geocode_meta.get("coordinate_confidence"),
                "coverage_available": False,
                "resolved_region_id": None,
                "selected_region_id": coverage.get("selected_region_id"),
                "selected_region_display_name": coverage.get("selected_region_display_name"),
                "reason": "no_prepared_region_for_location",
                "error_class": "outside_prepared_region",
                "unsupported_location_reason": coverage.get("reason"),
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
                "candidate_regions_containing_point": coverage.get("candidate_regions_containing_point"),
            },
        )

    result = _compute_assessment(
        payload,
        organization_id=organization_id,
        ruleset=ruleset,
        geocode_resolution=geocode_resolution,
        coverage_resolution=coverage_resolution,
        include_calibrated_outputs=bool(include_calibrated_outputs or payload.include_calibrated_outputs),
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
    if include_diagnostics:
        diagnostics = _build_assessment_trust_metadata(
            result=result,
            payload=payload,
            organization_id=organization_id,
            ruleset=ruleset,
            geocode_resolution=geocode_resolution,
            coverage_resolution=coverage_resolution,
        )
        return AssessmentWithDiagnosticsResponse(assessment=result, diagnostics=diagnostics)
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


@app.get(
    "/risk/improve/{assessment_id}",
    response_model=HomeownerImprovementOptions,
    dependencies=[Depends(require_api_key)],
)
def homeowner_improvement_options(
    assessment_id: str,
    ctx: ActorContext = Depends(get_actor_context),
) -> HomeownerImprovementOptions:
    existing = store.get(assessment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, existing.organization_id)
    return build_homeowner_improvement_options(existing)


@app.post(
    "/risk/improve/{assessment_id}",
    response_model=HomeownerImprovementRunResponse,
    dependencies=[Depends(require_api_key)],
)
def rerun_assessment_with_homeowner_inputs(
    assessment_id: str,
    payload: HomeownerImprovementRunRequest,
    ctx: ActorContext = Depends(get_actor_context),
) -> HomeownerImprovementRunResponse:
    _require_role(ctx, WRITE_ROLES, "Viewer role cannot update assessments")

    existing = store.get(assessment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, existing.organization_id)

    base_req = _payload_from_assessment(existing)
    improvement_attrs = payload.attributes.model_copy(deep=True)
    auto_confirmed_fields: list[str] = []
    mapped_defensible_space_ft = defensible_space_ft_from_condition(payload.defensible_space_condition)
    if mapped_defensible_space_ft is not None and improvement_attrs.defensible_space_ft is None:
        improvement_attrs.defensible_space_ft = mapped_defensible_space_ft
        auto_confirmed_fields.append("defensible_space_ft")

    merged_attrs = _merge_attributes(base_req.attributes, improvement_attrs)
    normalized_changes = normalized_attribute_changes(base_req.attributes, merged_attrs)
    if (
        not normalized_changes
        and not payload.confirmed_fields
        and mapped_defensible_space_ft is None
        and payload.property_anchor_point is None
        and payload.user_selected_point is None
        and payload.selected_structure_geometry is None
        and payload.selected_structure_id is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "No property-detail updates were provided. Submit roof_type, vent_type, "
                "window_type, defensible_space_ft, defensible_space_condition, map-point correction, "
                "or a selected/drawn building polygon."
            ),
        )

    merged_confirmed = sorted(
        set(
            base_req.confirmed_fields
            + payload.confirmed_fields
            + auto_confirmed_fields
            + list(normalized_changes.keys())
        )
    )
    ruleset = _get_ruleset_or_default(payload.ruleset_id or existing.ruleset_id)

    request_payload = AddressRequest(
        address=base_req.address,
        attributes=merged_attrs,
        confirmed_fields=merged_confirmed,
        structure_geometry_source=(
            payload.structure_geometry_source
            or (
                "user_modified"
                if isinstance(payload.selected_structure_geometry, dict)
                else base_req.structure_geometry_source
            )
        ),
        selection_mode=(
            payload.selection_mode
            if payload.selection_mode in {"polygon", "point"}
            else (
            "point"
            if (payload.property_anchor_point is not None or payload.user_selected_point is not None)
            else base_req.selection_mode
            )
        ),
        property_anchor_point=payload.property_anchor_point or base_req.property_anchor_point,
        user_selected_point=payload.user_selected_point or payload.property_anchor_point or base_req.user_selected_point,
        selected_structure_id=payload.selected_structure_id or base_req.selected_structure_id,
        selected_structure_geometry=(
            payload.selected_structure_geometry
            if isinstance(payload.selected_structure_geometry, dict)
            else base_req.selected_structure_geometry
        ),
        audience=payload.audience or base_req.audience,
        tags=base_req.tags,
        organization_id=existing.organization_id,
        ruleset_id=ruleset.ruleset_id,
        include_calibrated_outputs=bool(base_req.include_calibrated_outputs),
    )
    updated = _compute_assessment(
        request_payload,
        organization_id=existing.organization_id,
        ruleset=ruleset,
        portfolio_name=existing.portfolio_name,
        tags=existing.tags,
    )
    updated.review_status = existing.review_status
    updated.workflow_state = existing.workflow_state
    updated.assigned_reviewer = existing.assigned_reviewer
    updated.assigned_role = existing.assigned_role
    store.save(updated)

    before_options = build_homeowner_improvement_options(existing)
    after_options = build_homeowner_improvement_options(updated)
    what_changed, change_notes, what_changed_summary = build_improvement_change_set(
        existing,
        updated,
        changed_fields_hint=list(normalized_changes.keys()),
    )
    why_it_matters = build_improvement_why_it_matters(
        existing,
        updated,
        what_changed_summary,
    )
    if payload.defensible_space_condition and mapped_defensible_space_ft is not None:
        what_changed.setdefault(
            "defensible_space_condition_mapping",
            {"before": payload.defensible_space_condition, "after": mapped_defensible_space_ft},
        )
        change_notes.append(
            "Defensible space condition was converted to an approximate defensible-space distance for scoring."
        )
    if payload.property_anchor_point is not None or payload.user_selected_point is not None:
        previous_anchor = (
            base_req.user_selected_point
            or base_req.property_anchor_point
        )
        incoming_anchor = payload.user_selected_point or payload.property_anchor_point
        what_changed["map_point_correction"] = {
            "before": (
                previous_anchor.model_dump()
                if hasattr(previous_anchor, "model_dump")
                else (
                    {
                        "latitude": previous_anchor.latitude,
                        "longitude": previous_anchor.longitude,
                    }
                    if previous_anchor is not None
                    else None
                )
            ),
            "after": (
                incoming_anchor.model_dump()
                if hasattr(incoming_anchor, "model_dump")
                else (
                    {
                        "latitude": incoming_anchor.latitude,
                        "longitude": incoming_anchor.longitude,
                    }
                    if incoming_anchor is not None
                    else None
                )
            ),
        }
        change_notes.append(
            "Home location pin was updated and the assessment anchor was re-run."
        )
    if payload.selected_structure_geometry is not None or payload.selected_structure_id is not None:
        what_changed["structure_geometry_update"] = {
            "before_structure_id": (
                base_req.selected_structure_id
                if base_req.selected_structure_id
                else None
            ),
            "after_structure_id": (
                payload.selected_structure_id
                if payload.selected_structure_id
                else (updated.matched_structure_id if updated.matched_structure_id else None)
            ),
            "geometry_provided": bool(isinstance(payload.selected_structure_geometry, dict)),
        }
        change_notes.append(
            "Building geometry selection was updated and structure-relative rings were recomputed."
        )

    before_specificity = str((existing.specificity_summary or {}).specificity_tier) if hasattr(existing.specificity_summary, "specificity_tier") else str((existing.specificity_summary or {}).get("specificity_tier") if isinstance(existing.specificity_summary, dict) else "regional_estimate")
    after_specificity = str((updated.specificity_summary or {}).specificity_tier) if hasattr(updated.specificity_summary, "specificity_tier") else str((updated.specificity_summary or {}).get("specificity_tier") if isinstance(updated.specificity_summary, dict) else "regional_estimate")
    before_conf = float(existing.confidence_score or 0.0)
    after_conf = float(updated.confidence_score or 0.0)
    before_risk = float(existing.wildfire_risk_score or 0.0) if existing.wildfire_risk_score is not None else None
    after_risk = float(updated.wildfire_risk_score or 0.0) if updated.wildfire_risk_score is not None else None

    geometry_updated = bool(
        payload.property_anchor_point is not None
        or payload.user_selected_point is not None
        or payload.selected_structure_geometry is not None
        or payload.selected_structure_id is not None
    )
    score_change = {
        "before": before_risk,
        "after": after_risk,
        "delta": (
            round(float(after_risk) - float(before_risk), 2)
            if before_risk is not None and after_risk is not None
            else None
        ),
    }
    specificity_change = {
        "before": before_specificity,
        "after": after_specificity,
        "changed": before_specificity != after_specificity,
    }
    confidence_change = {
        "before": round(before_conf, 2),
        "after": round(after_conf, 2),
        "delta": round(after_conf - before_conf, 2),
    }
    what_changed["geometry_updated"] = geometry_updated
    what_changed["score_change"] = score_change
    existing_specificity_change = (
        what_changed.get("specificity_change")
        if isinstance(what_changed.get("specificity_change"), dict)
        else {}
    )
    what_changed["specificity_change"] = {**existing_specificity_change, **specificity_change}
    existing_confidence_change = (
        what_changed.get("confidence_change")
        if isinstance(what_changed.get("confidence_change"), dict)
        else {}
    )
    what_changed["confidence_change"] = {**existing_confidence_change, **confidence_change}
    what_changed_summary["geometry_updated"] = geometry_updated
    what_changed_summary["score_change"] = score_change
    existing_summary_specificity_change = (
        what_changed_summary.get("specificity_change")
        if isinstance(what_changed_summary.get("specificity_change"), dict)
        else {}
    )
    what_changed_summary["specificity_change"] = {
        **existing_summary_specificity_change,
        **specificity_change,
    }
    existing_summary_confidence_change = (
        what_changed_summary.get("confidence_change")
        if isinstance(what_changed_summary.get("confidence_change"), dict)
        else {}
    )
    what_changed_summary["confidence_change"] = {
        **existing_summary_confidence_change,
        **confidence_change,
    }

    confidence_improved = float(updated.confidence_score or 0.0) > float(existing.confidence_score or 0.0)
    recommendations_adjusted = (
        list(existing.top_recommended_actions or [])[:3] != list(updated.top_recommended_actions or [])[:3]
        or before_options.improve_your_result_suggestions != after_options.improve_your_result_suggestions
    )

    _log_audit(
        ctx=ctx,
        entity_type="assessment",
        entity_id=updated.assessment_id,
        action="assessment_improvement_rerun_created",
        organization_id=updated.organization_id,
        metadata={
            "source_assessment_id": assessment_id,
            "ruleset_id": ruleset.ruleset_id,
            "changed_fields": sorted(list(normalized_changes.keys())),
            "defensible_space_condition": payload.defensible_space_condition,
        },
    )
    return HomeownerImprovementRunResponse(
        baseline_assessment_id=assessment_id,
        updated_assessment_id=updated.assessment_id,
        before_summary=summarize_assessment_for_improvement(existing),
        after_summary=summarize_assessment_for_improvement(updated),
        what_changed=what_changed,
        what_changed_summary=what_changed_summary,
        why_it_matters=why_it_matters,
        confidence_improved=confidence_improved,
        recommendations_adjusted=recommendations_adjusted,
        improve_your_result_before=before_options,
        improve_your_result_after=after_options,
        change_notes=list(dict.fromkeys(change_notes)),
    )


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
    hardening_delta = _delta(baseline.home_hardening_readiness, simulated.home_hardening_readiness)

    sim_result = SimulationResult(
        scenario_name=payload.scenario_name,
        baseline=baseline,
        simulated=simulated,
        delta=SimulationDelta(
            wildfire_risk_score_delta=wildfire_delta,
            insurance_readiness_score_delta=readiness_delta,
            home_hardening_readiness_delta=hardening_delta,
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
            home_hardening_readiness_delta=hardening_delta,
        ),
        base_confidence=baseline.confidence,
        simulated_confidence=simulated.confidence,
        base_assumptions=baseline.assumptions,
        simulated_assumptions=simulated.assumptions,
        simulator_explanations=build_simulator_explanations(
            baseline=baseline.model_dump(mode="json"),
            simulated=simulated.model_dump(mode="json"),
        ),
        summary=(
            f"Wildfire risk changed by {wildfire_delta if wildfire_delta is not None else 'not computed'} "
            f"and home hardening readiness changed by "
            f"{hardening_delta if hardening_delta is not None else 'not computed'}."
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
    requested_anchor = _coerce_point_payload(payload.property_anchor_point or payload.user_selected_point)
    geocode_resolution, coverage_resolution, _, _ = _resolve_location_for_route(
        route_name="/risk/debug",
        purpose="assessment",
        address_input=payload.address,
        property_anchor_point=requested_anchor,
    )
    assert geocode_resolution is not None
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
    requested_anchor = _coerce_point_payload(payload.property_anchor_point or payload.user_selected_point)
    geocode_resolution, coverage_resolution, _, _ = _resolve_location_for_route(
        route_name="/risk/layer-diagnostics",
        purpose="assessment",
        address_input=payload.address,
        property_anchor_point=requested_anchor,
    )
    assert geocode_resolution is not None
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
            "property_specific_readiness": (debug_payload.get("property_level_context") or {}).get(
                "region_property_specific_readiness"
            ),
            "required_layers_missing": (debug_payload.get("property_level_context") or {}).get(
                "region_required_layers_missing",
                [],
            ),
            "optional_layers_missing": (debug_payload.get("property_level_context") or {}).get(
                "region_optional_layers_missing",
                [],
            ),
            "enrichment_layers_missing": (debug_payload.get("property_level_context") or {}).get(
                "region_enrichment_layers_missing",
                [],
            ),
            "missing_reason_by_layer": (debug_payload.get("property_level_context") or {}).get(
                "region_missing_reason_by_layer",
                {},
            ),
            "region_resolution": debug_payload.get("region_resolution", {}),
        },
        "structure_footprint": {
            "footprint_used": (debug_payload.get("property_level_context") or {}).get("footprint_used"),
            "footprint_status": (debug_payload.get("property_level_context") or {}).get("footprint_status"),
            "footprint_source": (debug_payload.get("property_level_context") or {}).get("footprint_source"),
            "footprint_source_name": (debug_payload.get("property_level_context") or {}).get("footprint_source_name"),
            "footprint_source_vintage": (debug_payload.get("property_level_context") or {}).get("footprint_source_vintage"),
            "geometry_basis": (debug_payload.get("property_level_context") or {}).get("geometry_basis"),
            "anchor_quality": (debug_payload.get("property_level_context") or {}).get("anchor_quality"),
            "anchor_quality_score": (debug_payload.get("property_level_context") or {}).get("anchor_quality_score"),
            "property_anchor_point": (debug_payload.get("property_level_context") or {}).get("property_anchor_point"),
            "property_anchor_source": (debug_payload.get("property_level_context") or {}).get("property_anchor_source"),
            "property_anchor_precision": (debug_payload.get("property_level_context") or {}).get("property_anchor_precision"),
            "parcel_id": (debug_payload.get("property_level_context") or {}).get("parcel_id"),
            "parcel_lookup_method": (debug_payload.get("property_level_context") or {}).get("parcel_lookup_method"),
            "parcel_lookup_distance_m": (debug_payload.get("property_level_context") or {}).get("parcel_lookup_distance_m"),
            "parcel_source_name": (debug_payload.get("property_level_context") or {}).get("parcel_source_name"),
            "parcel_source_vintage": (debug_payload.get("property_level_context") or {}).get("parcel_source_vintage"),
            "parcel_source": (debug_payload.get("property_level_context") or {}).get("parcel_source"),
            "structure_match_status": (debug_payload.get("property_level_context") or {}).get("structure_match_status"),
            "structure_match_method": (debug_payload.get("property_level_context") or {}).get("structure_match_method"),
            "structure_selection_method": (debug_payload.get("property_level_context") or {}).get("structure_selection_method"),
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
        "geometry_resolution": (
            debug_payload.get("geometry_resolution")
            if isinstance(debug_payload.get("geometry_resolution"), dict)
            else ((debug_payload.get("property_level_context") or {}).get("geometry_resolution") or {})
        ),
        "top_near_structure_risk_drivers": debug_payload.get("top_near_structure_risk_drivers", []),
        "prioritized_vegetation_actions": debug_payload.get("prioritized_vegetation_actions", []),
        "defensible_space_limitations_summary": debug_payload.get("defensible_space_limitations_summary", []),
        "feature_bundle": {
            "bundle_id": (debug_payload.get("property_level_context") or {}).get("feature_bundle_id"),
            "cache_hit": (debug_payload.get("property_level_context") or {}).get("feature_bundle_cache_hit"),
            "data_sources": (debug_payload.get("property_level_context") or {}).get("feature_bundle_data_sources", {}),
            "coverage_flags": (debug_payload.get("property_level_context") or {}).get(
                "feature_bundle_coverage_flags",
                {},
            ),
            "summary": (debug_payload.get("property_level_context") or {}).get("feature_bundle_summary", {}),
        },
        "layer_coverage_audit": debug_payload.get("layer_coverage_audit", []),
        "coverage_summary": debug_payload.get("coverage_summary", {}),
        "fallback_decisions": {
            "assumptions_used": debug_payload.get("assumptions_used", []),
            "assessment_blockers": ((debug_payload.get("eligibility") or {}).get("assessment_blockers") or []),
            "confidence_use_restriction": ((debug_payload.get("confidence_gating") or {}).get("use_restriction")),
        },
        "homeowner_summary": {
            "assessment_output_state": ((debug_payload.get("coverage") or {}).get("assessment_output_state")),
            "assessment_limitations": debug_payload.get("assessment_limitations", []),
            "what_was_observed": debug_payload.get("what_was_observed", []),
            "what_was_estimated": debug_payload.get("what_was_estimated", []),
            "what_was_missing": debug_payload.get("what_was_missing", []),
            "why_this_result_is_limited": debug_payload.get("why_this_result_is_limited"),
            "data_quality_summary": ((debug_payload.get("coverage") or {}).get("data_quality_summary") or {}),
            "confidence_improvement_actions": (
                ((debug_payload.get("homeowner_summary") or {}).get("confidence_improvement_actions"))
                if isinstance(debug_payload.get("homeowner_summary"), dict)
                else []
            ),
            "confidence_how_to_improve": (
                (((debug_payload.get("homeowner_summary") or {}).get("confidence_summary") or {}).get("how_to_improve_confidence"))
                if isinstance(((debug_payload.get("homeowner_summary") or {}).get("confidence_summary") or {}), dict)
                else []
            ),
        },
        "warnings": debug_payload.get("coverage_summary", {}).get("recommended_actions", []),
    }


def _internal_diagnostics_html_path() -> Path:
    return Path(__file__).resolve().parents[1] / "frontend" / "public" / "internal_diagnostics.html"


@app.get("/internal/diagnostics", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
def internal_diagnostics_page(_: ActorContext = Depends(get_actor_context)) -> HTMLResponse:
    path = _internal_diagnostics_html_path()
    if not path.exists():
        return HTMLResponse(
            "<h1>Internal diagnostics page not found.</h1>"
            "<p>Create frontend/public/internal_diagnostics.html or rebuild frontend assets.</p>",
            status_code=404,
        )
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/internal/diagnostics/api/runs", dependencies=[Depends(require_api_key)])
def internal_diagnostics_runs(_: ActorContext = Depends(get_actor_context)) -> dict[str, Any]:
    return list_no_ground_truth_runs()


@app.get("/internal/diagnostics/api/latest", dependencies=[Depends(require_api_key)])
def internal_diagnostics_latest(_: ActorContext = Depends(get_actor_context)) -> dict[str, Any]:
    bundle = load_no_ground_truth_run_bundle()
    summary = build_no_ground_truth_health_summary(bundle)
    return {
        "available": bool(bundle.get("available")),
        "run_id": bundle.get("run_id"),
        "artifact_root": bundle.get("artifact_root"),
        "summary": summary,
        "sections_available": {
            key: bool(((bundle.get("sections") or {}).get(key) or {}).get("available"))
            for key in SECTION_FILES.keys()
        },
        "message": bundle.get("message"),
    }


@app.get("/internal/diagnostics/api/run/{run_id}", dependencies=[Depends(require_api_key)])
def internal_diagnostics_run(
    run_id: str,
    _: ActorContext = Depends(get_actor_context),
) -> dict[str, Any]:
    bundle = load_no_ground_truth_run_bundle(run_id=run_id)
    summary = build_no_ground_truth_health_summary(bundle)
    return {
        "available": bool(bundle.get("available")),
        "run_id": bundle.get("run_id"),
        "artifact_root": bundle.get("artifact_root"),
        "summary": summary,
        "sections_available": {
            key: bool(((bundle.get("sections") or {}).get(key) or {}).get("available"))
            for key in SECTION_FILES.keys()
        },
        "message": bundle.get("message"),
    }


@app.get("/internal/diagnostics/api/compare", dependencies=[Depends(require_api_key)])
def internal_diagnostics_compare(
    run_id: str | None = Query(default=None, description="Current run to compare."),
    baseline_run_id: str | None = Query(default=None, description="Baseline run. Defaults to prior run."),
    _: ActorContext = Depends(get_actor_context),
) -> dict[str, Any]:
    return compare_no_ground_truth_runs(
        current_run_id=run_id,
        baseline_run_id=baseline_run_id,
    )


@app.get("/internal/diagnostics/api/public-outcomes", dependencies=[Depends(require_api_key)])
def internal_diagnostics_public_outcomes(
    validation_run_id: str | None = Query(default=None, description="Optional validation run id."),
    validation_baseline_run_id: str | None = Query(default=None, description="Optional validation baseline run id."),
    calibration_run_id: str | None = Query(default=None, description="Optional calibration run id."),
    calibration_baseline_run_id: str | None = Query(default=None, description="Optional calibration baseline run id."),
    _: ActorContext = Depends(get_actor_context),
) -> dict[str, Any]:
    return load_public_outcome_governance_snapshot(
        validation_run_id=validation_run_id,
        validation_baseline_run_id=validation_baseline_run_id,
        calibration_run_id=calibration_run_id,
        calibration_baseline_run_id=calibration_baseline_run_id,
    )


@app.get("/internal/diagnostics/api/latest/{section_key}", dependencies=[Depends(require_api_key)])
def internal_diagnostics_latest_section(
    section_key: str,
    _: ActorContext = Depends(get_actor_context),
) -> dict[str, Any]:
    normalized = str(section_key or "").strip().lower()
    if normalized not in SECTION_FILES:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown diagnostics section.",
                "valid_sections": sorted(SECTION_FILES.keys()),
            },
        )
    bundle = load_no_ground_truth_run_bundle()
    section = ((bundle.get("sections") or {}).get(normalized) or {}) if isinstance(bundle.get("sections"), dict) else {}
    return {
        "available": bool(section.get("available")),
        "run_id": bundle.get("run_id"),
        "section": normalized,
        "payload": section.get("payload") if bool(section.get("available")) else None,
        "message": section.get("message") or bundle.get("message"),
    }


@app.get("/internal/diagnostics/api/run/{run_id}/{section_key}", dependencies=[Depends(require_api_key)])
def internal_diagnostics_run_section(
    run_id: str,
    section_key: str,
    _: ActorContext = Depends(get_actor_context),
) -> dict[str, Any]:
    normalized = str(section_key or "").strip().lower()
    if normalized not in SECTION_FILES:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown diagnostics section.",
                "valid_sections": sorted(SECTION_FILES.keys()),
            },
        )
    bundle = load_no_ground_truth_run_bundle(run_id=run_id)
    section = ((bundle.get("sections") or {}).get(normalized) or {}) if isinstance(bundle.get("sections"), dict) else {}
    return {
        "available": bool(section.get("available")),
        "run_id": bundle.get("run_id"),
        "section": normalized,
        "payload": section.get("payload") if bool(section.get("available")) else None,
        "message": section.get("message") or bundle.get("message"),
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


@app.get(
    "/report/{assessment_id}",
    response_model=AssessmentResult | AssessmentWithDiagnosticsResponse,
    dependencies=[Depends(require_api_key)],
)
def get_report(
    assessment_id: str,
    audience: Audience | None = Query(default=None),
    audience_mode: Audience | None = Query(default=None),
    include_diagnostics: bool = Query(default=False),
    ctx: ActorContext = Depends(get_actor_context),
) -> AssessmentResult | AssessmentWithDiagnosticsResponse:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    _enforce_org_scope(ctx, result.organization_id)
    viewed = _apply_audience_view(result, actor=ctx, audience=audience, audience_mode=audience_mode)
    if not include_diagnostics:
        return viewed
    payload = _payload_from_assessment(viewed)
    ruleset = _get_ruleset_or_default(viewed.ruleset_id)
    diagnostics = _build_assessment_trust_metadata(
        result=viewed,
        payload=payload,
        organization_id=viewed.organization_id,
        ruleset=ruleset,
        geocode_resolution=_geocode_resolution_from_assessment(viewed),
        coverage_resolution=_coverage_resolution_from_assessment(viewed),
    )
    return AssessmentWithDiagnosticsResponse(assessment=viewed, diagnostics=diagnostics)


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
