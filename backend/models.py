from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.version import DEFAULT_RULESET_VERSION

Audience = Literal["homeowner", "agent", "inspector", "insurer"]
AnnotationRole = Literal["homeowner", "agent", "broker", "inspector", "insurer"]
AnnotationVisibility = Literal["internal", "shared"]
ReviewStatus = Literal["pending", "reviewed", "flagged", "approved"]
OrganizationType = Literal["insurer", "broker", "agent", "inspector", "admin", "demo"]
UserRole = Literal["admin", "underwriter", "broker", "inspector", "agent", "viewer"]
JobStatus = Literal["queued", "running", "completed", "failed", "partial"]
WorkflowState = Literal[
    "new",
    "triaged",
    "needs_inspection",
    "mitigation_pending",
    "ready_for_review",
    "approved",
    "declined",
    "escalated",
]
ConfidenceTier = Literal["high", "moderate", "low", "preliminary"]
AssessmentSpecificityTier = Literal["property_specific", "address_level", "regional_estimate"]
SpecificitySummaryTier = Literal[
    "property_specific",
    "address_level",
    "regional_estimate",
    "insufficient_data",
]
AssessmentOutputState = Literal[
    "property_specific_assessment",
    "address_level_estimate",
    "limited_regional_estimate",
    "insufficient_data",
]
AssessmentMode = Literal[
    "property_specific",
    "address_level",
    "limited_regional_estimate",
    "insufficient_data",
]
UseRestriction = Literal[
    "shareable",
    "homeowner_review_recommended",
    "agent_or_inspector_review_recommended",
    "not_for_underwriting_or_binding",
]
EligibilityStatus = Literal["full", "partial", "insufficient"]
AssessmentStatus = Literal["fully_scored", "partially_scored", "insufficient_data"]
SourceType = Literal[
    "observed",
    "footprint_derived",
    "user_provided",
    "public_record_inferred",
    "heuristic",
    "missing",
]
FreshnessStatus = Literal["current", "aging", "stale", "unknown"]
ProviderStatus = Literal["ok", "missing", "error"]
EvidenceStatus = Literal["observed", "inferred", "missing", "fallback"]
LedgerDirection = Literal[
    "increases_risk",
    "reduces_risk",
    "blocks_readiness",
    "improves_readiness",
    "composes_score",
    "data_quality",
]
EvidenceUseRestriction = Literal["consumer_estimate", "screening_only", "review_required"]
LayerCoverageStatus = Literal[
    "observed",
    "missing_file",
    "not_configured",
    "outside_extent",
    "sampling_failed",
    "fallback_used",
    "partial",
]
LayerSourceType = Literal["prepared_region", "runtime_env", "derived", "open_data"]
GeocodeStatus = Literal[
    "accepted",
    "matched",
    "no_match",
    "ambiguous_match",
    "low_confidence",
    "trust_filter_rejected",
    "missing_coordinates",
    "provider_error",
    "parser_error",
]
GeocodeOutcome = Literal["geocode_succeeded_trusted", "geocode_succeeded_untrusted", "geocode_failed"]
TrustedMatchStatus = Literal["trusted", "untrusted_fallback", "rejected"]
StructureGeometrySource = Literal["auto_detected", "user_selected", "user_modified"]
SelectionMode = Literal["polygon", "point"]


class PropertyAttributes(BaseModel):
    roof_type: Optional[str] = None
    vent_type: Optional[str] = None
    siding_type: Optional[str] = None
    window_type: Optional[str] = None
    defensible_space_ft: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5280.0,  # 1 mile; beyond this the input is almost certainly wrong
        description="Defensible space clearance in feet from the structure.",
    )
    vegetation_condition: Optional[str] = None
    driveway_access_notes: Optional[str] = None
    construction_year: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2100,
        description="Year the structure was built. Must be a plausible calendar year.",
    )
    inspection_notes: Optional[str] = None


class AddressRequest(BaseModel):
    address: str = Field(..., min_length=5)
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    structure_geometry_source: StructureGeometrySource = "auto_detected"
    selection_mode: SelectionMode = "polygon"
    property_anchor_point: Optional[Coordinates] = None
    user_selected_point: Optional[Coordinates] = None
    selected_parcel_id: Optional[str] = None
    selected_parcel_geometry: Optional[Dict[str, Any]] = None
    selected_structure_id: Optional[str] = None
    selected_structure_geometry: Optional[Dict[str, Any]] = None
    audience: Audience = "homeowner"
    tags: List[str] = Field(default_factory=list)
    organization_id: Optional[str] = None
    ruleset_id: str = "default"
    include_calibrated_outputs: bool = False


class ReassessmentRequest(BaseModel):
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"
    ruleset_id: Optional[str] = None


class HomeownerImprovementRunRequest(BaseModel):
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    defensible_space_condition: Optional[
        Literal["poor", "limited", "moderate", "good", "excellent"]
    ] = None
    structure_geometry_source: Optional[StructureGeometrySource] = None
    selection_mode: Optional[SelectionMode] = None
    property_anchor_point: Optional[Coordinates] = None
    user_selected_point: Optional[Coordinates] = None
    selected_parcel_id: Optional[str] = None
    selected_parcel_geometry: Optional[Dict[str, Any]] = None
    selected_structure_id: Optional[str] = None
    selected_structure_geometry: Optional[Dict[str, Any]] = None
    confirm_selected_parcel: bool = False
    confirm_selected_footprint: bool = False
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"
    ruleset_id: Optional[str] = None


class SimulationRequest(BaseModel):
    assessment_id: Optional[str] = None
    address: Optional[str] = None
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"
    scenario_name: str = "what_if"
    scenario_overrides: PropertyAttributes = Field(default_factory=PropertyAttributes)
    scenario_confirmed_fields: List[str] = Field(default_factory=list)


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class RegionBoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class RegionPrepareRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region_id: str = Field(..., min_length=2)
    display_name: Optional[str] = None
    bbox: RegionBoundingBox
    source_config_path: Optional[str] = None
    run_validation: bool = Field(default=True, alias="validate")
    overwrite: bool = False
    allow_partial_coverage_fill: bool = False
    skip_optional_layers: bool = False
    prefer_bbox_downloads: bool = True
    allow_full_download_fallback: bool = True
    require_core_layers: bool = True
    target_resolution: Optional[float] = None


class RegionPrepJobStatus(BaseModel):
    job_id: str
    region_id: str
    display_name: str
    requested_bbox: RegionBoundingBox
    requested_address: Optional[str] = None
    point_lat: Optional[float] = None
    point_lon: Optional[float] = None
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    manifest_path: Optional[str] = None
    dedupe_key: str
    reused_existing_job: bool = False
    result: Optional[Dict[str, Any]] = None


class RegionCoverageRequest(BaseModel):
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class GeocodeDebugRequest(BaseModel):
    address: str = Field(..., min_length=1)


class AddressCandidateSearchRequest(BaseModel):
    address: str = Field(..., min_length=3)
    zip_code: Optional[str] = None
    locality: Optional[str] = None
    state: Optional[str] = None
    limit: int = Field(default=8, ge=1, le=25)


class ManualAddressCandidate(BaseModel):
    candidate_id: str
    formatted_address: str
    locality: Optional[str] = None
    postal_code: Optional[str] = None
    state: Optional[str] = None
    source: str
    source_type: Optional[str] = None
    confidence: str
    match_method: Optional[str] = None
    latitude: float
    longitude: float
    coverage_available: bool = False
    resolved_region_id: Optional[str] = None
    resolved_region_display_name: Optional[str] = None
    region_reason: Optional[str] = None
    diagnostics: List[str] = Field(default_factory=list)


class AddressCandidateSearchResponse(BaseModel):
    status: str
    input_address: str
    normalized_address: str
    zip_code: Optional[str] = None
    inferred_localities: List[str] = Field(default_factory=list)
    selected_locality: Optional[str] = None
    candidates: List[ManualAddressCandidate] = Field(default_factory=list)
    map_click_fallback_recommended: bool = False
    diagnostics: List[str] = Field(default_factory=list)
    final_status: Optional[str] = None


class RegionCoverageStatus(BaseModel):
    covered: bool
    region_id: Optional[str] = None
    display_name: Optional[str] = None
    latitude: float
    longitude: float
    message: str
    diagnostics: List[str] = Field(default_factory=list)
    regions_root: str
    coverage_available: bool = False
    resolved_region_id: Optional[str] = None
    selected_region_id: Optional[str] = None
    selected_region_display_name: Optional[str] = None
    reason: str = ""
    recommended_action: Optional[str] = None
    geocode_status: Optional[GeocodeStatus] = None
    geocode_outcome: Optional[GeocodeOutcome] = None
    geocode_decision: Optional[str] = None
    geocode_trust_tier: Optional[str] = None
    trusted_match_status: Optional[TrustedMatchStatus] = None
    trusted_match_failure_reason: Optional[str] = None
    trusted_match_subchecks: Optional[Dict[str, Any]] = None
    fallback_eligibility: Optional[bool] = None
    normalized_address: Optional[str] = None
    geocode_source: Optional[str] = None
    geocode_precision: Optional[str] = None
    geocode_location_type: Optional[str] = None
    resolution_status: Optional[str] = None
    resolution_method: Optional[str] = None
    fallback_used: Optional[bool] = None
    final_location_confidence: Optional[str] = None
    provider_attempts: Optional[List[Dict[str, Any]]] = None
    provider_statuses: Optional[Dict[str, str]] = None
    candidate_sources_attempted: Optional[List[str]] = None
    candidates_found: Optional[int] = None
    coordinate_source: Optional[str] = None
    final_coordinate_source: Optional[str] = None
    final_coordinates_used: Optional[Dict[str, float]] = None
    match_confidence: Optional[str] = None
    match_method: Optional[str] = None
    unsupported_location_reason: Optional[str] = None
    local_fallback_attempted: Optional[bool] = None
    authoritative_fallback_result: Optional[Dict[str, Any]] = None
    local_fallback_result: Optional[Dict[str, Any]] = None
    region_distance_to_boundary_m: Optional[float] = None
    nearest_region_id: Optional[str] = None
    edge_tolerance_m: Optional[float] = None
    candidate_regions_containing_point: Optional[List[str]] = None
    address_exists: Optional[bool] = None
    address_confidence: Optional[str] = None
    address_validation_sources: Optional[List[str]] = None
    coordinate_confidence: Optional[str] = None
    error_class: Optional[str] = None
    needs_user_confirmation: Optional[bool] = None
    final_status: Optional[str] = None
    resolver_candidates: Optional[List[Dict[str, Any]]] = None
    candidate_disagreement_distances: Optional[List[Dict[str, Any]]] = None
    candidate_needs_confirmation: Optional[Dict[str, Any]] = None
    final_candidate_selected: Optional[Dict[str, Any]] = None
    resolver_settings: Optional[Dict[str, Any]] = None
    acceptance_threshold: Optional[float] = None
    medium_confidence_threshold: Optional[str] = None
    top_margin_threshold: Optional[float] = None
    top_candidate_score: Optional[float] = None
    second_candidate_score: Optional[float] = None
    final_acceptance_decision: Optional[bool] = None
    failure_reason: Optional[str] = None


class GeocodingDetails(BaseModel):
    geocode_status: GeocodeStatus = "accepted"
    geocode_outcome: Optional[GeocodeOutcome] = None
    geocode_decision: Optional[str] = None
    geocode_trust_tier: Optional[str] = None
    submitted_address: str = ""
    normalized_address: Optional[str] = None
    geocode_source: Optional[str] = None
    geocode_provider: Optional[str] = None
    provider: Optional[str] = None
    geocoded_address: Optional[str] = None
    matched_address: Optional[str] = None
    geocoded_point: Optional[Dict[str, float]] = None
    geocode_location_type: Optional[str] = None
    geocode_precision: Optional[str] = None
    resolution_status: Optional[str] = None
    resolution_method: Optional[str] = None
    fallback_used: Optional[bool] = None
    final_location_confidence: Optional[str] = None
    trusted_match_status: Optional[TrustedMatchStatus] = None
    trusted_match_failure_reason: Optional[str] = None
    trusted_match_subchecks: Optional[Dict[str, Any]] = None
    fallback_eligibility: Optional[bool] = None
    address_component_comparison: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    candidate_count: Optional[int] = None
    resolved_latitude: Optional[float] = None
    resolved_longitude: Optional[float] = None
    rejection_reason: Optional[str] = None
    provider_attempts: Optional[List[Dict[str, Any]]] = None
    provider_statuses: Optional[Dict[str, str]] = None
    candidate_sources_attempted: Optional[List[str]] = None
    candidates_found: Optional[int] = None
    coordinate_source: Optional[str] = None
    final_coordinate_source: Optional[str] = None
    final_coordinates_used: Optional[Dict[str, float]] = None
    match_confidence: Optional[str] = None
    match_method: Optional[str] = None
    local_fallback_attempted: Optional[bool] = None
    authoritative_fallback_result: Optional[Dict[str, Any]] = None
    local_fallback_result: Optional[Dict[str, Any]] = None
    candidate_regions_containing_point: Optional[List[str]] = None
    unsupported_location_reason: Optional[str] = None
    address_exists: Optional[bool] = None
    address_confidence: Optional[str] = None
    address_validation_sources: Optional[List[str]] = None
    coordinate_confidence: Optional[str] = None
    error_class: Optional[str] = None
    needs_user_confirmation: Optional[bool] = None
    final_status: Optional[str] = None
    resolver_candidates: Optional[List[Dict[str, Any]]] = None
    candidate_disagreement_distances: Optional[List[Dict[str, Any]]] = None
    candidate_needs_confirmation: Optional[Dict[str, Any]] = None
    final_candidate_selected: Optional[Dict[str, Any]] = None
    resolver_settings: Optional[Dict[str, Any]] = None
    acceptance_threshold: Optional[float] = None
    medium_confidence_threshold: Optional[str] = None
    top_margin_threshold: Optional[float] = None
    top_candidate_score: Optional[float] = None
    second_candidate_score: Optional[float] = None
    final_acceptance_decision: Optional[bool] = None
    failure_reason: Optional[str] = None


class RegionResolution(BaseModel):
    coverage_available: bool = False
    resolved_region_id: Optional[str] = None
    resolved_region_display_name: Optional[str] = None
    reason: str = "unknown"
    recommended_action: Optional[str] = None
    diagnostics: List[str] = Field(default_factory=list)


class RiskScores(BaseModel):
    site_hazard_score: Optional[float] = None
    home_ignition_vulnerability_score: Optional[float] = None
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    overall_wildfire_risk: Optional[float] = None
    home_hardening_readiness: Optional[float] = None
    site_hazard_score_available: bool = False
    home_ignition_vulnerability_score_available: bool = False
    wildfire_risk_score_available: bool = False
    insurance_readiness_score_available: bool = False
    home_hardening_readiness_score_available: bool = False


class RiskDrivers(BaseModel):
    environmental: float
    structural: float
    access_exposure: float


class FactorBreakdown(BaseModel):
    # Step 2 grouped breakdown aligned to factorized submodels.
    submodels: Dict[str, float] = Field(default_factory=dict)
    environmental: Dict[str, float] = Field(default_factory=dict)
    structural: Dict[str, float] = Field(default_factory=dict)
    component_scores: Dict[str, float] = Field(default_factory=dict)
    component_weight_fractions: Dict[str, float] = Field(default_factory=dict)

    # Legacy compatibility fields (deprecated): retained for older clients.
    environmental_risk: float = 0.0
    structural_risk: float = 0.0
    access_risk: float = 0.0
    access_risk_provisional: bool = True
    access_included_in_total: bool = False
    access_risk_note: str = (
        "Access exposure is advisory and excluded from weighted wildfire scoring. "
        "When available, it is derived from open road-network context."
    )


class EnvironmentalFactors(BaseModel):
    burn_probability: Optional[float] = None
    wildfire_hazard: Optional[float] = None
    slope: Optional[float] = None
    fuel_model: Optional[float] = None
    canopy_cover: Optional[float] = None
    historic_fire_distance: Optional[float] = None
    wildland_distance: Optional[float] = None

    # Legacy compatibility mirrors
    hazard_severity: Optional[float] = None
    slope_topography: Optional[float] = None
    aspect_exposure: Optional[float] = None
    vegetation_fuel: Optional[float] = None
    drought_moisture: Optional[float] = None
    canopy_density: Optional[float] = None
    fuel_proximity: Optional[float] = None
    historical_fire_recurrence: Optional[float] = None


class AssumptionsBlock(BaseModel):
    confirmed_inputs: Dict[str, object] = Field(default_factory=dict)
    observed_inputs: Dict[str, object] = Field(default_factory=dict)
    inferred_inputs: Dict[str, object] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    assumptions_used: List[str] = Field(default_factory=list)


class ConfidenceBlock(BaseModel):
    confidence_score: float
    data_completeness_score: float
    environmental_data_completeness_score: float = 0.0
    confidence_tier: ConfidenceTier
    use_restriction: UseRestriction
    assumption_count: int
    low_confidence_flags: List[str]
    requires_user_verification: bool
    environmental_data_present: bool = False
    property_context_present: bool = False
    confirmed_fields_count: int = 0
    inferred_fields_count: int = 0
    missing_critical_fields: List[str] = Field(default_factory=list)
    confidence_drivers: List[str] = Field(default_factory=list)
    confidence_limiters: List[str] = Field(default_factory=list)


class InputSourceMetadata(BaseModel):
    field_name: str = ""
    source_type: SourceType
    source_name: str
    provider_status: ProviderStatus = "ok"
    freshness_status: FreshnessStatus = "unknown"
    used_in_scoring: bool = True
    confidence_weight: float = 0.0
    observed_at: Optional[str] = None
    loaded_at: Optional[str] = None
    dataset_version: Optional[str] = None
    spatial_resolution_m: Optional[float] = None
    source_class: Optional[str] = None
    spatial_resolution: Optional[str] = None  # legacy compatibility
    details: Optional[str] = None


class DataProvenanceSummary(BaseModel):
    direct_data_coverage_score: float = 0.0
    inferred_data_coverage_score: float = 0.0
    missing_data_share: float = 0.0
    stale_data_share: float = 0.0
    heuristic_input_count: int = 0
    current_input_count: int = 0


class DataProvenanceBlock(BaseModel):
    inputs: List[InputSourceMetadata] = Field(default_factory=list)
    summary: DataProvenanceSummary = Field(default_factory=DataProvenanceSummary)
    environmental_inputs_used: Dict[str, InputSourceMetadata] = Field(default_factory=dict)
    property_inputs_used: Dict[str, InputSourceMetadata] = Field(default_factory=dict)
    inferred_inputs_used: List[str] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list)
    heuristic_inputs_used: List[str] = Field(default_factory=list)


class ScoreFamilyInputQuality(BaseModel):
    direct_coverage: float = 0.0
    inferred_coverage: float = 0.0
    stale_share: float = 0.0
    missing_share: float = 0.0
    heuristic_count: int = 0


class ConfidencePenalty(BaseModel):
    penalty_key: str
    reason: str
    amount: float


class ScoreEvidenceFactor(BaseModel):
    factor_key: str
    display_name: str
    category: str
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None
    weight: float = 0.0
    contribution: float = 0.0
    direction: LedgerDirection = "increases_risk"
    evidence_status: EvidenceStatus = "missing"
    source_layer: Optional[str] = None
    source_field: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ScoreEvidenceLedger(BaseModel):
    site_hazard_score: List[ScoreEvidenceFactor] = Field(default_factory=list)
    home_ignition_vulnerability_score: List[ScoreEvidenceFactor] = Field(default_factory=list)
    insurance_readiness_score: List[ScoreEvidenceFactor] = Field(default_factory=list)
    wildfire_risk_score: List[ScoreEvidenceFactor] = Field(default_factory=list)


class EvidenceQualitySummary(BaseModel):
    observed_factor_count: int = 0
    inferred_factor_count: int = 0
    missing_factor_count: int = 0
    fallback_factor_count: int = 0
    evidence_quality_score: float = 0.0
    confidence_penalties: List[ConfidencePenalty] = Field(default_factory=list)
    use_restriction: EvidenceUseRestriction = "screening_only"


class ModelGovernanceInfo(BaseModel):
    product_version: str = ""
    api_version: str = ""
    scoring_model_version: str = ""
    ruleset_version: str = ""
    rules_logic_version: str = ""
    factor_schema_version: str = ""
    benchmark_pack_version: Optional[str] = None
    calibration_version: str = ""
    region_data_version: Optional[str] = None
    data_bundle_version: Optional[str] = None


class LayerCoverageAuditItem(BaseModel):
    layer_key: str
    display_name: str
    required_for: List[str] = Field(default_factory=list)
    configured: bool = False
    present_in_region: bool = False
    sample_attempted: bool = False
    sample_succeeded: bool = False
    coverage_status: LayerCoverageStatus = "not_configured"
    source_type: LayerSourceType = "runtime_env"
    source_path: Optional[str] = None
    raw_value_preview: Optional[object] = None
    failure_reason: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class DataCoverageSummary(BaseModel):
    """Describes which data layers came from local vs. national sources.

    Added in Phase 5. Included in every AssessmentResult so clients can
    communicate data provenance to end users without parsing data_sources strings.
    """
    overall_coverage: Literal["full", "partial", "limited"] = "limited"
    local_data_available: bool = False
    layers_from_national_sources: List[str] = Field(default_factory=list)
    layers_unavailable: List[str] = Field(default_factory=list)
    coverage_note: str = ""


class LayerCoverageSummary(BaseModel):
    total_layers_checked: int = 0
    observed_count: int = 0
    partial_count: int = 0
    fallback_count: int = 0
    failed_count: int = 0
    not_configured_count: int = 0
    critical_missing_layers: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class ScoreEligibility(BaseModel):
    eligible: bool = False
    eligibility_status: EligibilityStatus = "insufficient"
    blocking_reasons: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)


class AssessmentDiagnostics(BaseModel):
    critical_inputs_present: List[str] = Field(default_factory=list)
    critical_inputs_missing: List[str] = Field(default_factory=list)
    stale_inputs: List[str] = Field(default_factory=list)
    inferred_inputs: List[str] = Field(default_factory=list)
    heuristic_inputs: List[str] = Field(default_factory=list)
    confidence_downgrade_reasons: List[str] = Field(default_factory=list)
    trust_tier_blockers: List[str] = Field(default_factory=list)
    fallback_decisions: List[Dict[str, object]] = Field(default_factory=list)


class ScoreSectionSummary(BaseModel):
    label: str = ""
    score: Optional[float] = None
    summary: str = ""
    explanation: str = ""
    top_drivers: List[str] = Field(default_factory=list)
    key_drivers: List[str] = Field(default_factory=list)
    protective_factors: List[str] = Field(default_factory=list)
    top_next_actions: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)


class ScoreSummaries(BaseModel):
    site_hazard: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)
    home_ignition_vulnerability: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)
    insurance_readiness: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)


class SubmodelScore(BaseModel):
    score: float
    weighted_contribution: float
    explanation: str
    key_inputs: Dict[str, object]
    assumptions: List[str] = Field(default_factory=list)

    # Legacy compatibility field
    key_contributing_inputs: Dict[str, object] = Field(default_factory=dict)


class WeightedContribution(BaseModel):
    weight: float
    score: float
    contribution: float
    base_weight: Optional[float] = None
    effective_weight: Optional[float] = None
    observed_fraction: Optional[float] = None
    availability_multiplier: Optional[float] = None
    basis: Optional[Literal["observed", "inferred", "fallback", "missing"]] = None
    factor_evidence_status: Optional[Literal["observed", "inferred", "fallback", "suppressed"]] = None
    support_level: Optional[Literal["high", "medium", "low"]] = None
    component: Optional[Literal["regional_context", "property_surroundings", "structure_specific", "unknown"]] = None
    omitted_due_to_missing: bool = False


class ReadinessFactor(BaseModel):
    name: str
    status: Literal["pass", "watch", "fail"]
    score_impact: float
    detail: str


class MitigationAction(BaseModel):
    title: str = ""
    reason: str = ""
    impacted_submodels: List[str] = Field(default_factory=list)
    impacted_readiness_factors: List[str] = Field(default_factory=list)
    estimated_risk_reduction_band: Literal["low", "medium", "high"] = "low"
    estimated_readiness_improvement_band: Literal["low", "medium", "high"] = "low"
    priority: int = 5
    insurer_relevance: Literal["required", "recommended", "nice_to_have"] = "recommended"

    # Legacy compatibility fields
    action: Optional[str] = None
    related_factor: Optional[str] = None
    impact_statement: Optional[str] = None
    estimated_risk_reduction: Optional[float] = None
    effort: Optional[Literal["low", "medium", "high"]] = None


class NearStructureAction(BaseModel):
    title: str = ""
    explanation: str = ""
    target_zone: str = ""
    why_it_matters: str = ""
    impact_category: Literal["low", "medium", "high"] = "low"
    priority: int = 5
    evidence_status: Literal["observed", "inferred", "missing", "unknown"] = "unknown"


class HomeownerRiskDriver(BaseModel):
    factor: str = ""
    impact: Literal["high", "medium", "low"] = "low"
    explanation: str = ""
    relative_contribution_pct: Optional[float] = None


class HomeownerPrioritizedAction(BaseModel):
    action: str = ""
    explanation: str = ""
    why_this_matters: str = ""
    what_it_reduces: str = ""
    expected_effect: Literal["small", "moderate", "significant"] = "moderate"
    data_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    impact_level: Literal["high", "medium", "low"] = "low"
    effort_level: Literal["low", "medium", "high"] = "medium"
    estimated_cost_band: Literal["low", "medium", "high"] = "medium"
    timeline: Literal["now", "this_season", "later"] = "later"
    priority: int = 5


class HomeownerConfidenceSummary(BaseModel):
    confidence: ConfidenceTier = "preliminary"
    observed_data: List[str] = Field(default_factory=list)
    estimated_data: List[str] = Field(default_factory=list)
    missing_data: List[str] = Field(default_factory=list)
    fallback_assumptions: List[str] = Field(default_factory=list)
    accuracy_improvements: List[str] = Field(default_factory=list)


class HomeownerFollowUpInput(BaseModel):
    input_key: str
    assessment_field: str
    label: str
    prompt: str
    input_type: Literal["select", "number", "map_point", "map_polygon"] = "select"
    options: List[str] = Field(default_factory=list)
    unit: Optional[str] = None


class HomeownerImprovementOptions(BaseModel):
    assessment_id: str
    missing_key_inputs: List[str] = Field(default_factory=list)
    prioritized_missing_key_inputs: List[str] = Field(default_factory=list)
    highest_value_next_question: Optional[HomeownerFollowUpInput] = None
    remaining_optional_input_count: int = 0
    geometry_issue_flags: List[str] = Field(default_factory=list)
    missing_property_fields: List[str] = Field(default_factory=list)
    structure_attribute_gaps: List[str] = Field(default_factory=list)
    geometry_uncertainty: Dict[str, Any] = Field(default_factory=dict)
    improve_your_result_suggestions: List[str] = Field(default_factory=list)
    optional_follow_up_inputs: List[HomeownerFollowUpInput] = Field(default_factory=list)


class HomeownerImprovementRunResponse(BaseModel):
    baseline_assessment_id: str
    updated_assessment_id: str
    before_summary: Dict[str, Any] = Field(default_factory=dict)
    after_summary: Dict[str, Any] = Field(default_factory=dict)
    what_changed: Dict[str, Any] = Field(default_factory=dict)
    what_changed_summary: Dict[str, Any] = Field(default_factory=dict)
    geometry_update_summary: Dict[str, Any] = Field(default_factory=dict)
    why_it_matters: List[str] = Field(default_factory=list)
    confidence_improved: bool = False
    recommendations_adjusted: bool = False
    improve_your_result_before: HomeownerImprovementOptions
    improve_your_result_after: HomeownerImprovementOptions
    change_notes: List[str] = Field(default_factory=list)
    homeowner_before_after_summary: Dict[str, Any] = Field(default_factory=dict)


class TrustDiagnosticsConfidence(BaseModel):
    tier: ConfidenceTier = "preliminary"
    score: float = 0.0
    evidence_completeness: float = 0.0
    fallback_heavy: bool = False
    fallback_weight_fraction: float = 0.0
    observed_feature_count: int = 0
    inferred_feature_count: int = 0
    fallback_feature_count: int = 0
    missing_feature_count: int = 0
    missing_critical_fields: List[str] = Field(default_factory=list)
    missing_critical_field_count: int = 0
    inferred_fields: List[str] = Field(default_factory=list)
    inferred_field_count: int = 0
    confidence_reduction_reasons: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TrustDiagnosticsStability(BaseModel):
    rating: Literal["stable", "moderate", "unstable"] = "moderate"
    local_sensitivity_score: float = 0.0
    geocode_jitter_swing: float = 0.0
    fallback_assumption_swing: float = 0.0
    assumption_sensitive: bool = False
    tier_flip_risk: Literal["low", "medium", "high"] = "medium"
    notes: List[str] = Field(default_factory=list)


class TrustDiagnosticsInterventionImpact(BaseModel):
    name: str
    estimated_risk_delta: float = 0.0
    estimated_readiness_delta: float = 0.0
    directionally_expected: bool = True
    notes: List[str] = Field(default_factory=list)


class TrustDiagnosticsMitigationSensitivity(BaseModel):
    top_interventions: List[TrustDiagnosticsInterventionImpact] = Field(default_factory=list)
    backwards_or_zero_impact_flags: List[str] = Field(default_factory=list)


class TrustDiagnosticsMonotonicity(BaseModel):
    checks_run: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    status: Literal["pass", "warn", "fail"] = "warn"


class TrustDiagnosticsBenchmarkAlignment(BaseModel):
    available: bool = False
    signals_used: List[str] = Field(default_factory=list)
    local_alignment: Literal["high", "moderate", "low", "unknown"] = "unknown"
    notes: List[str] = Field(default_factory=list)


class TrustDiagnosticsDistributionSegment(BaseModel):
    region: Optional[str] = None
    settlement_pattern: Optional[str] = None
    evidence_tier: Optional[str] = None


class TrustDiagnosticsDistributionContext(BaseModel):
    relative_risk_percentile: Optional[float] = None
    segment: TrustDiagnosticsDistributionSegment = Field(default_factory=TrustDiagnosticsDistributionSegment)
    notes: List[str] = Field(default_factory=list)


class TrustDiagnosticsVegetationSignal(BaseModel):
    major_driver: bool = False
    driver_strength: Literal["high", "moderate", "low", "unknown"] = "unknown"
    contribution_share: Optional[float] = None
    related_submodels: List[str] = Field(default_factory=list)
    related_risk_drivers: List[str] = Field(default_factory=list)
    near_structure_summary: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class TrustDiagnostics(BaseModel):
    version: str = "ngt_eval_v1"
    generated_at: datetime
    evaluation_basis: Literal["no_ground_truth"] = "no_ground_truth"
    caveat: str = (
        "These diagnostics measure model coherence, stability, evidence quality, and external alignment. "
        "They do not establish real-world predictive accuracy or insurer approval."
    )
    confidence: TrustDiagnosticsConfidence = Field(default_factory=TrustDiagnosticsConfidence)
    stability: TrustDiagnosticsStability = Field(default_factory=TrustDiagnosticsStability)
    mitigation_sensitivity: TrustDiagnosticsMitigationSensitivity = Field(
        default_factory=TrustDiagnosticsMitigationSensitivity
    )
    monotonicity: TrustDiagnosticsMonotonicity = Field(default_factory=TrustDiagnosticsMonotonicity)
    benchmark_alignment: TrustDiagnosticsBenchmarkAlignment = Field(
        default_factory=TrustDiagnosticsBenchmarkAlignment
    )
    distribution_context: TrustDiagnosticsDistributionContext = Field(
        default_factory=TrustDiagnosticsDistributionContext
    )
    differentiation_mode: Literal[
        "highly_local",
        "property_specific",
        "mixed",
        "mostly_regional",
    ] = "mostly_regional"
    property_specific_feature_count: int = 0
    proxy_feature_count: int = 0
    defaulted_feature_count: int = 0
    regional_feature_count: int = 0
    local_differentiation_score: float = 0.0
    neighborhood_differentiation_confidence: float = 0.0
    differentiation_notes: List[str] = Field(default_factory=list)
    vegetation_signal: TrustDiagnosticsVegetationSignal = Field(
        default_factory=TrustDiagnosticsVegetationSignal
    )
    explanations: List[str] = Field(default_factory=list)


class CalibratedPublicOutcomeMetadata(BaseModel):
    requested: bool = False
    available: bool = False
    availability_status: str = "not_requested"
    calibration_version: Optional[str] = None
    label_definition: str = (
        "structure_loss_or_major_damage (major_damage or destroyed = 1; minor_damage or no_damage = 0)"
    )
    calibrated_public_outcome_probability: Optional[float] = None
    calibration_basis_summary: str = (
        "Public observed wildfire structure-damage outcomes were used to calibrate this optional probability layer."
    )
    calibration_caveat: str = (
        "This calibrated value is based on public observed wildfire damage outcomes and should not be interpreted "
        "as carrier underwriting probability or claims likelihood."
    )
    calibration_data_coverage_tier: str = "unknown"
    calibration_data_coverage_note: Optional[str] = None
    raw_score_reference: Dict[str, Any] = Field(default_factory=dict)
    fallback_state: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class SpecificitySummary(BaseModel):
    specificity_tier: SpecificitySummaryTier = "regional_estimate"
    headline: str = ""
    what_this_means: str = ""
    comparison_allowed: bool = False


class PropertyConfidenceSummary(BaseModel):
    score: float = 0.0
    level: Literal[
        "verified_property_specific",
        "strong_property_specific",
        "address_level",
        "regional_estimate_with_anchor",
        "insufficient_property_identification",
        # Backward-compatible legacy levels
        "high",
        "medium",
        "low",
    ] = "insufficient_property_identification"
    key_reasons: List[str] = Field(default_factory=list)
    user_action_recommended: str = ""
    # Backward-compatible alias retained for older payload consumers.
    key_gaps: List[str] = Field(default_factory=list)


class GeometryResolutionSummary(BaseModel):
    anchor_source: str = "geocoded_address_point"
    anchor_quality_score: float = 0.0
    parcel_match_status: str = "not_found"
    footprint_match_status: str = "none"
    footprint_source: Optional[str] = None
    ring_generation_mode: str = "point_annulus_fallback"
    naip_structure_feature_status: str = "missing"
    near_structure_data_quality_tier: str = "point_proxy"
    near_structure_claim_strength: str = "coarse_directional"
    supports_property_specific_claims: bool = False
    property_mismatch_flag: bool = False
    mismatch_reason: Optional[str] = None
    geometry_limitations: List[str] = Field(default_factory=list)


class FootprintResolutionSummary(BaseModel):
    selected_source: Optional[str] = None
    confidence_score: float = 0.0
    candidates_considered: int = 0
    fallback_used: bool = True
    match_status: str = "none"
    match_method: Optional[str] = None
    match_distance_m: Optional[float] = None


class ParcelResolutionSummary(BaseModel):
    status: str = "not_found"
    confidence: float = 0.0
    source: Optional[str] = None
    geometry_used: str = "none"
    overlap_score: float = 0.0
    candidates_considered: int = 0
    lookup_method: Optional[str] = None
    lookup_distance_m: Optional[float] = None
    bounding_geometry: Optional[Dict[str, Any]] = None


class PropertyLinkageSummary(BaseModel):
    anchor_status: str = "unresolved"
    anchor_confidence: float = 0.0
    anchor_source: Optional[str] = None
    selected_structure_id: Optional[str] = None
    parcel_source: Optional[str] = None
    footprint_source: Optional[str] = None
    parcel_candidate_count: int = 0
    footprint_candidate_count: int = 0
    mismatch_flags: List[str] = Field(default_factory=list)

    # Legacy compatibility fields
    geocode_confidence: float = 0.0
    parcel_confidence: float = 0.0
    footprint_confidence: float = 0.0
    overall_property_confidence: float = 0.0
    parcel_status: str = "not_found"
    footprint_status: str = "none"
    footprint_match_method: Optional[str] = None
    multiple_footprints_on_parcel: bool = False
    footprint_outside_parcel: bool = False
    structure_candidate_count: int = 0
    selection_notes: List[str] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    assessment_id: str
    address: str
    audience: Audience = "homeowner"
    report_audience: Optional[Audience] = None
    audience_highlights: List[str] = Field(default_factory=list)

    organization_id: str = "default_org"
    portfolio_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    ruleset_id: str = "default"
    ruleset_name: str = "Default Carrier Profile"
    ruleset_version: str = DEFAULT_RULESET_VERSION
    ruleset_description: str = "Default underwriting-oriented readiness adjustments"

    review_status: ReviewStatus = "pending"
    workflow_state: WorkflowState = "new"
    assigned_reviewer: Optional[str] = None
    assigned_role: Optional[UserRole] = None

    property_facts: Dict[str, object] = Field(default_factory=dict)
    confirmed_fields: List[str] = Field(default_factory=list)

    latitude: float
    longitude: float
    geocoding: GeocodingDetails = Field(default_factory=GeocodingDetails)
    wildfire_risk_score: Optional[float] = None
    overall_wildfire_risk: Optional[float] = None
    legacy_weighted_wildfire_risk_score: Optional[float] = None
    site_hazard_score: Optional[float] = None
    home_ignition_vulnerability_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    home_hardening_readiness: Optional[float] = None
    calibrated_damage_likelihood: Optional[float] = None
    empirical_damage_likelihood_proxy: Optional[float] = None
    empirical_loss_likelihood_proxy: Optional[float] = None
    calibration_applied: bool = False
    calibration_method: Optional[str] = None
    calibration_status: str = "disabled"
    calibration_limitations: List[str] = Field(default_factory=list)
    calibration_scope_warning: Optional[str] = None
    calibrated_public_outcome_metadata: Optional[CalibratedPublicOutcomeMetadata] = None
    wildfire_risk_score_available: bool = False
    site_hazard_score_available: bool = False
    home_ignition_vulnerability_score_available: bool = False
    insurance_readiness_score_available: bool = False
    home_hardening_readiness_score_available: bool = False
    risk_drivers: RiskDrivers
    factor_breakdown: FactorBreakdown
    submodel_scores: Dict[str, SubmodelScore] = Field(default_factory=dict)
    weighted_contributions: Dict[str, WeightedContribution] = Field(default_factory=dict)
    submodel_explanations: Dict[str, str] = Field(default_factory=dict)
    property_findings: List[str] = Field(default_factory=list)
    defensible_space_analysis: Dict[str, object] = Field(default_factory=dict)
    top_near_structure_risk_drivers: List[str] = Field(default_factory=list)
    prioritized_vegetation_actions: List[NearStructureAction] = Field(default_factory=list)
    defensible_space_limitations_summary: List[str] = Field(default_factory=list)
    near_structure_features: Dict[str, Any] = Field(default_factory=dict)
    parcel_based_metrics: Dict[str, Any] = Field(default_factory=dict)
    directional_risk: Dict[str, Any] = Field(default_factory=dict)
    structure_relative_slope: Dict[str, Any] = Field(default_factory=dict)
    structure_attributes: Dict[str, Any] = Field(default_factory=dict)
    top_risk_drivers: List[str]
    top_risk_drivers_detailed: List[HomeownerRiskDriver] = Field(default_factory=list)
    prioritized_mitigation_actions: List[HomeownerPrioritizedAction] = Field(default_factory=list)
    confidence_summary: HomeownerConfidenceSummary = Field(default_factory=HomeownerConfidenceSummary)
    top_recommended_actions: List[str] = Field(default_factory=list)
    top_protective_factors: List[str]
    explanation_summary: str
    confirmed_inputs: Dict[str, object] = Field(default_factory=dict)
    observed_inputs: Dict[str, object] = Field(default_factory=dict)
    inferred_inputs: Dict[str, object] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    assumptions_used: List[str] = Field(default_factory=list)
    assumptions_and_unknowns: List[str] = Field(default_factory=list)
    confidence_score: float
    data_completeness_score: float = 0.0
    environmental_data_completeness_score: float = 0.0
    confidence_tier: ConfidenceTier = "preliminary"
    use_restriction: UseRestriction = "not_for_underwriting_or_binding"
    low_confidence_flags: List[str]
    data_sources: List[str]
    environmental_layer_status: Dict[str, str] = Field(default_factory=dict)
    input_source_metadata: Dict[str, InputSourceMetadata] = Field(default_factory=dict)
    direct_data_coverage_score: float = 0.0
    inferred_data_coverage_score: float = 0.0
    missing_data_share: float = 0.0
    data_provenance: DataProvenanceBlock = Field(default_factory=DataProvenanceBlock)
    site_hazard_input_quality: ScoreFamilyInputQuality = Field(default_factory=ScoreFamilyInputQuality)
    home_vulnerability_input_quality: ScoreFamilyInputQuality = Field(default_factory=ScoreFamilyInputQuality)
    insurance_readiness_input_quality: ScoreFamilyInputQuality = Field(default_factory=ScoreFamilyInputQuality)
    score_evidence_ledger: ScoreEvidenceLedger = Field(default_factory=ScoreEvidenceLedger)
    evidence_quality_summary: EvidenceQualitySummary = Field(default_factory=EvidenceQualitySummary)
    feature_coverage_summary: Dict[str, bool] = Field(default_factory=dict)
    feature_coverage_percent: float = 0.0
    assessment_specificity_tier: AssessmentSpecificityTier = "regional_estimate"
    specificity_summary: SpecificitySummary = Field(default_factory=SpecificitySummary)
    geometry_resolution: GeometryResolutionSummary = Field(default_factory=GeometryResolutionSummary)
    footprint_resolution: FootprintResolutionSummary = Field(default_factory=FootprintResolutionSummary)
    parcel_resolution: ParcelResolutionSummary = Field(default_factory=ParcelResolutionSummary)
    property_linkage: PropertyLinkageSummary = Field(default_factory=PropertyLinkageSummary)
    assessment_output_state: AssessmentOutputState = "insufficient_data"
    assessment_mode: AssessmentMode = "insufficient_data"
    scoring_status: str = "insufficient_data_to_score"
    computed_components: List[str] = Field(default_factory=list)
    blocked_components: List[str] = Field(default_factory=list)
    minimum_missing_requirements: List[str] = Field(default_factory=list)
    recommended_data_improvements: List[str] = Field(default_factory=list)
    limited_assessment_flag: bool = False
    confidence_not_meaningful: bool = False
    observed_factor_count: int = 0
    missing_factor_count: int = 0
    fallback_factor_count: int = 0
    observed_feature_count: int = 0
    inferred_feature_count: int = 0
    fallback_feature_count: int = 0
    missing_feature_count: int = 0
    observed_weight_fraction: float = 0.0
    fallback_dominance_ratio: float = 0.0
    fallback_weight_fraction: float = 0.0
    structure_data_completeness: float = 0.0
    structure_assumption_mode: Literal["observed", "mixed", "default_assumed", "unknown"] = "unknown"
    structure_score_confidence: float = 0.0
    geometry_quality_score: float = 0.0
    regional_context_coverage_score: float = 0.0
    property_specificity_score: float = 0.0
    property_data_confidence: float = 0.0
    property_confidence_summary: PropertyConfidenceSummary = Field(default_factory=PropertyConfidenceSummary)
    score_specificity_warning: Optional[str] = None
    data_quality_summary: Dict[str, str] = Field(default_factory=dict)
    assessment_limitations: List[Dict[str, str]] = Field(default_factory=list)
    what_was_observed: List[str] = Field(default_factory=list)
    what_was_estimated: List[str] = Field(default_factory=list)
    what_was_missing: List[str] = Field(default_factory=list)
    why_this_result_is_limited: Optional[str] = None
    developer_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    homeowner_summary: Dict[str, Any] = Field(default_factory=dict)
    insurability_status: str = ""
    insurability_status_reasons: List[str] = Field(default_factory=list)
    insurability_status_methodology_note: str = ""
    layer_coverage_audit: List[LayerCoverageAuditItem] = Field(default_factory=list)
    coverage_summary: LayerCoverageSummary = Field(default_factory=LayerCoverageSummary)
    data_coverage_summary: DataCoverageSummary = Field(default_factory=DataCoverageSummary)
    region_resolution: RegionResolution = Field(default_factory=RegionResolution)
    # Convenience mirrors for routing observability without parsing nested objects.
    coverage_available: bool = False
    resolved_region_id: Optional[str] = None
    property_anchor_point: Optional[Dict[str, float]] = None
    property_anchor_source: Optional[str] = None
    property_anchor_precision: Optional[str] = None
    assessed_property_display_point: Optional[Dict[str, float]] = None
    parcel_id: Optional[str] = None
    parcel_source: Optional[str] = None
    parcel_lookup_method: Optional[str] = None
    parcel_lookup_distance_m: Optional[float] = None
    source_conflict_flag: bool = False
    alignment_notes: List[str] = Field(default_factory=list)
    display_point_source: Optional[str] = None
    structure_match_status: Optional[str] = None
    structure_match_method: Optional[str] = None
    matched_structure_id: Optional[str] = None
    structure_match_confidence: Optional[float] = None
    building_source: Optional[str] = None
    building_source_version: Optional[str] = None
    building_source_confidence: Optional[float] = None
    structure_match_distance_m: Optional[float] = None
    candidate_structure_count: Optional[int] = None
    final_structure_geometry_source: Optional[str] = None
    structure_geometry_confidence: Optional[float] = None
    geometry_source: Optional[str] = None
    geometry_confidence: Optional[float] = None
    ring_generation_mode: Optional[str] = None
    property_mismatch_flag: bool = False
    mismatch_reason: Optional[str] = None
    snapped_structure_distance_m: Optional[float] = None
    selection_mode: Optional[SelectionMode] = None
    matched_structure_centroid: Optional[Dict[str, float]] = None
    matched_structure_footprint: Optional[Dict[str, Any]] = None
    user_selected_point: Optional[Dict[str, float]] = None
    site_hazard_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    home_vulnerability_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    insurance_readiness_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    assessment_status: AssessmentStatus = "insufficient_data"
    assessment_blockers: List[str] = Field(default_factory=list)
    assessment_limitations_summary: List[str] = Field(default_factory=list)
    assessment_diagnostics: AssessmentDiagnostics = Field(default_factory=AssessmentDiagnostics)
    property_level_context: Dict[str, object] = Field(default_factory=dict)
    mitigation_plan: List[MitigationAction]
    readiness_factors: List[ReadinessFactor] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    readiness_penalties: Dict[str, float] = Field(default_factory=dict)
    readiness_summary: str = ""
    score_summaries: ScoreSummaries = Field(default_factory=ScoreSummaries)
    site_hazard_section: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)
    home_ignition_vulnerability_section: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)
    insurance_readiness_section: ScoreSectionSummary = Field(default_factory=ScoreSectionSummary)
    model_version: str
    product_version: str = ""
    api_version: str = ""
    scoring_model_version: str = ""
    rules_logic_version: str = ""
    factor_schema_version: str = "1.0.0"
    benchmark_pack_version: Optional[str] = None
    calibration_version: str = ""
    region_data_version: Optional[str] = None
    data_bundle_version: Optional[str] = None
    model_governance: ModelGovernanceInfo = Field(default_factory=ModelGovernanceInfo)
    generated_at: datetime
    scoring_notes: List[str] = Field(default_factory=list)

    # Structured mirrors
    coordinates: Optional[Coordinates] = None
    risk_scores: Optional[RiskScores] = None
    assumptions: Optional[AssumptionsBlock] = None
    confidence: Optional[ConfidenceBlock] = None
    mitigation_recommendations: List[MitigationAction] = Field(default_factory=list)
    environmental_factors: Optional[EnvironmentalFactors] = None
    explanation: Optional[str] = None


class SimulationDelta(BaseModel):
    wildfire_risk_score_delta: Optional[float] = None
    insurance_readiness_score_delta: Optional[float] = None
    home_hardening_readiness_delta: Optional[float] = None


class SimulationResult(BaseModel):
    scenario_name: str
    baseline: AssessmentResult
    simulated: AssessmentResult
    delta: SimulationDelta
    changed_inputs: Dict[str, Dict[str, object]] = Field(default_factory=dict)
    next_best_actions: List[MitigationAction] = Field(default_factory=list)

    # Homeowner-friendly summary mirrors for easier UI rendering.
    base_assessment_id: Optional[str] = None
    simulated_assessment_id: Optional[str] = None
    base_scores: Optional[RiskScores] = None
    simulated_scores: Optional[RiskScores] = None
    score_delta: Optional[SimulationDelta] = None
    base_confidence: Optional[ConfidenceBlock] = None
    simulated_confidence: Optional[ConfidenceBlock] = None
    base_assumptions: Optional[AssumptionsBlock] = None
    simulated_assumptions: Optional[AssumptionsBlock] = None
    simulator_explanations: Dict[str, object] = Field(default_factory=dict)
    summary: str = ""
    homeowner_before_after_summary: Dict[str, object] = Field(default_factory=dict)


class AssessmentListItem(BaseModel):
    assessment_id: str
    created_at: str
    address: str
    organization_id: str = "default_org"
    audience: Audience = "homeowner"
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    model_version: str
    confidence_score: float = 0.0
    readiness_blockers: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending"
    workflow_state: WorkflowState = "new"
    assigned_reviewer: Optional[str] = None
    assigned_role: Optional[UserRole] = None
    ruleset_id: str = "default"


class SimulationScenarioItem(BaseModel):
    scenario_id: str
    assessment_id: str
    scenario_name: str
    created_at: str
    wildfire_risk_score_delta: float
    insurance_readiness_score_delta: float
    homeowner_before_after_summary: Dict[str, object] = Field(default_factory=dict)


class ReportExport(BaseModel):
    assessment_id: str
    generated_at: str
    model_version: str
    organization_id: str = "default_org"
    audience_mode: Audience = "homeowner"
    audience_highlights: List[str] = Field(default_factory=list)
    audience_focus: Dict[str, object] = Field(default_factory=dict)
    governance_metadata: Dict[str, object] = Field(default_factory=dict)
    model_governance: ModelGovernanceInfo = Field(default_factory=ModelGovernanceInfo)
    ruleset: Dict[str, object] = Field(default_factory=dict)
    property_summary: Dict[str, object]
    location_summary: Dict[str, object]
    wildfire_risk_summary: Dict[str, object]
    home_hardening_readiness_summary: Dict[str, object] = Field(default_factory=dict)
    insurance_readiness_summary: Dict[str, object]
    defensible_space_analysis: Dict[str, object] = Field(default_factory=dict)
    top_near_structure_risk_drivers: List[str] = Field(default_factory=list)
    prioritized_vegetation_actions: List[NearStructureAction] = Field(default_factory=list)
    defensible_space_limitations_summary: List[str] = Field(default_factory=list)
    top_risk_drivers: List[str]
    top_risk_drivers_detailed: List[HomeownerRiskDriver] = Field(default_factory=list)
    prioritized_mitigation_actions: List[HomeownerPrioritizedAction] = Field(default_factory=list)
    confidence_summary: HomeownerConfidenceSummary = Field(default_factory=HomeownerConfidenceSummary)
    top_protective_factors: List[str]
    assumptions_confidence: Dict[str, object]
    score_evidence_ledger: ScoreEvidenceLedger = Field(default_factory=ScoreEvidenceLedger)
    evidence_quality_summary: EvidenceQualitySummary = Field(default_factory=EvidenceQualitySummary)
    layer_coverage_audit: List[LayerCoverageAuditItem] = Field(default_factory=list)
    coverage_summary: LayerCoverageSummary = Field(default_factory=LayerCoverageSummary)
    mitigation_recommendations: List[MitigationAction]
    simulation: Optional[Dict[str, object]] = None


class AssessmentWithDiagnosticsResponse(BaseModel):
    assessment: AssessmentResult
    diagnostics: TrustDiagnostics


class HomeownerReportAction(BaseModel):
    title: str
    priority: int = 5
    target_zone: Optional[str] = None
    why_it_matters: str = ""
    why_this_matters: str = ""
    what_it_reduces: str = ""
    expected_effect: Literal["small", "moderate", "significant"] = "moderate"
    expected_impact_category: Literal["low", "medium", "high"] = "medium"
    evidence_status: Literal["observed", "inferred", "missing", "unknown"] = "unknown"
    explanation: str = ""


class HomeownerReport(BaseModel):
    assessment_id: str
    report_format_version: str = "1.1.0"
    generated_at: str
    insurability_status: str = ""
    insurability_status_reasons: List[str] = Field(default_factory=list)
    insurability_status_methodology_note: str = ""
    homeowner_focus_summary: Dict[str, object] = Field(default_factory=dict)
    internal_calibration_debug: Dict[str, object] = Field(default_factory=dict)
    advanced_details: Dict[str, object] = Field(default_factory=dict)
    first_screen: Dict[str, object] = Field(default_factory=dict)
    headline_risk_summary: str = ""
    top_risk_drivers: List[str] = Field(default_factory=list)
    prioritized_actions: List[Dict[str, object]] = Field(default_factory=list)
    ranked_actions: List[Dict[str, object]] = Field(default_factory=list)
    most_impactful_actions: List[Dict[str, object]] = Field(default_factory=list)
    what_to_do_first: Dict[str, object] = Field(default_factory=dict)
    limitations_notice: str = ""
    report_header: Dict[str, object] = Field(default_factory=dict)
    property_summary: Dict[str, object] = Field(default_factory=dict)
    score_summary: Dict[str, object] = Field(default_factory=dict)
    key_risk_drivers: List[str] = Field(default_factory=list)
    top_risk_drivers_detailed: List[HomeownerRiskDriver] = Field(default_factory=list)
    defensible_space_summary: Dict[str, object] = Field(default_factory=dict)
    top_recommended_actions: List[HomeownerReportAction] = Field(default_factory=list)
    prioritized_mitigation_actions: List[HomeownerPrioritizedAction] = Field(default_factory=list)
    mitigation_plan: List[HomeownerReportAction] = Field(default_factory=list)
    home_hardening_readiness_summary: Dict[str, object] = Field(default_factory=dict)
    insurance_readiness_summary: Dict[str, object] = Field(default_factory=dict)
    confidence_summary: HomeownerConfidenceSummary = Field(default_factory=HomeownerConfidenceSummary)
    confidence_and_limitations: Dict[str, object] = Field(default_factory=dict)
    metadata: Dict[str, object] = Field(default_factory=dict)
    professional_debug_metadata: Optional[Dict[str, object]] = None
    specificity_summary: SpecificitySummary = Field(default_factory=SpecificitySummary)


class AssessmentMapLayer(BaseModel):
    layer_key: str
    display_name: str
    available: bool = False
    default_visible: bool = False
    description: str = ""
    legend_label: str = ""
    geometry_type: str = "unknown"
    feature_count: int = 0
    reason_unavailable: Optional[str] = None


class AssessmentMapPayload(BaseModel):
    assessment_id: str
    center: Dict[str, float] = Field(default_factory=dict)
    resolved_region_id: Optional[str] = None
    coverage_available: bool = False
    basis_geometry_type: str = "point_proxy"
    geocode_provider: Optional[str] = None
    geocode_source_name: Optional[str] = None
    geocoded_address: Optional[str] = None
    geocode_location_type: Optional[str] = None
    geocode_precision: Optional[str] = None
    property_anchor_point: Optional[Dict[str, object]] = None
    property_anchor_source: Optional[str] = None
    property_anchor_precision: Optional[str] = None
    assessed_property_display_point: Optional[Dict[str, object]] = None
    parcel_address_point: Optional[Dict[str, object]] = None
    parcel_polygon: Optional[Dict[str, object]] = None
    parcel_id: Optional[str] = None
    parcel_lookup_method: Optional[str] = None
    parcel_lookup_distance_m: Optional[float] = None
    parcel_source_name: Optional[str] = None
    parcel_source_vintage: Optional[str] = None
    footprint_source_name: Optional[str] = None
    footprint_source_vintage: Optional[str] = None
    source_conflict_flag: bool = False
    alignment_notes: List[str] = Field(default_factory=list)
    structure_match_status: Optional[str] = None
    structure_match_method: Optional[str] = None
    matched_structure_id: Optional[str] = None
    structure_match_confidence: Optional[float] = None
    structure_match_distance_m: Optional[float] = None
    candidate_structure_count: Optional[int] = None
    final_structure_geometry_source: Optional[str] = None
    structure_geometry_confidence: Optional[float] = None
    snapped_structure_distance_m: Optional[float] = None
    selection_mode: Optional[SelectionMode] = None
    user_selected_point: Optional[Dict[str, object]] = None
    display_point_source: str = "property_anchor_point"
    geocoded_address_point: Optional[Dict[str, object]] = None
    matched_structure_centroid: Optional[Dict[str, object]] = None
    matched_structure_footprint: Optional[Dict[str, object]] = None
    layers: List[AssessmentMapLayer] = Field(default_factory=list)
    data: Dict[str, Dict[str, object]] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, object] = Field(default_factory=dict)


class BatchAssessmentItem(BaseModel):
    row_id: Optional[str] = None
    address: str
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"
    tags: List[str] = Field(default_factory=list)


class BatchAssessmentRequest(BaseModel):
    portfolio_name: Optional[str] = None
    organization_id: Optional[str] = None
    ruleset_id: str = "default"
    items: List[BatchAssessmentItem] = Field(default_factory=list)


class BatchAssessmentResultItem(BaseModel):
    row_id: Optional[str] = None
    address: str
    status: Literal["success", "failed"]
    error: Optional[str] = None
    assessment_id: Optional[str] = None
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    top_risk_drivers: List[str] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None


class BatchAssessmentResponse(BaseModel):
    portfolio_name: Optional[str] = None
    organization_id: Optional[str] = None
    ruleset_id: str = "default"
    job_id: Optional[str] = None

    # Step 4 contract fields.
    total_properties: int
    completed_count: int
    failed_count: int
    high_risk_count: int
    blocker_count: int
    average_wildfire_risk: float
    average_insurance_readiness: float

    # Backward-compatibility mirrors.
    total: int
    succeeded: int
    failed: int

    results: List[BatchAssessmentResultItem] = Field(default_factory=list)


class PortfolioSummary(BaseModel):
    total_count: int
    high_risk_count: int
    blocker_count: int
    avg_wildfire_risk: float
    avg_insurance_readiness: float


class PortfolioResponse(BaseModel):
    limit: int
    offset: int
    total: int
    items: List[AssessmentListItem] = Field(default_factory=list)
    summary: PortfolioSummary


class AssessmentSummaryResponse(BaseModel):
    summary: PortfolioSummary


class AssessmentAnnotationCreate(BaseModel):
    author_role: AnnotationRole
    note: str = Field(..., min_length=1, max_length=2000)
    tags: List[str] = Field(default_factory=list)
    visibility: AnnotationVisibility = "internal"
    review_status: Optional[ReviewStatus] = None


class AssessmentAnnotation(BaseModel):
    annotation_id: str
    assessment_id: str
    organization_id: str = "default_org"
    created_at: str
    author_role: AnnotationRole
    note: str
    tags: List[str] = Field(default_factory=list)
    visibility: AnnotationVisibility = "internal"
    review_status: ReviewStatus = "pending"


class AssessmentReviewStatusUpdate(BaseModel):
    review_status: ReviewStatus


class AssessmentReviewStatus(BaseModel):
    assessment_id: str
    organization_id: str = "default_org"
    review_status: ReviewStatus
    updated_at: str


class AssessmentAssignmentRequest(BaseModel):
    assigned_reviewer: Optional[str] = None
    assigned_role: Optional[UserRole] = None


class AssessmentWorkflowUpdateRequest(BaseModel):
    workflow_state: WorkflowState


class AssessmentWorkflowInfo(BaseModel):
    assessment_id: str
    organization_id: str = "default_org"
    workflow_state: WorkflowState
    assigned_reviewer: Optional[str] = None
    assigned_role: Optional[UserRole] = None
    updated_at: str


class AssessmentComparisonItem(BaseModel):
    assessment_id: str
    address: str
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    top_risk_drivers: List[str] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    mitigation_titles: List[str] = Field(default_factory=list)


class AssessmentComparisonResult(BaseModel):
    base: AssessmentComparisonItem
    other: AssessmentComparisonItem
    wildfire_risk_delta: Optional[float] = None
    insurance_readiness_delta: Optional[float] = None
    driver_differences: Dict[str, List[str]]
    blocker_differences: Dict[str, List[str]]
    mitigation_differences: Dict[str, List[str]]
    version_comparison: Dict[str, object] = Field(default_factory=dict)


class AssessmentComparisonResponse(BaseModel):
    requested_ids: List[str] = Field(default_factory=list)
    comparisons: List[AssessmentComparisonResult] = Field(default_factory=list)


class OrganizationCreate(BaseModel):
    organization_id: str = Field(..., min_length=2)
    organization_name: str = Field(..., min_length=2)
    organization_type: OrganizationType


class Organization(BaseModel):
    organization_id: str
    organization_name: str
    organization_type: OrganizationType
    created_at: str


class UnderwritingRuleset(BaseModel):
    ruleset_id: str
    ruleset_name: str
    ruleset_version: str
    ruleset_description: str
    config: Dict[str, object] = Field(default_factory=dict)


class UnderwritingRulesetCreate(BaseModel):
    ruleset_id: str = Field(..., min_length=2)
    ruleset_name: str = Field(..., min_length=2)
    ruleset_version: str = DEFAULT_RULESET_VERSION
    ruleset_description: str = ""
    config: Dict[str, object] = Field(default_factory=dict)


class PortfolioJobCreate(BaseModel):
    portfolio_name: Optional[str] = None
    organization_id: Optional[str] = None
    ruleset_id: str = "default"
    process_immediately: bool = True
    items: List[BatchAssessmentItem] = Field(default_factory=list)


class PortfolioJobStatus(BaseModel):
    job_id: str
    organization_id: str
    portfolio_name: Optional[str] = None
    ruleset_id: str = "default"
    created_at: str
    updated_at: str
    status: JobStatus
    total_properties: int = 0
    completed_count: int = 0
    failed_count: int = 0
    high_risk_count: int = 0
    blocker_count: int = 0
    average_wildfire_risk: float = 0.0
    average_insurance_readiness: float = 0.0
    error_summary: Optional[str] = None


class PortfolioJobResultsResponse(BaseModel):
    job: PortfolioJobStatus
    results: List[BatchAssessmentResultItem] = Field(default_factory=list)


class PortfolioJobsSummary(BaseModel):
    total_jobs: int
    queued_count: int
    running_count: int
    completed_count: int
    failed_count: int
    partial_count: int
    failure_rate: float


class CSVImportRequest(BaseModel):
    csv_text: str
    portfolio_name: Optional[str] = None
    organization_id: Optional[str] = None
    ruleset_id: str = "default"
    audience: Audience = "homeowner"
    tags: List[str] = Field(default_factory=list)
    process_immediately: bool = True


class CSVImportError(BaseModel):
    row_number: int
    address: Optional[str] = None
    error: str


class CSVImportResponse(BaseModel):
    row_count: int
    accepted_count: int
    rejected_count: int
    validation_errors: List[CSVImportError] = Field(default_factory=list)
    job: PortfolioJobStatus


class AuditEvent(BaseModel):
    audit_event_id: str
    entity_type: str
    entity_id: str
    organization_id: str
    user_role: UserRole
    action: str
    metadata: Dict[str, object] = Field(default_factory=dict)
    created_at: str


class AdminSummary(BaseModel):
    organization_id: Optional[str] = None
    assessments_created_recently: int
    high_risk_count: int
    blocker_count: int
    pending_review_count: int
    needs_inspection_count: int
    ready_for_review_count: int
    approved_count: int
    declined_count: int
    escalated_count: int
    avg_wildfire_risk: float
    avg_insurance_readiness: float
    jobs_summary: PortfolioJobsSummary
