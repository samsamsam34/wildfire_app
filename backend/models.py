from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

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


class PropertyAttributes(BaseModel):
    roof_type: Optional[str] = None
    vent_type: Optional[str] = None
    siding_type: Optional[str] = None
    window_type: Optional[str] = None
    defensible_space_ft: Optional[float] = None
    vegetation_condition: Optional[str] = None
    driveway_access_notes: Optional[str] = None
    construction_year: Optional[int] = None
    inspection_notes: Optional[str] = None


class AddressRequest(BaseModel):
    address: str = Field(..., min_length=5)
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"
    tags: List[str] = Field(default_factory=list)
    organization_id: Optional[str] = None
    ruleset_id: str = "default"


class ReassessmentRequest(BaseModel):
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
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


class RiskScores(BaseModel):
    site_hazard_score: Optional[float] = None
    home_ignition_vulnerability_score: Optional[float] = None
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    site_hazard_score_available: bool = False
    home_ignition_vulnerability_score_available: bool = False
    wildfire_risk_score_available: bool = False
    insurance_readiness_score_available: bool = False


class RiskDrivers(BaseModel):
    environmental: float
    structural: float
    access_exposure: float


class FactorBreakdown(BaseModel):
    # Step 2 grouped breakdown aligned to factorized submodels.
    submodels: Dict[str, float] = Field(default_factory=dict)
    environmental: Dict[str, float] = Field(default_factory=dict)
    structural: Dict[str, float] = Field(default_factory=dict)

    # Legacy compatibility fields (deprecated): retained for older clients.
    environmental_risk: float = 0.0
    structural_risk: float = 0.0
    access_risk: float = 0.0
    access_risk_provisional: bool = True
    access_included_in_total: bool = False
    access_risk_note: str = (
        "Access exposure is provisional and not included in total score "
        "until real parcel/egress inputs are integrated."
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
    ruleset_version: str = "1.0"
    ruleset_description: str = "Default underwriting-oriented readiness adjustments"

    review_status: ReviewStatus = "pending"
    workflow_state: WorkflowState = "new"
    assigned_reviewer: Optional[str] = None
    assigned_role: Optional[UserRole] = None

    property_facts: Dict[str, object] = Field(default_factory=dict)
    confirmed_fields: List[str] = Field(default_factory=list)

    latitude: float
    longitude: float
    wildfire_risk_score: Optional[float] = None
    legacy_weighted_wildfire_risk_score: Optional[float] = None
    site_hazard_score: Optional[float] = None
    home_ignition_vulnerability_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    wildfire_risk_score_available: bool = False
    site_hazard_score_available: bool = False
    home_ignition_vulnerability_score_available: bool = False
    insurance_readiness_score_available: bool = False
    risk_drivers: RiskDrivers
    factor_breakdown: FactorBreakdown
    submodel_scores: Dict[str, SubmodelScore] = Field(default_factory=dict)
    weighted_contributions: Dict[str, WeightedContribution] = Field(default_factory=dict)
    submodel_explanations: Dict[str, str] = Field(default_factory=dict)
    property_findings: List[str] = Field(default_factory=list)
    top_risk_drivers: List[str]
    top_protective_factors: List[str]
    explanation_summary: str
    confirmed_inputs: Dict[str, object] = Field(default_factory=dict)
    observed_inputs: Dict[str, object] = Field(default_factory=dict)
    inferred_inputs: Dict[str, object] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    assumptions_used: List[str] = Field(default_factory=list)
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
    site_hazard_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    home_vulnerability_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    insurance_readiness_eligibility: ScoreEligibility = Field(default_factory=ScoreEligibility)
    assessment_status: AssessmentStatus = "insufficient_data"
    assessment_blockers: List[str] = Field(default_factory=list)
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
    summary: str = ""


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


class ReportExport(BaseModel):
    assessment_id: str
    generated_at: str
    model_version: str
    organization_id: str = "default_org"
    audience_mode: Audience = "homeowner"
    audience_highlights: List[str] = Field(default_factory=list)
    audience_focus: Dict[str, object] = Field(default_factory=dict)
    ruleset: Dict[str, object] = Field(default_factory=dict)
    property_summary: Dict[str, object]
    location_summary: Dict[str, object]
    wildfire_risk_summary: Dict[str, object]
    insurance_readiness_summary: Dict[str, object]
    top_risk_drivers: List[str]
    top_protective_factors: List[str]
    assumptions_confidence: Dict[str, object]
    mitigation_recommendations: List[MitigationAction]
    simulation: Optional[Dict[str, object]] = None


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
    ruleset_version: str = "1.0"
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
