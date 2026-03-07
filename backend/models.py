from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Audience = Literal["homeowner", "agent", "inspector", "insurer"]
AnnotationRole = Literal["homeowner", "agent", "broker", "inspector", "insurer"]
AnnotationVisibility = Literal["internal", "shared"]
ReviewStatus = Literal["pending", "reviewed", "flagged", "approved"]


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


class ReassessmentRequest(BaseModel):
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)
    confirmed_fields: List[str] = Field(default_factory=list)
    audience: Audience = "homeowner"


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
    wildfire_risk_score: float
    insurance_readiness_score: float


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
    burn_probability: float
    hazard_severity: float
    slope_topography: float
    aspect_exposure: float
    vegetation_fuel: float
    drought_moisture: float
    canopy_density: float
    fuel_proximity: float
    historical_fire_recurrence: float


class AssumptionsBlock(BaseModel):
    confirmed_inputs: Dict[str, object] = Field(default_factory=dict)
    observed_inputs: Dict[str, object] = Field(default_factory=dict)
    inferred_inputs: Dict[str, object] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    assumptions_used: List[str] = Field(default_factory=list)


class ConfidenceBlock(BaseModel):
    confidence_score: float
    data_completeness_score: float
    assumption_count: int
    low_confidence_flags: List[str]
    requires_user_verification: bool


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
    portfolio_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending"
    property_facts: Dict[str, object] = Field(default_factory=dict)
    confirmed_fields: List[str] = Field(default_factory=list)

    latitude: float
    longitude: float
    wildfire_risk_score: float
    insurance_readiness_score: float
    risk_drivers: RiskDrivers
    factor_breakdown: FactorBreakdown
    submodel_scores: Dict[str, SubmodelScore] = Field(default_factory=dict)
    weighted_contributions: Dict[str, WeightedContribution] = Field(default_factory=dict)
    submodel_explanations: Dict[str, str] = Field(default_factory=dict)
    top_risk_drivers: List[str]
    top_protective_factors: List[str]
    explanation_summary: str
    confirmed_inputs: Dict[str, object] = Field(default_factory=dict)
    observed_inputs: Dict[str, object] = Field(default_factory=dict)
    inferred_inputs: Dict[str, object] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    assumptions_used: List[str] = Field(default_factory=list)
    confidence_score: float
    low_confidence_flags: List[str]
    data_sources: List[str]
    mitigation_plan: List[MitigationAction]
    readiness_factors: List[ReadinessFactor] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    readiness_penalties: Dict[str, float] = Field(default_factory=dict)
    readiness_summary: str = ""
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
    wildfire_risk_score_delta: float
    insurance_readiness_score_delta: float


class SimulationResult(BaseModel):
    scenario_name: str
    baseline: AssessmentResult
    simulated: AssessmentResult
    delta: SimulationDelta
    changed_inputs: Dict[str, Dict[str, object]] = Field(default_factory=dict)
    next_best_actions: List[MitigationAction] = Field(default_factory=list)


class AssessmentListItem(BaseModel):
    assessment_id: str
    created_at: str
    address: str
    audience: Audience = "homeowner"
    wildfire_risk_score: float
    insurance_readiness_score: float
    model_version: str
    confidence_score: float = 0.0
    readiness_blockers: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending"


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
    audience_mode: Audience = "homeowner"
    audience_highlights: List[str] = Field(default_factory=list)
    audience_focus: Dict[str, object] = Field(default_factory=dict)
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
    review_status: ReviewStatus
    updated_at: str


class AssessmentComparisonItem(BaseModel):
    assessment_id: str
    address: str
    wildfire_risk_score: float
    insurance_readiness_score: float
    top_risk_drivers: List[str] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    mitigation_titles: List[str] = Field(default_factory=list)


class AssessmentComparisonResult(BaseModel):
    base: AssessmentComparisonItem
    other: AssessmentComparisonItem
    wildfire_risk_delta: float
    insurance_readiness_delta: float
    driver_differences: Dict[str, List[str]]
    blocker_differences: Dict[str, List[str]]
    mitigation_differences: Dict[str, List[str]]


class AssessmentComparisonResponse(BaseModel):
    requested_ids: List[str] = Field(default_factory=list)
    comparisons: List[AssessmentComparisonResult] = Field(default_factory=list)
