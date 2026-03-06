from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PropertyAttributes(BaseModel):
    roof_type: Optional[str] = None
    vent_type: Optional[str] = None
    defensible_space_ft: Optional[float] = None
    construction_year: Optional[int] = None


class AddressRequest(BaseModel):
    address: str = Field(..., min_length=5)
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)


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
    observed_inputs: Dict[str, object]
    inferred_inputs: Dict[str, object]
    missing_inputs: List[str]
    assumptions_used: List[str]


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
    observed_inputs: Dict[str, object]
    inferred_inputs: Dict[str, object]
    missing_inputs: List[str]
    assumptions_used: List[str]
    confidence_score: float
    low_confidence_flags: List[str]
    data_sources: List[str]
    mitigation_plan: List[MitigationAction]
    readiness_factors: List[ReadinessFactor] = Field(default_factory=list)
    readiness_blockers: List[str] = Field(default_factory=list)
    readiness_penalties: Dict[str, float] = Field(default_factory=dict)
    readiness_summary: str = ""
    model_version: str
    scoring_notes: List[str] = Field(default_factory=list)

    # Structured mirrors
    coordinates: Optional[Coordinates] = None
    risk_scores: Optional[RiskScores] = None
    assumptions: Optional[AssumptionsBlock] = None
    confidence: Optional[ConfidenceBlock] = None
    mitigation_recommendations: List[MitigationAction] = Field(default_factory=list)
    environmental_factors: Optional[EnvironmentalFactors] = None
    explanation: Optional[str] = None
