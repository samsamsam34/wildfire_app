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


class FactorBreakdown(BaseModel):
    environmental_risk: float
    structural_risk: float
    access_risk: float
    access_risk_provisional: bool = True
    access_included_in_total: bool = False
    access_risk_note: str = "Access exposure is provisional and not included in total score until real parcel/egress inputs are integrated."


class RiskDrivers(BaseModel):
    environmental: float
    structural: float
    access_exposure: float


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


class MitigationAction(BaseModel):
    action: str
    related_factor: str
    impact_statement: str
    estimated_risk_reduction: float
    effort: Literal["low", "medium", "high"]
    insurer_relevance: Literal["required", "recommended", "nice_to_have"]
    reason: str


class AssessmentResult(BaseModel):
    assessment_id: str
    address: str
    latitude: float
    longitude: float
    wildfire_risk_score: float
    insurance_readiness_score: float
    risk_drivers: RiskDrivers
    factor_breakdown: FactorBreakdown
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
    model_version: str
    scoring_notes: List[str] = Field(default_factory=list)

    # Compatibility and richer structured mirrors
    coordinates: Optional[Coordinates] = None
    risk_scores: Optional[RiskScores] = None
    assumptions: Optional[AssumptionsBlock] = None
    confidence: Optional[ConfidenceBlock] = None
    mitigation_recommendations: List[MitigationAction] = Field(default_factory=list)
    environmental_factors: Optional[EnvironmentalFactors] = None
    explanation: Optional[str] = None
