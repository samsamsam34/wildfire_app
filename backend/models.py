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
    coordinates: Coordinates
    risk_scores: RiskScores
    factor_breakdown: FactorBreakdown
    mitigation_recommendations: List[MitigationAction]
    assumptions: AssumptionsBlock
    confidence: ConfidenceBlock
    top_risk_drivers: List[str]
    top_protective_factors: List[str]
    explanation_summary: str
    model_version: str

    # Backward compatibility fields
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    wildfire_risk_score: Optional[float] = None
    insurance_readiness_score: Optional[float] = None
    risk_drivers: Optional[RiskDrivers] = None
    environmental_factors: Optional[EnvironmentalFactors] = None
    assumptions_used: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    mitigation_plan: List[MitigationAction] = Field(default_factory=list)
    explanation: Optional[str] = None
