from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PropertyAttributes(BaseModel):
    roof_type: Optional[str] = None
    vent_type: Optional[str] = None
    defensible_space_ft: Optional[float] = None
    construction_year: Optional[int] = None


class AddressRequest(BaseModel):
    address: str = Field(..., min_length=5)
    attributes: PropertyAttributes = Field(default_factory=PropertyAttributes)


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


class DataQuality(BaseModel):
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
    environmental_factors: EnvironmentalFactors
    data_quality: DataQuality
    assumptions_used: List[str]
    data_sources: List[str]
    mitigation_plan: List[MitigationAction]
    explanation: str
