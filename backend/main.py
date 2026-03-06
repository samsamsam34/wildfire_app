from __future__ import annotations

from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import require_api_key
from backend.database import AssessmentStore
from backend.geocoding import Geocoder
from backend.mitigation import build_mitigation_plan
from backend.models import AddressRequest, AssessmentResult
from backend.risk_engine import RiskEngine
from backend.wildfire_data import WildfireDataClient

app = FastAPI(title="WildfireRisk Advisor API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

risk_engine = RiskEngine()
geocoder = Geocoder()
wildfire_data = WildfireDataClient()
store = AssessmentStore()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/risk/assess", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def assess_risk(payload: AddressRequest) -> AssessmentResult:
    assumptions = []

    try:
        lat, lon, geocode_source = geocoder.geocode(payload.address)
    except Exception:
        lat, lon = risk_engine.geocode_stub(payload.address)
        geocode_source = "Deterministic geocode fallback"
        assumptions.append("Geocoding provider unavailable; fallback coordinates used.")

    context = wildfire_data.collect_context(lat, lon)
    risk = risk_engine.score(payload.attributes, lat, lon, context)
    plan = build_mitigation_plan(payload.attributes, risk.total_score)

    mitigation_credit = sum(a.estimated_risk_reduction for a in plan[:3]) * 0.45
    readiness = max(0.0, min(100.0, round(100 - risk.total_score + mitigation_credit, 1)))

    explanation = (
        f"Wildfire risk is {risk.total_score}. "
        f"Environmental signal blends burn probability, hazard severity, slope/aspect, fuel/canopy, "
        f"wildland proximity, and fire recurrence. "
        f"Top mitigations can improve insurance readiness to about {readiness}."
    )

    result = AssessmentResult(
        assessment_id=str(uuid4()),
        address=payload.address,
        latitude=lat,
        longitude=lon,
        wildfire_risk_score=risk.total_score,
        insurance_readiness_score=readiness,
        risk_drivers=risk.drivers,
        assumptions_used=assumptions + risk.assumptions,
        data_sources=[geocode_source] + context.data_sources,
        mitigation_plan=plan,
        explanation=explanation,
    )
    store.save(result)
    return result


@app.get("/report/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
def get_report(assessment_id: str) -> AssessmentResult:
    result = store.get(assessment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result
