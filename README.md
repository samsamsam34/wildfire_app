# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI application for property-level wildfire risk and insurance-readiness workflows.

Current release: **Step 4 commercial workflow layer** on top of the Step 2/3 scoring/readiness/simulation foundation.

## What It Does

- Runs address-level wildfire assessments with editable property facts.
- Distinguishes confirmed facts from inferred defaults and missing inputs.
- Returns factorized submodel scores and weighted contributions.
- Returns separate insurance-readiness outputs (factors, blockers, penalties, summary).
- Supports reassessment and deterministic what-if simulation workflows.
- Persists assessments in SQLite with backward-compatible payload upgrades.
- Adds Step 4 B2B workflows: batch/portfolio, filtering, prioritization, audience-specific reports, annotations/review status, and assessment comparisons.

## Core Endpoints

### Health and Core Risk
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`

### Portfolio / Batch
- `POST /portfolio/assess`
- `GET /portfolio`
- `GET /assessments`
- `GET /assessments/summary`

### Reports
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view`

All report endpoints accept:
- `?audience=homeowner|agent|inspector|insurer`
- (compat) `?audience_mode=...`

### Review and Annotation Workflows
- `POST /assessments/{assessment_id}/annotations`
- `GET /assessments/{assessment_id}/annotations`
- `PUT /assessments/{assessment_id}/review-status`
- `GET /assessments/{assessment_id}/review-status`

Compatibility aliases are also available:
- `POST /assessment/{assessment_id}/annotations`
- `GET /assessment/{assessment_id}/annotations`

### Comparison Workflows
- `GET /assessments/{assessment_id}/compare/{other_assessment_id}`
- `GET /assessment/{assessment_id}/compare/{other_assessment_id}` (alias)
- `GET /assessments/compare?ids=id1,id2,...`
- `GET /assessments/{assessment_id}/scenarios`

All non-health endpoints use API key auth (`X-API-Key`) when `WILDFIRE_API_KEYS` is configured.

## Setup

```bash
pip install -r requirements.txt

export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"

# Optional scoring calibration overrides
export WILDFIRE_SUBMODEL_WEIGHTS_JSON='{"ember_exposure_risk":0.15,"flame_contact_risk":0.14}'
export WILDFIRE_READINESS_PENALTIES_JSON='{"roof_fail":26}'
export WILDFIRE_READINESS_BONUSES_JSON='{"roof_pass":4}'

# Optional geospatial layer paths
export WF_LAYER_BURN_PROB_TIF="/path/to/burn_probability.tif"
export WF_LAYER_HAZARD_SEVERITY_TIF="/path/to/hazard_severity.tif"
export WF_LAYER_SLOPE_TIF="/path/to/slope_degrees.tif"
export WF_LAYER_ASPECT_TIF="/path/to/aspect_degrees.tif"
export WF_LAYER_DEM_TIF="/path/to/dem.tif"
export WF_LAYER_FUEL_TIF="/path/to/fuel_model.tif"
export WF_LAYER_CANOPY_TIF="/path/to/canopy_density.tif"
export WF_LAYER_MOISTURE_TIF="/path/to/moisture_or_dryness.tif"
export WF_LAYER_FIRE_PERIMETERS_GEOJSON="/path/to/fire_perimeters.geojson"

uvicorn backend.main:app --reload
```

Frontend file: `frontend/public/index.html`

## Step 2+ Scoring Architecture

### Wildfire Risk (factorized submodels)
- `vegetation_intensity_risk`
- `fuel_proximity_risk`
- `slope_topography_risk`
- `ember_exposure_risk`
- `flame_contact_risk`
- `historic_fire_risk`
- `structure_vulnerability_risk`
- `defensible_space_risk`

Each submodel returns score, weighted contribution, explanation, key inputs, and assumptions.

### Insurance Readiness (separate rules engine)
Readiness is computed independently from wildfire risk and returns:
- `readiness_factors`
- `readiness_blockers`
- `readiness_penalties`
- `readiness_summary`

### Factor Breakdown
`factor_breakdown` includes:
- `submodels`
- `environmental`
- `structural`

Legacy coarse compatibility fields remain:
- `environmental_risk`
- `structural_risk`
- `access_risk`

## Step 4 B2B Workflows

### 1) Batch / Portfolio Assessment
Use `POST /portfolio/assess` with multiple properties.

Per-row output includes:
- `address`
- `assessment_id` (if success)
- `wildfire_risk_score`
- `insurance_readiness_score`
- `top_risk_drivers`
- `readiness_blockers`
- `confidence_score`
- `status`
- `error` (if failed)

Portfolio-level summary fields include:
- `total_properties`
- `completed_count`
- `failed_count`
- `high_risk_count`
- `blocker_count`
- `average_wildfire_risk`
- `average_insurance_readiness`

### 2) Listing / Filtering / Prioritization
`GET /portfolio` and `GET /assessments` support:
- `sort_by`, `sort_dir`
- risk/readiness range filters
- `readiness_blocker` contains filter
- `confidence_min`
- `audience`
- `tag`
- `created_after`, `created_before`, `recent_days`
- `limit`, `offset`

`GET /portfolio` and `GET /assessments/summary` return metrics:
- total count
- high-risk count
- blocker count
- average wildfire risk
- average insurance readiness

### 3) Audience-Specific Report Outputs
Same assessment, different presentation emphasis via query parameter:
- `audience=homeowner|agent|inspector|insurer`

Emphasis model:
- `homeowner`: clear explanation and next steps
- `agent`: disclosure-ready summary and mitigation talking points
- `inspector`: observed vs inferred facts and assumptions/verification focus
- `insurer`: blockers/penalties, confidence, and factorized contributions

### 4) Annotation and Review Workflow
Annotations support:
- `author_role`
- `note`
- `tags`
- `visibility`
- optional `review_status`

Review status values:
- `pending`
- `reviewed`
- `flagged`
- `approved`

### 5) Comparison Workflows
Comparison outputs include:
- wildfire/readiness score deltas
- driver differences
- blocker differences
- mitigation title differences

Supported by:
- pair compare endpoints
- multi-compare endpoint with `ids=...`
- scenario listing endpoint for baseline/simulation audit trails

## Persistence and Compatibility

SQLite tables:
- `assessments`
- `assessment_scenarios`
- `assessment_annotations`
- `assessment_review_status`

Compatibility behavior:
- old rows remain readable
- model version defaults safely for legacy rows
- missing Step 2/3/4 fields are backfilled on read

## Model Versioning
- Current model version: `1.4.0`
- Legacy fallback for old rows: `1.0.0`

## Dependencies
`requirements.txt` includes:
- Runtime: `fastapi`, `uvicorn`, `pydantic`
- Geospatial: `numpy`, `rasterio`, `pyproj`, `shapely`
- Test: `pytest`, `httpx`

Without geospatial packages and configured layer files, the app runs in fallback mode rather than true layer-backed mode.

## Tests
Deterministic tests in `tests/test_risk_assessment.py` cover:
- core assess/report shape parity
- simulation/reassessment
- batch portfolio + partial failure handling
- filtering/sorting/summary workflows
- audience report outputs
- annotation/review workflows
- comparison endpoints
- legacy compatibility

## Current Limitations

- Scoring/readiness logic are transparent MVP heuristics, not carrier-approved underwriting models.
- Access/egress scoring remains provisional and not weighted into total wildfire risk.
- Batch execution is synchronous.
- API-key auth is shared-environment only (no tenant/user identity model yet).
- No native PDF generation; HTML report view and JSON export are provided.

## Ready for Step 5
The Step 4 foundation now includes:
- batch and portfolio operations with prioritization filters/metrics
- audience-specific reporting layers
- inspector/broker/insurer annotation + review status workflows
- structured comparison endpoints for underwriting and broker review
- deterministic, test-covered, backward-compatible persistence and API contracts
