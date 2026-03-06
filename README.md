# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI application for property-level wildfire risk and insurance-readiness workflows.

Current release: **Step 3 productization layer** on top of the factorized Step 2 scoring engine.

## What It Does

- Runs address-level wildfire assessments with editable property facts.
- Distinguishes confirmed facts from inferred defaults and missing inputs.
- Returns factorized submodel scores and weighted contributions.
- Returns separate insurance-readiness logic (factors, blockers, penalties, summary).
- Supports what-if simulation scenarios for mitigation planning.
- Persists assessments in SQLite and supports report export/view and assessment history listing.

## Core Endpoints

- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view` (print-friendly HTML)
- `GET /assessments`

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

## Step 2/3 Scoring + Readiness Architecture

### Wildfire Risk (factorized)

The scoring engine uses explicit submodels:
- `vegetation_intensity_risk`
- `fuel_proximity_risk`
- `slope_topography_risk`
- `ember_exposure_risk`
- `flame_contact_risk`
- `historic_fire_risk`
- `structure_vulnerability_risk`
- `defensible_space_risk`

Each returns score, weighted contribution, explanation, key inputs, assumptions.

### Insurance Readiness (separate rules engine)

Readiness is computed independently from wildfire risk and returns:
- `readiness_factors`
- `readiness_blockers`
- `readiness_penalties`
- `readiness_summary`

### Factor Breakdown

`factor_breakdown` includes grouped Step 2 data:
- `submodels`
- `environmental`
- `structural`

Legacy coarse fields remain for compatibility:
- `environmental_risk`
- `structural_risk`
- `access_risk`

## Editable Property Facts and Assumptions

Assessment requests support richer facts:
- `roof_type`, `vent_type`, `siding_type`, `window_type`
- `defensible_space_ft`, `vegetation_condition`
- `driveway_access_notes`, `construction_year`, `inspection_notes`

You can also pass `confirmed_fields`.

Responses include:
- `confirmed_inputs`
- `observed_inputs`
- `inferred_inputs`
- `missing_inputs`
- `assumptions_used`
- `confidence_score`
- `low_confidence_flags`

## What-If Simulation Workflow

Use `POST /risk/simulate` with either:
- an `assessment_id` (recommended), or
- an address + baseline attributes.

Then provide:
- `scenario_name`
- `scenario_overrides`
- `scenario_confirmed_fields`

Simulation returns:
- `baseline`
- `simulated`
- `delta` (risk/readiness)
- `changed_inputs`
- `next_best_actions`

## Reports and Export

- `GET /report/{assessment_id}` returns the full structured assessment payload.
- `GET /report/{assessment_id}/export` returns an export-oriented report contract with:
  - property summary
  - location summary
  - wildfire risk summary
  - insurance readiness summary
  - assumptions/confidence
  - mitigation recommendations
- `GET /report/{assessment_id}/view` returns print-friendly HTML.

## B2B-Friendly Workflow Support

- Reassessment endpoint for updated facts on an existing property.
- Scenario simulation for before/after comparison.
- Assessment history listing (`GET /assessments`) for agents/inspectors/insurers.
- `audience` tagging (`homeowner|agent|inspector|insurer`) on assessments.

## Persistence and Compatibility

SQLite table: `assessments`

- Stores full payload JSON + `model_version`.
- Legacy rows are upgraded in read path with safe defaults.
- Missing Step 3 fields are backfilled (for example: `generated_at`, grouped `factor_breakdown`, readiness defaults).

## Model Versioning

- Current model version: `1.3.0`
- Legacy fallback for old rows: `1.0.0`

## Dependencies

`requirements.txt` includes:
- Runtime: `fastapi`, `uvicorn`, `pydantic`
- Geospatial stack: `numpy`, `rasterio`, `pyproj`, `shapely`
- Testing: `pytest`, `httpx`

## Tests

Deterministic tests are in `tests/test_risk_assessment.py`.
They cover:
- assessment/report shape parity
- confirmed facts flow
- reassessment flow
- what-if simulation deltas
- report export/view endpoints
- assessment listing endpoint
- deterministic outputs
- legacy row compatibility
- provisional access remains non-authoritative

## Current Limitations

- Scoring and readiness logic are transparent MVP heuristics, not carrier-approved underwriting models.
- Access/egress scoring remains provisional and is not weighted into total wildfire risk.
- No user-account/multi-tenant system yet; API key auth is shared-environment only.
- No built-in PDF generator yet (HTML report view is provided for print/export pipeline integration).
