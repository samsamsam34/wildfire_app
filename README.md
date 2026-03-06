# WildfireRisk Advisor (MVP Scaffold)

Now includes:
- Real geocoding integration via OpenStreetMap Nominatim (with fallback)
- Layer-backed wildfire context extraction
- Step 2.5 factorized wildfire scoring architecture
- Separate insurance readiness rules and readiness blockers
- Structured explainability and confidence outputs
- API key authentication
- Persistent SQLite storage for reports

## Setup

```bash
pip install -r requirements.txt
export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"

# Optional scoring calibration overrides (JSON maps)
export WILDFIRE_SUBMODEL_WEIGHTS_JSON='{"ember_exposure_risk":0.15,"flame_contact_risk":0.14}'
export WILDFIRE_READINESS_PENALTIES_JSON='{"roof_fail":26}'
export WILDFIRE_READINESS_BONUSES_JSON='{"roof_pass":4}'

# Layer paths (point these to your real datasets)
export WF_LAYER_BURN_PROB_TIF="/path/to/burn_probability.tif"
export WF_LAYER_HAZARD_SEVERITY_TIF="/path/to/hazard_severity.tif"
export WF_LAYER_SLOPE_TIF="/path/to/slope_degrees.tif"            # optional if DEM set
export WF_LAYER_ASPECT_TIF="/path/to/aspect_degrees.tif"          # optional if DEM set
export WF_LAYER_DEM_TIF="/path/to/dem.tif"                        # used to derive slope/aspect
export WF_LAYER_FUEL_TIF="/path/to/fuel_model.tif"
export WF_LAYER_CANOPY_TIF="/path/to/canopy_density.tif"
export WF_LAYER_MOISTURE_TIF="/path/to/moisture_or_dryness.tif"   # optional; recommended
export WF_LAYER_FIRE_PERIMETERS_GEOJSON="/path/to/fire_perimeters.geojson"

uvicorn backend.main:app --reload
```

## Endpoints

- `GET /health` (public)
- `POST /risk/assess` (requires `X-API-Key` when keys configured)
- `POST /risk/debug` (requires `X-API-Key`; returns intermediate context/submodel/readiness/config payload)
- `GET /report/{assessment_id}` (requires `X-API-Key` when keys configured)

## Step 2.5 Scoring Architecture

Wildfire risk is composed from explicit submodels:
- `vegetation_intensity_risk`
- `fuel_proximity_risk`
- `slope_topography_risk`
- `ember_exposure_risk`
- `flame_contact_risk`
- `historic_fire_risk`
- `structure_vulnerability_risk`
- `defensible_space_risk`

Each submodel returns a score, weighted contribution, deterministic explanation, key inputs, and assumptions.

The API also returns:
- `submodel_scores`
- `weighted_contributions`
- `factor_breakdown`
- `top_risk_drivers`
- `top_protective_factors`

## Wildfire Risk vs Insurance Readiness

`wildfire_risk_score` and `insurance_readiness_score` are separate systems.

- Wildfire risk: weighted submodel composition.
- Insurance readiness: rule checks on roof, vents, defensible space, fuel pressure, and severe environmental hazard signals.

Readiness outputs include:
- `readiness_factors`
- `readiness_blockers`
- `readiness_penalties`

## Calibration Fixtures

Step 2 calibration fixtures are stored in `tests/fixtures/`:
- `step2_calibration_low.json`
- `step2_calibration_medium.json`
- `step2_calibration_high.json`

Regression tests verify deterministic alignment between mocked profile outputs and these expected submodel/readiness values.

## Mitigation Linkage

Mitigations are tied to submodels and blockers and include:
- `title`
- `reason`
- `impacted_submodels`
- `impacted_readiness_factors`
- `estimated_risk_reduction_band`
- `estimated_readiness_improvement_band`
- `insurer_relevance`
- `priority`

## Layer-Backed vs Fallback Mode

True layer-backed mode requires:
- geospatial packages from `requirements.txt` (`numpy`, `rasterio`, `pyproj`, `shapely`)
- valid configured `WF_LAYER_*` file paths

If missing, the API still runs in fallback/proxy mode with explicit assumptions and lower confidence.

## Current MVP Limitations

- Scoring weights and readiness checks are transparent MVP insurer-oriented heuristics, not underwriting-approved models.
- Access/egress scoring remains provisional and is not weighted into final wildfire score.
- Model version is `1.2.0`; legacy rows without version metadata default to `1.0.0` when read.

## Tests

Deterministic regression tests are in `tests/test_risk_assessment.py`.
They cover:
- low / medium / high risk profiles
- weak structure + moderate environment
- strong structure + high environment
- readiness blockers
- deterministic outputs for fixed mocked inputs
- debug endpoint payload structure
- legacy row compatibility
