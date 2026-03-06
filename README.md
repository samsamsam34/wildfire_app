# WildfireRisk Advisor (MVP Scaffold)

Now includes:
- Real geocoding integration via OpenStreetMap Nominatim (with fallback)
- Layer-based wildfire environmental model
- Structured explainability and confidence outputs
- API key authentication
- Persistent SQLite storage for reports

## Setup

```bash
pip install -r requirements.txt
export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"

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
- `GET /report/{assessment_id}` (requires `X-API-Key` when keys configured)

## Assessment Transparency

Each assessment now includes:
- `model_version`: Scoring model identifier (`1.1.0` for current outputs).
- `factor_breakdown`: Deterministic component scores (`environmental_risk`, `structural_risk`, `access_risk`).
- `confidence`: Includes `confidence_score`, `data_completeness_score`, `assumption_count`, and confidence flags.
- `assumptions`: Structured `observed_inputs`, `inferred_inputs`, `missing_inputs`, and `assumptions_used`.

This makes outputs more auditable and easier to debug for underwriting and QA workflows.

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/risk/assess" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-1" \
  -d '{"address":"123 Main St, Boulder, CO","attributes":{"roof_type":"class a","defensible_space_ft":20}}'
```

## Storage

Assessments are persisted to `wildfire_app.db` in the project root.
Legacy rows without `model_version` are treated as `1.0.0`.

## Tests

Regression tests are in `tests/test_risk_assessment.py`.
