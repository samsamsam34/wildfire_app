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

Each assessment includes:
- `model_version`: scoring model identifier (`1.1.0` current, legacy rows default `1.0.0`).
- `observed_inputs`, `inferred_inputs`, `missing_inputs`, `assumptions_used`.
- `confidence_score` and `low_confidence_flags` for trust/debug workflow.
- `factor_breakdown` with explicit provisional metadata for access scoring.

Current limitation:
- `access_risk` is **provisional** and currently **excluded from final weighted scoring**.
- Real parcel/road/egress access modeling is planned for the next major refactor.

## Storage

Assessments are persisted to `wildfire_app.db`.
Rows lacking `model_version` are read as `1.0.0` automatically.

## Tests

Deterministic regression tests are in `tests/test_risk_assessment.py`.
They cover low/medium/high scenarios, model version presence, transparency fields, and provisional access behavior.

## Baseline Fixtures

Deterministic benchmark fixtures live in `tests/fixtures/`:
- `low_risk_baseline.json`
- `medium_risk_baseline.json`
- `high_risk_baseline.json`

Regression tests compare assessment outputs against these fixtures to detect unintended model/output drift before larger refactors.

## Scoring Notes

Each response includes `scoring_notes`, a centralized list of model caveats/conditions for downstream consumers (for example, provisional access scoring status and fallback usage).
