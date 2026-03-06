# WildfireRisk Advisor (MVP Scaffold)

Now includes:
- Real geocoding integration via OpenStreetMap Nominatim (with fallback)
- Layer-based wildfire environmental model (burn probability, hazard severity, slope/aspect, fuel/canopy, wildland distance, fire recurrence)
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
export WF_LAYER_FIRE_PERIMETERS_GEOJSON="/path/to/fire_perimeters.geojson"

uvicorn backend.main:app --reload
```

## Endpoints

- `GET /health` (public)
- `POST /risk/assess` (requires `X-API-Key` when keys configured)
- `GET /report/{assessment_id}` (requires `X-API-Key` when keys configured)

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/risk/assess" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-1" \
  -d '{"address":"123 Main St, Boulder, CO","attributes":{"roof_type":"class a","defensible_space_ft":20}}'
```

## Storage

Assessments are persisted to `wildfire_app.db` in the project root.

## Layer notes

- If a layer is missing at runtime, the API still responds using explicit fallback assumptions.
- For defensible property-grade scoring, provide all layer paths and ensure consistent CRS/coverage.

## Frontend

A basic static page is in `frontend/public/index.html`.
