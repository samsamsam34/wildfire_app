# WildfireRisk Advisor (MVP Scaffold)

Now includes:
- Real geocoding integration via OpenStreetMap Nominatim (with fallback)
- Real environmental data wiring via USGS elevation, Open-Meteo precipitation, and NASA EONET wildfire events
- API key authentication
- Persistent SQLite storage for reports

## Setup

```bash
pip install -r requirements.txt
export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"
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

## Frontend

A basic static page is in `frontend/public/index.html`.
