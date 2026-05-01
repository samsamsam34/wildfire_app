# Deployment Guide

This guide covers Railway (primary), Render (alternative), and local deployment checks.

## 1. Prerequisites

Required for production:
- `WF_ALLOWED_ORIGINS` (frontend origin allowlist)
- `WILDFIRE_API_KEYS` (API auth keys)
- `WF_REGRID_API_KEY` (parcel API)

Recommended:
- `WF_GOOGLE_GEOCODE_API_KEY` (fallback geocoding quality)
- `WF_MTBS_GPKG_PATH` (defaults to `data/national/mtbs_perimeters.gpkg`)
- cache DB path vars (`WF_*_CACHE_DB`) if using mounted persistent volumes

Rate limiting defaults:
- `WF_RATE_LIMIT_ASSESS=10/minute`
- `WF_RATE_LIMIT_ASSESS_DAILY=100/day`
- `WF_RATE_LIMIT_SIMULATE=20/minute`

## 2. Railway (Primary)

1. Connect repo to Railway.
2. Railway detects `railway.toml` and deploys with Nixpacks.
3. Set environment variables in Railway:
   - `WF_ALLOWED_ORIGINS`
   - `WILDFIRE_API_KEYS`
   - `WF_REGRID_API_KEY`
   - optional: `WF_GOOGLE_GEOCODE_API_KEY`, rate-limit overrides.
4. Deploy service.
5. Verify:
   - `GET /health` returns `200` and `"status": "ok"`.
   - App responds on Railway-provided `$PORT`.

## 3. Data Setup (Post-Deploy)

Runtime data should be mounted or downloaded into persistent storage.

Recommended order:
1. `data/zip_centroids.csv` (if used by your geocoding validation flow).
2. National MTBS:
   - `python scripts/download_national_mtbs.py`
3. Optional prepared regions:
   - `python scripts/prep_region.py ...` or your region prep workflow
   - validate with `python scripts/validate_prepared_region.py ...`

Notes:
- `data/national/` and `data/regions/` are runtime data, not source-controlled.
- MTBS national file is large; persistent disk is required.

## 4. Render (Alternative)

1. Create a new Render Web Service from this repo.
2. Use `render.yaml` (Docker runtime).
3. Set `sync: false` env vars in Render UI (secrets/origin values).
4. Deploy and verify `GET /health`.

## 5. Local Development

Use existing project flow from `README.md`:
1. Create/activate virtual environment.
2. `pip install -r requirements.txt`
3. Set `.env` values (`WILDFIRE_API_KEYS`, `WF_ALLOWED_ORIGINS`, etc.).
4. Run:
   - `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`

## 6. Monitoring and Health

`GET /health` returns:
- `status`
- version fields
- component statuses (`geocoder`, `parcel_client`, `national_footprint`, `fire_history`, `whp_client`)

Use it for:
- load balancer health checks
- startup diagnostics after deploy
- quick confirmation of optional integration availability

## 7. Scaling Considerations

- Worker count vs memory:
  - MTBS national dataset is large (hundreds of MB on disk, high in-memory footprint).
  - Start with **1 worker** on small plans (Railway starter-class memory).
- SQLite constraints:
  - per-process local file DB is fine for small single-instance deployment.
  - for multi-instance or higher write concurrency, migrate assessment state to PostgreSQL.
- Cache DB files:
  - place on persistent volume for better warm-cache behavior across restarts.
