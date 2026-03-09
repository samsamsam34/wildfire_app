# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI backend plus a lightweight static frontend for deterministic, property-level wildfire assessment.

It focuses on three related outputs:
- `site_hazard_score` (landscape/environment around the property)
- `home_ignition_vulnerability_score` (home and near-structure susceptibility)
- `insurance_readiness_score` (rules-based readiness signal)

`wildfire_risk_score` is also returned as a blended summary for compatibility.

## Project Overview

The app geocodes an address, builds wildfire context from prepared regional layers, scores risk/readiness, and returns explainable results with assumptions, confidence, and mitigation actions.

It also supports reassessment, simulation, report retrieval/export, portfolio workflows, and lightweight operational review features (annotations, workflow status, assignments, audit events).

Runtime scoring reads prepared local data. Large GIS download/prep is handled offline by scripts.

## What It Does

- Runs factorized wildfire scoring across environmental and structure-focused submodels.
- Computes insurance readiness through a separate rules path with blockers and penalties.
- Supports homeowner inputs (`roof_type`, `vent_type`, `defensible_space_ft`, etc.), reassessment, and what-if simulation.
- Uses optional building-footprint ring metrics (`0-5 ft`, `5-30 ft`, `30-100 ft`) when available.
- Returns trust-oriented outputs:
  - score availability flags (distinguish “not scored” from real low scores)
  - confidence tier and use restriction
  - score eligibility, blockers, diagnostics, and provenance metadata
  - factor-level score evidence ledger (`score_evidence_ledger`) with weight/contribution/evidence status per factor
  - evidence-quality summary (`evidence_quality_summary`) with observed/inferred/missing/fallback counts and confidence penalties
- Persists assessment/report payloads in SQLite with compatibility handling for older rows.

## Main API Capabilities

Full route docs are available at `/docs` when running locally.

Core assessment:
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`

Reports:
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view`

Portfolio and batch:
- `POST /portfolio/assess`
- `POST /portfolio/jobs`, `GET /portfolio/jobs/{job_id}`, `GET /portfolio/jobs/{job_id}/results`
- `POST /portfolio/import/csv`
- `GET /portfolio`, `GET /assessments`, `GET /assessments/summary`

Review and operations:
- annotations, review status, assignment, workflow, comparison, scenario history
- organizations and underwriting rulesets
- audit and summary endpoints (`/audit/events`, `/admin/summary`)

## Local Development / Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enforce API keys. If unset, auth is open for local dev.
export WILDFIRE_API_KEYS="dev-key-1"

uvicorn backend.main:app --reload
```

Frontend:
- file: `frontend/public/index.html`
- default API base: `http://127.0.0.1:8000`
- open directly or serve with a static server, for example:

```bash
python3 -m http.server 8080 --directory frontend/public
```

## Configuration / Environment

Common runtime settings:
- `WILDFIRE_API_KEYS` (optional local auth)
- `WF_REGION_DATA_DIR` (prepared region root, default `data/regions`)
- `WF_USE_PREPARED_REGIONS` (default true)
- `WF_ALLOW_LEGACY_LAYER_FALLBACK` (optional direct-layer fallback mode)
- `WILDFIRE_APP_CACHE_ROOT`, `WILDFIRE_APP_DATA_ROOT`, `WILDFIRE_APP_TMP_ROOT` (offline prep script paths)

Legacy direct-layer paths are still supported via `WF_LAYER_*` env vars (`DEM`, `SLOPE`, `FUEL`, `CANOPY`, fire perimeters, building footprints, etc.), but prepared-region runtime is the primary path.

Geocoding uses OpenStreetMap Nominatim (`backend/geocoding.py`). If geocoding fails, assessment returns an error instead of scoring synthetic coordinates.

## Data / Storage Notes

SQLite (`wildfire_app.db`) stores:
- assessments and scenarios
- organizations and underwriting rulesets
- annotations, review status, workflow/assignment state
- portfolio jobs
- audit events

Prepared region layout:

```text
data/regions/<region_id>/
  dem.tif
  slope.tif
  fuel.tif
  canopy.tif
  fire_perimeters.geojson
  building_footprints.geojson
  manifest.json
```

Offline prep/validation scripts:
- `scripts/prepare_region_layers.py`
- `scripts/stage_landfire_assets.py`
- `scripts/build_landfire_region.py`
- `scripts/validate_prepared_region.py`

Example pilot prep flow:

```bash
python scripts/prepare_region_layers.py --region-id test_region --bbox -111.2 45.5 -110.9 45.8
python scripts/validate_prepared_region.py --region-id test_region --sample-lat 45.67 --sample-lon -111.04
```

Key point: runtime endpoints do not download large GIS datasets.

Trust/transparency behavior:
- score families may be unavailable (`null`) when evidence is insufficient
- availability flags are included for each score
- confidence/provenance fields explain missing, inferred, stale, or provisional inputs
- each score family can be audited through factor-level ledger entries (inputs, weights, contributions, evidence status, source references)
- evidence quality summary exposes confidence penalties and insurer-facing interpretation guardrails
- access exposure is explicitly provisional and not parcel-verified

## Testing

Run the full suite:

```bash
pytest
```

Main coverage areas:
- assessment/report contract and trust gating
- reassessment/simulation flows
- portfolio/jobs/CSV paths
- roles, org scoping, review/workflow, audit summaries
- region prep, LANDFIRE handling, and prepared-region validation

## Limitations

- Scoring and readiness are deterministic heuristics; this is not a carrier-approved underwriting model.
- Report export is JSON/HTML oriented (no built-in PDF generator).
- Building-footprint/ring enrichment depends on local footprint data availability.
