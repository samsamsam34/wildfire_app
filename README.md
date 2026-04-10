# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI backend plus a lightweight static frontend for deterministic, property-level wildfire assessment.

It focuses on three related outputs:
- `site_hazard_score` (landscape/environment around the property)
- `home_ignition_vulnerability_score` (home and near-structure susceptibility)
- `home_hardening_readiness` (rules-based homeowner readiness signal)

`wildfire_risk_score` is also returned as a blended summary for compatibility.
Legacy `insurance_readiness_score` remains available as an optional/future-facing compatibility mirror.

## Project Overview

The app geocodes an address, builds wildfire context from prepared regional layers, scores risk/home-hardening readiness, and returns explainable results with assumptions, confidence, and mitigation actions.

It also supports reassessment, simulation, report retrieval/export, portfolio workflows, and lightweight operational review features (annotations, workflow status, assignments, audit events).

Runtime scoring reads prepared local data. Large GIS download/prep is handled offline by scripts.

## Product Focus

Primary audience right now is homeowners. The default product path is:
1. assess an address
2. view the homeowner report
3. improve the result with missing home details
4. simulate mitigation upgrades

Insurer/portfolio/internal diagnostics/calibration capabilities remain available for advanced use, but they are secondary/internal surfaces and are not the default homeowner flow.

## What It Does

- Runs factorized wildfire scoring across environmental and structure-focused submodels.
- Computes home hardening readiness through a separate rules path with blockers and penalties.
- Supports homeowner inputs (`roof_type`, `vent_type`, `defensible_space_ft`, etc.), reassessment, and what-if simulation.
- Uses structure-based ring metrics (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`) when footprint data is available.
- Supports optional NAIP imagery-derived near-structure features (prepared offline):
  - ring vegetation cover/canopy/high-fuel/continuity proxies
  - local percentile context within the prepared region
  - nearest high-fuel patch distance proxy
- Adds defensible-space zone analysis for near-structure vegetation/fuel context:
  - `defensible_space_analysis` (zone metrics, basis geometry, quality/limitations, mitigation flags)
  - `top_near_structure_risk_drivers`
  - `prioritized_vegetation_actions`
  - `defensible_space_limitations_summary`
- Enriches context from open datasets when configured:
  - USFS WHP for hazard/burn context
  - MTBS perimeter/severity context for historical fire exposure
  - gridMET-derived dryness proxy
  - OpenStreetMap road-network features for access exposure
- Returns trust-oriented outputs:
  - score availability flags (distinguish “not scored” from real low scores)
  - confidence tier and use restriction
  - score eligibility, blockers, diagnostics, and provenance metadata
  - per-layer coverage audit (`layer_coverage_audit`) and coverage summary (`coverage_summary`) to explain data gaps vs sampling/config issues
  - factor-level score evidence ledger (`score_evidence_ledger`) with weight/contribution/evidence status per factor
  - evidence-quality summary (`evidence_quality_summary`) with observed/inferred/missing/fallback counts and confidence penalties
- Includes a homeowner-facing assessment map panel in the frontend:
  - property point and building footprint (when available)
  - defensible-space rings (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`)
  - nearby wildfire context overlays (historical fire perimeters and nearby structures when available)
  - layer toggles, legends, and limitations text for missing/partial geometry
- Persists assessment/report payloads in SQLite with compatibility handling for older rows.

## Running Locally

1. Clone the repo and enter it.

```bash
git clone https://github.com/samsamsam34/wildfire_app.git
cd wildfire_app
```

2. Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set environment variables (directly or in a local `.env` you source).

```bash
export WILDFIRE_API_KEYS="dev-key-1"     # optional local auth
export WF_REGION_DATA_DIR="data/regions" # prepared region root
```

4. Start the backend.

```bash
uvicorn backend.main:app --reload
```

5. Access the app.
- API docs: `http://127.0.0.1:8000/docs`
- Static frontend:

```bash
python3 -m http.server 4173 --directory frontend/public
```

Then open `http://127.0.0.1:4173`.

## Requirements

- Python `3.10+`
- Key Python packages: FastAPI, Uvicorn, Pydantic, NumPy, Rasterio, PyProj, Shapely
- Common geospatial system dependencies: GDAL, PROJ, GEOS

## Data Sources (High-Level)

- Building footprints (Overture, Microsoft, OSM): identify structures and near-structure context for property-specific analysis.
- Parcel data (county/state sources): improve address-to-property matching and property-boundary-aware feature extraction.
- Wildfire layers (LANDFIRE, USFS WHP, MTBS, gridMET): provide fuels, terrain, burn history, and dryness context.
- Imagery (NAIP): supports near-structure vegetation and defensible-space feature extraction in prepared workflows.

Data sources are open/public where available. Coverage and quality vary by region, and missing or weak local data reduces specificity and confidence.

## Data Preparation (Template)

```bash
# scripts/download_data.sh

mkdir -p data/raw
mkdir -p data/processed

# Example: download building footprints
# wget or curl command here

# Example: download parcel data
# region-specific

# Example: download wildfire layers
# LANDFIRE / WHP / MTBS
```

## Limitations

- Scoring and readiness are deterministic heuristics; this is not a carrier-approved underwriting model.
- Report outputs are decision-support guidance and do not guarantee insurability or wildfire outcomes.
- Open-data enrichment depends on local prepared layers and configured sources; missing datasets still trigger partial/fallback paths.

## Release Notes

Use `CHANGELOG.md` for structured release notes. Each release should record:
- versions bumped
- reason for bump
- expected output impact
- interpretation/comparability notes

## License
This project is licensed under the MIT License – see the LICENSE file for details.
