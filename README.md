# WildfireRisk Advisor
A homeowner-focused wildfire risk and mitigation tool that aims to be property-specific when data supports it, and explicit about limits when it does not.

## What this is
WildfireRisk Advisor helps homeowners understand wildfire risk at their property and decide what to do next.

It uses deterministic scoring logic and open geospatial data, then returns:
- risk and readiness summaries
- plain-language drivers and limitations
- prioritized mitigation actions
- confidence and specificity signals

## Current status
### What works
- Homeowner-first flow: assess, review, improve inputs, and simulate mitigation changes.
- Action-oriented outputs: top risk drivers, prioritized actions, and improvement guidance.
- Trust framing: explicit confidence, assumptions, and data-gap limitations.
- Specificity tiers reflect evidence quality: `property_specific` (strong parcel/footprint/near-structure support), `address_level` (partial property evidence), `regional_estimate` (limited local property evidence).
- Prepared-region workflow for deterministic runtime scoring on local data snapshots.

### Known limitations
- Coverage is only as good as prepared regional data.
- Property-level quality depends heavily on parcel polygons, building footprints, and near-structure context.
- Some addresses require manual confirmation when candidates conflict or confidence is low.
- Data quality can vary by county/region, so outputs may be less specific in some locations.
- This is a decision-support tool, not a guarantee of real-world outcomes.

## Core principle
Be useful without pretending certainty.

If evidence is strong, return property-specific guidance. If evidence is weak, reduce specificity, surface uncertainty clearly, and still provide practical next steps.

## What users get
- Overall wildfire risk summary
- Home hardening readiness summary
- Top risk drivers
- Prioritized mitigation actions
- Confidence and specificity summary
- Assumptions and unknowns
- Before/after simulation feedback for mitigation scenarios

## How it works
1. User submits an address.
2. The app resolves a location and checks prepared-region coverage.
3. It pulls available property and regional context (parcel, footprint, vegetation/fuel, hazard, access, etc.).
4. Deterministic scoring computes risk/readiness outputs.
5. The app returns scores, drivers, limitations, confidence/specificity tier, and recommended actions.

## Running Locally
1. Clone and enter the repo.
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
3. Set basic environment variables.
```bash
# Optional: set API keys for auth; if omitted, local dev runs open.
export WILDFIRE_API_KEYS="dev-key-1"

# Optional: override prepared-region directory.
export WF_REGION_DATA_DIR="data/regions"
```
4. Run the backend API.
```bash
uvicorn backend.main:app --reload
```
5. Open the app.
API docs: `http://127.0.0.1:8000/docs`

Static frontend: serve `frontend/public` and open it in a browser:
```bash
python3 -m http.server 4173 --directory frontend/public
```
Then open `http://127.0.0.1:4173`.

## Requirements
- Python: `3.10+`
- Core runtime libraries: FastAPI, Uvicorn, Pydantic
- Geo/data libraries: NumPy, Rasterio, PyProj, Shapely
- Test libraries: Pytest, HTTPX
- System libraries (commonly needed for geo stack): GDAL, PROJ, GEOS

## Data Sources (High-Level)
- Building footprints (for example Overture, Microsoft, OSM): anchors near-structure context and improves property specificity.
- Parcel data (county/state parcel polygons and parcel-address points): improves address-to-property matching and boundary-aware feature extraction.
- Wildfire layers (for example LANDFIRE fuels/canopy, hazard and burn history layers): provides regional hazard context.
- Imagery (for example NAIP): supports near-structure vegetation/fuel proxies in prepared workflows.

This project uses open data where available. Coverage and quality vary by region, and that directly affects confidence and specificity tier.

## Data Pipeline Overview
- Offline preparation: region data is downloaded/ingested, clipped/validated, and written to prepared region folders (`data/regions/<region_id>`).
- Runtime assessment: API calls read prepared region files, extract features for the target location, and run deterministic scoring.
- Practical split: heavy GIS work happens offline; runtime stays fast, deterministic, and transparent about missing/partial evidence.

## Data Download / Preparation (Template)
Use this as a starting point for regional data staging scripts.

```bash
# Example: scripts/download_data.sh

set -euo pipefail

# Create directories
mkdir -p data/raw/footprints
mkdir -p data/raw/parcels
mkdir -p data/raw/wildfire
mkdir -p data/processed

# Download building footprints (placeholder)
# curl -L "<footprint_source_url>" -o data/raw/footprints/footprints.geojson

# Download parcel data (region-specific placeholder)
# curl -L "<parcel_source_url>" -o data/raw/parcels/parcels.geojson

# Download wildfire layers (placeholder)
# curl -L "<wildfire_layer_url>" -o data/raw/wildfire/fuel.tif
```

Then prepare a region with the project script:

```bash
python3 scripts/prepare_region_from_catalog_or_sources.py \
  --region-id my_region \
  --display-name "My Region" \
  --bbox min_lon min_lat max_lon max_lat \
  --validate
```

## Current focus / next steps
- Improve property-specific reliability where parcel and footprint quality are inconsistent.
- Reduce false precision by tightening coordinate and candidate selection safety.
- Expand regional data quality and validation so more addresses can stay in `property_specific` mode.
- Keep homeowner guidance clear, practical, and tied to observable evidence.

## What this is not
- Not an underwriting engine.
- Not a fire spread simulator.
- Not a probabilistic loss forecast.
- Not a substitute for local fire officials, defensible space inspections, or code requirements.

## Philosophy
Trust is earned by clarity, not complexity.

The product should:
- prioritize homeowner decisions over internal platform detail
- expose assumptions instead of hiding them
- separate strong evidence from fallback inference
- make limitations visible at the same level as scores

## Bottom line
WildfireRisk Advisor is built to give homeowners actionable wildfire guidance with honest confidence boundaries. It should be specific when the data is strong, and transparent when it is not.
