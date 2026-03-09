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
- Uses structure-based ring metrics (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`) when footprint data is available.
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
- Persists assessment/report payloads in SQLite with compatibility handling for older rows.

## Main API Capabilities

Full route docs are available at `/docs` when running locally.

Core assessment:
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`
- `POST /risk/layer-diagnostics`

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

## Model Governance / Versioning

Version metadata is centralized in `backend/version.py` and returned as `model_governance` in:
- `GET /health`
- assessment responses (`POST /risk/assess`, reassess/simulate mirrors, debug payloads)
- report export payloads (`GET /report/{assessment_id}/export`)
- benchmark artifacts (`scripts/run_benchmark_suite.py`)

Tracked dimensions:
- `product_version`
- `api_version`
- `scoring_model_version`
- `ruleset_version` and `rules_logic_version`
- `factor_schema_version`
- `benchmark_pack_version`
- `calibration_version`
- `region_data_version` / `data_bundle_version`

Version utilities:

```bash
python scripts/print_model_versions.py
python scripts/check_version_consistency.py
```

Bump guidance:
- `product_version` / `api_version`:
  - patch: internal bug fix without meaningful schema/output impact
  - minor: backward-compatible field additions or behavior changes
  - major: breaking contract or materially incompatible semantics
- `scoring_model_version`: scoring formulas/weights/submodel math changed
- `rules_logic_version` or `ruleset_version`: readiness/blocker logic changed
- `factor_schema_version`: factor/evidence ledger field meaning changed
- `benchmark_pack_version`: canonical benchmark scenarios/expectations changed
- `calibration_version`: empirical calibration method/dataset policy changed
- `region_data_version` / `data_bundle_version`: prepared layer snapshot changed materially

For cross-assessment comparisons, `/assessments/.../compare/...` includes a `version_comparison` block and compatibility label.

## Layer Diagnostics / Coverage Audit

Use `POST /risk/layer-diagnostics` (or `POST /risk/debug`) to inspect runtime data coverage before tuning scores.

Key response blocks:
- `layer_coverage_audit`: per-layer status (`configured`, `present_in_region`, `sample_attempted`, `sample_succeeded`, `coverage_status`, and failure notes)
- `coverage_summary`: totals plus `critical_missing_layers` and actionable `recommended_actions`

`coverage_status` interpretation:
- `observed`: sampled successfully
- `not_configured`: no source configured for that layer
- `missing_file`: configured path is missing
- `outside_extent`: point/ring is outside layer coverage or sampled nodata
- `sampling_failed`: read/CRS/runtime sampling failure
- `fallback_used`: scoring fallback path was used
- `partial`: layer exists but only partial evidence is available

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
- `WILDFIRE_APP_CATALOG_ROOT` (canonical catalog root, default `data/catalog`)
- `WF_USE_PREPARED_REGIONS` (default true)
- `WF_ALLOW_LEGACY_LAYER_FALLBACK` (optional direct-layer fallback mode)
- Optional open-data runtime sources:
  - `WF_LAYER_WHP_TIF`
  - `WF_LAYER_MTBS_SEVERITY_TIF`
  - `WF_LAYER_GRIDMET_DRYNESS_TIF`
  - `WF_LAYER_OSM_ROADS_GEOJSON`
  - `WF_LAYER_FEMA_STRUCTURES_GEOJSON`
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
  # optional enrichment layers
  whp.tif
  mtbs_severity.tif
  gridmet_dryness.tif
  roads.geojson
  manifest.json
```

Offline prep/validation scripts:
- `scripts/prepare_region_layers.py`
- `scripts/stage_landfire_assets.py`
- `scripts/build_landfire_region.py`
- `scripts/validate_prepared_region.py`
- `scripts/catalog_ingest_raster.py`
- `scripts/catalog_ingest_vector.py`
- `scripts/build_region_from_catalog.py`
- `scripts/prepare_region_from_catalog_or_sources.py`

BBox-first region prep:
- Region prep remains offline/admin-only; runtime API still uses prepared local files.
- `prepare_region_layers.py` now supports provider-aware acquisition with bbox-first behavior.
- For supported providers (for example ArcGIS ImageServer/FeatureServer), prep requests only the region bbox first.
- If bbox export/query is unsupported or fails, prep can fall back to full-download + local clip.
- Manifest layer metadata records acquisition details (`acquisition_method`, `provider_type`, `source_endpoint`, `bbox_used`, cache hits, fallbacks).

Example pilot prep flow:

```bash
python scripts/prepare_region_layers.py \
  --region-id test_region \
  --bbox -111.2 45.5 -110.9 45.8 \
  --prefer-bbox-downloads \
  --allow-full-download-fallback \
  --source-config path/to/source_config.json
python scripts/validate_prepared_region.py --region-id test_region --sample-lat 45.67 --sample-lon -111.04
```

Useful prep flags:
- `--prefer-bbox-downloads`
- `--allow-full-download-fallback` / `--no-allow-full-download-fallback`
- `--require-core-layers` / `--no-require-core-layers`
- `--skip-optional-layers`
- `--target-resolution`
- `--source-config`
- `--cache-root`

Minimal `source_config.json` example:

```json
{
  "layers": {
    "fuel": {
      "provider_type": "arcgis_image_service",
      "source_endpoint": "https://.../ImageServer",
      "full_download_url": "https://.../fuel_full.zip"
    },
    "building_footprints": {
      "provider_type": "arcgis_feature_service",
      "source_endpoint": "https://.../FeatureServer/0"
    }
  }
}
```

Canonical catalog and region build workflow:
- Use `data/catalog/` as a reusable canonical cache of normalized raster/vector layers.
- Ingestion (slow/path-provider aware) is separate from region assembly (fast/bbox subset from catalog).
- Runtime API still reads `data/regions/<region_id>/...` only; it does not download GIS data.

Catalog layout:

```text
data/catalog/
  rasters/<layer_name>/
  vectors/<layer_name>/
  metadata/<layer_name>/
  index/catalog_index.json
```

Catalog ingest examples:

```bash
python scripts/catalog_ingest_raster.py --layer dem --source-path path/to/dem.tif
python scripts/catalog_ingest_raster.py --layer fuel --source-endpoint https://.../ImageServer --bbox -111.2 45.5 -110.9 45.8 --prefer-bbox-downloads
python scripts/catalog_ingest_vector.py --layer fire_perimeters --source-endpoint https://.../FeatureServer/0 --bbox -111.2 45.5 -110.9 45.8 --prefer-bbox-downloads
```

Build region from catalog:

```bash
python scripts/build_region_from_catalog.py \
  --region-id bozeman_pilot \
  --display-name "Bozeman Pilot" \
  --bbox -111.2 45.5 -110.9 45.8 \
  --validate
```

`scripts/prepare_region_layers.py` also supports catalog mode via `--use-catalog`.

Key point: runtime endpoints do not download large GIS datasets.

Preparing a new area on demand:
- Use `scripts/prepare_region_from_catalog_or_sources.py` when a bbox is not yet fully covered in local catalog.
- The workflow is:
  1) inspect catalog coverage for required layers,
  2) fetch/ingest missing coverage from configured sources,
  3) build the prepared region from catalog,
  4) optionally validate.
- Runtime still reads only prepared region outputs.
- Default source registry: `config/source_registry.json`.
  - If `--source-config` is omitted, the script auto-loads the default registry.
  - You can override with `--source-config <path>` or `WF_SOURCE_CONFIG_PATH`.
  - Registry values support env references like `${WF_DEFAULT_DEM_ENDPOINT}`.

Plan-only check:

```bash
python scripts/prepare_region_from_catalog_or_sources.py \
  --region-id missoula_pilot \
  --display-name "Missoula Pilot" \
  --bbox -114.2 46.75 -113.8 47.0 \
  --plan-only
```

Optional plan helper:

```bash
python scripts/plan_region_build.py \
  --region-id missoula_pilot \
  --display-name "Missoula Pilot" \
  --bbox -114.2 46.75 -113.8 47.0
```

Prepare/build/validate:

```bash
python scripts/prepare_region_from_catalog_or_sources.py \
  --region-id bozeman_pilot \
  --display-name "Bozeman Pilot" \
  --bbox -111.2 45.5 -110.9 45.8 \
  --prefer-bbox-downloads \
  --allow-full-download-fallback \
  --allow-partial-coverage-fill \
  --validate
```

Plan output includes:
- required layers covered/missing/partial
- optional layers missing/partial
- which layers will use existing catalog vs acquisition
- required blockers (for example missing source config for a core layer)
- buildability from current catalog vs buildability with current source registry

Trust/transparency behavior:
- score families may be unavailable (`null`) when evidence is insufficient
- availability flags are included for each score
- confidence/provenance fields explain missing, inferred, stale, or provisional inputs
- layer diagnostics distinguish `not_configured`, `missing_file`, `outside_extent`, `sampling_failed`, `fallback_used`, and `observed`
- each score family can be audited through factor-level ledger entries (inputs, weights, contributions, evidence status, source references)
- evidence quality summary exposes confidence penalties and insurer-facing interpretation guardrails
- access exposure uses observable OSM road-network features when available and remains advisory (not part of weighted wildfire total)

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

## Benchmark Suite

The repo includes a versioned benchmark scenario pack for calibration discipline and drift checks:
- `benchmark/scenario_pack_v1.json`
- runner: `scripts/run_benchmark_suite.py`

Run the suite:

```bash
python scripts/run_benchmark_suite.py
```

Compare against a previous artifact:

```bash
python scripts/run_benchmark_suite.py \
  --compare-to benchmark/results/benchmark_run_YYYYMMDDTHHMMSSZ.json \
  --fail-on-drift
```

What it checks:
- scenario expectations (risk band, confidence tier/restriction, fallback behavior, warnings)
- relative ordering assertions
- monotonic sanity assertions (for mitigation and insurer-facing directional logic)
- release drift summary (score deltas, confidence deltas, warnings/blockers, factor contribution shifts)

Benchmark artifacts include aggregated governance metadata plus a `model_governance` block.

Use `POST /risk/debug?include_benchmark_hints=true` or `GET /report/{assessment_id}/export?include_benchmark_hints=true` to include lightweight benchmark resemblance and sanity-check hints in diagnostics/export output.

## Limitations

- Scoring and readiness are deterministic heuristics; this is not a carrier-approved underwriting model.
- Report export is JSON/HTML oriented (no built-in PDF generator).
- Open-data enrichment depends on local prepared layers and configured sources; missing datasets still trigger partial/fallback paths.

## Release Notes

Use `CHANGELOG.md` for structured release notes. Each release should record:
- versions bumped
- reason for bump
- expected output impact
- interpretation/comparability notes
