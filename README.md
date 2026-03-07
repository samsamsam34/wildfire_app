# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI application for property-level wildfire risk and insurance-readiness workflows.

Current release: **Step 5 enterprise operations layer** on top of the Step 2/3/4 scoring, simulation, and portfolio foundations.

## What It Does

- Runs deterministic, factorized wildfire risk assessments at the property level.
- Computes insurance readiness with a separate rules engine and explicit blockers/penalties.
- Supports confirmed property facts, reassessment, and what-if simulation.
- Persists reports in SQLite with backward-compatible payload upgrades.
- Supports batch/portfolio workflows, filtering, and prioritization.
- Adds enterprise foundations: organizations, role-aware behavior, portfolio jobs, CSV import/export, underwriting rulesets, audit events, assignment/workflow states, and admin summaries.

## Core API Endpoints

### Health and Assessment
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`

### Reports
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view`

Report endpoints accept `?audience=homeowner|agent|inspector|insurer` (and compatibility alias `audience_mode`).

### Homeowner Simulation API
`POST /risk/simulate` supports:
- simulate from an existing assessment (`assessment_id`)
- or simulate directly from `address` + base `attributes`

Example request:

```json
{
  "assessment_id": "abc123",
  "scenario_name": "Defensible space + roof upgrade",
  "scenario_overrides": {
    "roof_type": "class_a",
    "defensible_space_ft": 30
  }
}
```

Example response excerpt:

```json
{
  "base_scores": {
    "wildfire_risk_score": 71,
    "insurance_readiness_score": 58
  },
  "simulated_scores": {
    "wildfire_risk_score": 48,
    "insurance_readiness_score": 79
  },
  "score_delta": {
    "wildfire_risk_score_delta": -23,
    "insurance_readiness_score_delta": 21
  }
}
```

The response also includes `changed_inputs`, baseline/simulated assumptions, and baseline/simulated confidence blocks for homeowner explainability.

### Portfolio and Batch
- `POST /portfolio/assess`
- `GET /portfolio`
- `GET /assessments`
- `GET /assessments/summary`
- `POST /portfolio/jobs`
- `GET /portfolio/jobs/{job_id}`
- `GET /portfolio/jobs/{job_id}/results`
- `GET /portfolio/jobs/{job_id}/export/csv`
- `GET /portfolio/jobs/{job_id}/report-pack`
- `GET /portfolio/jobs/summary`
- `POST /portfolio/import/csv`
- `GET /portfolio/{portfolio_name}/export/csv`

### Organizations and Underwriting Rulesets
- `GET /organizations`
- `POST /organizations`
- `GET /organizations/{organization_id}`
- `GET /underwriting/rulesets`
- `GET /underwriting/rulesets/{ruleset_id}`
- `POST /underwriting/rulesets`

### Review, Workflow, and Comparison
- `POST /assessments/{assessment_id}/annotations`
- `GET /assessments/{assessment_id}/annotations`
- `PUT /assessments/{assessment_id}/review-status`
- `GET /assessments/{assessment_id}/review-status`
- `POST /assessment/{assessment_id}/assign`
- `POST /assessment/{assessment_id}/workflow`
- `GET /assessment/{assessment_id}/workflow`
- `GET /assessments/{assessment_id}/compare/{other_assessment_id}`
- `GET /assessments/compare?ids=id1,id2,...`
- `GET /assessments/{assessment_id}/scenarios`

### Audit and Ops
- `GET /audit/events`
- `GET /admin/summary`
- `GET /organizations/{organization_id}/summary`

## Organization and Role Model (Step 5)

All non-health endpoints continue using API key auth. Step 5 adds lightweight role/org context through headers:

- `X-API-Key`
- `X-User-Role`: `admin|underwriter|broker|inspector|agent|viewer`
- `X-Organization-Id`
- `X-User-Id`

Role-aware behavior includes:
- only `admin|underwriter` can update review status
- workflow/assignment edits restricted to `admin|underwriter|broker|inspector`
- viewer role is read-only for assessments/batch/simulation mutations
- organization scope is enforced for non-admin users

## Scoring and Readiness (unchanged core)

### Factorized Wildfire Risk Submodels
- `vegetation_intensity_risk`
- `fuel_proximity_risk`
- `slope_topography_risk`
- `ember_exposure_risk`
- `flame_contact_risk`
- `historic_fire_risk`
- `structure_vulnerability_risk`
- `defensible_space_risk`

Each submodel includes score, weighted contribution, deterministic explanation, key inputs, and assumptions.

### Score Decomposition (Trustworthiness)
Assessments now return three distinct score families:
- `site_hazard_score` (0-100): landscape/environmental pressure around the property
- `home_ignition_vulnerability_score` (0-100): structure and near-structure susceptibility
- `insurance_readiness_score` (0-100): rules-based insurer-oriented readiness advisory

`wildfire_risk_score` is retained for compatibility and presented as a blended summary (`site_hazard_score` + `home_ignition_vulnerability_score`).
`legacy_weighted_wildfire_risk_score` is included for traceability with prior weighting output.

### Property-Level Structure Ring Context (Sprint 2)
The data layer now attempts building-footprint-based vegetation analysis before scoring:
- footprint lookup from local vector data (`WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON`)
- structure-relative rings: `ring_0_5_ft`, `ring_5_30_ft`, `ring_30_100_ft`
- canopy/vegetation summaries by ring (`canopy_mean`, `canopy_max`, `coverage_pct`, `vegetation_density`, `fuel_presence_proxy`)

These ring metrics are used as additional inputs for flame-contact, defensible-space, fuel-proximity, and vegetation-intensity submodels, and to sharpen mitigation recommendations.
If footprint data is unavailable, the system falls back to point-based context and records an assumption.

### Separate Insurance Readiness
Readiness is computed separately and returns:
- `readiness_factors`
- `readiness_blockers`
- `readiness_penalties`
- `readiness_summary`

### Underwriting Rulesets / Carrier Profiles
Assessments can specify `ruleset_id` (for example: `default`, `strict_carrier_demo`, `inspection_first_demo`).
Rulesets can adjust readiness penalty scaling, blocker thresholds, and mitigation prioritization emphasis.
Response/report payloads include `ruleset_id`, `ruleset_name`, `ruleset_version`, and `ruleset_description`.

## Step 5 Enterprise Workflows

### Portfolio Jobs
`POST /portfolio/jobs` creates a persisted job with status lifecycle:
- `queued`
- `running`
- `completed`
- `failed`
- `partial`

Jobs include totals, completion/failure counts, summary metrics, and error summary.

### CSV Import / Export
- CSV import (`POST /portfolio/import/csv`) parses address + optional property facts, validates rows, and returns row-level errors.
- CSV export endpoints provide portfolio/job summaries for operational workflows.

### Audit Trail
Important actions are logged as append-only audit events, including assessment creation, simulation, report export/view, status/workflow changes, annotation writes, job lifecycle events, and ruleset usage.

### Reviewer Assignment and Workflow States
Assessments support:
- assignment (`assigned_reviewer`, `assigned_role`)
- workflow states: `new`, `triaged`, `needs_inspection`, `mitigation_pending`, `ready_for_review`, `approved`, `declined`, `escalated`
- deterministic transition validation in API logic

## Response Trust/Transparency Fields

Assessment/report payloads include:
- assumptions: `confirmed_inputs`, `observed_inputs`, `inferred_inputs`, `missing_inputs`, `assumptions_used`
- confidence: `confidence_score`, `low_confidence_flags`
- explainability: `submodel_scores`, `weighted_contributions`, `factor_breakdown`, `top_risk_drivers`, `top_protective_factors`, `explanation_summary`
- homeowner insights: `property_findings` (plain-language findings derived from structure-ring vegetation context)
- confidence gating: `confidence_tier` (`high|moderate|low|preliminary`) and `use_restriction`
- assessment eligibility guardrails:
  - score-family eligibility blocks:
    - `site_hazard_eligibility`
    - `home_vulnerability_eligibility`
    - `insurance_readiness_eligibility`
  - assessment-level status:
    - `assessment_status` (`fully_scored|partially_scored|insufficient_data`)
    - `assessment_blockers`
  - `assessment_diagnostics` for developer/debug trust tracing
- insufficient-data score handling:
  - score values are `null` when minimum evidence is insufficient (not `0` as a fake no-risk signal)
  - availability flags are included:
    - `wildfire_risk_score_available`
    - `site_hazard_score_available`
    - `home_ignition_vulnerability_score_available`
    - `insurance_readiness_score_available`
- environmental quality: `environmental_layer_status` (`ok|missing|error` per key layer) and `environmental_data_completeness_score`
- data coverage/provenance:
  - `direct_data_coverage_score`, `inferred_data_coverage_score`, `missing_data_share`
  - `input_source_metadata` and `data_provenance` (per-input metadata + machine-readable summary)
  - freshness/source quality metadata per input:
    - `provider_status`: `ok|missing|error`
    - `freshness_status`: `current|aging|stale|unknown`
    - `used_in_scoring`, `confidence_weight`, `dataset_version`, `observed_at`, `loaded_at`
  - provenance summary metrics:
    - `stale_data_share`, `heuristic_input_count`, `current_input_count`
  - score-family input quality rollups:
    - `site_hazard_input_quality`
    - `home_vulnerability_input_quality`
    - `insurance_readiness_input_quality`
- property-level context: `property_level_context.footprint_used` and `property_level_context.ring_metrics`
  - includes `footprint_status` (`used`, `not_found`, `provider_unavailable`, `error`) and `fallback_mode` (`footprint`, `point_based`) for fallback transparency
  - includes `fallback_mode` (`footprint` or `point_based`) for clear interpretation context
- readiness and mitigation linkage fields

`score_summaries` is included with three sections (`site_hazard`, `home_ignition_vulnerability`, `insurance_readiness`), each containing label, score, explanation, top drivers, protective factors, and next actions.

If address geocoding cannot be verified, `/risk/assess` returns an error (no synthetic coordinate scoring in default behavior).
Missing environmental layers are surfaced explicitly and do not silently default to neutral point values in the data layer.
If critical inputs are stale/unknown or provider errors occur, confidence tier and use restriction are downgraded deterministically.
Hard trust blockers force `not_for_underwriting_or_binding` when minimum evidence is insufficient.

## Setup

```bash
pip install -r requirements.txt

export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"

# Optional geospatial layer paths
export WF_LAYER_BURN_PROB_TIF="/path/to/burn_probability.tif"
export WF_LAYER_HAZARD_SEVERITY_TIF="/path/to/hazard_severity.tif"
export WF_LAYER_SLOPE_TIF="/path/to/slope_degrees.tif"
export WF_LAYER_ASPECT_TIF="/path/to/aspect_degrees.tif"
export WF_LAYER_DEM_TIF="/path/to/dem.tif"
export WF_LAYER_FUEL_TIF="/path/to/fuel_model.tif"
export WF_LAYER_CANOPY_TIF="/path/to/canopy_density.tif"
export WF_LAYER_MOISTURE_TIF="/path/to/moisture_or_dryness.tif"
export WF_LAYER_FIRE_PERIMETERS_GEOJSON="/path/to/fire_perimeters.geojson"
export WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON="/path/to/building_footprints.geojson"
export WF_REGION_DATA_DIR="./data/regions"
export WF_USE_PREPARED_REGIONS="true"
# Optional compatibility fallback for legacy direct paths when no prepared region matches:
export WF_ALLOW_LEGACY_LAYER_FALLBACK="true"

# Optional provenance/freshness metadata (recommended for trust gating)
export WF_LAYER_BURN_PROB_VERSION="v2025.10"
export WF_LAYER_BURN_PROB_DATE="2025-10-01"
export WF_LAYER_HAZARD_SEVERITY_VERSION="v2025.08"
export WF_LAYER_HAZARD_SEVERITY_DATE="2025-08-15"
export WF_LAYER_SLOPE_DATE="2024-06-01"
export WF_LAYER_FUEL_DATE="2025-07-01"
export WF_LAYER_CANOPY_DATE="2025-07-01"
export WF_LAYER_FIRE_PERIMETERS_DATE="2025-01-01"
export WF_BUILDING_FOOTPRINT_VERSION="county_release_2025"
export WF_BUILDING_FOOTPRINT_DATE="2025-09-01"

# Optional freshness policy overrides (days)
export WF_FRESHNESS_ENVIRONMENTAL_RASTER_CURRENT_DAYS=180
export WF_FRESHNESS_ENVIRONMENTAL_RASTER_AGING_DAYS=365
export WF_FRESHNESS_FIRE_HISTORY_LAYER_CURRENT_DAYS=365
export WF_FRESHNESS_FIRE_HISTORY_LAYER_AGING_DAYS=730

uvicorn backend.main:app --reload
```

Frontend file: `frontend/public/index.html`.

## Regional Layer Prep Pipeline (separate from runtime)

Layer ingestion/prep runs **outside** the API app. Runtime scoring reads prepared local region assets and does **not** download large GIS layers during `/risk/assess`.

### CLI Modes

`scripts/prepare_region_layers.py` now supports:
- `local-source prepare mode` (existing local files)
- `download-and-prepare mode` (URL provided per layer, then clipped/validated locally)
- `auto-discovery mode` (resolve dataset URLs from bbox when explicit URLs/files are not provided)

### Local-source prepare mode (fully working)

```bash
python scripts/prepare_region_layers.py \
  --region-id marin_county_ca \
  --display-name "Marin County, CA" \
  --bbox -123.05,37.70,-122.20,38.35 \
  --dem /path/to/dem.tif \
  --fuel /path/to/fuel.tif \
  --canopy /path/to/canopy.tif \
  --fire-perimeters /path/to/fire_perimeters.geojson \
  --building-footprints /path/to/building_footprints.geojson
```

`--slope` is optional. If omitted, `slope.tif` is derived from `dem.tif`.

### Download-and-prepare mode (pilot)

```bash
python scripts/prepare_region_layers.py \
  --region-id pilot_demo \
  --bbox -122.6 37.8 -122.2 38.1 \
  --dem-url https://example.org/dem.tif \
  --fuel-url https://example.org/fuel.tif \
  --canopy-url https://example.org/canopy.tif \
  --fire-perimeters-url https://example.org/perimeters.geojson \
  --building-footprints-url https://example.org/footprints.geojson
```

For local testing, `--skip-download` can be used with local file inputs.

Common hardening flags:
- `--dry-run`: validates config/sources and reports what would run, without writing outputs
- `--download-timeout`: per-request timeout seconds
- `--download-retries`: retry count
- `--retry-backoff-seconds`: exponential backoff base
- `--keep-temp-on-failure`: preserve `_downloads`/`_extracted` for debugging
- `--clean-download-cache`: remove staging folders after run
- `--allow-partial`: write partial manifest with explicit failed/missing layers
- `--no-auto-discovery`: disable dataset discovery adapters and require explicit sources
- optional per-layer checksum flags (for example `--dem-checksum sha256:<hex>`, `--fuel-checksum sha256:<hex>`)

Auto-discovery pilot flow:

```bash
python scripts/prepare_region_layers.py \
  --region-id marin_ca \
  --bbox -123.1 37.8 -122.2 38.3
```

In this mode, the prep pipeline attempts to resolve source assets using adapters, then downloads/caches/clips/validates outputs.

Archive handling:
- `.zip` inputs are supported for raster/vector layer sources.
- The preparer uses deterministic selection rules and fails clearly on ambiguous archives.
- If no valid candidate exists (or multiple candidates cannot be resolved safely), prep fails clearly.

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

`manifest.json` stores region metadata (bounds/CRS/status), per-layer source metadata, freshness timestamps, and file mappings.

Manifest layer metadata includes fields such as:
- `source_name`, `source_type`, `source_mode`, `source_url`
- `dataset_version`, `freshness_timestamp`, `downloaded_at`, `download_status`
- `bytes_downloaded`, `retry_count_used`, `timeout_seconds`
- `extraction_performed`, `extracted_path`, `checksum_status`
- `dataset_source`, `dataset_provider`, `tile_ids`, `download_url`, `mosaic_performed`
- `cache_hit`, `clipped_to_bbox`, `validation_status`
- layer notes/warnings

Optional checksum verification:
- Add `checksum` in `source_metadata` (for example `sha256:<hex>`).
- If provided, the preparer verifies checksum and fails on mismatch.

### Automation honesty (current state)

- Fully automated adapters in this pass:
  - USGS 3DEP discovery path for DEM assets (API-driven discovery + download/caching/clip)
  - NIFC fire perimeter adapter when `WF_NIFC_FIRE_PERIMETERS_URL` is configured
  - Microsoft building-footprint tile-index adapter when `WF_MS_BUILDINGS_INDEX_URL` is configured
- Partially automated adapters in this pass:
  - LANDFIRE fuel/canopy adapters are template/URL driven and may require explicit URL templates
- Full end-to-end provider orchestration (catalog crawling, authed provider variants, advanced mosaicking strategies) is still deferred.
- This is intentional for a pilot-region ingestion workflow.

URL behavior guidance:
- Direct file URLs (or predictable archive URLs) work best.
- Catalog scraping/discovery orchestration is intentionally deferred.

Caching:
- Download cache is stored in `data/cache/` by default.
- Reused cached assets are tracked per layer (`cache_hit` in manifest metadata).

Partial-mode caveat:
- `--allow-partial` does **not** indicate full readiness.
- Manifest status remains `partial`, with explicit `failed_layers` and warnings.

Runtime behavior:
- geocode address
- resolve region by lat/lon from `manifest.json` bounds
- load prepared local files for that region
- score deterministically using local assets

If no prepared region matches and no legacy direct layer paths are configured, assessment returns explicit insufficient-data blockers (`region_not_prepared`) instead of attempting live large-data ingestion.

## Dependencies

`requirements.txt` includes:
- Runtime: `fastapi`, `uvicorn`, `pydantic`
- Geospatial stack: `numpy`, `rasterio`, `pyproj`, `shapely`
- Test: `pytest`, `httpx`

Without geospatial packages and configured layer files, the app runs in deterministic fallback mode instead of true layer-backed mode.
Without a configured building-footprint layer, the app still runs but `property_level_context.footprint_used` is `false`, `footprint_status` is `source_unavailable`, and ring metrics are omitted.

## Persistence and Compatibility

SQLite tables include:
- `assessments`
- `organizations`
- `underwriting_rulesets`
- `assessment_scenarios`
- `assessment_annotations`
- `assessment_review_status`
- `assessment_workflow`
- `portfolio_jobs`
- `audit_events`

Compatibility behavior:
- old rows remain readable
- missing fields are backfilled safely at read time
- legacy model rows default to `1.0.0`

## Model Versioning
- Current model version: `1.5.0`
- Legacy fallback for older rows: `1.0.0`

## Tests

Deterministic tests in `tests/test_risk_assessment.py` cover:
- assess/report contract and Step 2+ explainability fields
- reassessment and simulation flows
- batch/portfolio and partial failure behavior
- organization scoping and role-aware permissions
- portfolio job lifecycle and CSV import/export
- ruleset selection behavior
- assignment/workflow transitions
- audit and admin summary endpoints
- legacy compatibility

## Current Limitations

- Scoring/readiness remain transparent MVP heuristics, not carrier-approved underwriting models.
- Outputs are advisory and should not be treated as calibrated premium or binding/underwriting decisions.
- Access/egress scoring is still provisional and excluded from weighted wildfire total.
- Building footprint support currently expects a local GeoJSON footprint source; no external footprint API integration yet.
- Portfolio jobs are SQLite/background-task based; no distributed queue yet.
- API-key + role headers are lightweight controls, not full identity/access management.
- Report export is JSON/HTML oriented; native PDF generation is not included.

## Ready for Step 6

Step 5 now provides a practical enterprise foundation:
- org-aware operational data model
- role-aware review and workflow controls
- portfolio job abstraction with CSV import/export
- configurable underwriting ruleset profiles
- audit event logging and admin/organization summaries
- deterministic, test-covered APIs that retain backward compatibility
