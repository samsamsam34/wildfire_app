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

## Main API Capabilities

Full route docs are available at `/docs` when running locally.

Core assessment:
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`
- `POST /risk/layer-diagnostics`
- `POST /regions/coverage-check` (check whether a location is covered by prepared regions)
- `POST /regions/prepare`, `GET /regions/prepare/{job_id}` (queue/poll offline region-prep jobs)

Reports:
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view`
- `GET /report/{assessment_id}/homeowner`
- `GET /report/{assessment_id}/homeowner/pdf`
- `GET /report/{assessment_id}/map`

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
python scripts/print_release_note_template.py --version 0.10.1 --date 2026-03-09
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

Release note format:
- Keep one `CHANGELOG.md` entry per `product_version` with required sections:
  - `Version changes`
  - `Reason`
  - `Expected effect on outputs`
  - `Migration/interpretation notes`
  - `Historical comparison validity`

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

Region resolution fields:
- Assessment responses include `region_resolution` with `coverage_available`, `resolved_region_id`, `reason`, and `recommended_action`.
- Uncovered locations can return `region_not_ready` details (HTTP 409 when prepared coverage is required) or `insufficient_data` with `region_resolution.reason=no_prepared_region_for_location`.
- When footprint geometry is unavailable, defensible-space zone metrics can still run in point-proxy mode, and responses include explicit limitations.

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
- `WF_REQUIRE_PREPARED_REGION_COVERAGE` (if true, uncovered addresses return `region_not_ready` instead of scoring fallback)
- `WF_AUTO_QUEUE_REGION_PREP_ON_MISS` (when true, `/risk/assess` queues a prep job and returns `region_not_ready` for uncovered addresses)
- `WF_AUTO_REGION_PREP_TILE_DEG` (bbox tile size used for auto-queued region prep, default `0.25`)
- `WF_REGION_PREP_SOURCE_CONFIG` (optional source config path for queued prep jobs)
- `WF_REGION_PREP_VALIDATE`, `WF_REGION_PREP_REQUIRE_CORE_LAYERS`, `WF_REGION_PREP_SKIP_OPTIONAL_LAYERS` (queued prep behavior)
- `WILDFIRE_SCORING_PARAMETERS_PATH` (optional scoring/tuning parameter file, default `config/scoring_parameters.yaml`)
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
- Preferred (canonical): `scripts/prepare_region_from_catalog_or_sources.py` (plan/fill/build/validate in one command)
- Validation: `scripts/validate_prepared_region.py`
- Legacy/manual helpers (still available, but not the primary operator flow): `scripts/prepare_region_layers.py`, `scripts/stage_landfire_assets.py`, `scripts/build_landfire_region.py`, `scripts/catalog_ingest_raster.py`, `scripts/catalog_ingest_vector.py`, `scripts/build_region_from_catalog.py`
- Local queue worker: `scripts/run_region_prep_worker.py`

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

Preferred new-region workflow (canonical path):
- Runtime still reads prepared files only; it does not perform heavy GIS prep at request time.
- `scripts/prepare_region_from_catalog_or_sources.py` is the canonical entrypoint for new regions.
- The command checks existing prepared coverage, checks catalog coverage, acquires missing layers, builds the region, and can validate in one run.
- Default source registry: `config/source_registry.json`.
  - If `--source-config` is omitted, this registry is loaded automatically.
  - Override with `--source-config <path>` or `WF_SOURCE_CONFIG_PATH`.
  - Registry values support env references, including defaults, for example `${WF_DEFAULT_DEM_ENDPOINT:-https://...}`.
  - Required core layers (`dem`, `fuel`, `canopy`, `fire_perimeters`, `building_footprints`) ship with non-empty starter source details so `--plan-only` can evaluate buildability without custom config.
  - Optional defaults are included for `whp`, `mtbs_severity`, and `roads`; `gridmet_dryness` usually requires an explicit endpoint/URL override.
  - Optional layers remain non-blocking; missing/invalid optional config is surfaced in `optional_layer_diagnostics` and `optional_config_warnings`.

Required vs optional layers:
- Required core: `dem`, `fuel`, `canopy`, `fire_perimeters`, `building_footprints`
- Derived core: `slope` (from `dem`)
- Optional enrichment: `whp`, `mtbs_severity`, `gridmet_dryness`, `roads`
- Missing required layers fail the build; missing optional layers are reported as warnings/omissions.

Plan-only check:

```bash
python scripts/prepare_region_from_catalog_or_sources.py \
  --region-id missoula_pilot \
  --display-name "Missoula Pilot" \
  --bbox -114.2 46.75 -113.8 47.0 \
  --plan-only
```

Prepare/build/validate:

```bash
python scripts/prepare_region_from_catalog_or_sources.py \
  --region-id missoula_pilot \
  --display-name "Missoula Pilot" \
  --bbox -114.2 46.75 -113.8 47.0 \
  --prefer-bbox-downloads \
  --allow-full-download-fallback \
  --allow-partial-coverage-fill \
  --validate
```

Operator diagnostics in command output include:
- prepared-region status (`covered`, `not_found`, `present_outside_bbox`, `invalid_manifest`)
- catalog coverage sufficiency and acquisition plan
- required blockers vs optional omissions
- stage status (`prepared_region_check`, `coverage_plan`, `acquisition`, `region_build`, `validation`)
- per-layer execution diagnostics during run (`provider_type`, request mode, fetch/ingest success, failure reason, actionable error)
- compact summary (`final_status`, missing layers after run, validation status)

Manual uncovered-region workflow:
- Set `WF_REQUIRE_PREPARED_REGION_COVERAGE=true` to require prepared coverage for assessment requests.
- When uncovered, runtime returns `region_not_ready` with a suggested bbox.
- Operator runs the preferred prep command above.
- Retry `POST /risk/assess` after prep/validation completes.

Optional auto-queue workflow:
- Set `WF_AUTO_QUEUE_REGION_PREP_ON_MISS=true` to enqueue prep jobs automatically on uncovered addresses.
- Start local worker:

```bash
python scripts/run_region_prep_worker.py --once
```

Developer checklist:
1. Plan: run `prepare_region_from_catalog_or_sources.py --plan-only`.
2. Verify required-layer blockers and source registry values.
3. Execute with `--validate`.
4. Inspect `data/regions/<region_id>/manifest.json` (`catalog`, acquisition method, omissions, validation status).
5. If runtime still reports `region_not_ready`, run `POST /regions/coverage-check` for point-level coverage diagnostics.

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

## Event Backtesting

For empirical validation against labeled wildfire outcomes, run the event backtest harness:

- sample dataset: `benchmark/event_backtest_sample_v1.json`
- runner: `scripts/run_event_backtest.py`
- supported dataset formats: JSON, GeoJSON, CSV

Run with the sample dataset:

```bash
python scripts/run_event_backtest.py
```

Run with one or more custom datasets:

```bash
python scripts/run_event_backtest.py \
  --dataset path/to/event_a.json \
  --dataset path/to/event_b.csv \
  --output-dir benchmark/event_backtest_results
```

Backtest artifacts include:
- per-record scores, availability flags, confidence/use restriction, coverage/evidence summaries
- score distributions by outcome label
- rank correlations and risk-bucket adverse-outcome rates
- confidence stratification (`high_evidence`, `mixed_evidence`, `fallback_heavy`)
- false-low and false-high review sets
- deterministic tuning review recommendations (no auto-applied tuning)
- `model_governance` and aggregated version metadata

Interpretation guardrails:
- event labels are proxy outcomes, not carrier claims truth
- use results for directional calibration and threshold review
- do not treat fallback-heavy records as primary tuning anchors

## Model Tuning

The repo includes a deterministic tuning harness that turns event-backtest output into bounded, explainable parameter experiments.

- parameter file: `config/scoring_parameters.yaml`
- runner: `scripts/run_model_tuning.py`

Run with the sample backtest dataset:

```bash
python scripts/run_model_tuning.py
```

Run with custom datasets and enforce objective improvement:

```bash
python scripts/run_model_tuning.py \
  --dataset path/to/event_a.json \
  --dataset path/to/event_b.csv \
  --max-candidates 8 \
  --require-improvement
```

What the tuning harness does:
- computes structured false-low/false-high error analysis from backtest records
- evaluates bounded parameter variations (weights/thresholds) against rank, bucket, and false-rate metrics
- enforces monotonic guardrails before recommending any candidate
- writes JSON + markdown artifacts with:
  - `tuning_run_id`, `parameter_set_id`, timestamps, metrics
  - before/after comparison and recommended review changes
  - `model_governance` version metadata for release-to-release traceability

Governance guidance:
- tuning artifacts are recommendations only; no weights are auto-applied
- promote parameter changes only after benchmark + backtest review
- bump governance versions (`scoring_model_version`, `calibration_version`, etc.) when accepted tuning changes materially affect outputs

## Scoring Completeness And Fallbacks

Homeowner assessments now prioritize graceful degradation when a trusted geocode and supported prepared region are available.

- Hard blockers (assessment may return `insufficient_data`): no trusted geocode, no prepared-region coverage, or total absence of enough core evidence to score both site hazard and home vulnerability.
- Soft blockers (assessment still returns): missing/partial layers, nodata/outside-extent sampling, missing structure fields, and footprint/ring gaps.

Fallback hierarchy is deterministic and explicit:
- observed value
- derived proxy (for example ring-based defensible-space estimate)
- conservative or neutral default
- component exclusion with transparent note if evidence is still insufficient

Near-structure defensible-space behavior:
- Preferred geometry basis: building footprint rings (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`).
- Fallback geometry basis: point-proxy annulus sampling when footprint geometry is unavailable.
- New response fields:
  - `defensible_space_analysis` (zone metrics, basis geometry, mitigation flags, quality)
  - `top_near_structure_risk_drivers`
  - `prioritized_vegetation_actions`
  - `defensible_space_limitations_summary`

Response transparency is preserved in existing structures:
- `score_*_available` flags
- `assessment_status` and `assessment_blockers`
- `layer_coverage_audit` / `coverage_summary`
- `score_evidence_ledger` / `evidence_quality_summary`
- `assessment_diagnostics.fallback_decisions`
- `assessment_limitations_summary`

When one component is unavailable, `wildfire_risk_score` can still be computed from available component evidence, with explicit scoring notes and confidence penalties.

## Homeowner Reports

Completed assessments can be transformed into a homeowner-facing report and downloaded as PDF.

- JSON report view: `GET /report/{assessment_id}/homeowner`
- PDF download: `GET /report/{assessment_id}/homeowner/pdf`

Homeowner report sections include:
- property summary
- wildfire risk and insurance readiness score summary
- key risk drivers
- defensible-space zone findings and vegetation actions
- prioritized mitigation plan
- confidence and limitations summary
- model/region metadata

Use `include_professional_debug_metadata=true` on the homeowner JSON endpoint when you need internal diagnostics alongside consumer-facing content.
See `docs/homeowner_report.md` for details.

## Frontend Map

After a successful assessment, the frontend shows a map card with property and wildfire-context layers.

- Map payload endpoint: `GET /report/{assessment_id}/map`
- Geometry contract:
  - map geometries are GeoJSON in WGS84 (`EPSG:4326`) with `[longitude, latitude]` coordinates
  - `display_point_source` identifies whether the main property marker is from
    `matched_structure_centroid` (preferred when available) or `geocoded_address_point`
  - map payload includes geocode/structure-match diagnostics (`geocode_precision`,
    `structure_match_status`, `structure_match_method`, `structure_match_distance_m`,
    `candidate_structure_count`) for routing/alignment QA
- Graceful degradation:
  - if footprint geometry is unavailable, rings use point-proxy geometry
  - if overlays are unavailable, map still renders available layers with limitations text

See `docs/frontend_map.md` for payload details and layer behavior.

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
