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

## Homeowner-First Primary Fields

Primary homeowner-facing assessment fields now emphasize actionability:
- `overall_wildfire_risk`
- `home_hardening_readiness`
- `top_risk_drivers`
- `top_risk_drivers_detailed`
- `prioritized_mitigation_actions`
- `top_recommended_actions`
- `simulator_explanations` (on `/risk/simulate`)
- `confidence_summary`
- `specificity_summary` (property-specific vs address-level vs regional estimate status)
- `assumptions_and_unknowns`
- `confidence_tier`

Compatibility fields such as `insurance_readiness_score` remain available, but are treated as optional/future-facing in homeowner views.

## Main API Capabilities

Full route docs are available at `/docs` when running locally.

Homeowner primary flow:
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `GET /risk/improve/{assessment_id}` (homeowner input gap check + follow-up prompts)
- `POST /risk/improve/{assessment_id}` (rerun with added homeowner details + before/after summary)
- `POST /risk/simulate`
- `GET /report/{assessment_id}/homeowner`
- `GET /report/{assessment_id}/homeowner/pdf`

Homeowner report and map views:
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/view`
- `GET /report/{assessment_id}/map`

Advanced/internal (secondary surfaces):
- `POST /risk/assess?include_diagnostics=true` (opt-in trust metadata)
- `POST /risk/assess?include_calibrated_outputs=true` (opt-in calibrated public-outcome metadata)
- calibrated public-outcome metadata is optional/additive governance context; homeowner-facing guidance remains based on deterministic risk drivers, actions, specificity, and limitations
- `POST /risk/debug`
- `POST /risk/layer-diagnostics`
- `GET /report/{assessment_id}?include_diagnostics=true`
- `GET /report/{assessment_id}/export`
- `POST /regions/coverage-check` (prepared-region coverage diagnostics)
- `POST /regions/prepare`, `GET /regions/prepare/{job_id}` (offline region-prep jobs)

Portfolio and batch (secondary/internal):
- `POST /portfolio/assess`
- `POST /portfolio/jobs`, `GET /portfolio/jobs/{job_id}`, `GET /portfolio/jobs/{job_id}/results`
- `POST /portfolio/import/csv`
- `GET /portfolio`, `GET /assessments`, `GET /assessments/summary`

Review, governance, and operations (secondary/internal):
- annotations, review status, assignment, workflow, comparison, scenario history
- organizations and underwriting rulesets
- audit and summary endpoints (`/audit/events`, `/admin/summary`)
- internal diagnostics dashboard: `GET /internal/diagnostics` (internal trust metadata view)
- internal diagnostics APIs: `/internal/diagnostics/api/*` (including `/internal/diagnostics/api/public-outcomes` for validation/calibration governance snapshots)

Diagnostics opt-in note:
- `include_diagnostics=true` returns an envelope with:
  - `assessment` (the existing `AssessmentResult`)
  - `diagnostics` (no-ground-truth trust metadata)
- `include_calibrated_outputs=true` adds optional `calibrated_public_outcome_metadata` to `AssessmentResult`:
  - raw model scores remain unchanged and primary
  - calibrated values are public-outcome-based metadata only
- default behavior remains unchanged when the flag is absent.
- trust diagnostics are coherence/evidence metadata only, not claims-validation proof.

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
- `feature_coverage_summary`: preflight availability flags for parcel/footprint/hazard/burn/dryness/roads/near-structure metrics
- `feature_coverage_percent`: percent of core preflight features observed
- `assessment_specificity_tier`: `property_specific` | `address_level` | `regional_estimate`
- `limited_assessment_flag`: true when missing-core coverage forces a lower-specificity assessment path
- `feature_bundle`: canonical enrichment snapshot with `bundle_id`, cache hit status, and source-by-feature diagnostics
- `feature_bundle_data_sources`: simplified source map used by homeowner/debug surfaces (e.g., building footprint, parcel, vegetation, roads, climate)
- `feature_bundle_summary.coverage_metrics`: observed/inferred/fallback/missing feature counts plus
  `observed_weight_fraction`, `fallback_dominance_ratio`, `structure_geometry_quality_score`,
  `environmental_layer_coverage_score`, `regional_enrichment_consumption_score`,
  and `property_specificity_score`
- `feature_bundle_summary.geometry_provenance`: anchor quality (`property_anchor_quality`), anchor method,
  structure selection method, and geometry basis provenance
- `feature_bundle_summary.enrichment_runtime_status`: per-layer runtime classification
  (`not_configured`, `configured_but_fetch_failed`, `configured_but_no_coverage`,
  `present_but_not_consumed`, `present_and_consumed`)

`coverage_status` interpretation:
- `observed`: sampled successfully
- `not_configured`: no source configured for that layer
- `missing_file`: configured path is missing
- `outside_extent`: point/ring is outside layer coverage or sampled nodata
- `sampling_failed`: read/CRS/runtime sampling failure
- `fallback_used`: scoring fallback path was used
- `partial`: layer exists but only partial evidence is available
  - This now includes nearby raster sample recovery when point samples hit nodata/edge cells.

Region resolution fields:
- Assessment responses include `region_resolution` with `coverage_available`, `resolved_region_id`, `reason`, and `recommended_action`.
- Uncovered locations can return `region_not_ready` details (HTTP 409 when prepared coverage is required) or `insufficient_data` with `region_resolution.reason=no_prepared_region_for_location`.
- When footprint geometry is unavailable, defensible-space zone metrics can still run in point-proxy mode, and responses include explicit limitations.

Score variance diagnostics:
- `POST /risk/debug` now includes:
  - `score_variance_diagnostics`
  - `raw_feature_vector`
  - `transformed_feature_vector`
  - `factor_contribution_breakdown`
  - `compression_flags`
  - `fallback_dominance_ratio`
  - `missing_core_layer_count`
  - `observed_weight_fraction`
- Use these fields to inspect score compression, fallback-heavy factors, and contribution spread for a single assessment.

Fallback policy (current):
- Missing data now reduces specificity/confidence more than it flattens numeric scores.
- Missing factors are omitted from direct weighting where possible.
- Active factor weights are renormalized across observed evidence.
- Omission-driven uncertainty is surfaced as an explicit confidence penalty (`missing_factor_uncertainty`), not hidden as conservative numeric defaults.
- Developer note: see [docs/fallback_specificity_policy.md](docs/fallback_specificity_policy.md).
- Assumption-reduction map: [docs/assumption_reduction_gap_map.md](docs/assumption_reduction_gap_map.md).
- Batch variance check script:

```bash
python scripts/analyze_score_variance.py \
  --fixture tests/fixtures/score_variance_scenarios.json \
  --csv-out /tmp/score_variance.csv

# Open-model spread regression
python scripts/analyze_open_model_score_spread.py \
  --fixture tests/fixtures/score_variance_scenarios.json \
  --json-out /tmp/open_model_spread.json \
  --csv-out /tmp/open_model_spread.csv

# Assumption/fallback reduction analysis on saved assessment payloads
python scripts/analyze_assumption_reduction.py \
  --input /tmp/assessment_batch.json \
  --csv-out /tmp/assumption_reduction.csv
```

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
- `WF_TRUST_REFERENCE_ARTIFACT_DIR` (optional directory with precomputed no-ground-truth diagnostics artifacts for API percentile/alignment context)
- `WF_NO_GROUND_TRUTH_EVAL_DIR` (optional root for offline no-ground-truth evaluation runs used by internal dashboard APIs)
- Optional open-data runtime sources:
  - `WF_LAYER_WHP_TIF`
  - `WF_LAYER_MTBS_SEVERITY_TIF`
  - `WF_LAYER_GRIDMET_DRYNESS_TIF`
  - `WF_LAYER_OSM_ROADS_GEOJSON`
  - `WF_LAYER_NAIP_IMAGERY_TIF`
  - `WF_LAYER_NAIP_STRUCTURE_FEATURES_JSON`
  - `WF_LAYER_FEMA_STRUCTURES_GEOJSON`
  - `WF_LAYER_BUILDING_FOOTPRINTS_OVERTURE_GEOJSON`
  - `WF_LAYER_BUILDING_FOOTPRINTS_MICROSOFT_GEOJSON`
  - `WF_LAYER_ADDRESS_POINTS_GEOJSON`
  - `WF_LAYER_PARCELS_GEOJSON`
  - `WF_LAYER_PARCEL_ADDRESS_POINTS_GEOJSON`
  - `WF_LAYER_PARCEL_POLYGONS_GEOJSON`
- Geometry source selection:
  - `WF_GEOMETRY_SOURCE_REGISTRY_PATH` (optional override path; default `config/geometry_source_registry.json`)
  - `WF_BUILDING_SOURCE_PRIORITY` (legacy fallback order when region manifest source order is unavailable)
  - `WF_OVERTURE_BUILDINGS_VERSION` (optional version tag shown in debug metadata)
  - `WF_POINT_SELECTION_USE_PARCEL_CONTEXT` (default `true`; when true, point-selected structure matching uses parcel association if available)
  - `WF_NAIP_FEATURE_MATCH_MAX_DISTANCE_M` (default `45`; max nearest-centroid NAIP feature fallback distance)
- Runtime enrichment fallback source hooks (optional, file-path based):
  - `WF_ENRICH_OVERTURE_BUILDINGS_PATH`, `WF_ENRICH_MICROSOFT_BUILDINGS_PATH`
  - `WF_ENRICH_PARCELS_PATH`, `WF_ENRICH_ADDRESS_POINTS_PATH`
  - `WF_ENRICH_LANDFIRE_FUEL_TIF`, `WF_ENRICH_LANDFIRE_CANOPY_TIF`
  - `WF_ENRICH_MTBS_PERIMETERS_GEOJSON`, `WF_ENRICH_MTBS_SEVERITY_TIF`
  - `WF_ENRICH_OSM_ROADS_GEOJSON`
  - `WF_ENRICH_GRIDMET_DRYNESS_TIF`
  - `WF_ENRICH_NAIP_IMAGERY_TIF`, `WF_ENRICH_NAIP_STRUCTURE_FEATURES_JSON`
- Property feature-bundle cache:
  - `WF_FEATURE_BUNDLE_CACHE_ENABLED` (default `true`)
  - `WF_FEATURE_BUNDLE_CACHE_READ` / `WF_FEATURE_BUNDLE_CACHE_WRITE` (default `true`)
  - `WF_FEATURE_BUNDLE_CACHE_DIR` (default `data/cache/feature_bundles`)
  - `WF_FEATURE_BUNDLE_CACHE_TTL_SEC` (default `21600`)
- Optional public-outcome calibration:
  - `WF_PUBLIC_CALIBRATION_ARTIFACT` (path to logistic/piecewise calibration artifact JSON)
- Geocoding trust controls:
  - `WF_GEOCODE_ALLOW_LOW_CONFIDENCE_FALLBACK` (default: `true`, allows medium-confidence fallback when a candidate has usable coordinates)
  - `WF_GEOCODE_ALLOW_AMBIGUOUS_FALLBACK` (default: `false`)
  - `WF_GEOCODE_SECONDARY_ENABLED` + `WF_GEOCODE_SECONDARY_SEARCH_URL` (optional secondary provider stage)
  - `WF_GEOCODE_SECONDARY_PROVIDER_NAME`, `WF_GEOCODE_SECONDARY_USER_AGENT` (secondary provider metadata)
  - `WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY` (default: `true`, enables street/locality backoff queries when full-address lookup returns no match)
  - `WF_LOCAL_ADDRESS_FALLBACK_PATH` (optional local alias/address-point file, default: `config/local_address_fallbacks.json`)
  - `WF_LOCAL_ADDRESS_MATCH_MIN_SCORE` (default: `0.76`, fuzzy-match threshold for local fallback candidates)
  - `WF_LOCATION_RESOLUTION_SOURCE_CONFIG` (optional source config, default: `config/location_resolution_sources.json`)
  - `WF_WA_STATEWIDE_PARCEL_PATH` (optional direct path override for Washington statewide parcel/address dataset exports)
  - `WF_OKANOGAN_ADDRESS_POINTS_PATH` (optional direct path override for Okanogan local addressing exports)
  - `WF_RESOLVER_CONFLICT_DISTANCE_M` (default: `1500`, cross-source disagreement distance threshold for ambiguity protection)
  - `WF_RESOLVER_CONFLICT_SCORE_MARGIN` (default: `18`, score-gap threshold used with disagreement distance checks)
  - `WF_RESOLVER_IN_REGION_BOOST` (default: `35`, score boost for candidates inside prepared-region coverage)
  - `WF_RESOLVER_AUTHORITATIVE_SOURCE_BONUS` (default: `18`, bonus for county/prepared/parcel authoritative sources)
  - `WF_RESOLVER_CLEAR_WINNER_MIN_MARGIN` (default: `12`, minimum score margin to auto-promote a clearly best in-region medium candidate)
  - `WF_RESOLVER_CLEAR_WINNER_MIN_SCORE` (default: `230`, minimum score required for clear-winner auto-promotion)
  - `WF_RESOLVER_IN_REGION_PREFERENCE_MARGIN` (default: `18`, allows close-score in-region candidates to outrank outside-region candidates)
  - `WF_RESOLVER_MIN_AUTO_CANDIDATE_SCORE` (default: `150`, minimum candidate rank score for automatic coordinate use)
  - `WF_RESOLVER_MIN_GEOCODER_TOKEN_COVERAGE` (default: `0.72`, minimum submitted-token coverage for geocoder candidate safety checks)
  - `WF_RESOLVER_EMERGENCY_IN_REGION_MEDIUM_AUTORESOLVE` (default: `true`, emergency guardrail to auto-resolve clearly best in-region medium candidates)
  - `WF_RESOLVER_EMERGENCY_MIN_SCORE` (default: `155`, guardrail minimum score)
  - `WF_RESOLVER_EMERGENCY_MIN_MARGIN` (default: `8`, guardrail minimum margin over next candidate)
  - `WF_RESOLVER_ALLOW_INTERPOLATED_AUTO` (default: `true`, controls whether interpolated geocoder candidates can auto-resolve)
  - `WF_RESOLVER_MIN_GEOCODER_TOKEN_SIMILARITY` (default: `0.55`, minimum token-similarity required before geocoder-only candidates can auto-resolve property coordinates)
  - `WF_REGION_EDGE_TOLERANCE_M` (default: `0`, optional region-boundary tolerance for near-edge points)
- `WILDFIRE_APP_CACHE_ROOT`, `WILDFIRE_APP_DATA_ROOT`, `WILDFIRE_APP_TMP_ROOT` (offline prep script paths)

Legacy direct-layer paths are still supported via `WF_LAYER_*` env vars (`DEM`, `SLOPE`, `FUEL`, `CANOPY`, fire perimeters, building footprints, etc.), but prepared-region runtime is the primary path.

Address resolution uses a staged, confidence-scored pipeline:
1. Primary geocoder
2. Optional secondary geocoder
3. Local authoritative datasets (`address_points`, `parcel_address_points`, `parcels`, plus optional configured statewide/county sources)
4. Statewide parcel-address lookup (when configured)
5. Explicit local fallback records (`config/local_address_fallbacks.json`)
6. Provider backoff query variants
7. User-selected `property_anchor_point` fallback (for assessment routes)

The runtime now evaluates these stages separately:
1. Address existence validation (`address_exists`, `address_confidence`)
2. Coordinate resolution (`final_coordinates_used`, `coordinate_source`, `coordinate_confidence`)
3. Prepared-region containment check (`coverage_available`, `resolved_region_id`)
4. Assessment execution (only after the first 3 stages succeed)

Candidate guardrails:
- only `high`/`medium` confidence candidates are auto-used for property coordinates
- street-only/locality-only matches are not auto-used
- cross-source disagreement checks prevent silently selecting materially conflicting candidates
- geocoder-only candidates with weak address-token similarity are downgraded and require confirmation
- geocoder similarity checks use both token overlap similarity and submitted-token coverage to avoid false downgrades from extra provider tokens
- if a candidate is outside prepared coverage, resolver keeps searching later stages before finalizing
- if multiple prepared regions contain a point, the smallest covering region is selected

Geocode outcomes remain explicit:
- `geocode_failed`: no usable location after all enabled stages
- `geocode_succeeded_untrusted`: fallback location was used (secondary/local/user point)
- `geocode_succeeded_trusted`: accepted trusted match from a geocoder/local authoritative source

Resolver status labels:
- `resolved_high_confidence`
- `resolved_medium_confidence`
- `ambiguous_conflict` (requires map confirmation)
- `candidates_found_but_not_safe_enough`
- `unresolved`

Error classes are now explicit:
- `address_not_found`: no candidate from any enabled source
- `address_unresolved`: address appears to exist but no safe coordinates were auto-selected
- `outside_prepared_region`: coordinates resolved but no prepared region contains the point
- `ready_for_assessment`: coordinates resolved and point is inside a prepared region

`unsupported_location` is only returned for `outside_prepared_region` cases (resolved coordinates outside prepared coverage). Invalid or unresolved addresses are no longer collapsed into unsupported-location responses.

`/risk/assess`, `/risk/debug`, and `/regions/coverage-check` now share this same canonical resolution flow and expose:
- `resolution_status`, `resolution_method`, `fallback_used`
- `error_class`, `address_exists`, `address_confidence`, `address_validation_sources`
- `provider_attempts`, `provider_statuses`
- `candidate_sources_attempted`, `candidates_found`
- `final_coordinates_used`, `final_coordinate_source`, `coordinate_confidence`
- `candidate_regions_containing_point`
- `local_fallback_attempted`, `authoritative_fallback_result`, `local_fallback_result`
- `resolver_candidates` and `candidate_disagreement_distances` in debug payloads

For local geocode/trust troubleshooting (including Winthrop edge cases), use:
- `POST /risk/geocode-debug` (or `/debug/geocode`) for candidate/trust/region diagnostics
- `POST /risk/address-candidates` for ZIP/locality-based manual address disambiguation candidates before map-click fallback
- `python scripts/debug_geocode_trust_pipeline.py --address "6 Pineview Rd, Winthrop, WA 98862" --pretty`
- `python scripts/debug_address_resolution.py --address "6 Pineview Rd, Winthrop, WA 98862" --pretty`
- `python scripts/debug_address_resolution.py --csv path/to/wa_addresses.csv --pretty`
- `python scripts/ingest_county_address_points.py --input path/to/okanogan_address_points.csv --output data/address_points/okanogan/okanogan_address_points.geojson`

Frontend verify-before-submit flow now branches as:
1. auto verification succeeds -> confirm and run assessment
2. auto verification uncertain -> show manual `Choose your address` candidate list (ZIP/locality-aware), then confirm selected candidate
3. no usable candidates -> show `Can’t find it? Click your home on the map` fallback

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
  # optional primary building source
  building_footprints_overture.geojson
  # optional backup footprint sources
  building_footprints_microsoft.geojson
  fema_structures.geojson
  # optional region-specific parcel override
  parcel_polygons_override.geojson
  # optional enrichment layers
  whp.tif
  mtbs_severity.tif
  gridmet_dryness.tif
  roads.geojson
  manifest.json
```

Geometry source registry workflow:
- Prepared-region manifests now include `geometry_source_manifest` with explicit:
  - `parcel_sources`
  - `footprint_sources`
  - `default_source_order`
  - `source_versions`
  - `known_limitations`
- Runtime geometry resolution reads region-specific source precedence from this manifest instead of ad hoc fallback.
- `building_sources` and `parcel_sources` remain in manifest for backward compatibility and quick inspection.
- Structure-match diagnostics include:
  - `structure_match_method` (`parcel_intersection` or `nearest_building_fallback`)
  - `matched_structure_id`
  - `building_source`, `building_source_version`, `building_source_confidence`
  - `structure_match_distance_m` (used to flag potential alignment issues)
- Full registry reference: [`docs/geometry_source_registry.md`](docs/geometry_source_registry.md)

Optional public-record structure enrichment:
- Regions can optionally provide parcel/address public-record attributes (for example `year_built`, `land_use`, gross building area, assessor roof/material fields).
- Runtime normalizes these into `property_level_context.structure_attributes`:
  - `year_built`
  - `building_area_sqft`
  - `land_use_class`
  - `roof_material_public_record`
  - `attribute_provenance`
  - `attribute_confidence`
- Field provenance is explicit and constrained to:
  - `observed_public_record`
  - `inferred_from_geometry`
  - `user_provided`
  - `missing`
- These enrichments are optional by region; missing public-record coverage does not block assessment.

Property confidence (separate from wildfire model confidence):
- Assessments now include a dedicated property-identification confidence ladder in `property_confidence_summary`:
  - `verified_property_specific`
  - `strong_property_specific`
  - `address_level`
  - `regional_estimate_with_anchor`
  - `insufficient_property_identification`
- This score measures how confidently the system identified the actual home geometry (anchor/parcel/footprint consistency, local feature availability, and user confirmation), not wildfire outcome confidence.
- Low property-confidence levels automatically cap nearby-home comparison behavior and recommend geometry-correction actions.

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
  - Optional defaults are included for `whp`, `mtbs_severity`, `roads`, and `gridmet_dryness` (annual fm1000 NetCDF URL; override as needed).
  - Optional layers remain non-blocking; missing/invalid optional config is surfaced in `optional_layer_diagnostics` and `optional_config_warnings`.

Required vs optional vs enrichment layers:
- Required core: `dem`, `fuel`, `canopy`, `fire_perimeters`, `building_footprints`
- Derived core: `slope` (from `dem`)
- Optional context: `whp`, `mtbs_severity`, `gridmet_dryness`, `roads`
- Enrichment hooks: `building_footprints_overture`, `parcel_polygons`, `parcel_address_points`, `naip_imagery`
- Missing required layers fail the build; missing optional layers are reported as warnings/omissions.
- A region can be valid for core scoring while still not `property_specific_ready`.

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

Missoula helper wrapper (larger default bbox, overwrite enabled):

```bash
bash scripts/download_landfire_missoula.sh
```

Optional environment overrides for the helper:
- `WF_BBOX_MIN_LON`, `WF_BBOX_MIN_LAT`, `WF_BBOX_MAX_LON`, `WF_BBOX_MAX_LAT`
- `WF_VALIDATE_AFTER_PREP=1` (runs post-prep validation)
- `WF_DOWNLOAD_TIMEOUT`, `WF_DOWNLOAD_RETRIES`

Winthrop/WA optional-layer bootstrap (WHP/MTBS/geoplatform endpoints + gridMET + WA parcels + Overture fallback):

```bash
bash scripts/download_optional_winthrop_wa.sh
```

Optional overrides for the Winthrop helper:
- `WF_ALLOW_PARTIAL_COVERAGE_FILL=1` to force re-acquisition of partially covered layers
- `WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT`, `WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT`
- `WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT`, `WF_DEFAULT_OVERTURE_BUILDINGS_PATH`
- `WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL`

Operator diagnostics in command output include:
- prepared-region status (`covered`, `not_found`, `present_outside_bbox`, `invalid_manifest`)
- catalog coverage sufficiency and acquisition plan
- required blockers vs optional omissions
- per-layer status classification:
  - `not_configured`
  - `configured_but_fetch_failed`
  - `configured_but_outside_coverage`
  - `configured_but_empty_result`
  - `optional_and_skipped`
  - `present_from_existing_catalog`
  - `present_after_acquisition`
- stage status (`prepared_region_check`, `coverage_plan`, `acquisition`, `region_build`, `validation`)
- per-layer execution diagnostics during run (`provider_type`, request mode, fetch/ingest success, failure reason, actionable error)
- compact summary (`final_status`, missing layers after run, validation status, property-specific readiness)
- `operator_next_steps` with:
  - core build status
  - optional/enrichment status
  - source-config blockers
  - exact rerun commands
  - env override keys and registry keys to fill

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
4. Inspect `data/regions/<region_id>/manifest.json` (`catalog`, acquisition method, omissions, `property_specific_readiness`, validation summary).
   - Region onboarding readiness is explicitly reported under `catalog.property_specific_readiness`:
     - `parcel_ready`
     - `footprint_ready`
     - `parcel_footprint_linkage_quality`
     - `naip_ready` (NAIP structure-feature artifact prepared)
     - `structure_enrichment_ready` (public-record structure enrichment availability)
     - `overall_readiness` (`property_specific` | `address_level` | `limited_regional`)
5. If runtime still reports `region_not_ready`, run `POST /regions/coverage-check` for point-level coverage diagnostics.
6. See `docs/region_onboarding_readiness.md` for readiness interpretation and operator runbook.

Example source overrides (commonly needed before rerun):

```bash
export WF_DEFAULT_GRIDMET_DRYNESS_FULL_URL="https://www.northwestknowledge.net/metdata/data/fm1000_2026.nc"
export WF_DEFAULT_OVERTURE_BUILDINGS_ENDPOINT="https://<org>/arcgis/rest/services/<overture_layer>/FeatureServer/0"
export WF_DEFAULT_PARCEL_POLYGONS_ENDPOINT="https://<org>/arcgis/rest/services/<parcel_polygons>/FeatureServer/0"
export WF_DEFAULT_PARCEL_ADDRESS_POINTS_ENDPOINT="https://<org>/arcgis/rest/services/<parcel_points>/FeatureServer/0"
```

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
- `benchmark/scenario_pack_confidence_v2.json` (geometry/enrichment/fallback-confidence stress pack)
- `benchmark/scenario_pack_nearby_differentiation_v1.json` (adjacent-home local-differentiation stress pack)
- `benchmark/scenario_pack_nearby_differentiation_v2.json` (release-grade adjacent-home differentiation + honest-abstention pack)
- runner: `scripts/run_benchmark_suite.py`
- confidence runner: `scripts/run_confidence_benchmark_pack.py`

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

Run the confidence-focused pack (score spread + fallback/suppression diagnostics):

```bash
python scripts/run_confidence_benchmark_pack.py
```

Run the nearby-home differentiation pack (local-factor separation + cautious missing-geometry behavior):

```bash
python scripts/run_benchmark_suite.py \
  --pack benchmark/scenario_pack_nearby_differentiation_v2.json
```

Run the nearby-home release gate (CI-safe blocking mode):

```bash
python scripts/run_benchmark_suite.py \
  --nearby-suite \
  --enforce-nearby-release-gate
```

What it checks:
- scenario expectations (risk band, confidence tier/restriction, fallback behavior, warnings)
- relative ordering assertions
- monotonic sanity assertions (for mitigation and insurer-facing directional logic)
- nearby-home differentiation assertions (local sub-score separation and confidence/differentiation caution under missing geometry)
- release drift summary (score deltas, confidence deltas, warnings/blockers, factor contribution shifts)

Benchmark artifacts include aggregated governance metadata, a `model_governance` block, and
`nearby_differentiation_performance` when nearby-home scenarios are present.
When nearby scenarios are present, the runner also writes:
- `benchmark_run_<timestamp>_nearby_release_gate.json`
- `benchmark_run_<timestamp>_nearby_release_gate.md`

These summarize:
- separation success rate
- false similarity cases (collapsed without low-specificity warning)
- abstention success rate when data is weak

Internal property-specificity scorecard (latest vs previous baseline):

```bash
python scripts/build_property_specificity_scorecard.py
```

This writes:
- `benchmark/property_specificity_scorecard/<run_id>/property_specificity_scorecard.json`
- `benchmark/property_specificity_scorecard/<run_id>/summary.md`

Scorecard metrics include:
- percent `property_specific`, `address_level`, `regional_estimate`
- percent with footprint match
- percent with NAIP structure features (from multi-region runtime artifacts)
- percent passing nearby-home differentiation assertions

To pin current/baseline artifacts explicitly:

```bash
python scripts/build_property_specificity_scorecard.py \
  --benchmark-artifact benchmark/results/benchmark_run_YYYYMMDDTHHMMSSZ.json \
  --baseline-benchmark-artifact benchmark/results/benchmark_run_YYYYMMDDTHHMMSSZ.json \
  --multi-region-artifact benchmark/multi_region_runtime/multi_region_regression_YYYYMMDDTHHMMSSZ.json \
  --baseline-multi-region-artifact benchmark/multi_region_runtime/multi_region_regression_YYYYMMDDTHHMMSSZ.json
```

Use `POST /risk/debug?include_benchmark_hints=true` or `GET /report/{assessment_id}/export?include_benchmark_hints=true` to include lightweight benchmark resemblance and sanity-check hints in diagnostics/export output.

## No-Ground-Truth Evaluation

When outcome labels are sparse or unavailable, run the trustworthiness/coherence evaluation workflow:

```bash
python scripts/run_no_ground_truth_evaluation.py
```

This writes reproducible artifacts under `benchmark/no_ground_truth_evaluation/<run_id>/`:

- `evaluation_manifest.json`
- `monotonicity_results.json`
- `counterfactual_results.json`
- `stability_results.json`
- `distribution_results.json`
- `benchmark_alignment_results.json`
- `confidence_diagnostics.json`
- `summary.md`

Use this as a directional model-quality check only. It does not claim predictive accuracy.
Confidence diagnostics are evidence-quality signals: confidence should decrease as fallback, inferred/proxy,
or missing critical evidence increases.

See `docs/no_ground_truth_evaluation.md` for commands, caveats, and interpretation guidance.
See `docs/api_diagnostics.md` for API opt-in diagnostics fields and caveats.
See `docs/internal_diagnostics_dashboard.md` for internal dashboard usage and caveats.

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

## Public Outcome Calibration

The model supports an optional, additive empirical calibration workflow using public structure-damage outcomes (for example CAL FIRE DINS-style datasets). The deterministic scoring engine remains the primary scorer.

For the reproducible trust-validation bundle, use the single orchestrator:

```bash
python scripts/run_public_outcome_validation.py
```

This writes a validation bundle under `benchmark/public_outcomes/validation/<timestamp>/` using the latest labeled dataset from `benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl` (or an explicitly supplied dataset path).

End-to-end workflow:

```bash
# 1) Normalize public structure-damage outcomes
python scripts/ingest_public_outcomes.py \
  --input path/to/public_damage_records.csv

# 2) Build labeled public-outcome evaluation dataset
python scripts/build_public_outcome_evaluation_dataset.py \
  --outcomes benchmark/public_outcomes/normalized/<run_id>/normalized_outcomes.json \
  --feature-artifact benchmark/event_backtest_results/event_backtest_YYYYMMDDTHHMMSSZ.json

# 3) Validate discrimination/calibration against public observed outcomes
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl

# 4) Check whether the current dataset supports predictive modeling (directional viability)
python scripts/evaluate_dataset_model_viability.py \
  --evaluation-dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl

# 5) Fit optional calibration artifact bundle
python scripts/fit_public_outcome_calibration.py \
  --dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl \
  --output config/public_outcome_calibration.json
```

Synthetic sensitivity workflow (controlled variation, not ground truth):

```bash
python scripts/run_synthetic_sensitivity_evaluation.py \
  --evaluation-dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl
```

This writes synthetic sensitivity artifacts under:
- `benchmark/public_outcomes/synthetic_sensitivity/<run_id>/`
- Includes synthetic scenario rows (`vegetation_up`, `slope_up`, `fuel_near`, `combined_high`, `mitigation_low`),
  baseline-vs-synthetic metric comparison, and directionality pass/fail checks.

Calibration bundles are written under:
- `benchmark/public_outcomes/calibration/<run_id>/`
- Includes `calibration_model.json`, `pre_vs_post_metrics.json`, `calibration_curve.json`, `summary.md`, and `manifest.json`

Runtime calibration (optional):

```bash
export WF_PUBLIC_CALIBRATION_ARTIFACT=config/public_outcome_calibration.json
```

When enabled, `/risk/assess` and `/risk/debug` include calibration metadata (`calibration_status`, method/artifact metadata, and calibrated likelihood proxies) while preserving deterministic raw scores and evidence outputs.

See `docs/public_outcome_calibration.md` for details and caveats.
See `docs/api_calibrated_outputs.md` for API opt-in behavior and response examples.
See `docs/public_outcome_validation.md` for the v1 reproducible validation workflow and guardrails.
See `docs/synthetic_sensitivity_evaluation.md` for synthetic variation sensitivity checks and caveats.
See `docs/calibration_gap_analysis.md` for the empirical-gap rationale behind this calibration step.

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

Homeowner assessments now prioritize graceful degradation when a usable geocode and supported prepared region are available.

- Hard blockers (assessment may return `insufficient_data`): no usable location after all enabled resolution stages, no prepared-region coverage, or total absence of enough core evidence to score both site hazard and home vulnerability.
- Soft blockers (assessment still returns): missing/partial layers, nodata/outside-extent sampling, missing structure fields, and footprint/ring gaps.
- Low-confidence geocode candidates can continue as `geocode_succeeded_untrusted` with `trusted_match_status=untrusted_fallback`, explicit diagnostics, and confidence penalties.

Fallback hierarchy is deterministic and explicit:
- observed value
- derived proxy (for example ring-based defensible-space estimate)
- conservative or neutral default
- component exclusion with transparent note if evidence is still insufficient

Near-structure defensible-space behavior:
- Preferred geometry basis: building footprint rings (`0-5 ft`, `5-30 ft`, `30-100 ft`, `100-300 ft`).
- Fallback geometry basis: point-proxy annulus sampling when footprint geometry is unavailable.
- Scoring emphasis: dense vegetation in `0-5 ft` is weighted more strongly than wider rings in ignition/vulnerability logic, while `5-30 ft` remains a secondary spread-pressure signal.
- Mitigation sensitivity: clearing the `0-5 ft` zone is expected to produce a larger risk reduction than equivalent `5-30 ft` thinning in otherwise similar conditions.
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
- `assessment_mode` (`property_specific` | `address_level` | `limited_regional_estimate` | `insufficient_data`)
- `homeowner_summary` (grouped limitations, observed/estimated/missing rollups, concise confidence headline)
- `developer_diagnostics` (full technical fallback/source diagnostics without homeowner-facing noise)

When one component is unavailable, `wildfire_risk_score` can still be computed from available component evidence, with explicit scoring notes and confidence penalties.

Component scoring status is now explicit:
- `scoring_status`:
  - `full_property_assessment`
  - `limited_homeowner_estimate`
  - `insufficient_data_to_score`
- `computed_components` and `blocked_components` list which score families were actually computed.
- `minimum_missing_requirements` and `recommended_data_improvements` explain exactly what blocked higher-specificity scoring.

`assessment_mode` / `assessment_output_state` remain backward-compatible and continue to represent specificity tiering.

Adaptive scoring now separates three evidence families before blending:
- `regional_context_score` (hazard/burn/dryness/terrain context)
- `property_surroundings_score` (near-home vegetation/fuel pressure)
- `structure_specific_score` (defensible space + structure hardening)

Weighting is evidence-aware:
- missing-factor submodels are omitted or downweighted instead of filled with broad numeric defaults
- footprint/parcel gaps downweight structure-specific factors more aggressively
- fallback-heavy runs surface both `fallback_dominance_ratio` and `fallback_weight_fraction`
- blended wildfire risk blends an additive core with an explicit hazard × structure-vulnerability interaction core
- blended wildfire risk also applies a bounded contrast-expansion step to reduce mid-band score compression

For diagnostics and QA, inspect:
- `factor_breakdown.component_scores`
- `factor_breakdown.component_weight_fractions`
- `weighted_contributions[*].basis` / `support_level` / `component`
- `developer_diagnostics.adaptive_component_scores`

## Homeowner Reports

Completed assessments can be transformed into a homeowner-facing report and downloaded as PDF.

- JSON report view: `GET /report/{assessment_id}/homeowner`
- PDF download: `GET /report/{assessment_id}/homeowner/pdf`

Homeowner report sections include:
- property summary
- wildfire risk and home hardening readiness summary
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
  - map payload separates `geocoded_address_point`, `property_anchor_point`,
    optional `parcel_polygon`/`parcel_address_point`, optional `user_selected_point`,
    and `matched_structure_centroid`
  - `display_point_source` identifies whether the main marker is from
    `matched_structure_centroid` (high-confidence only) or `property_anchor_point`
  - selection fallback fields include `selection_mode`, `final_structure_geometry_source`,
    `structure_geometry_confidence`, and `snapped_structure_distance_m`
  - point snaps can report `structure_selection_method=point_parcel_intersection_snap`
    when parcel-assisted structure matching is used
  - map payload includes geocode/structure-match diagnostics (`geocode_precision`,
    `parcel_lookup_method`, `parcel_lookup_distance_m`, `structure_match_status`,
    `structure_match_method`, `matched_structure_id`, `structure_match_distance_m`,
    `candidate_structure_count`) for routing/alignment QA
- Graceful degradation:
  - if footprint geometry is unavailable, rings use point-proxy geometry
  - if selectable polygons are unavailable or weakly matched, user point selection can be
    used as the property anchor without forcing an incorrect building snap
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
