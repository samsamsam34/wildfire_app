# Changelog

This file tracks release-level governance changes for `WildfireRisk Advisor`.

## [0.17.1] - 2026-04-17
### Version changes
- `product_version`: `0.17.1` (patch)
- `api_version`: `1.4.0` (unchanged)
- `scoring_model_version`: `1.9.0` (unchanged)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.3.0` (unchanged)
- `benchmark_pack_version`: `1.1.0` (patch; scenario pack updated to reflect expanded geocoding coverage)
- `calibration_version`: `0.3.0` (unchanged)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Phase 1a/1b: Add US Census TIGER geocoder as primary provider ahead of Nominatim, with a configurable `GeocodeFallbackChain`. Add ZIP centroid validation to reject cross-state geocoding mismatches (e.g. Winthrop WA → Dutchess County NY).
- Phase 2a: Add Regrid Terrain API parcel client with 90-day SQLite caching as a national on-demand fallback when no local parcel data exists. Add STRtree spatial index for O(log n) candidate pre-filtering on local parcel datasets. Properties outside prepared regions now receive a real parcel polygon (confidence 72) instead of a 25 m bounding-box approximation (confidence 18).

### Expected effect on outputs
- Geocoding precision improves for standard US residential addresses through Census TIGER interpolation as primary provider.
- ZIP validation prevents cross-state geocode mismatches from reaching the scoring engine; affected requests now advance to the next provider or surface a validation failure.
- Assessments for properties outside prepared regions that have `WF_REGRID_API_KEY` configured will receive a parcel-polygon-anchored result with confidence 72 rather than a bounding-box fallback.
- No change to confidence or output shape for assessments within prepared regions.

### Migration/interpretation notes
- Set `WF_REGRID_API_KEY` to enable national parcel coverage; the system degrades gracefully to the existing bounding-box fallback when the key is absent.
- Geocoding provider chain is configurable via `config/geocoding_config.yaml`; existing default behaviour (Nominatim) is preserved when Census TIGER returns no match.
- `benchmark_pack_version` bump reflects updated scenario coverage; benchmark scores from `1.0.0` packs remain valid for comparison within the same scoring model version.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.16.0] - 2026-03-11
### Version changes
- `product_version`: `0.16.0` (minor)
- `api_version`: `1.4.0` (minor; additive calibration metadata and status fields)
- `scoring_model_version`: `1.9.0` (unchanged)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.3.0` (unchanged)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.3.0` (minor; public-outcome ingest/evaluate workflow and runtime calibration status metadata)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Add outcome-based validation and optional empirical calibration workflow using public structure-damage records while preserving deterministic core scoring and explainability.

### Expected effect on outputs
- New public-outcome scripts support ingest, calibration-dataset building, and model evaluation against labeled structure outcomes.
- `/risk/assess` adds additive optional calibration fields (`empirical_damage_likelihood_proxy`, `calibration_status`, `calibration_limitations`, `calibration_scope_warning`).
- `/risk/debug` now returns richer calibration status/metadata, including out-of-scope warnings when applicable.

### Migration/interpretation notes
- Core deterministic score fields are unchanged; new calibration fields are additive and optional.
- Calibrated likelihood values are directional guidance derived from public outcome datasets and should not be treated as carrier claims probabilities.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.15.0] - 2026-03-11
### Version changes
- `product_version`: `0.15.0` (minor)
- `api_version`: `1.3.0` (minor; additive calibration/output metadata)
- `scoring_model_version`: `1.9.0` (minor; open-data feature integration and blend rebalance)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.3.0` (minor; new near-structure provenance fields)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.2.0` (minor; optional public-outcome calibration artifact support)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Improve property-level discrimination and competitiveness using open-data near-structure features (NAIP-derived ring metrics) while preserving deterministic explainability.

### Expected effect on outputs
- More separation across materially different properties via added near-structure continuity/canopy/high-fuel distance signals.
- `POST /risk/debug` includes richer parcel-level feature context through existing diagnostics surfaces.
- Optional `calibrated_damage_likelihood` can be returned when a public calibration artifact is configured.

### Migration/interpretation notes
- Historical score comparisons against `scoring_model_version` `1.8.x` should be treated as directional.
- Calibration remains optional; when enabled, treat calibrated likelihood as additive guidance, not underwriting truth.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.14.0] - 2026-03-11
### Version changes
- `product_version`: `0.14.0` (minor)
- `api_version`: `1.2.0` (minor; additive debug diagnostics fields)
- `scoring_model_version`: `1.8.0` (minor; variance/discriminativeness update)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.2.0` (unchanged)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.1.0` (unchanged)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Reduce score compression so property/location differences produce meaningful separation and improve model competitiveness for property-level wildfire guidance.

### Expected effect on outputs
- Wider spread in `wildfire_risk_score`, `site_hazard_score`, and `home_ignition_vulnerability_score` across materially different properties.
- Blended wildfire scoring now includes a bounded readiness-risk component.
- `POST /risk/debug` includes new score-variance diagnostics blocks (`score_variance_diagnostics`, raw/transformed vectors, contribution breakdown, compression flags).

### Migration/interpretation notes
- Historical comparisons against `scoring_model_version` `1.7.x` should be treated as directional due to transform/weight changes.
- Debug consumers can safely ignore new additive diagnostics fields if unused.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.13.0] - 2026-03-10
### Version changes
- `product_version`: `0.13.0` (minor)
- `api_version`: `1.1.0` (minor; homeowner report/PDF endpoints added)
- `scoring_model_version`: `1.7.0` (unchanged)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.2.0` (unchanged)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.1.0` (unchanged)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Add a homeowner-facing report view-model and downloadable PDF export so completed assessments can be shared as consumer-ready documents.

### Expected effect on outputs
- New endpoints: `GET /report/{assessment_id}/homeowner` and `GET /report/{assessment_id}/homeowner/pdf`.
- Existing report/export endpoints remain backward-compatible.
- No scoring formula changes in this release.

### Migration/interpretation notes
- Homeowner report payload is additive and separate from technical report export payloads.
- Clients can opt into `include_professional_debug_metadata=true` for advanced diagnostics in homeowner report JSON.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.12.0] - 2026-03-10
### Version changes
- `product_version`: `0.12.0` (minor)
- `api_version`: `1.0.0` (unchanged)
- `scoring_model_version`: `1.7.0` (minor; near-structure vegetation scoring updates)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (unchanged)
- `factor_schema_version`: `1.2.0` (minor; defensible-space analysis fields added)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.1.0` (unchanged)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Add structured defensible-space and near-structure vegetation analysis so homeowner outputs include zone-specific findings and mitigation actions.

### Expected effect on outputs
- Assessment responses now include `defensible_space_analysis`, `top_near_structure_risk_drivers`, `prioritized_vegetation_actions`, and `defensible_space_limitations_summary`.
- When footprint geometry is missing, point-proxy ring metrics can still be computed from canopy/fuel data, with explicit limitation notes.
- Home ignition and insurance readiness outcomes can shift due to new near-structure vegetation distance/zone signals.

### Migration/interpretation notes
- New defensible-space fields are additive and optional for clients.
- Compare historical assessments directionally across `scoring_model_version` `1.6.0` vs `1.7.0`.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.11.0] - 2026-03-09
### Version changes
- `product_version`: `0.11.0` (minor)
- `api_version`: `1.0.0` (unchanged)
- `scoring_model_version`: `1.6.0` (minor; completeness/fallback behavior update)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.1.0` (minor; eligibility and partial-scoring policy updates)
- `factor_schema_version`: `1.1.0` (minor; fallback decision diagnostics + limitations summary)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.1.0` (unchanged)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Improve homeowner scoring completeness so non-critical data gaps degrade confidence/diagnostics instead of failing scoring paths.

### Expected effect on outputs
- More assessments return partial or fully scored outputs when soft data gaps exist.
- `wildfire_risk_score` can remain available when one component score is unavailable, with transparent reweighting notes.
- Responses include `assessment_limitations_summary` and richer `assessment_diagnostics.fallback_decisions`.

### Migration/interpretation notes
- Downstream consumers should treat new fallback diagnostics as additive fields.
- Historical comparisons across `scoring_model_version` `1.5.0` and `1.6.0` are directional, not exact, due to fallback and availability policy changes.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.

## [0.10.0] - 2026-03-08
### Version changes
- `product_version`: `0.10.0` (minor)
- `api_version`: `1.0.0` (initial explicit API governance version)
- `scoring_model_version`: `1.5.0` (unchanged)
- `ruleset_version`: tracked per assessment ruleset
- `rules_logic_version`: `1.0.0` (unchanged)
- `factor_schema_version`: `1.0.0` (unchanged)
- `benchmark_pack_version`: `1.0.0` (unchanged)
- `calibration_version`: `0.1.0` (initialized governance dimension)
- `region_data_version`: tracked per assessment/region build
- `data_bundle_version`: `unversioned` default unless overridden

### Reason
- Introduce repo-wide model governance metadata to separate API/scoring/rules/benchmark/calibration/data changes.

### Expected effect on outputs
- Adds explicit `model_governance` metadata across health, assessments, debug/report export, and benchmark artifacts.
- Adds version-comparison helpers and compatibility labeling for cross-assessment comparisons.
- No scoring formula changes in this release.

### Migration/interpretation notes
- Scores from assessments with different `scoring_model_version`, `ruleset_version`, `rules_logic_version`, or
  `factor_schema_version` should be treated as not directly comparable.
- Differences in `calibration_version`, `region_data_version`, or `data_bundle_version` require review before
  operational comparison.

### Historical comparison validity
- `not_directly_comparable` when scoring/rules/schema dimensions differ.
- `comparable_with_review` when data/calibration dimensions differ.
- `directly_comparable` when governance dimensions match.
