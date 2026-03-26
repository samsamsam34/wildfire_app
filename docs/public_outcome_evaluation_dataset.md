# Public Outcome Evaluation Dataset Builder

This workflow builds a reproducible labeled dataset by joining:

- historical model score/feature artifacts (event backtests)
- normalized public observed wildfire structure-damage outcomes

It is for directional validation/calibration support. It is not insurer claims validation.

## Command

```bash
python scripts/build_public_outcome_evaluation_dataset.py \
  --outcomes benchmark/public_outcomes/normalized/<run_id>/normalized_outcomes.json \
  --feature-artifact benchmark/event_backtest_results/event_backtest_<stamp>.json
```

Multiple inputs are supported:

- repeat `--outcomes` to aggregate multiple normalized outcome files (multi-event/multi-year).
- repeat `--feature-artifact` to aggregate multiple scored feature artifacts.
- optionally use discovery flags:
  - `--outcomes-root benchmark/public_outcomes/normalized --outcomes-run-id <run_id>` (repeat run ids)
  - `--outcomes-root benchmark/public_outcomes/normalized --outcomes-root-mode all` (use all normalized runs)
  - `--feature-artifact-dir <dir> --feature-artifact-glob "*.json"`

If feature artifacts do not already contain model score fields, the builder can auto-backfill
scores by running event-backtest scoring directly on those records:

```bash
python scripts/build_public_outcome_evaluation_dataset.py \
  --outcomes benchmark/public_outcomes/normalized/<run_id>/normalized_outcomes.json \
  --feature-artifact benchmark/event_backtest_sample_v1.json \
  --auto-score-missing
```

Use `--no-auto-score-missing` to disable this behavior.

Rapid coverage mode (recommended when validation sample size is too small):

```bash
python scripts/build_public_outcome_evaluation_dataset.py \
  --rapid-max-coverage \
  --outcomes-root benchmark/public_outcomes/normalized \
  --run-id public_eval_ds_rapid_coverage
```

`--rapid-max-coverage` does all of the following:
- uses all normalized outcomes runs under `--outcomes-root`
- discovers all auto-scored feature artifacts under `--feature-artifact-search-root`
- includes normalized outcomes as additional feature-artifact candidates
- allows duplicate outcome matches (coverage-first mode) for larger dataset size

## Join Logic (priority order)

1. `exact_parcel_event`
2. `exact_source_record_id`
3. `exact_event_record_id`
4. `exact_event_address`
5. `approx_event_address_token_overlap` (bounded by `--address-token-overlap-min`)
6. `exact_event_coordinates` (bounded by `--exact-match-distance-m`)
7. `near_event_coordinates` (bounded by `--near-match-distance-m`)
8. `extended_event_coordinates` (bounded by `--max-distance-m`)
9. `buffered_event_coordinates` (bounded by `--buffer-match-radius-m`)
10. `nearest_event_name_coordinates_tolerant_year` (bounded, year tolerance configurable)
11. `approx_global_address_token_overlap` (global fallback using address token overlap)
12. `nearest_global_coordinates` (bounded by `--global-max-distance-m`, optional via `--enable-global-nearest-fallback`)

Each joined row includes:

- `join_method`
- `join_confidence_score`
- `join_confidence_tier`
- `match_tier` (`exact` / `near` / `extended` / `fallback`)
- `join_distance_m`
- distance-tier thresholds used for classification:
  - `high`: `< 20m` (configurable via `--high-confidence-distance-m`)
  - `moderate`: `20–100m` (configurable via `--medium-confidence-distance-m`)
  - `low`: `> 100m` (or fallback methods capped to lower tiers)
- `event_year_consistent`
- `evaluation.row_confidence_tier` (`high-confidence` / `medium-confidence` / `low-confidence`)

Coordinate handling:
- incoming coordinates are normalized to WGS84 latitude/longitude with fixed precision
- invalid/out-of-range coordinates are treated as missing rather than silently used
- Web Mercator (`x`/`y` or projected `latitude`/`longitude`) is converted to WGS84 when detected

Confidence-distance controls:
- `--high-confidence-distance-m` (default `20`)
- `--medium-confidence-distance-m` (default `100`)
- `--buffer-match-radius-m` (default `80`)

Duplicate handling:
- one outcome is matched to at most one feature row by default
- override with `--allow-duplicate-outcome-matches` only for diagnostics/debug workflows

## Leakage Guardrails

The builder flags potential leakage when outcome-like tokens appear in:

- `raw_feature_vector` keys
- `transformed_feature_vector` keys
- `factor_contribution_breakdown` keys

It also flags potential post-event timestamp issues when score timestamps appear later than event year.

Flags are emitted per-row (`leakage_flags`) and aggregated in join-quality artifacts.

## Output Location

`benchmark/public_outcomes/evaluation_dataset/<run_id>/`

Artifacts:

- `evaluation_dataset.jsonl`
- `evaluation_dataset.csv`
- `join_quality_metrics.json`
- `join_quality_report.md`
- `dataset_quality_report.json`
- `dataset_quality_report.md`
- `manifest.json`
- `join_quality_report.json` (backward-compatible alias of `join_quality_metrics.json`)
- `summary.md`

## Quality Diagnostics

`join_quality_metrics.json` (and alias `join_quality_report.json`) includes:

- total outcomes loaded
- outcomes loaded by source file path
- total feature rows loaded
- total joined records + join rate
- match rate percent
- join method counts
- join confidence tier counts
- match tier counts
- row confidence tier counts
- fallback usage summary (`high_evidence` / `mixed_evidence` / `fallback_heavy`)
- fallback-heavy reason counts and fallback-weight summary stats
- diversity spread summary across hazard/vegetation/terrain terciles plus region share
- join confidence score stats (min/mean/max)
- join confidence distance stats by tier
- join confidence examples by tier
- average/median join distance
- min/max join distance
- distance histogram
- distance outlier threshold + examples
- join-quality warnings
  - no high-confidence matches
  - high average join distance
- coordinate normalization summary
- duplicate-prevention counts
- low-confidence join count
- by-event join counts
- outcomes-by-event counts
- feature-rows-by-event counts
- unmatched feature rows by event
- by-label join counts
- excluded rows and reasons
- leakage warnings
- score-backfill diagnostics (attempted/backfilled/remaining-missing)
- structure-feature and near-structure-vegetation variation diagnostics:
  - non-zero-variance feature counts
  - per-feature stddev maps
  - available vs missing proxy-feature counts

Each joined row now also includes `evaluation.feature_observation_summary` with:
- `observed_fields`
- `inferred_fields`
- `fallback_fields`
- `missing_fields`
- per-feature source tags (for example `observed_defensible_space_zone`, `observed_neighboring_structure_metrics`, `observed_feature_sampling_property_specific`, `observed_feature_sampling_region_level`, `fallback_feature_sampling`, `spatial_proxy`, `derived_proxy`, `missing`)

The builder now preferentially consumes nested property-level context before deriving proxies, including:
- `property_level_context.defensible_space_analysis.zones.*.vegetation_density`
- `property_level_context.neighboring_structure_metrics.*`
- `property_level_context.feature_sampling.*` (`raw_point_value`/`index` + `scope`)

When available locally in the repo catalog, the builder also consumes external public geospatial assets:
- `data/catalog/vectors/building_footprints/*.geojson`
- `data/catalog/vectors/building_footprints_overture/*.geojson`
- `data/catalog/vectors/parcel_polygons/*.geojson`

These are used to derive per-property structure/parcel proxies such as:
- `building_footprint_area_proxy_sqft`
- `mean_nearby_footprint_area_sqft`
- `lot_size_proxy_sqft`
- `parcel_distance_ft`

Each joined row includes `evaluation.fallback_usage` for per-row auditing:
- factor counts (observed/inferred/fallback/missing)
- coverage fallback/failed counts
- fallback weight fraction and ratios
- classification + reason codes used to classify `fallback_heavy`

`join_quality_metrics.json` also includes `diversity_spread`:
- `hazard_bin_counts`, `vegetation_bin_counts`, `terrain_bin_counts` (`low` / `medium` / `high` / `unknown`)
- `region_counts` and `max_region_share` to detect clustering in one region/event
- `strata_combo_count` and top strata combinations

`join_quality_report.md` is a concise operator-facing summary with:
- total outcomes, matched rows, and match rate
- counts by confidence tier
- distance min/mean/max/median plus histogram
- sample high-confidence and low-confidence joins
- clear warning section for weak matching patterns

`dataset_quality_report.json` includes:
- independent sample counts (`total_labeled_rows`, `unique_property_event_id_count`)
- fallback-heavy fraction
- structure and near-structure variation summaries
- minimal high-signal feature diagnostics (`high_signal_feature_diagnostics`) covering:
  - `ring_0_5_ft_vegetation_density`
  - `ring_5_30_ft_vegetation_density`
  - `nearest_high_fuel_patch_distance_ft`
  - `distance_to_nearest_structure_ft`
  - `structure_density`
  - `slope`
  - `canopy_adjacency_proxy_pct`
  - `building_age_proxy_year`
  with per-feature `min`/`max`/`stddev` and non-zero-variance checks
  (`building_age_proxy_year` is sourced from assessor/year-built fields when present,
  otherwise inferred deterministically from local structure/vegetation/slope signals)
- `top_discriminative_features` (top absolute positive-vs-negative mean deltas, directional only)

Row-level feature provenance is also exposed under:
- `evaluation.feature_observation_summary.provenance_by_feature`
- `evaluation.feature_observation_summary.high_signal_feature_provenance`
with normalized provenance classes: `observed`, `proxy`, `inferred`, `missing`.

## Graceful Degradation

- Missing feature artifact paths are recorded as exclusions.
- Unmatched feature rows are excluded with explicit reasons.
- Builder succeeds with partial joins when at least one joined row exists.
