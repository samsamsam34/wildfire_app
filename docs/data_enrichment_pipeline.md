# Data Enrichment Pipeline (Pre-Scoring)

This pipeline enriches property context before `RiskEngine.score(...)` runs and avoids changing core scoring formulas.

## Objectives

- Increase feature coverage (footprints, parcels, vegetation/fuels, fire history, roads, dryness, NAIP features).
- Prefer structured source attribution over silent fallbacks.
- Cache deterministic feature bundles for repeat assessments.

## Runtime Flow

1. Resolve runtime layer paths (`prepared region` first, then legacy/runtime env paths).
2. Apply enrichment source fallbacks (`backend/feature_enrichment.py`):
   - Building footprints: Overture -> Microsoft -> existing footprints -> FEMA.
   - Parcels/address points: county/regrid style polygon + point sources.
   - LANDFIRE fuel/canopy, MTBS/perimeters, OSM roads, gridMET dryness, NAIP artifacts.
3. Build/refresh property anchor + structure rings (`backend/wildfire_data.py`).
4. Sample raster/vector features and compute derived context fields.
5. Build `feature_bundle_summary` (source map, coverage flags, geometry basis, feature snapshot, quality metrics).
6. Cache the full `WildfireContext` artifact (`backend/feature_bundle_cache.py`) keyed by:
   - coordinates
   - active runtime paths + mtimes
   - region context
   - selection/geometry inputs
7. Pass enriched `WildfireContext` into scoring.

## Diagnostics Surfaces

- `property_level_context.feature_bundle_summary`
- `property_level_context.feature_bundle_data_sources`
- `property_level_context.feature_bundle_coverage_flags`
- `property_level_context.feature_bundle_summary.coverage_metrics`
- `property_level_context.feature_bundle_summary.geometry_provenance`
- `property_level_context.feature_bundle_id`
- `property_level_context.feature_bundle_cache_hit`
- `/risk/layer-diagnostics` now includes a `feature_bundle` block.

## Assumption-Reduction Mechanics

- Anchor resolution is precision-aware (`rooftop` vs `interpolated`/`approximate`) and emits:
  - `property_anchor_quality`
  - `property_anchor_quality_score`
  - `property_anchor_selection_method`
- Structure matching keeps ambiguity protection, but allows slightly wider matching for low-precision anchors.
- Raster sampling uses a staged lookup:
  1. exact point sample
  2. nearby valid sample (small-radius search)
  3. source-adapter fallback (WHP/gridMET/MTBS/OSM)
- Nearby raster fallback is marked in diagnostics as partial coverage rather than silently treated as exact.

## Key Modules

- `backend/feature_enrichment.py`: source-group fallbacks + feature-bundle summary builder.
- `backend/feature_bundle_cache.py`: file-backed cache with TTL and read/write toggles.
- `backend/wildfire_data.py`: orchestration and pre-scoring integration.

## Env Controls

- `WF_FEATURE_BUNDLE_CACHE_ENABLED`
- `WF_FEATURE_BUNDLE_CACHE_READ`
- `WF_FEATURE_BUNDLE_CACHE_WRITE`
- `WF_FEATURE_BUNDLE_CACHE_DIR`
- `WF_FEATURE_BUNDLE_CACHE_TTL_SEC`
- `WF_ENRICH_*` source hooks (see README).
