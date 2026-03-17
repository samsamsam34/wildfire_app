# Assumption Reduction Gap Map

This document maps where assessments still rely on fallback logic and what was implemented to reduce that dependence.

## Architecture Plan

1. Resolve geometry first: geocode -> anchor -> parcel/address points -> structure footprint.
2. Collect environmental layers with staged lookup: direct sample -> nearby sample proxy -> open-data adapter fallback.
3. Assemble one canonical feature bundle before scoring.
4. Track observed vs inferred vs fallback coverage in bundle metrics.
5. Downgrade specificity/confidence when fallback dominates.

## Gap Map

| feature_name | current_source | current_failure_mode | current_fallback_behavior | score_quality_impact | best_improvement_path |
|---|---|---|---|---|---|
| property anchor point | geocoder + county address points + parcels | interpolated/approx geocode misses nearby address point | geocode point used directly | wrong/weak structure alignment | dynamic precision-aware lookup tolerance + anchor quality scoring |
| building footprint | Overture -> Microsoft -> prepared footprints -> FEMA | low-precision anchor too far from footprint | point-based ring approximation | high (near-home risk flattening) | relaxed distance for interpolated anchors + preserve ambiguity guardrails |
| parcel polygon | prepared parcel layer / env overrides | no containing parcel at strict tolerance | parcel omitted | medium-high | precision-aware nearest-parcel tolerance + parcel diagnostics |
| burn probability / hazard | burn_prob raster + WHP | nodata at exact point / edge-of-extent | missing -> generic scoring omission | high (regional hazard flattening) | nearby raster sample fallback before marking missing |
| slope / aspect | slope/aspect raster + DEM fallback | nodata at exact cell | missing or DEM derivation only | medium | nearby raster sample fallback with partial-coverage audit |
| fuel + canopy | LANDFIRE rasters | nodata at exact point | neighborhood sample may still miss | high (vegetation context) | nearby sample recovery + stronger ring-based metrics when footprint present |
| climate dryness | moisture raster + gridMET | path missing/not configured | dryness missing | medium-high | source alias normalization + staged fallback remains explicit |
| roads/access | prepared roads + OSM adapter | layer key/source mismatch | access missing or provisional | medium | runtime path alias normalization + explicit source provenance |
| NAIP structure features | prepared NAIP artifact | artifact not prepared for structure | ring/fuel proxy only | medium-high | keep footprint ID stable + prepared artifact QA in region prep |
| structural attributes | user-provided fields | absent owner inputs | inferred/omitted in submodels | medium | expand public-record hooks and maintain explicit inferred labels |

## Implemented In This Pass

- Precision-aware anchor lookup tolerances for address points/parcels.
- Anchor quality metadata (`property_anchor_quality`, `property_anchor_quality_score`, `property_anchor_selection_method`).
- Interpolated/approximate footprint matching no longer hard-clamped to ultra-short distance.
- Nearby raster sampling fallback (`ok_nearby`) to reduce false missing for edge/nodata cells.
- Runtime alias normalization for enrichment layer keys (roads/parcels/address points/building aliases).
- Feature bundle coverage metrics:
  - `observed_feature_count`
  - `inferred_feature_count`
  - `fallback_feature_count`
  - `missing_feature_count`
  - `observed_weight_fraction`
  - `fallback_dominance_ratio`
  - `structure_geometry_quality_score`
  - `environmental_layer_coverage_score`
  - `property_specificity_score`

## Remaining Highest-Impact Follow-ups

1. Expand county parcel/address point ingestion coverage beyond configured regions.
2. Increase NAIP structure-feature preparation coverage in offline region builds.
3. Add county/local road centerline ingestion where OSM is sparse.
4. Add richer structural attribute enrichment (year/material proxies) from local public records.
