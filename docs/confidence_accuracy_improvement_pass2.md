# Confidence/Accuracy Improvement Pass 2

## Scope
This pass focuses on:
- stronger structure-geometry reliability (footprint/parcel/anchor handling)
- stricter evidence-aware factor gating
- benchmark coverage for fallback and confidence behavior

This pass does **not** redesign the full scoring model.

## Gap Map (Top Confidence Blockers)

| Feature family | Direct source | Fallback mode | Downstream factors | Confidence impact | Priority |
|---|---|---|---|---|---|
| Structure geometry | Building footprints + parcel association | Point annulus proxy | `defensible_space_risk`, `flame_contact_risk`, `vegetation_intensity_risk`, `fuel_proximity_risk`, `ember_exposure_risk` | High | P0 |
| Property anchor quality | Address/parcel anchor resolver | Geocode/selected-point only | Geometry basis, display point, ring fidelity | High | P0 |
| Regional enrichment context | WHP, MTBS, gridMET | Generic hazard/moisture fallback | `historic_fire_risk`, `slope_topography_risk`, environmental component balance | High | P1 |
| Near-structure vegetation evidence | Ring metrics + NAIP enrichment | Coarse vegetation/fuel proxy | `vegetation_intensity_risk`, `fuel_proximity_risk`, defensible-space narratives | Medium/High | P1 |
| Weighting under sparse evidence | Weighted submodel blend | Fallback features still weighted | Final score separation, readiness precision, factor explanations | High | P0 |

## What Changed

## 1. Geometry quality improvements
- Point-selection mode can now use parcel context for structure matching (`WF_POINT_SELECTION_USE_PARCEL_CONTEXT=true` by default).
- Point-mode snapping now distinguishes parcel-assisted snaps with `structure_selection_method=point_parcel_intersection_snap`.
- Parcel-backed unsnapped runs now preserve `geometry_basis=parcel` instead of collapsing to pure point basis.
- `parcel_source` is now propagated in property context and result payloads for clearer provenance.

## 2. Evidence-aware gating tightened
- Geometry-sensitive factors are now downweighted more aggressively when:
  - geometry basis is point/parcel without strong structure confidence
  - anchor quality is weak
  - direct near-structure evidence is missing
- `ember_exposure_risk` can now be suppressed under very weak geometry evidence (not just downweighted).
- Regional-context factors are downweighted more when burn/hazard/dryness are simultaneously missing.
- Blend composition now further limits structure/readiness influence under weak geometry quality.
- Readiness now adds a stronger provisional penalty when structure evidence is weak even if geometry basis is not strictly point-only.

## 3. Benchmark instrumentation
- Benchmark snapshots now include `evidence_metrics`:
  - `observed_feature_count`, `inferred_feature_count`, `fallback_feature_count`, `missing_feature_count`
  - `fallback_weight_fraction`, `observed_weight_fraction`
  - `geometry_quality_score`, `regional_context_coverage_score`, `property_specificity_score`
  - `suppressed_factor_count`
- Added confidence benchmark pack:
  - `benchmark/scenario_pack_confidence_v2.json`
- Added runner:
  - `scripts/run_confidence_benchmark_pack.py`

## How To Run

```bash
python scripts/run_confidence_benchmark_pack.py
```

Optional drift comparison:

```bash
python scripts/run_confidence_benchmark_pack.py \
  --compare-to benchmark/results/benchmark_run_YYYYMMDDTHHMMSSZ.json
```

## Interpretation
- Higher confidence and specificity should coincide with:
  - lower `fallback_weight_fraction`
  - fewer suppressed factors in footprint-resolved scenarios
  - stronger separation across materially different fire-regime scenarios
- Low-evidence runs should remain constrained and explicitly provisional rather than appearing parcel-precise.
