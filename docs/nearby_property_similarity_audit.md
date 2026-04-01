# Nearby Property Similarity Audit

Date: 2026-04-01
Scope: why nearby homes can receive very similar wildfire assessments when local geometry and enrichment data are missing.

## Pipeline audit (where similarity comes from)

Primary path:
- `backend/main.py:_run_assessment` (context fetch, coverage preflight, confidence/specificity gating)
- `backend/risk_engine.py:_build_submodels` (feature assembly + proxy/fallback assumptions)
- `backend/risk_engine.py:score` (availability multipliers, effective weights, fallback dominance)
- `backend/risk_engine.py:compute_site_hazard_score`, `compute_home_ignition_vulnerability_score`, `compute_blended_wildfire_score`
- `backend/main.py:_build_feature_coverage_preflight`, `_build_fallback_decisions`, `_build_score_evidence_ledger`

## Feature inventory by spatial specificity

| Group | Major features | Typical source | Behavior when missing |
|---|---|---|---|
| Structure-level | `roof_type`, `vent_type`, `construction_year`, `defensible_space_ft`, structure-ring metrics (0-5/5-30/30-100), nearest high-fuel patch | User attributes + footprint/ring extraction | Contribution shrinks or is omitted; assumptions include missing/proxy notes |
| Parcel-level | `parcel_id`, `parcel_geometry`, parcel lookup confidence | Parcel/address layers | If absent, geometry basis falls back to point mode; structure-sensitive models are heavily downweighted |
| Property-specific (near-home) | `near_structure_vegetation_0_5_pct`, `canopy_adjacency_proxy_pct`, `vegetation_continuity_proxy_pct`, ring vegetation density | Footprint/parcels + NAIP/ring prep | Falls back to coarse canopy/fuel proxies; near-structure variation drops |
| Neighborhood/regional | `burn_probability_index`, `hazard_severity_index`, `slope_index`, `fuel_index`, `canopy_index`, `historic_fire_index`, `moisture_index` | Regional prepared rasters | If partial/missing, conservative fallback assumptions are applied; remaining layers dominate |
| Defaulted/proxied | Missing-input fallbacks, layer proxies, point-based ring approximations | `_build_fallback_decisions`, `_availability_multiplier` | Increases fallback dominance and suppresses geometry-sensitive submodels |

## Quantified convergence under missing-geometry conditions

Regression scenario (added test):
- Two nearby addresses (`~60 m` apart) with:
  - no building footprint
  - no parcel geometry
  - hazard layer missing
  - dryness missing
  - no structure attributes provided

Observed in pipeline output:
- `assessment_specificity_tier = regional_estimate` for both
- `fallback_weight_fraction = 1.0` for both
- `home_ignition_vulnerability_score = None` for both
- `wildfire_risk_score` delta only `2.8` points (49.2 vs 52.0) despite mild local context differences
- In canonical submodels, `~99.72%` of effective weight came from just:
  - `slope_topography_risk`
  - `historic_fire_risk`

Interpretation:
- When geometry-sensitive inputs are missing, the model can collapse toward a shared regional signature for nearby homes.
- Adjacent properties can therefore look very similar unless structure/parcel/near-home signals are restored.

## Missing inputs that most reduce local differentiation

Highest impact gaps:
1. Building footprint + ring geometry (removes near-home structure/vegetation separation).
2. Parcel geometry (prevents parcel-based fallback when footprints are absent).
3. Hazard severity + dryness enrichment (partial regional context flattens variation).
4. User structure attributes (`roof_type`, `vent_type`, `defensible_space_ft`) when not provided.

## Recommended fixes (priority order)

1. Increase structure geometry coverage first (footprints, then parcel fallback).
2. Ensure near-structure ring metrics are available even in point workflows (with strict confidence labels).
3. Raise enrichment completeness for hazard + dryness before treating runs as property-specific.
4. Continue prompting for missing roof/vent/defensible-space details in homeowner flows.
5. Add an internal alert when canonical effective-weight concentration exceeds a threshold (for example, >90% in <=2 factors) so similarity-risk is explicit to operators.

## Regression test added

- `tests/test_risk_assessment.py::test_nearby_properties_collapse_toward_similar_scores_when_geometry_and_layers_are_missing`

This test intentionally demonstrates convergence behavior under degraded evidence and should fail if that specific collapse pattern changes.
