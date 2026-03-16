# Fallback and Specificity Policy

Policy statement:

`missing data should reduce specificity and confidence more than it should numerically flatten the score`

## What changed

- The assessment now runs a **coverage preflight** before final score presentation.
- Missing core evidence no longer defaults to pseudo-observed numeric substitutes for key structure factors.
- Missing factors are omitted from direct weighting where possible.
- Submodel weights are renormalized across observed evidence.
- Omission uncertainty is surfaced as an explicit confidence penalty (`missing_factor_uncertainty`).

## New response fields

- `feature_coverage_summary`
- `feature_coverage_percent`
- `assessment_specificity_tier` (`property_specific` | `address_level` | `regional_estimate`)
- `limited_assessment_flag`
- `observed_factor_count`
- `missing_factor_count`
- `fallback_factor_count`
- `observed_weight_fraction`
- `fallback_dominance_ratio`
- `score_specificity_warning`

## Interpretation

- `property_specific`: strong parcel/footprint/near-structure evidence available.
- `address_level`: mixed evidence; usable but not fully property-specific.
- `regional_estimate`: substantial data gaps; treat as broad context guidance.

If `limited_assessment_flag=true`, the assessment should be treated as lower-specificity guidance and confidence/restriction tiers will be downgraded accordingly.
