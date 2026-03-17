# Fallback and Specificity Policy

Policy statement:

`missing data should reduce specificity and confidence more than it should numerically flatten the score`

## What changed

- The assessment now runs a **coverage preflight** before final score presentation.
- Runtime now also consumes prepared-region readiness from `manifest.catalog.property_specific_readiness`.
- Missing core evidence no longer defaults to pseudo-observed numeric substitutes for key structure factors.
- Missing factors are omitted from direct weighting where possible.
- Submodel weights are renormalized across observed evidence.
- Omission uncertainty is surfaced as an explicit confidence penalty (`missing_factor_uncertainty`).
- Confidence/use-restriction guardrails are stricter when prepared data is weak:
  - `limited_regional_ready` regions cap confidence more aggressively
  - required-layer manifest gaps can force `preliminary` confidence and `not_for_underwriting_or_binding`

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
- `assessment_mode` (`property_specific` | `address_level` | `limited_regional_estimate` | `insufficient_data`)
- `homeowner_summary.confidence_summary`
  - `score`
  - `label`
  - `assessment_type`
  - `headline`
  - `why_confidence_is_limited`
- `homeowner_summary.assessment_limitations` (grouped, deduplicated homeowner categories)
- `developer_diagnostics` (full technical fallback/source details for debugging)
- `property_level_context.region_property_specific_readiness`
- `property_level_context.region_required_layers_missing`
- `property_level_context.region_optional_layers_missing`
- `property_level_context.region_enrichment_layers_missing`

## Interpretation

- `property_specific`: strong parcel/footprint/near-structure evidence available.
- `address_level`: mixed evidence; usable but not fully property-specific.
- `regional_estimate`: substantial data gaps; treat as broad context guidance.

If `limited_assessment_flag=true`, the assessment should be treated as lower-specificity guidance and confidence/restriction tiers will be downgraded accordingly.

Homeowner output policy:
- Low-coverage runs now prioritize concise grouped limitation summaries instead of per-model repetitive technical messages.
- Detailed source/provider/fallback diagnostics remain available under `developer_diagnostics`.
- `limited_regional_estimate` and `insufficient_data` modes are presented as estimates, not full property-specific score claims.

Prepared-region readiness tiers:
- `property_specific_ready`: enough prepared evidence for property-level behavior (still subject to per-property coverage checks).
- `address_level_only`: prepared data supports address-level estimates but not full property-specific confidence.
- `limited_regional_ready`: prepared data should be treated as regional context only; stricter confidence caps apply.
