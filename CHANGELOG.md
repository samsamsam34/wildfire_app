# Changelog

This file tracks release-level governance changes for `WildfireRisk Advisor`.

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
