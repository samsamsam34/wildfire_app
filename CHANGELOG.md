# Changelog

This file tracks release-level governance changes for `WildfireRisk Advisor`.

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
