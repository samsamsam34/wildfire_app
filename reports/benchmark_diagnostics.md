# Benchmark Scaffold Diagnostics

This run uses the existing `POST /risk/assess` pipeline through FastAPI TestClient.
No backend scoring logic was modified.

## Input Summary
- Fixture path: `/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/fixtures/benchmark_properties.csv`
- Total properties loaded: **13**
- Successful assessments: **13**
- Failed assessments: **0**
- Scenario groups present:
  - `forest_edge`: 3
  - `high_regional_hazard`: 2
  - `hillside`: 2
  - `missing_geometry`: 1
  - `mitigation_improved`: 2
  - `suburban_low_fuel`: 3
- Rows with invalid `optional_inputs_json`: **0**
  - IDs: None
- Very low confidence rows (< 0.30 normalized): **12**
  - IDs: ex_001, ex_002, ex_004, ex_005, ex_006, ex_007, ex_008, ex_009, ex_010, ex_011, ex_012, ex_013
- Suspiciously low risk in `high_regional_hazard` rows (risk < 35.0): **2**
  - IDs: ex_006, ex_007
- Optional inputs with little/no observable score effect: **0**
  - IDs: None
- Missing geometry indicators: **12**
  - IDs: ex_001, ex_002, ex_004, ex_005, ex_006, ex_007, ex_008, ex_009, ex_010, ex_011, ex_012, ex_013
- Fallback-heavy indicators: **12**
  - IDs: ex_001, ex_002, ex_004, ex_005, ex_006, ex_007, ex_008, ex_009, ex_010, ex_011, ex_012, ex_013
- Rows with optional inputs that could not map to scoring attributes: **9**
  - IDs: ex_002, ex_003, ex_005, ex_006, ex_007, ex_008, ex_010, ex_011, ex_012

## Failed Rows
- None

## TODOs for Scoring Integration
- TODO: Calibrate anomaly thresholds using observed benchmark distributions over multiple runs.
- TODO: Add commit hash + environment metadata to make regressions easier to compare.
- TODO: Optionally add paired baseline-vs-override deltas for every row as separate report columns.
