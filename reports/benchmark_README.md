# Benchmark Properties Scaffold

This folder contains generated outputs from the benchmark harness:

- Input fixture: `tests/fixtures/benchmark_properties.csv`
- Runner: `scripts/run_benchmark_properties.py`
- Generated results: `reports/benchmark_results.csv`
- Generated diagnostics: `reports/benchmark_diagnostics.md`

## Run

```bash
python scripts/run_benchmark_properties.py
```

## What this scaffold does

- Validates required CSV columns.
- Parses `optional_inputs_json` safely per row.
- Calls the existing backend scoring pipeline through `POST /risk/assess`
  using in-process FastAPI `TestClient` (no scoring logic changes).
- Emits benchmark result rows with:
  - `risk_score`
  - `insurance_readiness_score`
  - `confidence_score`
  - `key_drivers`
  - `missing_data_flags`
  - `notes`
- Summarizes diagnostics, including:
  - very low confidence rows
  - suspiciously low risk in `high_regional_hazard` rows
  - optional input rows with little/no observed effect
  - fallback-heavy and missing-geometry indicators
  - per-row failures

## Current scope

This pass **does execute wildfire scoring via the existing API path**, but it is still
benchmark-oriented and does not modify model behavior. See TODO notes in
`reports/benchmark_diagnostics.md` for follow-up improvements.
