# Internal Diagnostics Dashboard

This page is for internal engineering and model-operations review.

It visualizes trust metadata and diagnostics signals. It is not a claims-validation or underwriting-approval dashboard.

## Purpose

The internal dashboard helps teams:

- inspect latest offline no-ground-truth model-health artifacts
- inspect per-property diagnostics from the live assessment API
- compare risk/readiness outputs with trust metadata
- identify unstable assumptions and strongest mitigation levers
- inspect whether near-structure vegetation is a major modeled driver and why confidence is reduced

Model Health view highlights:

- monotonicity checks with violation table
- mitigation impact ranking and backwards/zero-impact flags
- stability swing summaries (average/median/max) and unstable factors/tests
- distribution spread and evidence-tier context
- benchmark alignment summaries (signals, correlation/agreement, disagreement counts) with explicit caveats
- run comparison panel for latest-vs-previous or explicit run-vs-run drift checks

## Route

- `GET /internal/diagnostics`

The page is served by the backend and reads internal diagnostics APIs:

- `GET /internal/diagnostics/api/runs`
- `GET /internal/diagnostics/api/latest`
- `GET /internal/diagnostics/api/run/{run_id}`
- `GET /internal/diagnostics/api/compare?run_id=<current>&baseline_run_id=<baseline>`
- `GET /internal/diagnostics/api/latest/{section_key}`
- `GET /internal/diagnostics/api/run/{run_id}/{section_key}`

## Required/Optional Data Sources

### Offline model-health artifacts (optional but recommended)

Default artifact root:

- `benchmark/no_ground_truth_evaluation`

Override with:

- `WF_NO_GROUND_TRUTH_EVAL_DIR=/path/to/artifacts`

Expected files inside each run folder:

- `evaluation_manifest.json`
- `summary.md`
- `monotonicity_results.json`
- `counterfactual_results.json`
- `stability_results.json`
- `distribution_results.json`
- `benchmark_alignment_results.json`
- `confidence_diagnostics.json`
- `comparison_to_previous.json`
- `comparison_to_previous.md`

If missing, the dashboard degrades gracefully and shows clear empty-state guidance.

### Property diagnostics

The page calls:

- `POST /risk/assess?include_diagnostics=true`

This returns risk/readiness outputs plus diagnostics metadata in an envelope:

- `assessment`
- `diagnostics`

## Caveats

- Diagnostics are coherence/stability/evidence signals.
- They do not establish real-world predictive accuracy.
- They do not indicate insurer approval or underwriting equivalence.
- Benchmark alignment is a sanity check only and not ground truth.

## Run Locally

1. Start backend:

```bash
uvicorn backend.main:app --reload
```

2. Optionally generate offline evaluation artifacts:

```bash
python scripts/run_no_ground_truth_evaluation.py
```

3. Open dashboard:

- `http://127.0.0.1:8000/internal/diagnostics`

## Interpretation Notes

- Use Model Health to spot systematic issues (monotonicity violations, unstable tests, clustering).
- Use Property Diagnostics for per-address confidence, sensitivity, and mitigation diagnostics.
- Property Diagnostics now explicitly surfaces confidence-reduction reasons, fallback pressure, and
  near-structure vegetation driver diagnostics (`major_driver`, `driver_strength`, and related notes).
- Treat all outputs as internal decision-support diagnostics, not external performance claims.
