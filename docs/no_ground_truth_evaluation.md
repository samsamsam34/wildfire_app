# No-Ground-Truth Evaluation

This workflow evaluates trustworthiness signals for the deterministic wildfire model when reliable labeled outcomes are not yet available.

It is explicitly **not** an accuracy claim. It checks internal coherence, directional behavior, robustness, and evidence-aware confidence behavior.

## Scope

The evaluation bundle covers:

- monotonicity and directional checks
- counterfactual mitigation sensitivity checks
- stability under small perturbations
- score distribution and segmentation health
- external benchmark alignment sanity checks (not truth)
- confidence and evidence-quality diagnostics

## Run

```bash
python scripts/run_no_ground_truth_evaluation.py \
  --fixture benchmark/fixtures/no_ground_truth/scenario_pack_v1.json \
  --output-root benchmark/no_ground_truth_evaluation
```

Deterministic reruns for CI/regression checks:

```bash
python scripts/run_no_ground_truth_evaluation.py \
  --fixture benchmark/fixtures/no_ground_truth/scenario_pack_v1.json \
  --output-root benchmark/no_ground_truth_evaluation \
  --run-id fixed_no_gt_eval \
  --seed 17 \
  --overwrite
```

Compare against a specific prior run while generating a new run:

```bash
python scripts/run_no_ground_truth_evaluation.py \
  --fixture benchmark/fixtures/no_ground_truth/scenario_pack_v1.json \
  --output-root benchmark/no_ground_truth_evaluation \
  --run-id after_fix_v2 \
  --compare-to-run before_fix_v1
```

## Artifacts

Each run writes:

`benchmark/no_ground_truth_evaluation/<run_id>/`

Files:

- `evaluation_manifest.json`
- `monotonicity_results.json`
- `counterfactual_results.json`
- `stability_results.json`
- `distribution_results.json`
- `benchmark_alignment_results.json`
- `confidence_diagnostics.json`
- `comparison_to_previous.json`
- `comparison_to_previous.md`
- `summary.md`

Comparison behavior:

- By default, each run compares to the most recent previous run in the artifact root.
- Use `--compare-to-run <run_id>` to force a specific baseline.
- If no baseline exists, comparison artifacts still write with `available: false` and a clear reason.

## Interpretation

- Treat this report as a model trust/coherence check.
- Treat external-signal agreement as directional sanity checking only.
- Use violations/warnings to prioritize data-quality and model-governance improvements.
- Follow with labeled public-outcome validation and calibration when enough outcome data is available.
- Confidence diagnostics are evidence-quality checks: confidence should increase with observed evidence
  and decrease with fallback, inferred/proxy inputs, and missing critical fields.
- Risk-bucket diagnostics are triage-oriented resolution checks (`low` / `medium` / `high`) using configured
  thresholds (`risk_bucket_thresholds` in `config/scoring_parameters.yaml`), not claims-calibrated probability bands.
- Better bucket separation indicates improved screening resolution, but it is not itself predictive-accuracy validation.
- Use `comparison_to_previous.md` for quick before/after drift review across monotonicity, interventions, stability,
  confidence warnings, and score distribution behavior.
