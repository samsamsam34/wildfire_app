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
- `summary.md`

## Interpretation

- Treat this report as a model trust/coherence check.
- Treat external-signal agreement as directional sanity checking only.
- Use violations/warnings to prioritize data-quality and model-governance improvements.
- Follow with labeled public-outcome validation and calibration when enough outcome data is available.

