# Synthetic Sensitivity Evaluation

This workflow creates **synthetic feature variations** per property to test whether the deterministic model responds in expected directions.

It is a behavior/sensitivity check only. It is **not** real-world outcome validation.

## Command

```bash
python scripts/run_synthetic_sensitivity_evaluation.py
```

Optional explicit dataset:

```bash
python scripts/run_synthetic_sensitivity_evaluation.py \
  --evaluation-dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl \
  --run-id synthetic_sensitivity_YYYYMMDDTHHMMSSZ
```

## What It Does

For each base property row, it generates deterministic synthetic scenarios:
- `baseline_observed`
- `vegetation_up`
- `slope_up`
- `fuel_near`
- `combined_high`
- `mitigation_low`

Each synthetic record is explicitly tagged:
- `source_name: synthetic_sensitivity`
- `source_metadata.synthetic_variation: true`
- `source_metadata.synthetic_profile: <profile>`

Then it:
1. Scores synthetic scenarios via the event backtesting engine.
2. Evaluates synthetic discrimination metrics.
3. Compares synthetic metrics vs the baseline evaluation dataset.
4. Emits directionality checks (risk-up/risk-down monotonic expectations).

## Output Bundle

`benchmark/public_outcomes/synthetic_sensitivity/<run_id>/`

- `synthetic_event_dataset.json`
- `synthetic_backtest/event_backtest_*.json`
- `synthetic_validation_metrics.json`
- `baseline_validation_metrics.json`
- `comparison_to_baseline.json`
- `sensitivity_response.json`
- `summary.md`

## Interpretation

- Use this to confirm model **sensitivity** and monotonic behavior under controlled variation.
- Do not treat synthetic AUC/Brier as empirical public-outcome performance.
- Continue to use `scripts/run_public_outcome_validation.py` for public observed outcome validation.
