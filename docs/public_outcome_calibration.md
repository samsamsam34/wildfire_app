# Public Outcome Calibration

This workflow fits an optional calibration layer that maps raw `wildfire_risk_score` to a public-observed adverse-outcome probability proxy.

Calibration here is directional and based on public outcomes. It is not carrier claims truth.

## Command

```bash
python scripts/fit_public_outcome_calibration.py
```

By default this:
- loads the latest labeled dataset from `benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl`
- writes a timestamped calibration bundle to `benchmark/public_outcomes/calibration/<run_id>/`

Use an explicit dataset:

```bash
python scripts/fit_public_outcome_calibration.py \
  --dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl
```

Copy/export the fitted artifact for runtime use:

```bash
python scripts/fit_public_outcome_calibration.py \
  --dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl \
  --output config/public_outcome_calibration.json
```

## Methods

Supported fitting methods:
- `logistic` (Platt-style)
- `isotonic` (piecewise monotonic fit)
- `auto` (default): selects a cautious method using validation Brier comparison and data sufficiency checks

If isotonic sample support is weak, the fitter falls back to logistic with warnings.

## Output Bundle

Each run writes:

`benchmark/public_outcomes/calibration/<run_id>/`

Artifacts:
- `calibration_model.json`
- `calibration_config.json`
- `pre_vs_post_metrics.json`
- `calibration_curve.json`
- `summary.md`
- `manifest.json`

## Guardrails

The fitter warns or skips fitting when:
- sample size is too small
- positive/negative labels are too sparse
- low-confidence / low-quality joins dominate
- calibrated metrics degrade vs raw baseline

Raw deterministic scores remain preserved and distinct from calibrated outputs.

## Runtime Use

Set:

```bash
export WF_PUBLIC_CALIBRATION_ARTIFACT=config/public_outcome_calibration.json
```

Runtime continues to expose raw score fields and applies calibration additively when artifact scope/method is valid.

## Semantics

Calibrated output semantic:
- `calibrated_adverse_outcome_probability_public`

Target definition:
- `structure_loss_or_major_damage = 1` for `major_damage` or `destroyed`
- `0` for `minor_damage` or `no_damage`

Backward-compatible runtime fields remain:
- `calibrated_damage_likelihood`
- `empirical_damage_likelihood_proxy`
- `empirical_loss_likelihood_proxy`

## Caveats

- Public outcomes are heterogeneous and incomplete.
- Join confidence and evidence quality directly affect calibration trust.
- Calibration should be treated as directional model alignment, not underwriting validation.
