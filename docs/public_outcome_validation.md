# Public Outcome Validation v1

This workflow provides a reproducible trustworthiness check against public wildfire structure-impact outcomes before changing core model logic.

It is directional model validation. It is not insurer claims truth and not pricing validation.

## One-Command Run

```bash
python scripts/run_public_outcome_validation.py
```

By default this uses `benchmark/event_backtest_sample_v1.json` as both:
- normalization input for public outcomes
- event-backtest dataset input (if no `--feature-artifact` is provided)

Run with explicit inputs:

```bash
python scripts/run_public_outcome_validation.py \
  --outcomes-input path/to/public_damage_records.csv \
  --dataset path/to/event_backtest_dataset.json \
  --fit-calibration
```

Run against existing feature artifacts (skip backtest execution):

```bash
python scripts/run_public_outcome_validation.py \
  --outcomes-input path/to/public_damage_records.csv \
  --feature-artifact benchmark/event_backtest_results/event_backtest_20260318T120000Z.json
```

## Output Bundle

Each run writes to:

`benchmark/public_outcome_validation/<timestamp_or_run_id>/`

Artifacts:
- `public_outcomes_normalized.json`
- `public_outcome_calibration_dataset.json`
- `public_outcome_calibration_dataset.csv`
- `public_outcome_evaluation.json`
- `public_outcome_evaluation_rows.csv`
- `public_outcome_validation_summary.md`
- `manifest.json`
- `public_outcome_calibration_artifact.json` (only when `--fit-calibration`)

## What Is Evaluated

`public_outcome_evaluation.json` includes:
- sample counts overall and by event/region
- score distributions by outcome class
- ROC AUC, threshold precision/recall, confusion summaries
- Brier score
- calibration-by-bin (decile/quantile) and ECE
- rank correlation between risk score and outcome rank
- slices by confidence tier, evidence-quality tier, and evidence group
- fallback-heavy vs high-evidence diagnostics
- false-low and false-high review sets with top factor contributions
- leakage-risk checks and guardrail warnings

## Guardrails

The workflow explicitly:
- fails if required fields are missing from usable rows (`structure_loss_or_major_damage`, `scores.wildfire_risk_score`)
- warns on small sample sizes and severe class imbalance
- warns on leakage-risk patterns (outcome-like tokens in feature vectors, suspiciously perfect small-sample separation)
- warns/skips calibration fitting when fallback-heavy share is too high unless explicitly overridden

## Calibration Fit Policy

Calibration fitting is optional (`--fit-calibration`) and keeps deterministic raw scores unchanged.

Default fitting guardrails:
- minimum rows (`--min-rows-for-fit`, default `50`)
- fallback-heavy cap (`--fallback-heavy-fit-threshold`, default `0.5`)
- explicit override flag: `--allow-fallback-heavy-fit`

## Assumptions and Caveats

- Public outcome datasets are partial and heterogeneous.
- Labels are proxy outcomes and should be interpreted directionally.
- Regional/temporal coverage gaps can dominate results in fallback-heavy cohorts.
- Use this workflow to decide whether additional data quality work is required before production calibration adoption.
