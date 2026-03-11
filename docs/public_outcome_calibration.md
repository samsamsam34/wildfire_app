# Public Outcome Calibration

This repo supports an optional, additive empirical calibration workflow using public wildfire structure-damage outcomes (for example CAL FIRE DINS-style records).

The deterministic rules-based scoring engine remains the primary model. Calibration maps the existing `wildfire_risk_score` to an empirical damage-likelihood proxy for benchmarking and directional decision support.

## What Public Data Is Used

Primary target source classes:
- CAL FIRE structure-damage / DINS-style records
- Other public structure-impact datasets with usable coordinates and damage categories

Expected input fields (or synonyms):
- damage status/label
- structure coordinates (`latitude`, `longitude`)
- event/incident metadata (`event_id`, `event_name`, `event_date`)
- optional address and quality/confidence fields

## Label Definitions

Canonical normalized labels:
- `no_damage`
- `minor_damage`
- `major_damage`
- `destroyed`

Binary calibration target:
- `structure_loss_or_major_damage`:
  - `1` for `major_damage` or `destroyed`
  - `0` for `no_damage` or `minor_damage`

Unknown/unusable labels are retained in normalized artifacts but excluded from calibration fitting.

## Workflow

### 1) Normalize public outcome records

```bash
python scripts/ingest_public_structure_damage.py \
  --input path/to/public_damage_records.csv \
  --output-json benchmark/calibration/public_structure_damage_normalized.json \
  --output-csv benchmark/calibration/public_structure_damage_normalized.csv \
  --source-name calfire_dins
```

### 2) Produce scored feature artifacts

Run backtesting with one or more event datasets so each record has model scores + debug feature vectors:

```bash
python scripts/run_event_backtest.py \
  --dataset benchmark/event_backtest_sample_v1.json \
  --output-dir benchmark/event_backtest_results
```

### 3) Build a calibration/evaluation dataset

```bash
python scripts/build_calibration_dataset.py \
  --outcomes benchmark/calibration/public_structure_damage_normalized.json \
  --feature-artifact benchmark/event_backtest_results/event_backtest_YYYYMMDDTHHMMSSZ.json \
  --output benchmark/calibration/public_outcome_calibration_dataset.json \
  --output-csv benchmark/calibration/public_outcome_calibration_dataset.csv
```

### 4) Evaluate discrimination and calibration quality

```bash
python scripts/evaluate_model_against_public_outcomes.py \
  --dataset benchmark/calibration/public_outcome_calibration_dataset.json \
  --output-json benchmark/calibration/public_outcome_evaluation.json \
  --output-csv benchmark/calibration/public_outcome_evaluation_rows.csv
```

Outputs include:
- score distributions by outcome class
- ROC AUC for top-level scores
- threshold precision/recall and confusion matrices
- calibration table for wildfire risk score
- fallback/default usage diagnostics
- factor contribution summaries by TP/FP/FN/TN class

### 5) Fit an optional calibration artifact

```bash
python scripts/fit_public_outcome_calibration.py \
  --dataset benchmark/calibration/public_outcome_calibration_dataset.json \
  --output config/public_outcome_calibration.json
```

Enable runtime calibration:

```bash
export WF_PUBLIC_CALIBRATION_ARTIFACT=config/public_outcome_calibration.json
```

## Runtime Behavior

When `WF_PUBLIC_CALIBRATION_ARTIFACT` is configured:
- `/risk/assess` includes:
  - `calibrated_damage_likelihood`
  - `empirical_damage_likelihood_proxy`
  - `empirical_loss_likelihood_proxy`
  - `calibration_applied`
  - `calibration_method`
  - `calibration_status`
  - `calibration_limitations`
- `/risk/debug` includes calibration metadata:
  - enabled/applied state
  - artifact metadata (`artifact_version`, `generated_at`, dataset metadata)
  - scope/out-of-scope warning when applicable

If no valid artifact is configured, calibration remains disabled and core deterministic scoring is unchanged.

## What Calibration Does Not Mean

- It is **not** an insurer claims-probability model.
- It does **not** replace the deterministic score engine.
- It is geographically/temporally limited by public outcome coverage.
- It should be described as an empirical proxy for directional risk validation.

## Coverage and Interpretation Limits

- Public outcome data is incomplete and uneven by region/event.
- Damage labels are proxies and may not reflect full insured loss outcomes.
- Calibration artifacts can become stale as hazard, fuels, and mitigation patterns change.

Use `calibration_status`, `calibration_limitations`, and governance metadata when interpreting calibrated values.
