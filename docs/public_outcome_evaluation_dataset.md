# Public Outcome Evaluation Dataset Builder

This workflow builds a reproducible labeled dataset by joining:

- historical model score/feature artifacts (event backtests)
- normalized public observed wildfire structure-damage outcomes

It is for directional validation/calibration support. It is not insurer claims validation.

## Command

```bash
python scripts/build_public_outcome_evaluation_dataset.py \
  --outcomes benchmark/public_outcomes/normalized/<run_id>/normalized_outcomes.json \
  --feature-artifact benchmark/event_backtest_results/event_backtest_<stamp>.json
```

Multiple feature artifacts are supported by repeating `--feature-artifact`.

If feature artifacts do not already contain model score fields, the builder can auto-backfill
scores by running event-backtest scoring directly on those records:

```bash
python scripts/build_public_outcome_evaluation_dataset.py \
  --outcomes benchmark/public_outcomes/normalized/<run_id>/normalized_outcomes.json \
  --feature-artifact benchmark/event_backtest_sample_v1.json \
  --auto-score-missing
```

Use `--no-auto-score-missing` to disable this behavior.

## Join Logic (priority order)

1. `exact_parcel_event`
2. `exact_source_record_id`
3. `exact_event_record_id`
4. `exact_event_address`
5. `nearest_event_coordinates` (bounded by `--max-distance-m`)
6. `nearest_event_name_year_coordinates` (bounded)
7. `nearest_global_coordinates` (bounded, low confidence)

Each joined row includes:

- `join_method`
- `join_confidence_score`
- `join_confidence_tier`
- `join_distance_m`
- `event_year_consistent`

## Leakage Guardrails

The builder flags potential leakage when outcome-like tokens appear in:

- `raw_feature_vector` keys
- `transformed_feature_vector` keys
- `factor_contribution_breakdown` keys

It also flags potential post-event timestamp issues when score timestamps appear later than event year.

Flags are emitted per-row (`leakage_flags`) and aggregated in `join_quality_report.json`.

## Output Location

`benchmark/public_outcomes/evaluation_dataset/<run_id>/`

Artifacts:

- `evaluation_dataset.jsonl`
- `evaluation_dataset.csv`
- `manifest.json`
- `join_quality_report.json`
- `summary.md`

## Quality Diagnostics

`join_quality_report.json` includes:

- total outcomes loaded
- total feature rows loaded
- total joined records + join rate
- join method counts
- join confidence tier counts
- join confidence score stats (min/mean/max)
- average/median join distance
- low-confidence join count
- by-event join counts
- by-label join counts
- excluded rows and reasons
- leakage warnings
- score-backfill diagnostics (attempted/backfilled/remaining-missing)

## Graceful Degradation

- Missing feature artifact paths are recorded as exclusions.
- Unmatched feature rows are excluded with explicit reasons.
- Builder succeeds with partial joins when at least one joined row exists.
