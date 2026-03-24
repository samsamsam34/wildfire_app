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

Multiple inputs are supported:

- repeat `--outcomes` to aggregate multiple normalized outcome files (multi-event/multi-year).
- repeat `--feature-artifact` to aggregate multiple scored feature artifacts.
- optionally use discovery flags:
  - `--outcomes-root benchmark/public_outcomes/normalized --outcomes-run-id <run_id>` (repeat run ids)
  - `--outcomes-root benchmark/public_outcomes/normalized --outcomes-root-mode all` (use all normalized runs)
  - `--feature-artifact-dir <dir> --feature-artifact-glob "*.json"`

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
5. `approx_event_address_token_overlap` (bounded by `--address-token-overlap-min`)
6. `exact_event_coordinates` (bounded by `--exact-match-distance-m`)
7. `near_event_coordinates` (bounded by `--near-match-distance-m`)
8. `extended_event_coordinates` (bounded by `--max-distance-m`)
9. `buffered_event_coordinates` (bounded by `--buffer-match-radius-m`)
10. `nearest_event_name_coordinates_tolerant_year` (bounded, year tolerance configurable)
11. `approx_global_address_token_overlap` (global fallback using address token overlap)
12. `nearest_global_coordinates` (bounded by `--global-max-distance-m`, optional via `--enable-global-nearest-fallback`)

Each joined row includes:

- `join_method`
- `join_confidence_score`
- `join_confidence_tier`
- `match_tier` (`exact` / `near` / `extended` / `fallback`)
- `join_distance_m`
- distance-tier thresholds used for classification:
  - `high`: `< 20m` (configurable via `--high-confidence-distance-m`)
  - `moderate`: `20–100m` (configurable via `--medium-confidence-distance-m`)
  - `low`: `> 100m` (or fallback methods capped to lower tiers)
- `event_year_consistent`
- `evaluation.row_confidence_tier` (`high-confidence` / `medium-confidence` / `low-confidence`)

Coordinate handling:
- incoming coordinates are normalized to WGS84 latitude/longitude with fixed precision
- invalid/out-of-range coordinates are treated as missing rather than silently used
- Web Mercator (`x`/`y` or projected `latitude`/`longitude`) is converted to WGS84 when detected

Confidence-distance controls:
- `--high-confidence-distance-m` (default `20`)
- `--medium-confidence-distance-m` (default `100`)
- `--buffer-match-radius-m` (default `80`)

Duplicate handling:
- one outcome is matched to at most one feature row by default
- override with `--allow-duplicate-outcome-matches` only for diagnostics/debug workflows

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
- outcomes loaded by source file path
- total feature rows loaded
- total joined records + join rate
- match rate percent
- join method counts
- join confidence tier counts
- match tier counts
- row confidence tier counts
- join confidence score stats (min/mean/max)
- join confidence distance stats by tier
- join confidence examples by tier
- average/median join distance
- distance histogram
- distance outlier threshold + examples
- coordinate normalization summary
- duplicate-prevention counts
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
