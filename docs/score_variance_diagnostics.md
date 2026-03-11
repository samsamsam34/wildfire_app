# Score Variance Diagnostics

## Root Causes Found

The prior score-compression behavior was primarily caused by:

1. Coarse/bucketed transforms in core factors (fuel code mapping, aspect classes, age buckets).
2. Heavy reuse of shared fallback proxies when ring metrics or feature details were missing.
3. Neighborhood-average inputs dominating point-specific signal for fuel/canopy.
4. Broad clamping/rounding in intermediate steps.
5. Final wildfire blending favoring environmental context so strongly that structure/readiness differences were muted.

## What Changed

### Scoring sensitivity updates

- Added finer-grained fuel combustibility transform (less bucket collapse).
- Replaced coarse aspect classes with a continuous southwest exposure transform.
- Added center-point + local-percentile blending for fuel/canopy indices in `collect_context`.
- Expanded ring influence in submodels (`ring_100_300_ft` support + stronger near-structure terms).
- Reduced quantization in submodel weighted averages and tracked raw/clamped submodel scores.
- Updated blended wildfire composition to include a bounded readiness-risk component.

### Weight/composition rebalance

- Updated `config/scoring_parameters.yaml` and defaults in `backend/scoring_config.py`:
  - More balanced structural influence.
  - Added `risk_blending_weights.readiness`.
  - Added optional `zone_100_300_ft` ring penalty config.

### Debug instrumentation

`POST /risk/debug` now includes:

- `score_variance_diagnostics`
- `raw_feature_vector`
- `transformed_feature_vector`
- `factor_contribution_breakdown`
- `compression_flags`

These expose raw sampled values, transformed indices, fallback/default signals, contribution spread, and compression warnings.

## How to Run Diagnostics

### Single assessment diagnostics

Use the existing debug endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/risk/debug \
  -H 'Content-Type: application/json' \
  -d '{
    "address":"6 Pineview Rd, Winthrop, WA 98862",
    "attributes": {"roof_type":"class a","vent_type":"ember-resistant","defensible_space_ft":30},
    "confirmed_fields":["roof_type","vent_type","defensible_space_ft"],
    "audience":"homeowner"
  }' | jq '.score_variance_diagnostics'
```

### Batch score-spread analysis

```bash
python scripts/analyze_score_variance.py \
  --fixture tests/fixtures/score_variance_scenarios.json \
  --csv-out /tmp/score_variance.csv
```

Open-data upgrade spread harness:

```bash
python scripts/analyze_open_model_score_spread.py \
  --fixture tests/fixtures/score_variance_scenarios.json \
  --json-out /tmp/open_model_spread.json \
  --csv-out /tmp/open_model_spread.csv
```

This prints min/max/mean/stddev for top-level scores, fallback frequency, and contribution-variance ranking.

## Before/After Examples (Representative)

- Before: materially different scenarios often clustered in narrow wildfire-score bands with low contribution spread.
- After: synthetic benchmark fixtures now show broad score range across:
  - dense WUI slope vs suburban low-vegetation cases
  - burn-scar-adjacent vs grassland contexts
  - same-site hardened vs poor-hardening structure profiles

Use the script output (`wildfire_score_range`, `stddev`, and contribution variance ranking) as the objective before/after check.
