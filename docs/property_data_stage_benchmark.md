# Property-Data Stage Benchmark

This benchmark is an internal comparison workflow for measuring whether richer property data improves property-specific differentiation quality.

It compares the same nearby-home scenarios across five staged data conditions:

1. geocode-only
2. parcel matched
3. footprint matched
4. footprint + NAIP
5. footprint + NAIP + public-record/user structure attributes

It does **not** claim real-world predictive accuracy. It is a trust/coherence progression check for property-identification quality.

## Run

```bash
python scripts/run_property_data_stage_benchmark.py
```

Optional:

```bash
python scripts/run_property_data_stage_benchmark.py \
  --benchmark-artifact benchmark/results/benchmark_run_YYYYMMDDTHHMMSSZ.json
```

Fail CI on stage regression:

```bash
python scripts/run_property_data_stage_benchmark.py --fail-on-stage-regression
```

Also fail when the underlying benchmark pack has scenario expectation failures:

```bash
python scripts/run_property_data_stage_benchmark.py \
  --fail-on-stage-regression \
  --fail-on-benchmark-failures
```

## Outputs

The command writes sidecar artifacts next to the benchmark run artifact:

- `benchmark_run_<timestamp>_property_data_stages.json`
- `benchmark_run_<timestamp>_property_data_stages.md`

Each report includes:

- `property_confidence_summary` by stage/variant
- `assessment_specificity_tier`
- `local_differentiation_score`
- top-risk-driver stability across stages
- recommendation changes across stages
- transition-level trust-wording regression checks

## Interpretation

- Preferred outcome: richer data stages improve differentiation and/or improve honest abstention quality when data remains weak.
- Guardrail: trust wording should not silently become less confident when property-data confidence and specificity are improving.
- If differentiation remains flat, use the transition report to identify which stage failed to add new local signal.
