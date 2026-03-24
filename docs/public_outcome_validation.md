# Public Outcome Validation

This workflow evaluates model discrimination and calibration against **public observed wildfire outcomes**.

It is directional validation and does **not** establish carrier-claims truth, underwriting truth, or insurer approval.

## Command

```bash
python scripts/run_public_outcome_validation.py
```

Default behavior:
- looks for the latest labeled dataset under `benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl`
- writes a new validation bundle under `benchmark/public_outcomes/validation/<run_id>/`

Use an explicit dataset path:

```bash
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset benchmark/public_outcomes/evaluation_dataset/<run_id>/evaluation_dataset.jsonl
```

Use a specific dataset run id:

```bash
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset-run-id <run_id>
```

You can lower/raise the minimum required usable rows:

```bash
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset-run-id <run_id> \
  --min-labeled-rows 1
```

Compare explicitly against a baseline validation run:

```bash
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset-run-id <run_id> \
  --baseline-run-id <previous_validation_run_id>
```

## Inputs

Expected labeled dataset:
- joined output from `scripts/build_public_outcome_evaluation_dataset.py`
- supported formats: `.jsonl` (preferred), `.json`, `.csv`

The evaluator preserves raw, uncalibrated model outputs and computes metrics on those values.

## Output Bundle

Each run writes:

`benchmark/public_outcomes/validation/<timestamp_or_run_id>/`

Artifacts:
- `validation_metrics.json`
- `calibration_table.json`
- `threshold_metrics.json`
- `false_low_review_set.jsonl`
- `false_high_review_set.jsonl`
- `evaluation_rows.csv`
- `comparison_to_previous.json`
- `comparison_to_previous.md`
- `summary.md`
- `manifest.json`

`comparison_to_previous.*` compares the current run against:
- `--baseline-run-id <run_id>` when provided
- otherwise the most recent prior run in the same output root

## Metrics Reported

Where label/sample coverage allows:
- sample counts and prevalence
- ROC AUC
- PR AUC
- precision/recall/F1 at configured thresholds
- confusion summaries
- Brier score
- calibration-by-bin and ECE
- rank correlation (Spearman)

Sliced metrics include:
- event
- region (if available)
- confidence tier
- evidence tier/group
- fallback-heavy vs non-fallback-heavy
- join-confidence tier

Error-analysis review sets include:
- false lows
- false highs
- unstable but outcome-positive rows
- low-confidence but outcome-positive rows

## Guardrails

The workflow warns when:
- sample size is small
- class balance is highly skewed
- fallback-heavy rows dominate
- leakage-like patterns are detected

If the sample is small, the report still computes available metrics and avoids overstating precision.
The runner also logs stage counts:
- loaded dataset rows
- prepared/usable rows
- dropped rows with missing required fields
- join-stage counts when `join_quality_report.json` is available.

## Run-to-run Governance

Each validation `manifest.json` now includes:
- model/scoring governance versions
- calibration version
- dataset lineage metadata (evaluation dataset + normalized outcomes manifest when available)
- observed feature/input version rollups (if present in joined rows)
- command configuration used for the run

This supports reproducible before/after comparisons over time.

## Caveats

- Public outcomes are incomplete and heterogeneous.
- Join quality and event-time consistency influence reliability.
- Use this as directional model-quality evidence, not claims-performance validation.
