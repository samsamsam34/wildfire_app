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

Filtering strictness is configurable and defaults to row-retention with tagging:

```bash
python scripts/run_public_outcome_validation.py \
  --evaluation-dataset-run-id <run_id> \
  --min-join-confidence-score-for-metrics 0.7 \
  --allow-label-derived-target \
  --allow-surrogate-wildfire-score \
  --retain-unusable-rows
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
- `baseline_model_comparison.json`
- `feature_diagnostics.json`
- `segment_metrics.json`
- `segment_report.md`
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
- ROC/PR AUC 95% bootstrap confidence intervals
- precision/recall/F1 at configured thresholds
- confusion summaries
- Brier score
- Brier 95% bootstrap confidence interval
- calibration-by-bin and ECE
- rank correlation (Spearman)

Feature signal diagnostics are also emitted from the labeled rows:
- per-feature Pearson/Spearman correlations against adverse outcome label
- univariate rank/AUC-style directional signal checks
- plot-ready feature-vs-outcome quantile curves
- ranked lists for:
  - `top_predictive_features`
  - `weak_or_noisy_features`
  - `potentially_harmful_features` (expected-direction conflicts)

These are directional signal/noise diagnostics only. They do not establish causality or insurer-claims predictive truth.

Minimum viable diagnostics are emitted for usable datasets (including small samples):
- pairwise rank-order hit rate
- simple accuracy at default threshold
- top-risk-bucket adverse hit rate and lift vs baseline
- adverse outcome rate by score decile/bucket (when computable)
- narrative summary and data sufficiency flags

Simple baseline comparisons are also emitted to justify model complexity:
- `random` baseline (deterministic pseudo-random by record identity)
- `hazard_only` baseline
- `vegetation_only` baseline

For each baseline:
- ROC AUC / PR AUC / Brier
- missing-signal count

And full-model comparison fields:
- `beats_all_baselines_by_auc`
- `best_baseline_name`
- `best_baseline_auc`
- `auc_margin_vs_best_baseline`
- `complexity_justified_signal`

These comparisons are directional checks only; they are not ground-truth claims-performance validation.

Validation outputs now include a explicit sufficiency indicator object:
- `data_sufficiency_indicator.total_dataset`
- `data_sufficiency_indicator.high_confidence_subset`

Sufficiency tiers:
- `insufficient`: `< 20` samples
- `limited`: `20–99` samples
- `moderate`: `100–500` samples
- `strong`: `> 500` samples

Each tier includes a short explanation string so operators can immediately judge reliability.

Supplemental (non-ground-truth) streams are also emitted:
- `synthetic_validation`: deterministic stress-scenario monotonic checks
- `proxy_validation`: weak-label alignment against proxy wildfire signals (e.g., burn/hazard/distance context)

`synthetic_validation` now includes an explicit extreme-scenario ranking check:
- extreme high-risk synthetic scenario should score materially above extreme low-risk synthetic scenario
- this is a behavioral sanity check, not empirical accuracy evidence

These streams are explicitly separated from real-outcome validation and should be treated as behavioral diagnostics, not ground-truth accuracy evidence.

Sliced metrics include:
- event
- region (if available)
- hazard-level segments
- vegetation-density segments
- confidence tier
- validation confidence tier (`high-confidence`, `medium-confidence`, `low-confidence`)
- evidence tier/group
- fallback-heavy vs non-fallback-heavy
- join-confidence tier

Subset metrics are also reported for:
- full usable dataset
- high-confidence subset
- medium-confidence subset
- high-evidence subset

Validation output also includes an explicit confidence-tier comparison block:
- `confidence_tier_performance.tiers.all_data`
- `confidence_tier_performance.tiers.high_confidence`
- `confidence_tier_performance.tiers.medium_confidence`
- `confidence_tier_performance.deltas_vs_all_data` (AUC/Brier deltas)
- `confidence_tier_performance.warnings` (for undersized high/medium slices)

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
Small-sample runs include explicit stability metadata:
- `metric_stability.auc_stable=false` when discrimination metrics are not statistically stable for interpretation
- `discrimination_metrics.wildfire_discrimination_stability=unstable_small_sample`
- threshold constants used for warnings (`small_sample_threshold`, `stable_auc_min_class_count`)

The runner also logs stage counts:
- loaded dataset rows
- retained rows
- usable rows
- unusable rows retained with exclusion flags
- dropped rows with missing required fields
- join-stage counts when `join_quality_report.json` is available.

Rows below metric usability thresholds are retained (unless `--no-retain-unusable-rows` is set) with explicit flags:
- `low_confidence_join`
- `missing_features`
- `fallback_heavy`
- `row_usable_for_metrics`
- `exclusion_reasons`

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
