# Public Outcomes Ingestion

This workflow normalizes multi-source public observed wildfire structure-damage records into a reproducible artifact bundle for directional model evaluation and calibration prep.

It is **not** carrier claims truth and **not** insurer-grade validation by itself.

## Command

```bash
python scripts/ingest_public_outcomes.py \
  --input path/to/public_damage_source_a.csv \
  --source-name source_a \
  --input path/to/public_damage_source_b.geojson \
  --source-name source_b
```

Optional deterministic run id:

```bash
python scripts/ingest_public_outcomes.py \
  --input path/to/public_damage_source.csv \
  --run-id fixed_public_outcomes_run \
  --overwrite
```

## Output Location

`benchmark/public_outcomes/normalized/<run_id>/`

Artifacts:

- `normalized_outcomes.json`
- `normalized_outcomes.jsonl`
- `normalized_outcomes.csv`
- `source_summary_counts.json`
- `normalization_report.md`
- `manifest.json`

## Normalized Schema (core fields)

Each normalized record includes:

- `source_name`
- `event_name`, `event_date`, `event_year`
- `latitude`, `longitude`
- `address_text` (if available)
- `source_record_id` (if available)
- `damage_label` (normalized canonical label)
- `damage_severity_class` (`none` / `minor` / `major` / `destroyed` / `unknown`)
- `structure_loss_or_major_damage` (`1`/`0`/`null`)
- `adverse_outcome_binary` (`true`/`false`/`null`)
- `source_native_label`
- `label_confidence` (if available)
- `match_confidence` (`high` / `moderate` / `low` / `unknown`)
- `provenance_notes`

## Label Standardization

Canonical normalized damage labels:

- `no_damage`
- `minor_damage`
- `major_damage`
- `destroyed`
- `unknown`

Derived severity and binary target are retained together so downstream tools can use either conservative class labels or binary adverse outcomes.

## Quality Diagnostics

`source_summary_counts.json` and `normalization_report.md` include:

- counts by source
- counts by event
- counts by label/severity
- dedupe count/rate
- invalid coordinate drops
- missing address rate
- unknown label rate
- match-confidence distribution
- included vs excluded source list

## Graceful Degradation

- Missing or unreadable sources are logged as excluded with reasons.
- Ingestion still succeeds as long as at least one source yields usable normalized records.

## Limitations

- Public outcomes are heterogeneous and geographically incomplete.
- Public observed outcomes are directional validation signals, not equivalent to insurer claims outcomes.
