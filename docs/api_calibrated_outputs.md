# API Calibrated Public-Outcome Outputs

The assessment API supports an optional calibrated public-outcome metadata block.

This is an **internal/advanced** signal derived from public observed wildfire damage outcomes.
It is **not** carrier underwriting probability, claims likelihood, or insurer approval.

## Opt-in usage

You can request calibrated metadata using either:

- query flag: `include_calibrated_outputs=true`
- request body flag: `"include_calibrated_outputs": true`

Default behavior is unchanged when the flag is omitted.

## Request examples

Query-flag opt-in:

```bash
curl -sS -X POST "http://127.0.0.1:8000/risk/assess?include_calibrated_outputs=true" \
  -H "Content-Type: application/json" \
  -d '{
    "address": "6 Pineview Rd, Winthrop, WA 98862",
    "audience": "homeowner"
  }'
```

Body-flag opt-in:

```bash
curl -sS -X POST "http://127.0.0.1:8000/risk/assess" \
  -H "Content-Type: application/json" \
  -d '{
    "address": "6 Pineview Rd, Winthrop, WA 98862",
    "audience": "homeowner",
    "include_calibrated_outputs": true
  }'
```

## Response shape

When requested, `AssessmentResult` may include:

```json
{
  "calibrated_public_outcome_metadata": {
    "requested": true,
    "available": true,
    "availability_status": "available_applied",
    "calibration_version": "2026.03.18",
    "label_definition": "structure_loss_or_major_damage (major_damage or destroyed = 1; minor_damage or no_damage = 0)",
    "calibrated_public_outcome_probability": 0.37,
    "calibration_basis_summary": "Public observed wildfire structure-damage outcomes were used to fit an optional calibration layer on top of raw deterministic wildfire risk scores.",
    "calibration_caveat": "This calibrated value is based on public observed wildfire damage outcomes and should not be interpreted as carrier underwriting probability or claims likelihood. Availability depends on calibration artifact coverage and model version compatibility.",
    "calibration_data_coverage_tier": "moderate",
    "calibration_data_coverage_note": "Calibration artifact was fit using 412 labeled public-outcome rows.",
    "raw_score_reference": {
      "raw_wildfire_risk_score": 62.1,
      "calibration_raw_score_input": 62.1,
      "raw_score_units": "0-100 wildfire_risk_score"
    },
    "fallback_state": "calibration_applied",
    "notes": []
  }
}
```

If unavailable, the assessment still succeeds and returns:

- `requested: true`
- `available: false`
- `availability_status` such as:
  - `unavailable_no_artifact`
  - `unavailable_incompatible_artifact`
  - `unavailable_out_of_scope`
  - `unavailable_raw_score_missing`
- `fallback_state: calibration_unavailable_using_raw_scores_only`

## Version compatibility + graceful degradation

Calibration artifacts are checked for scoring-model compatibility.
If the artifact is missing or incompatible, calibration is skipped and raw deterministic scores remain authoritative.

## Caveats

- Calibrated metadata is tied to public outcome label definitions in the artifact.
- It is directional and dataset-dependent.
- Keep raw and calibrated outputs separate in downstream consumers.
