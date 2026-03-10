# Homeowner Report

WildfireRisk Advisor supports a homeowner-facing report view that transforms an existing assessment into a concise, shareable format.

## Endpoints

- `GET /report/{assessment_id}/homeowner`
  - Returns a homeowner-oriented report JSON view.
- `GET /report/{assessment_id}/homeowner/pdf`
  - Returns a downloadable PDF (`application/pdf`).

Optional query flag:
- `include_professional_debug_metadata=true`
  - Adds internal diagnostic blocks to the JSON view for review/debug use.
  - Omitted by default for consumer-facing output.

## Report sections

The report JSON includes:

- `report_header`
- `property_summary`
- `score_summary`
- `key_risk_drivers`
- `defensible_space_summary`
- `mitigation_plan`
- `insurance_readiness_summary`
- `confidence_and_limitations`
- `metadata`

## Confidence and limitations

The report explicitly summarizes confidence tier, missing inputs/fallback limitations, and includes a decision-support disclaimer.

## Notes

- The report is generated from a completed assessment result; it does not rerun scoring.
- PDF export is deterministic for a given stored assessment payload.
- The report is designed for homeowner communication and should not be treated as a guarantee of insurability or wildfire safety.
