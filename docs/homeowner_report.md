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
- `top_recommended_actions`
- `mitigation_plan`
- `home_hardening_readiness_summary`
- `insurance_readiness_summary`
- `confidence_and_limitations`
- `metadata`

Primary homeowner-facing score fields are:
- `score_summary.overall_wildfire_risk`
- `score_summary.home_hardening_readiness`
- `key_risk_drivers`
- `top_recommended_actions`
- `prioritized_mitigation_actions`
- `confidence_and_limitations`
- `confidence_summary`

`insurance_readiness_summary` is retained as an optional/future-facing compatibility block.

The report presentation is organized for homeowner usability:
1. Property summary
2. Overall wildfire risk level
3. Top risk drivers
4. Top mitigation actions
5. Mitigation simulator examples (guidance section)
6. Confidence and assumptions
7. Next-step checklist

## Confidence and limitations

The report explicitly summarizes confidence tier, missing inputs/fallback limitations, and includes a decision-support disclaimer.

## Notes

- The report is generated from a completed assessment result; it does not rerun scoring.
- PDF export is deterministic for a given stored assessment payload.
- The report is designed for homeowner communication and should not be treated as a guarantee of insurability or wildfire safety.
