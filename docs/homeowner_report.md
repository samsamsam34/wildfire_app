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

## Programmatic export

For a clean, shareable non-technical payload (or PDF bytes), use:

- `backend.homeowner_report.export_homeowner_report(result, output_format="structured")`
- `backend.homeowner_report.export_homeowner_report(result, output_format="pdf")`

The structured export intentionally focuses on homeowner-facing sections and omits technical diagnostics.

## Report sections

The report JSON includes:

- `headline_risk_summary`
- `top_risk_drivers` (plain-language, top 3)
- `prioritized_actions` (top 3 composed from existing mitigation recommendations)
- `ranked_actions` (practical homeowner ranking)
- `most_impactful_actions` (top 1-2 highlighted)
  - each action includes:
    - `why_this_matters`
    - `what_it_reduces`
    - `expected_effect` (`small` / `moderate` / `significant`)
- `what_to_do_first`
- `limitations_notice`
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
- `headline_risk_summary`
- `score_summary.overall_wildfire_risk`
- `score_summary.home_hardening_readiness`
- `top_risk_drivers`
- `prioritized_actions`
- `what_to_do_first`
- `limitations_notice`
- `key_risk_drivers`
- `top_recommended_actions`
- `prioritized_mitigation_actions`
- `confidence_and_limitations`
- `confidence_summary`

`insurance_readiness_summary` is retained as an optional/future-facing compatibility block.

The report presentation is organized for homeowner usability:
1. Property summary
2. Overall wildfire risk level
3. Top 3 risk drivers
4. Top 3 mitigation actions and what to do first
5. Mitigation simulator examples (guidance section)
6. Confidence and assumptions
7. Next-step checklist

## Confidence and limitations

The report explicitly summarizes confidence tier, missing inputs/fallback limitations, and includes a decision-support disclaimer.

Mitigation phrasing is confidence-aware:
- stronger evidence can use direct phrasing like "helps reduce"
- weak/inferred evidence uses hedged phrasing like "may help reduce"
- mitigation effects are qualitative and directional (no unsupported percentage claims)

Tone is selected deterministically from:
- confidence tier
- fallback usage
- missing-data burden

The same property with degraded evidence will render more cautious language in:
- `headline_risk_summary`
- `top_risk_drivers`
- mitigation rationale fields (`why_this_matters`)

Action ranking is explainable and deterministic. It combines:
- risk contribution (`impact_level`)
- proximity to structure (near-home actions rank higher)
- action feasibility (`effort_level`)
- data confidence (`data_confidence` / evidence tone)

## Notes

- The report is generated from a completed assessment result; it does not rerun scoring.
- PDF export is deterministic for a given stored assessment payload.
- The report is designed for homeowner communication and should not be treated as a guarantee of insurability or wildfire safety.
