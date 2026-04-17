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
- `include_optional_calibration_metadata=true`
  - Adds optional public-outcome calibration metadata to homeowner report output.
  - Omitted by default so homeowner output stays focused on property confidence, specificity, risk drivers, and actions.

## Homeowner Flow Analytics Events

The app reuses the existing durable audit log (`audit_events`) for lightweight homeowner analytics.  
No score/subscore/diagnostic/calibration fields are removed or renamed.

Tracked event actions:

- `homeowner_assessment_submitted`
  - Emitted by `POST /risk/assess` (homeowner audience).
- `homeowner_report_viewed`
  - Emitted by `GET /report/{assessment_id}/homeowner`.
- `homeowner_pdf_generated`
  - Emitted by `GET /report/{assessment_id}/homeowner/pdf`.
- `homeowner_simulation_started`
  - Emitted by `POST /risk/simulate` before simulation compute starts.
- `homeowner_simulation_completed`
  - Emitted by `POST /risk/simulate` after simulation completes.
- `homeowner_improvement_flow_opened`
  - Emitted by `GET /risk/improve/{assessment_id}`.
- `homeowner_input_submitted`
  - Emitted by `POST /risk/improve/{assessment_id}` and homeowner `POST /risk/reassess/{assessment_id}`.
- `homeowner_advanced_details_opened`
  - Emitted by frontend `toggle` handlers for advanced details panels via `POST /analytics/homeowner/event`.

Frontend UI-only analytics endpoint:

- `POST /analytics/homeowner/event`
  - Accepts `event_name`, optional `assessment_id`, and optional `metadata`.
  - Validates `event_name` against the homeowner event allowlist and writes to audit log.

## Programmatic export

For a clean, shareable non-technical payload (or PDF bytes), use:

- `backend.homeowner_report.export_homeowner_report(result, output_format="structured")`
- `backend.homeowner_report.export_homeowner_report(result, output_format="pdf")`
- Optional internal context:
  - `backend.homeowner_report.export_homeowner_report(result, output_format="structured", include_optional_calibration_metadata=True)`

The structured export intentionally focuses on homeowner-facing sections and omits technical diagnostics.

## Report sections

The report JSON includes:

- `homeowner_focus_summary`
  - `status_label` (`Likely Insurable` | `At Risk` | `High Risk of Insurance Issues`)
    - Heuristic screening label based on observable wildfire/property factors; not an insurer underwriting decision.
  - `one_sentence_summary`
  - `top_risk_drivers` (top 3)
  - `top_recommended_actions` (top 3)
  - `before_after_summary` (when simulation context exists, or when a persisted homeowner improvement rerun snapshot exists for the current `assessment_id`)
  - `confidence_limitations_summary`
- `internal_calibration_debug` (grouped technical/internal block; additive and compatibility-safe)
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
- `specificity_summary`
  - `specificity_tier` (`property_specific` | `address_level` | `regional_estimate` | `insufficient_data`)
  - `headline`
  - `what_this_means`
  - `comparison_allowed`
- `trust_summary` (user-facing confidence language + key uncertainty drivers)
  - for low-differentiation runs, includes `low_differentiation_explanation` with:
    - what was measured directly
    - what was estimated from regional context
    - why nearby homes may look similar
    - what additional details make the result more property-specific
- `improve_your_result` (diagnostic-gap-driven suggestions derived from existing evidence/coverage/fallback signals)
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
- `specificity_summary`
- `trust_summary`
- `improve_your_result`

`insurance_readiness_summary` is retained as an optional/future-facing compatibility block and should be interpreted as a heuristic screening indicator, not insurer approval probability.

The report presentation is organized for homeowner usability:
1. Property summary
2. Overall wildfire risk level
3. Top 3 risk drivers
4. Top 3 mitigation actions and what to do first
5. Mitigation simulator examples (guidance section)
6. Confidence and assumptions
7. Next-step checklist

## Confidence and limitations

The report explicitly summarizes confidence tier, missing inputs/fallback limitations, and includes a decision-support disclaimer stating this is screening guidance, not a prediction/guarantee of underwriting approval.

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
- The report is designed for homeowner communication and should not be treated as insurer-specific underwriting prediction, approval guidance, or a guarantee of insurability/wildfire safety.
