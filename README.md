# WildfireRisk Advisor

WildfireRisk Advisor is a FastAPI application for property-level wildfire risk and insurance-readiness workflows.

Current release: **Step 5 enterprise operations layer** on top of the Step 2/3/4 scoring, simulation, and portfolio foundations.

## What It Does

- Runs deterministic, factorized wildfire risk assessments at the property level.
- Computes insurance readiness with a separate rules engine and explicit blockers/penalties.
- Supports confirmed property facts, reassessment, and what-if simulation.
- Persists reports in SQLite with backward-compatible payload upgrades.
- Supports batch/portfolio workflows, filtering, and prioritization.
- Adds enterprise foundations: organizations, role-aware behavior, portfolio jobs, CSV import/export, underwriting rulesets, audit events, assignment/workflow states, and admin summaries.

## Core API Endpoints

### Health and Assessment
- `GET /health`
- `POST /risk/assess`
- `POST /risk/reassess/{assessment_id}`
- `POST /risk/simulate`
- `POST /risk/debug`

### Reports
- `GET /report/{assessment_id}`
- `GET /report/{assessment_id}/export`
- `GET /report/{assessment_id}/view`

Report endpoints accept `?audience=homeowner|agent|inspector|insurer` (and compatibility alias `audience_mode`).

### Homeowner Simulation API
`POST /risk/simulate` supports:
- simulate from an existing assessment (`assessment_id`)
- or simulate directly from `address` + base `attributes`

Example request:

```json
{
  "assessment_id": "abc123",
  "scenario_name": "Defensible space + roof upgrade",
  "scenario_overrides": {
    "roof_type": "class_a",
    "defensible_space_ft": 30
  }
}
```

Example response excerpt:

```json
{
  "base_scores": {
    "wildfire_risk_score": 71,
    "insurance_readiness_score": 58
  },
  "simulated_scores": {
    "wildfire_risk_score": 48,
    "insurance_readiness_score": 79
  },
  "score_delta": {
    "wildfire_risk_score_delta": -23,
    "insurance_readiness_score_delta": 21
  }
}
```

The response also includes `changed_inputs`, baseline/simulated assumptions, and baseline/simulated confidence blocks for homeowner explainability.

### Portfolio and Batch
- `POST /portfolio/assess`
- `GET /portfolio`
- `GET /assessments`
- `GET /assessments/summary`
- `POST /portfolio/jobs`
- `GET /portfolio/jobs/{job_id}`
- `GET /portfolio/jobs/{job_id}/results`
- `GET /portfolio/jobs/{job_id}/export/csv`
- `GET /portfolio/jobs/{job_id}/report-pack`
- `GET /portfolio/jobs/summary`
- `POST /portfolio/import/csv`
- `GET /portfolio/{portfolio_name}/export/csv`

### Organizations and Underwriting Rulesets
- `GET /organizations`
- `POST /organizations`
- `GET /organizations/{organization_id}`
- `GET /underwriting/rulesets`
- `GET /underwriting/rulesets/{ruleset_id}`
- `POST /underwriting/rulesets`

### Review, Workflow, and Comparison
- `POST /assessments/{assessment_id}/annotations`
- `GET /assessments/{assessment_id}/annotations`
- `PUT /assessments/{assessment_id}/review-status`
- `GET /assessments/{assessment_id}/review-status`
- `POST /assessment/{assessment_id}/assign`
- `POST /assessment/{assessment_id}/workflow`
- `GET /assessment/{assessment_id}/workflow`
- `GET /assessments/{assessment_id}/compare/{other_assessment_id}`
- `GET /assessments/compare?ids=id1,id2,...`
- `GET /assessments/{assessment_id}/scenarios`

### Audit and Ops
- `GET /audit/events`
- `GET /admin/summary`
- `GET /organizations/{organization_id}/summary`

## Organization and Role Model (Step 5)

All non-health endpoints continue using API key auth. Step 5 adds lightweight role/org context through headers:

- `X-API-Key`
- `X-User-Role`: `admin|underwriter|broker|inspector|agent|viewer`
- `X-Organization-Id`
- `X-User-Id`

Role-aware behavior includes:
- only `admin|underwriter` can update review status
- workflow/assignment edits restricted to `admin|underwriter|broker|inspector`
- viewer role is read-only for assessments/batch/simulation mutations
- organization scope is enforced for non-admin users

## Scoring and Readiness (unchanged core)

### Factorized Wildfire Risk Submodels
- `vegetation_intensity_risk`
- `fuel_proximity_risk`
- `slope_topography_risk`
- `ember_exposure_risk`
- `flame_contact_risk`
- `historic_fire_risk`
- `structure_vulnerability_risk`
- `defensible_space_risk`

Each submodel includes score, weighted contribution, deterministic explanation, key inputs, and assumptions.

### Score Decomposition (Trustworthiness)
Assessments now return three distinct score families:
- `site_hazard_score` (0-100): landscape/environmental pressure around the property
- `home_ignition_vulnerability_score` (0-100): structure and near-structure susceptibility
- `insurance_readiness_score` (0-100): rules-based insurer-oriented readiness advisory

`wildfire_risk_score` is retained for compatibility and presented as a blended summary (`site_hazard_score` + `home_ignition_vulnerability_score`).
`legacy_weighted_wildfire_risk_score` is included for traceability with prior weighting output.

### Property-Level Structure Ring Context (Sprint 2)
The data layer now attempts building-footprint-based vegetation analysis before scoring:
- footprint lookup from local vector data (`WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON`)
- structure-relative rings: `ring_0_5_ft`, `ring_5_30_ft`, `ring_30_100_ft`
- canopy/vegetation summaries by ring (`canopy_mean`, `canopy_max`, `coverage_pct`, `vegetation_density`, `fuel_presence_proxy`)

These ring metrics are used as additional inputs for flame-contact, defensible-space, fuel-proximity, and vegetation-intensity submodels, and to sharpen mitigation recommendations.
If footprint data is unavailable, the system falls back to point-based context and records an assumption.

### Separate Insurance Readiness
Readiness is computed separately and returns:
- `readiness_factors`
- `readiness_blockers`
- `readiness_penalties`
- `readiness_summary`

### Underwriting Rulesets / Carrier Profiles
Assessments can specify `ruleset_id` (for example: `default`, `strict_carrier_demo`, `inspection_first_demo`).
Rulesets can adjust readiness penalty scaling, blocker thresholds, and mitigation prioritization emphasis.
Response/report payloads include `ruleset_id`, `ruleset_name`, `ruleset_version`, and `ruleset_description`.

## Step 5 Enterprise Workflows

### Portfolio Jobs
`POST /portfolio/jobs` creates a persisted job with status lifecycle:
- `queued`
- `running`
- `completed`
- `failed`
- `partial`

Jobs include totals, completion/failure counts, summary metrics, and error summary.

### CSV Import / Export
- CSV import (`POST /portfolio/import/csv`) parses address + optional property facts, validates rows, and returns row-level errors.
- CSV export endpoints provide portfolio/job summaries for operational workflows.

### Audit Trail
Important actions are logged as append-only audit events, including assessment creation, simulation, report export/view, status/workflow changes, annotation writes, job lifecycle events, and ruleset usage.

### Reviewer Assignment and Workflow States
Assessments support:
- assignment (`assigned_reviewer`, `assigned_role`)
- workflow states: `new`, `triaged`, `needs_inspection`, `mitigation_pending`, `ready_for_review`, `approved`, `declined`, `escalated`
- deterministic transition validation in API logic

## Response Trust/Transparency Fields

Assessment/report payloads include:
- assumptions: `confirmed_inputs`, `observed_inputs`, `inferred_inputs`, `missing_inputs`, `assumptions_used`
- confidence: `confidence_score`, `low_confidence_flags`
- explainability: `submodel_scores`, `weighted_contributions`, `factor_breakdown`, `top_risk_drivers`, `top_protective_factors`, `explanation_summary`
- homeowner insights: `property_findings` (plain-language findings derived from structure-ring vegetation context)
- confidence gating: `confidence_tier` (`high|moderate|low|preliminary`) and `use_restriction`
- environmental quality: `environmental_layer_status` (`ok|missing|error` per key layer) and `environmental_data_completeness_score`
- property-level context: `property_level_context.footprint_used` and `property_level_context.ring_metrics`
  - includes `footprint_status` (`used`, `not_found`, `provider_unavailable`, `error`) and `fallback_mode` (`footprint`, `point_based`) for fallback transparency
  - includes `fallback_mode` (`footprint` or `point_based`) for clear interpretation context
- readiness and mitigation linkage fields

`score_summaries` is included with three sections (`site_hazard`, `home_ignition_vulnerability`, `insurance_readiness`), each containing label, score, explanation, top drivers, protective factors, and next actions.

If address geocoding cannot be verified, `/risk/assess` returns an error (no synthetic coordinate scoring in default behavior).
Missing environmental layers are surfaced explicitly and do not silently default to neutral point values in the data layer.

## Setup

```bash
pip install -r requirements.txt

export WILDFIRE_API_KEYS="dev-key-1,dev-key-2"

# Optional geospatial layer paths
export WF_LAYER_BURN_PROB_TIF="/path/to/burn_probability.tif"
export WF_LAYER_HAZARD_SEVERITY_TIF="/path/to/hazard_severity.tif"
export WF_LAYER_SLOPE_TIF="/path/to/slope_degrees.tif"
export WF_LAYER_ASPECT_TIF="/path/to/aspect_degrees.tif"
export WF_LAYER_DEM_TIF="/path/to/dem.tif"
export WF_LAYER_FUEL_TIF="/path/to/fuel_model.tif"
export WF_LAYER_CANOPY_TIF="/path/to/canopy_density.tif"
export WF_LAYER_MOISTURE_TIF="/path/to/moisture_or_dryness.tif"
export WF_LAYER_FIRE_PERIMETERS_GEOJSON="/path/to/fire_perimeters.geojson"
export WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON="/path/to/building_footprints.geojson"

uvicorn backend.main:app --reload
```

Frontend file: `frontend/public/index.html`.

## Dependencies

`requirements.txt` includes:
- Runtime: `fastapi`, `uvicorn`, `pydantic`
- Geospatial stack: `numpy`, `rasterio`, `pyproj`, `shapely`
- Test: `pytest`, `httpx`

Without geospatial packages and configured layer files, the app runs in deterministic fallback mode instead of true layer-backed mode.
Without a configured building-footprint layer, the app still runs but `property_level_context.footprint_used` is `false`, `footprint_status` is `source_unavailable`, and ring metrics are omitted.

## Persistence and Compatibility

SQLite tables include:
- `assessments`
- `organizations`
- `underwriting_rulesets`
- `assessment_scenarios`
- `assessment_annotations`
- `assessment_review_status`
- `assessment_workflow`
- `portfolio_jobs`
- `audit_events`

Compatibility behavior:
- old rows remain readable
- missing fields are backfilled safely at read time
- legacy model rows default to `1.0.0`

## Model Versioning
- Current model version: `1.5.0`
- Legacy fallback for older rows: `1.0.0`

## Tests

Deterministic tests in `tests/test_risk_assessment.py` cover:
- assess/report contract and Step 2+ explainability fields
- reassessment and simulation flows
- batch/portfolio and partial failure behavior
- organization scoping and role-aware permissions
- portfolio job lifecycle and CSV import/export
- ruleset selection behavior
- assignment/workflow transitions
- audit and admin summary endpoints
- legacy compatibility

## Current Limitations

- Scoring/readiness remain transparent MVP heuristics, not carrier-approved underwriting models.
- Outputs are advisory and should not be treated as calibrated premium or binding/underwriting decisions.
- Access/egress scoring is still provisional and excluded from weighted wildfire total.
- Building footprint support currently expects a local GeoJSON footprint source; no external footprint API integration yet.
- Portfolio jobs are SQLite/background-task based; no distributed queue yet.
- API-key + role headers are lightweight controls, not full identity/access management.
- Report export is JSON/HTML oriented; native PDF generation is not included.

## Ready for Step 6

Step 5 now provides a practical enterprise foundation:
- org-aware operational data model
- role-aware review and workflow controls
- portfolio job abstraction with CSV import/export
- configurable underwriting ruleset profiles
- audit event logging and admin/organization summaries
- deterministic, test-covered APIs that retain backward compatibility
