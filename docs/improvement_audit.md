### Executive Summary
1. `No rate limiting on high-cost assessment endpoints` (Weak): `/risk/assess` and related flows are API-key protected but not throttled, so one client can drive repeated expensive geocode/parcel/national lookups. Evidence: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:14162), [backend/auth.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/auth.py:48).
2. `CORS is fully open in API server` (Weak): wildcard origins/methods/headers are enabled in production code. Evidence: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:197).
3. `No global exception handler for consistent API error envelopes` (Weak): route-level `HTTPException` usage is extensive, but no `@app.exception_handler(...)` is registered. Evidence: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:197) (middleware only).
4. `Frontend defaults to localhost API base` (High deployment risk): default base is `http://127.0.0.1:8000` unless overridden, which can break production if not injected. Evidence: [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:2103).
5. `Homeowner PDF endpoint has no local fail-safe wrapper` (Reliability gap): report build/render is called directly; render failure bubbles to 500. Evidence: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:15902).
6. `PDF report module/path expectations drift` (Maintainability): `backend/report_pdf.py` is missing while PDF logic lives in `backend/homeowner_report.py`; `backend/templates/` is also absent. Evidence: diagnostics section #8 and #2 output.
7. `Frontend tests are mostly string-presence checks, not runtime UX behavior tests` (Coverage quality gap): many assertions verify literal snippets in HTML. Evidence: [tests/test_frontend_region_handling.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_frontend_region_handling.py:11), [tests/test_frontend_safe_dom_rendering.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_frontend_safe_dom_rendering.py:10).
8. `PDF content is robust and text-searchable, but visual polish is only adequate for broker/agent sharing` (Product quality): strong section coverage and deterministic generation, but no true charting/score bars and map is schematic (not real map tiles). Evidence: [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:2962), [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:3284).
9. `Address input validation is minimal for extreme payload lengths` (Abuse/perf gap): `AddressRequest.address` has `min_length=5` but no max length constraint. Evidence: [backend/models.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/models.py:124).
10. `Deployment packaging is incomplete` (No-go for production hardening): no Dockerfile/compose/ASGI process config beyond local uvicorn reload docs. Evidence: [README.md](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/README.md:92), repository file scan.

### Section 1: PDF Report Quality
1a. Content completeness — `Adequate`
- Present: risk scores/subscores, top drivers, top actions, before/after snapshot, confidence, limitations, defensible-space context, and “improve result” guidance.
- Evidence: section assembly includes “Top 3 Risk Drivers,” “Top 3 Recommended Actions,” “Confidence and Limitations,” “Property Context and Map,” and mitigation details in [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:2987).
- Missing or weak: explicit “data coverage summary” is indirect (counts/limitations) rather than a dedicated homeowner table; confidence-improvement actions exist but are blended into narrative.

1b. Visual design quality — `Adequate`
- Professional enough for internal/client-facing sharing, but not premium report design.
- No quantitative visual score bars/charts for risk dimensions.
- Local map in PDF is synthetic vector illustration (schematic), not parcel/satellite tile capture.
- Evidence: map drawing primitives in [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:3284); styles are typographic blocks in [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:2285).

1c. Plain-English language — `Strong`
- Good homeowner wording and jargon-reduction helpers.
- Explanations for drivers/actions are narrative and action-oriented.
- Evidence: de-jargon and homeowner explanation logic in [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:81), [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:1965).

1d. PDF technical quality — `Adequate`
- Typical file size estimate: roughly low tens of KB (text/vector only, no embedded raster imagery).
- Text-searchable: yes (PDF text operators written directly).
- Page size: fixed US Letter (`612x792`), no A4 mode.
- Pagination: explicit page-break and dynamic pagination reduce truncation risk.
- Evidence: text stream serialization [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:3415), media box [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:3561), pagination [backend/homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/homeowner_report.py:3243).

1e. Report generation reliability — `Adequate`
- Positive: extensive PDF/report tests, including low-confidence and missing-data scenarios.
- Gap: endpoint lacks local `try/except` around report render for graceful fallback message.
- Evidence tests: [tests/test_homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_homeowner_report.py:406), [tests/test_homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_homeowner_report.py:547).
- Endpoint render call without fallback envelope: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:15908).

1f. Shareability — `Weak`
- Regeneration by assessment ID is supported via `/report/{assessment_id}/homeowner` and `/homeowner/pdf`.
- No built-in share-link UX (copy/share URL), and full report opening uses blob URLs.
- Assessment ID is not prominently surfaced in primary frontend summary flow.
- Evidence: endpoints [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:15846), blob open/download flow [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:8922).

### Section 2: Frontend UX
2a. Address input flow — `Strong`
- Clear entry point and optional home-details funnel.
- Good progressive address verification modal with candidate search and map-pin fallback.
- Geocoding and coverage errors have explicit homeowner language and fallback flows.
- Evidence: address/start UI [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1492), verify modal [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:2001), error routing [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:8570).

2b. Results display — `Adequate`
- Risk/status and next actions are clearly surfaced in stepwise flow.
- Confidence split exists in backend model but homeowner explanation could still be simplified further in-screen.
- “Add home details” is discoverable via expandable section, but could be more prominent post-result.
- Evidence: summary section [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1596), optional details section [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1500), model split fields [backend/models.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/models.py:1125).

2c. Map panel — `Adequate`
- Map loads with basemap toggle, ring layers, and layer controls; behavior appears robust.
- Satellite base layer available in frontend map panel.
- Evidence: map controls/UI [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1792), map initialization [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:2362).

2d. Mobile experience — `Adequate`
- Multiple responsive breakpoints and touch-target sizing are present.
- No runtime/mobile browser tests validate real behavior at 375px.
- Evidence: breakpoints and sizing [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1422), [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:1461).

2e. Missing screens/flows — `Weak`
- Loading/error states are mostly present for assessment/simulation/report actions.
- Share URL flow is missing.
- Frontend tests do not validate full-state transitions with a browser harness.
- Evidence: report actions and status updates [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:8904), no copy-link flow found in API/share grep diagnostics.

### Section 3: API Robustness
3a. Input validation — `Adequate`
- Pydantic request models provide structural validation.
- Address has minimum length only; no max length/guardrail for very large payload strings.
- SQL injection risk is low in audited paths due parameterized SQLite usage.
- Evidence: request model [backend/models.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/models.py:124), SQLite parameterization [backend/database.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/database.py:306).

3b. Error handling completeness — `Adequate`
- Many explicit `HTTPException` responses and defensive `try/except` blocks exist.
- No global exception handler means non-HTTP exceptions can return generic 500 shape inconsistently.
- Evidence: widespread route-level exceptions [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:14162), no registered exception handlers in file scan.

3c. Rate limiting and abuse prevention — `Weak`
- API-key auth exists, including org-scoped keys.
- No request-rate limiting/throttling was found for heavy endpoints.
- If keys are unset, auth is bypassed for local/dev operation by design.
- Evidence: auth bypass behavior [backend/auth.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/auth.py:57), heavy route [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:14162).

3d. Response time — `Adequate`
- Route path is mostly sequential for geocode -> context collection -> scoring.
- Expected national-fallback latency is likely several seconds to low tens of seconds depending on cache and remote service conditions.
- Per-client network timeouts exist in geocode/national clients, but no global per-request timeout wrapper for the full assessment transaction.
- Evidence: sequential assessment core [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:7291), client timeouts in geocoding and national clients [backend/geocoding.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/geocoding.py:93), [backend/national_nlcd_client.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/national_nlcd_client.py:99).

3e. Caching effectiveness — `Adequate`
- Service-level caches exist (parcel, NLCD, elevation, footprint) with TTL and coarse coordinate/bbox keys.
- Neighborhood hit-rate should be moderate for 3-decimal caches (NLCD/elevation), lower for 5-decimal parcel key unless repeated same-parcel requests.
- No full assessment-response cache for identical `/risk/assess` payloads.
- Evidence: cache key precision [backend/parcel_api_client.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/parcel_api_client.py:247), [backend/national_elevation_client.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/national_elevation_client.py:331), [backend/national_nlcd_client.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/national_nlcd_client.py:322), [backend/national_footprint_index.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/national_footprint_index.py:183).

### Section 4: Test Coverage
4a. Major code paths with little/no direct coverage
- Auth enforcement with non-empty real API key sets appears under-tested; many tests explicitly disable keys (`auth.API_KEYS = set()`).
- No true browser-driven E2E coverage for frontend async states/mobile rendering.
- No tests for production CORS policy behavior.

4b. Tests over implementation details
- Frontend tests often assert raw HTML/JS string presence rather than behavior outcomes.
- Evidence: [tests/test_frontend_region_handling.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_frontend_region_handling.py:11), [tests/test_frontend_safe_dom_rendering.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_frontend_safe_dom_rendering.py:10).

4c. Integration tests for full assessment pipeline
- Present with mocked external dependencies and API-level calls (`TestClient`), especially in `test_risk_assessment.py`, `test_homeowner_report.py`, and map/geocoding suites.

4d. PDF report endpoint/output tests
- Strong: many tests verify endpoint responses and generated PDF byte signatures/sections.
- Evidence: [tests/test_homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_homeowner_report.py:406), [tests/test_homeowner_report.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/test_homeowner_report.py:547).

4e. Mobile/responsive layout tests
- Missing as runtime tests; only static string assertions.

4f. Approximate overall coverage estimate (read-based)
- Estimated `~55%–70%` behavior coverage for core homeowner/API flows.
- Lower confidence for deployment/auth-hardening/frontend-runtime/browser behavior.

### Section 5: Deployment Readiness
5a. Docker/container config — `No-Go`
- No `Dockerfile`/compose artifacts detected.

5b. Production ASGI server config — `No-Go`
- Docs show local `uvicorn --reload`; no production server/run profile documented.
- Evidence: [README.md](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/README.md:92).

5c. Required env vars documented — `No-Go`
- Code references ~108 env vars; only a subset appears in README/docs.
- Evidence from env-var scan.

5d. Health check endpoint — `Go`
- `/health` exists and returns status/version/governance metadata.
- Evidence: [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py:13967).

5e. SQLite strategy viability — `Conditional Go`
- WAL and thread-local connections help, but single-file SQLite at this scale will be a bottleneck under multi-worker write-heavy production traffic.
- Evidence: [backend/database.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/database.py:103).

5f. Dev-only/hardcoded URLs — `No-Go`
- Frontend fallback uses localhost API base.
- Evidence: [frontend/public/index.html](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/frontend/public/index.html:2103).

### Section 6: Quick Wins Table
| # | What | Why | Files touched | Effort |
|---|------|-----|---------------|--------|
| 1 | Add API rate limiting for `/risk/assess`, `/risk/simulate`, `/risk/reassess` keyed by API key/IP with burst + sustained windows | Prevents abuse/cost spikes and improves service stability | `backend/main.py`, `backend/auth.py`, `tests/test_risk_assessment.py` | 8-14h |
| 2 | Replace wildcard CORS with env-configured allowlist and safe defaults | Reduces cross-origin exposure and deployment risk | `backend/main.py`, `README.md`, `docs/deployment_hardening.md` (new), `tests/test_risk_assessment.py` | 4-8h |
| 3 | Add global FastAPI exception handlers for consistent error envelope and sanitized 500 responses | Better UX/debuggability; avoids inconsistent server errors | `backend/main.py`, `tests/test_risk_assessment.py`, `docs/api_diagnostics.md` | 6-10h |
| 4 | Add max-length and normalization constraints for address and selected geometry IDs | Prevents oversized payload abuse and edge-case failures | `backend/models.py`, `backend/main.py`, `tests/test_risk_assessment.py`, `tests/test_geocoding.py` | 4-8h |
| 5 | Add homeowner share-link UX (copy deep link with `assessment_id`) and load-by-query on frontend | Improves real-world report sharing and follow-up workflows | `frontend/public/index.html`, `backend/main.py` (optional tiny helper route), `tests/test_frontend_region_handling.py` | 10-16h |
| 6 | Surface assessment ID clearly in results header and report action area | Easier support/debug/share references for users | `frontend/public/index.html`, `tests/test_frontend_region_handling.py` | 2-4h |
| 7 | Add accessibility pass for form labels (`for`/`id`) and status live regions consistency checks | Better keyboard/screen-reader usability and compliance readiness | `frontend/public/index.html`, `tests/test_frontend_region_handling.py` | 6-10h |
| 8 | Add production deployment docs + baseline Dockerfile using uvicorn workers | Moves project from local-only to repeatable deployment baseline | `Dockerfile` (new), `docs/deployment_hardening.md` (new), `README.md` | 6-12h |
| 9 | Add auth-enforcement tests with non-empty API key config and negative cases | Closes security regression gap in CI | `tests/test_risk_assessment.py`, `tests/test_homeowner_report.py`, `tests/test_api_diagnostics.py` | 4-8h |
| 10 | Add PDF generation fallback in endpoint (`try/except`) with actionable error detail and event logging | Prevents opaque 500 on render edge cases | `backend/main.py`, `tests/test_homeowner_report.py` | 3-6h |

All quick wins above avoid off-limits files.

### Prioritized Implementation Queue
1. Title: Add API Rate Limiting
- Problem it solves: Unbounded expensive calls to assessment/report paths.
- Files to create or modify: `backend/main.py`, `backend/auth.py`, `tests/test_risk_assessment.py`.
- Estimated effort: 8-14h.
- Dependencies: None.

2. Title: Harden CORS by Environment
- Problem it solves: Wildcard CORS in production risk profile.
- Files to create or modify: `backend/main.py`, `README.md`, `docs/deployment_hardening.md` (new).
- Estimated effort: 4-8h.
- Dependencies: None.

3. Title: Standardize Global Exception Handling
- Problem it solves: Inconsistent 500 response envelopes.
- Files to create or modify: `backend/main.py`, `tests/test_risk_assessment.py`, `docs/api_diagnostics.md`.
- Estimated effort: 6-10h.
- Dependencies: Task 2 recommended first for shared middleware/error policy review.

4. Title: Enforce Input Size/Shape Limits
- Problem it solves: Oversized payload and malformed-id risk.
- Files to create or modify: `backend/models.py`, `backend/main.py`, `tests/test_risk_assessment.py`.
- Estimated effort: 4-8h.
- Dependencies: Task 3.

5. Title: Add Auth-Required Negative Tests
- Problem it solves: Missing CI coverage for configured-key rejection behavior.
- Files to create or modify: `tests/test_risk_assessment.py`, `tests/test_homeowner_report.py`, `tests/test_api_diagnostics.py`.
- Estimated effort: 4-8h.
- Dependencies: None.

6. Title: Add Homeowner Share Link Flow
- Problem it solves: No direct copy/share URL for users.
- Files to create or modify: `frontend/public/index.html`, optionally `backend/main.py`, tests in `tests/test_frontend_region_handling.py`.
- Estimated effort: 10-16h.
- Dependencies: Task 4 (validation hardening for query params).

7. Title: Improve Results Surface with Assessment ID
- Problem it solves: Weak user/support traceability.
- Files to create or modify: `frontend/public/index.html`, `tests/test_frontend_region_handling.py`.
- Estimated effort: 2-4h.
- Dependencies: None.

8. Title: Add Frontend Accessibility Hardening Pass
- Problem it solves: Label association/live-region and keyboard quality gaps.
- Files to create or modify: `frontend/public/index.html`, `tests/test_frontend_region_handling.py`.
- Estimated effort: 6-10h.
- Dependencies: Task 6/7 can be bundled.

9. Title: Add PDF Endpoint Fallback Handling
- Problem it solves: Unhandled render exceptions return opaque 500.
- Files to create or modify: `backend/main.py`, `tests/test_homeowner_report.py`.
- Estimated effort: 3-6h.
- Dependencies: Task 3.

10. Title: Add Deployment Artifacts Baseline
- Problem it solves: No reproducible production packaging.
- Files to create or modify: `Dockerfile` (new), `README.md`, `docs/deployment_hardening.md` (new).
- Estimated effort: 6-12h.
- Dependencies: Task 2 and task 3 guidance should be finalized first.

### Diagnostic Command Output (Verbatim)
```text
### 1. PDF report: find all report-related files
./benchmark/public_outcomes/normalized/public_outcomes_multi_source_20260324/normalization_report.md
./benchmark/public_outcomes/normalized/full_eval_20260326T153810Z/normalization_report.md
./benchmark/public_outcomes/normalized/20260325T190854Z/normalization_report.md
./benchmark/public_outcomes/normalized/ingest_full_20260326T043832Z/normalization_report.md
./benchmark/public_outcomes/normalized/ingest_full_20260326T042308Z/normalization_report.md
./benchmark/public_outcomes/normalized/diag_ingest_20260325T204451Z/normalization_report.md
./benchmark/public_outcomes/normalized/public_outcomes_smoke/normalization_report.md
./benchmark/public_outcomes/normalized/20260325T184730Z/normalization_report.md
./benchmark/public_outcomes/model_viability/model_viability_rerun_after_dataset_fix_20260325_rescored/model_viability_report.json
./benchmark/public_outcomes/model_viability/model_viability_20260330T180630Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_suite_viability_20260326T153810Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_eval_public_structure_integration_viability_20260327T013500Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_eval_enriched_viability_20260327T000500Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_eval_enriched_allnorm_v2_viability_20260327T002500Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/metrics_feature_importance_20260326T/model_viability_report.json
./benchmark/public_outcomes/model_viability/model_viability_full_20260326T043832Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_eval_input_upgrade_viability_20260327T011500Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/full_eval_model_viability_20260326T174900Z/model_viability_report.json
./benchmark/public_outcomes/model_viability/model_viability_full_20260326T042308Z/model_viability_report.json
./benchmark/public_outcomes/validation/full_suite_validation_20260326T153810Z/segment_report.md
./benchmark/public_outcomes/validation/weighted_feature_filtering_default_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/weighted_feature_filtering_default_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/weighted_feature_filtering_default_20260330/segment_report.md
./benchmark/public_outcomes/validation/diag_public_validation_after_rebuild_20260325T204451Z/segment_report.md
./benchmark/public_outcomes/validation/full_eval_public_validation_20260326T174900Z/segment_report.md
./benchmark/public_outcomes/validation/feature_signal_report_20260326
./benchmark/public_outcomes/validation/feature_signal_report_20260326/feature_signal_report.json
./benchmark/public_outcomes/validation/feature_signal_report_20260326/segment_report.md
./benchmark/public_outcomes/validation/20260330T180616Z/direction_alignment_report.json
./benchmark/public_outcomes/validation/20260330T180616Z/feature_signal_report.json
./benchmark/public_outcomes/validation/20260330T180616Z/segment_report.md
./benchmark/public_outcomes/validation/full_eval_enriched_allnorm_v2_validation_20260327T002500Z/segment_report.md
./benchmark/public_outcomes/validation/full_eval_input_upgrade_validation_20260327T011500Z/segment_report.md
./benchmark/public_outcomes/validation/validation_structure_proxy_light_20260326T152638Z/segment_report.md
./benchmark/public_outcomes/validation/interaction_validation_fast_20260325T210634Z/segment_report.md
./benchmark/public_outcomes/validation/high_signal_weighted_20260330T1807Z/direction_alignment_report.json
./benchmark/public_outcomes/validation/high_signal_weighted_20260330T1807Z/feature_signal_report.json
./benchmark/public_outcomes/validation/high_signal_weighted_20260330T1807Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_pack_20260325_v2_validation/segment_report.md
./benchmark/public_outcomes/validation/direction_alignment_20260330T2/direction_alignment_report.json
./benchmark/public_outcomes/validation/direction_alignment_20260330T2/feature_signal_report.json
./benchmark/public_outcomes/validation/direction_alignment_20260330T2/segment_report.md
./benchmark/public_outcomes/validation/simplified_unweighted_20260330_v2/direction_alignment_report.json
./benchmark/public_outcomes/validation/simplified_unweighted_20260330_v2/feature_signal_report.json
./benchmark/public_outcomes/validation/simplified_unweighted_20260330_v2/segment_report.md
./benchmark/public_outcomes/validation/simplified_unweighted_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/simplified_unweighted_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/simplified_unweighted_20260330/segment_report.md
./benchmark/public_outcomes/validation/filtered_features_weighted_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/filtered_features_weighted_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/filtered_features_weighted_20260330/segment_report.md
./benchmark/public_outcomes/validation/segment_slice_interaction_20260325T220900Z/segment_report.md
./benchmark/public_outcomes/validation/validation_diverse_eval_20260326T223000Z/segment_report.md
./benchmark/public_outcomes/validation/full_eval_enriched_validation_20260327T000500Z/segment_report.md
./benchmark/public_outcomes/validation/diag_public_validation_20260325T203746Z/segment_report.md
./benchmark/public_outcomes/validation/direction_alignment_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/direction_alignment_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/direction_alignment_20260330/segment_report.md
./benchmark/public_outcomes/validation/viability_guardrail_20260326T230500Z/segment_report.md
./benchmark/public_outcomes/validation/near_structure_validation_20260325T211928Z/segment_report.md
./benchmark/public_outcomes/validation/dev_fallback_reduction_validation_20260326/segment_report.md
./benchmark/public_outcomes/validation/simplified_high_signal_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/simplified_high_signal_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/simplified_high_signal_20260330/segment_report.md
./benchmark/public_outcomes/validation/validation_structure_proxy_wide_20260326T152638Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_pack_20260325_validation/segment_report.md
./benchmark/public_outcomes/validation/validation_structure_proxy_20260326T152638Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_full_20260326T042308Z_validation/segment_report.md
./benchmark/public_outcomes/validation/diverse_full_20260326T043832Z_validation/segment_report.md
./benchmark/public_outcomes/validation/20260325T201954Z/segment_report.md
./benchmark/public_outcomes/validation/full_eval_public_structure_integration_validation_20260327T013500Z/segment_report.md
./benchmark/public_outcomes/validation/validation_full_20260326T043832Z/segment_report.md
./benchmark/public_outcomes/validation/baseline_compare_20260325T221600Z/segment_report.md
./benchmark/public_outcomes/validation/segment_slice_map_20260325T220500Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_eval_validation_20260326T200000Z/segment_report.md
./benchmark/public_outcomes/validation/validation_full_20260326T042308Z/segment_report.md
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330/direction_alignment_report.json
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330/feature_signal_report.json
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_baseline_ref_20260325/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_featurevar_20260325_validation/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_structure_proxy_20260325_v2_validation/segment_report.md
./benchmark/public_outcomes/validation/metrics_feature_importance_20260326T/feature_signal_report.json
./benchmark/public_outcomes/validation/metrics_feature_importance_20260326T/segment_report.md
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330_v2
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330_v2/direction_alignment_report.json
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330_v2/feature_signal_report.json
./benchmark/public_outcomes/validation/simplified_weighted_from_report_20260330_v2/segment_report.md
./benchmark/public_outcomes/validation/full_eval_minimal_features_validation_20260327T024500Z/segment_report.md
./benchmark/public_outcomes/validation/rerun_after_dataset_fix_20260325_rescored/segment_report.md
./benchmark/public_outcomes/validation/segment_slice_map_large_20260325T220700Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_structure_proxy_20260325_validation/segment_report.md
./benchmark/public_outcomes/validation/validation_structure_proxy_tuned_light_20260326T153234Z/segment_report.md
./benchmark/public_outcomes/validation/all_diag_public_validation_20260325T223300Z/segment_report.md
./benchmark/public_outcomes/validation/baseline_gap_check_20260325T222500Z/segment_report.md
./benchmark/public_outcomes/validation/diverse_conditions_featurevar_baseline_ref_20260325/segment_report.md
./benchmark/public_outcomes/validation/near_structure_veg_validation_20260326T160929Z/segment_report.md
./benchmark/public_outcomes/validation/dev_diverse_try_validation_20260326/segment_report.md
./benchmark/public_outcomes/validation/segment_strength_map_20260325T211928Z/segment_report.md
./benchmark/public_outcomes/validation/hazard_vuln_split_validation_fast_20260325T2356/segment_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_conditions_pack_20260325/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_full_20260326T042308Z/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_full_20260326T043832Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_full_20260326T043832Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/feature_variation_debug_20260325/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/feature_variation_debug_20260325/feature_variation_debug_report.md
./benchmark/public_outcomes/evaluation_dataset/feature_variation_debug_20260325/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/feature_variation_runtime_context_probe/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/feature_variation_runtime_context_probe/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/debug_high_conf_check/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/debug_high_conf_check/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_20260325T210634Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_20260325T210634Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_tuned_light_20260326T153234Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_tuned_light_20260326T153234Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_multi_source_ingested_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_20260327T001500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_20260327T001500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_20260327T001500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_20260327T001500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260326T235900Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260326T235900Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260326T235900Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260326T235900Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260326T235900Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_04/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_04/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_multi_source_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_03/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_03/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_rescore_all_probe_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_rescore_all_probe_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_fast_20260325T2355/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_fast_20260325T2355/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_02/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_02/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_full_20260326T043832Z/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_05/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_05/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1848/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1848/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1848/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_full_20260326T042308Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_full_20260326T042308Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_fallback_reduction_20260326/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_fallback_reduction_20260326/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_dataset_20260326T153810Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_dataset_20260326T153810Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_dataset_20260326T153810Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_geo_match_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_check/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_check/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_check/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_check/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diag_eval_ds_20260325T204451Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/diag_eval_ds_20260325T204451Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/diag_eval_ds_20260325T204451Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_conditions_pack_20260325_v2/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_n0_fix_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_rapid_20260326T224500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_rapid_20260326T224500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_smoke/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260327T000500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260327T000500Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260327T000500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260327T000500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_20260327T000500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_noise_reduced_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_noise_reduced_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/near_structure_veg_eval_20260326T160929Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/near_structure_veg_eval_20260326T160929Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_high_signal_20260327T021500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_high_signal_20260327T021500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_high_signal_20260327T021500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_high_signal_20260327T021500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_high_conf_boost_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_rapid_20260326T223500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_rapid_20260326T223500Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_rapid_20260326T223500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/feature_variation_smoke/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/feature_variation_smoke/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_replaced/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_replaced/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_replaced/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_minimal_high_signal_replaced/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/feature_variation_fix_20260325/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/feature_variation_fix_20260325/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_20260326T200000Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_20260326T200000Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/dev_diverse_try_20260326/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/dev_diverse_try_20260326/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/rerun_after_dataset_fix_20260325/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/rerun_after_dataset_fix_20260325/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_20260325T2350/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_20260325T2350/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/near_structure_eval_ds_20260325T211928Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/near_structure_eval_ds_20260325T211928Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_input_upgrade_20260327T011500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_input_upgrade_20260327T011500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_input_upgrade_20260327T011500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_input_upgrade_20260327T011500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_conditions_structure_proxy_20260325/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_tuned_20260326T153234Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_tuned_20260326T153234Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_sample_20260325T210634Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_sample_20260325T210634Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1909/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1909/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/all_diag_eval_ds_20260325T1909/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_rapid_coverage_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_v2_20260327T002500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_v2_20260327T002500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_v2_20260327T002500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_enriched_allnorm_v2_20260327T002500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_allnorm_20260325T2358/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/hazard_vuln_split_eval_allnorm_20260325T2358/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_fast_20260325T210634Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_fast_20260325T210634Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_20260326T223000Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_20260326T223000Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/diverse_eval_20260326T223000Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_all_outcomes_root_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_after_missing_only_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_after_missing_only_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_09/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_09/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_00/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_00/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_rescore_all_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_rescore_all_ds/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_rescore_all_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_07/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_07/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_smoke_fix/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_06/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_06/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_01/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_01/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_20260326T152638Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_20260326T152638Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_scan_08/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_scan_08/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/20260326T005143Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/20260326T005143Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/rerun_after_dataset_fix_20260325_rescored/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/rerun_after_dataset_fix_20260325_rescored/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_features_20260327T024500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_features_20260327T024500Z/pipeline_audit_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_features_20260327T024500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_features_20260327T024500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_minimal_features_20260327T024500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_conditions_structure_proxy_20260325_v2/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_rescore_all_ctx_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_tuning_rescore_all_ctx_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_public_structure_integration_20260327T013500Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_public_structure_integration_20260327T013500Z/dataset_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/full_eval_public_structure_integration_20260327T013500Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/full_eval_public_structure_integration_20260327T013500Z/dataset_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_sample_alloutcomes_20260325T210634Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/interaction_eval_ds_sample_alloutcomes_20260325T210634Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/public_eval_ds_join_expansion_20260324/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/auc_rescore_all_probe_dup_ds/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/auc_rescore_all_probe_dup_ds/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_light_20260326T152638Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_light_20260326T152638Z/join_quality_report.json
./benchmark/public_outcomes/evaluation_dataset/diverse_conditions_featurevar_20260325/quick_validation_report.json
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_wide_20260326T152638Z/join_quality_report.md
./benchmark/public_outcomes/evaluation_dataset/eval_structure_proxy_wide_20260326T152638Z/join_quality_report.json
./tests/test_homeowner_report.py
./backend/homeowner_report.py
./docs/homeowner_report.md
./.venv/lib/python3.12/site-packages/rasterio/gdal_data/bag_template.xml
./.venv/lib/python3.12/site-packages/rasterio/gdal_data/pds4_template.xml
./.venv/lib/python3.12/site-packages/rasterio/gdal_data/pdfcomposition.xsd
./.venv/lib/python3.12/site-packages/numpy/f2py/_backends/meson.build.template
./.venv/lib/python3.12/site-packages/pip/_internal/models/installation_report.py
./.venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/reporter.py
./.venv/lib/python3.12/site-packages/pip/_vendor/resolvelib/reporters.py
./.venv/lib/python3.12/site-packages/_pytest/reports.py
./scripts/generate_benchmark_report.py
./scripts/print_release_note_template.py
./reports

### 2. What does the existing PDF generation look like?

### 3a. Current test coverage by file (collect only)
tests/test_catalog_orchestration.py::test_prepare_any_region_plan_only_no_name_error_and_policy_structure
tests/test_catalog_orchestration.py::test_cli_error_payload_flags_internal_missing_constant
tests/test_catalog_orchestration.py::test_cli_error_payload_includes_layer_execution_diagnostics
tests/test_frontend_region_handling.py::test_frontend_handles_structured_region_not_ready_errors
tests/test_geocoding.py::test_fallback_chain_raises_geocoding_error_when_all_providers_fail

### 3b. Current test run tail
....................
### 4a. Frontend API usage
frontend/public/index.html:2103:      const apiBase = (window.WILDFIRE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
frontend/public/index.html:3106:          || lowered.includes("failed to fetch")
frontend/public/index.html:4127:      async function fetchAssessmentMap(assessmentId) {
frontend/public/index.html:4148:          const payload = await fetchAssessmentMap(assessmentId);
frontend/public/index.html:4299:        const res = await fetch(`${apiBase}${path}`, opts);
frontend/public/index.html:7062:      async function fetchCoverageForAddress(address) {
frontend/public/index.html:7066:      async function fetchCoverageForCoordinates(latitude, longitude, address = "") {
frontend/public/index.html:7078:      async function fetchManualAddressCandidates(address, zipCode = "", locality = "") {
frontend/public/index.html:7712:          const result = await fetchManualAddressCandidates(
frontend/public/index.html:8045:          const coverageResult = await fetchCoverageForCoordinates(lat, lon, pendingAssessmentPayload.address);
frontend/public/index.html:8287:          const verification = await fetchCoverageForAddress(payload.address);
frontend/public/index.html:8582:              const coverage = await fetchCoverageForAddress(submittedAddress);
frontend/public/index.html:8602:              const coverage = await fetchCoverageForAddress(submittedAddress);
frontend/public/index.html:8989:          const res = await fetch(`${apiBase}/report/${latestAssessment.assessment_id}/homeowner/pdf`, {

### 4b. Backend routes
13325:@app.post("/risk/geocode-debug", dependencies=[Depends(require_api_key)])
13342:@app.post("/debug/geocode", dependencies=[Depends(require_api_key)])
13348:@app.post(
13967:@app.get("/health")
13980:@app.get("/organizations", response_model=list[Organization], dependencies=[Depends(require_api_key)])
13988:@app.post("/organizations", response_model=Organization, dependencies=[Depends(require_api_key)])
14006:@app.get("/organizations/{organization_id}", response_model=Organization, dependencies=[Depends(require_api_key)])
14018:@app.get("/underwriting/rulesets", response_model=list[UnderwritingRuleset], dependencies=[Depends(require_api_key)])
14023:@app.get(
14035:@app.post(
14057:@app.post("/regions/prepare", response_model=RegionPrepJobStatus, dependencies=[Depends(require_api_key)])
14102:@app.get("/regions/prepare/{job_id}", response_model=RegionPrepJobStatus, dependencies=[Depends(require_api_key)])
14110:@app.post("/regions/coverage-check", response_model=RegionCoverageStatus, dependencies=[Depends(require_api_key)])
14162:@app.post(
14411:@app.post("/risk/reassess/{assessment_id}", response_model=AssessmentResult, dependencies=[Depends(require_api_key)])
14471:@app.get(
14494:@app.post(
14927:@app.post("/risk/simulate", response_model=SimulationResult, dependencies=[Depends(require_api_key)])
15095:@app.post("/risk/debug", dependencies=[Depends(require_api_key)])
15138:@app.post("/risk/layer-diagnostics", dependencies=[Depends(require_api_key)])
15280:@app.get("/internal/diagnostics", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
15298:@app.get("/internal/diagnostics/api/runs", dependencies=[Depends(require_api_key)])
15303:@app.get("/internal/diagnostics/api/latest", dependencies=[Depends(require_api_key)])
15320:@app.get("/internal/diagnostics/api/run/{run_id}", dependencies=[Depends(require_api_key)])
15340:@app.get("/internal/diagnostics/api/compare", dependencies=[Depends(require_api_key)])
15352:@app.get("/internal/diagnostics/api/public-outcomes", dependencies=[Depends(require_api_key)])
15368:@app.get("/internal/diagnostics/api/latest/{section_key}", dependencies=[Depends(require_api_key)])
15393:@app.get("/internal/diagnostics/api/run/{run_id}/{section_key}", dependencies=[Depends(require_api_key)])
15419:@app.post("/portfolio/assess", response_model=BatchAssessmentResponse, dependencies=[Depends(require_api_key)])
15447:@app.post("/portfolio/jobs", response_model=PortfolioJobStatus, dependencies=[Depends(require_api_key)])
15506:@app.get("/portfolio/jobs/{job_id}", response_model=PortfolioJobStatus, dependencies=[Depends(require_api_key)])
15515:@app.get(
15532:@app.get(
15557:@app.get("/portfolio/jobs/{job_id}/report-pack", dependencies=[Depends(require_api_key)])
15600:@app.get("/portfolio/jobs/summary", response_model=PortfolioJobsSummary, dependencies=[Depends(require_api_key)])
15610:@app.post("/portfolio/import/csv", response_model=CSVImportResponse, dependencies=[Depends(require_api_key)])
15678:@app.get(
15704:@app.get(
15736:@app.get("/report/{assessment_id}/export", response_model=ReportExport, dependencies=[Depends(require_api_key)])
15766:@app.get("/report/{assessment_id}/view", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
15799:@app.get("/report/{assessment_id}/map", response_model=AssessmentMapPayload, dependencies=[Depends(require_api_key)])
15846:@app.get("/report/{assessment_id}/homeowner", response_model=HomeownerReport, dependencies=[Depends(require_api_key)])
15884:@app.get("/report/{assessment_id}/homeowner/pdf", dependencies=[Depends(require_api_key)])
15929:@app.post("/analytics/homeowner/event", dependencies=[Depends(require_api_key)])
15957:@app.get("/portfolio", response_model=PortfolioResponse, dependencies=[Depends(require_api_key)])
16007:@app.get("/assessments", response_model=list[AssessmentListItem], dependencies=[Depends(require_api_key)])
16055:@app.get("/assessments/summary", response_model=AssessmentSummaryResponse, dependencies=[Depends(require_api_key)])
16097:@app.post(
16138:@app.get(
16164:@app.post(
16177:@app.get(
16191:@app.put(
16220:@app.get(
16240:@app.post(
16274:@app.post(
16315:@app.get(
16342:@app.get(
16365:@app.get(
16378:@app.get(
16404:@app.get(

### 5. Frontend TODO/FIXME markers
frontend/public/index.html:550:      input::placeholder {
frontend/public/index.html:1496:              <input id="address" placeholder="123 Main St, Boulder, CO" />
frontend/public/index.html:1516:              <input id="roof_type_other" class="other-input" placeholder="Enter roof type" />
frontend/public/index.html:1528:              <input id="vent_type_other" class="other-input" placeholder="Enter vent type" />
frontend/public/index.html:1541:              <input id="siding_type_other" class="other-input" placeholder="Enter siding material" />
frontend/public/index.html:1554:              <input id="deck_attachment_other" class="other-input" placeholder="Enter deck/fence detail" />
frontend/public/index.html:1578:              <input id="year_built_other" class="other-input" placeholder="Enter year or range" />
frontend/public/index.html:1584:            <input id="confirmed_fields" placeholder="roof_type, vent_type, defensible_space_ft" />
frontend/public/index.html:1861:                <input id="sim_roof_type_other" class="other-input" placeholder="Enter roof type" />
frontend/public/index.html:1873:                <input id="sim_vent_type_other" class="other-input" placeholder="Enter vent type" />
frontend/public/index.html:1886:                <input id="sim_siding_type_other" class="other-input" placeholder="Enter siding material" />
frontend/public/index.html:1899:                <input id="sim_deck_attachment_other" class="other-input" placeholder="Enter deck/fence detail" />
frontend/public/index.html:1923:                <input id="sim_year_built_other" class="other-input" placeholder="Enter year or range" />
frontend/public/index.html:1929:              <input id="scenario_confirmed_fields" placeholder="roof_type, vent_type, defensible_space_ft" />
frontend/public/index.html:2049:                <input id="verifyZipInput" type="text" placeholder="e.g., 98862" />
frontend/public/index.html:3315:        // TODO(calibration): replace these text heuristics with empirically validated action-to-override mappings.
frontend/public/index.html:5685:        const placeholderSnippets = [
frontend/public/index.html:5690:        if (reasonLines.length > 0 && placeholderSnippets.some((snippet) => String(reasonsEl.textContent || "").includes(snippet))) {
frontend/public/index.html:5691:          console.warn("[homeowner-step2-dom] reasons list failed to replace placeholder", {
frontend/public/index.html:5698:        if (driverLines.length > 0 && placeholderSnippets.some((snippet) => String(driversEl.textContent || "").includes(snippet))) {
frontend/public/index.html:5699:          console.warn("[homeowner-step2-dom] drivers list failed to replace placeholder", {
frontend/public/index.html:5706:        if (actionRows.length > 0 && placeholderSnippets.some((snippet) => String(actionsEl.textContent || "").includes(snippet))) {
frontend/public/index.html:5707:          console.warn("[homeowner-step2-dom] actions list failed to replace placeholder", {

### 6a. Backend error handling signals count
     599

### 6b. Frontend error handling signals count
     128

### 7. Auth and rate limiting signals
backend/main.py:21:from backend.auth import get_key_org, require_api_key
backend/main.py:514:    x_api_key: str | None = Header(default=None),
backend/main.py:520:    if x_api_key:
backend/main.py:521:        key_org = get_key_org(x_api_key)
backend/main.py:629:    """Shared auth/org/ruleset setup for write endpoints.
backend/main.py:5391:    token = str(layer_key or "").strip()
backend/main.py:5392:    if not token:
backend/main.py:5394:    if token in LAYER_FRIENDLY_NAMES:
backend/main.py:5395:        return LAYER_FRIENDLY_NAMES[token]
backend/main.py:5396:    return token.replace("_", " ")
backend/main.py:5437:        token = text.lower()
backend/main.py:5438:        if not text or token in seen:
backend/main.py:5440:        seen.add(token)
backend/main.py:5659:        token = reason.strip().lower()
backend/main.py:5660:        if not token or token in seen:
backend/main.py:5662:        seen.add(token)
backend/main.py:5878:    token = str(field_name or "").strip().lower()
backend/main.py:5887:    if token in mapping:
backend/main.py:5888:        return mapping[token]
backend/main.py:5889:    return token.replace("_", " ")

### 8. PDF report implementation hooks

### 9. Frontend PDF download flow
frontend/public/index.html:742:      .report-journey-surface {
frontend/public/index.html:1632:          <p id="primaryDecisionCtaPriorityNote" class="action-priority-note">This is the main next step. More report and detail options are available below.</p>
frontend/public/index.html:1638:            <p class="post-simulation-action-note">Open your full report first. Download/share options stay available in the report section.</p>
frontend/public/index.html:1649:      <section id="reportJourneySection" class="card secondary-panel" style="display:none;">
frontend/public/index.html:1650:        <section id="reportShareSection" class="report-journey-surface">
frontend/public/index.html:1651:          <h3>Review your full report</h3>
frontend/public/index.html:1652:          <p class="muted">Save or share your results with homeowner-friendly report formats.</p>
frontend/public/index.html:1654:          <div id="reportPrimaryActions" class="actions actions-low-emphasis">
frontend/public/index.html:1656:            <button id="downloadPdfBtn" type="button" class="secondary">Download Homeowner-Friendly PDF</button>
frontend/public/index.html:1658:          <p id="reportActionHierarchyHint" class="section-help muted" style="display:none;">
frontend/public/index.html:1659:            Your main report actions are shown above in the homeowner summary.
frontend/public/index.html:1661:          <div id="reportInlineActionRow" class="actions actions-low-emphasis">
frontend/public/index.html:1662:            <button id="reportInlineOpenBtn" type="button" class="secondary">Open full report</button>
frontend/public/index.html:1663:            <button id="reportInlineDownloadBtn" type="button" class="context-link-btn">Download homeowner-friendly PDF</button>
frontend/public/index.html:1665:          <p id="reportActionStatus" class="status-line status-secondary tone-muted">Report review actions will appear after simulation.</p>
frontend/public/index.html:1668:            <p id="homeownerReportStatus" class="status-line status-secondary tone-muted">Load highlights to see a short report preview.</p>
frontend/public/index.html:1670:              <button id="loadReportHighlightsBtn" type="button" class="context-link-btn">Load report highlights</button>
frontend/public/index.html:1672:            <ul id="homeownerReportHighlights"><li class="muted">No report highlights yet.</li></ul>
frontend/public/index.html:1673:            <p class="mini-label mt-sm">More report notes</p>
frontend/public/index.html:2173:      let reportPreviewLoadedAssessmentId = null;
frontend/public/index.html:2174:      let reportPrimarySurfaceOpened = false;
frontend/public/index.html:2993:        const reportJourneySection = document.getElementById("reportJourneySection");
frontend/public/index.html:3003:        if (reportJourneySection) {
frontend/public/index.html:3004:          reportJourneySection.style.display = (hasAssessment && hasSimulationResult) ? "block" : "none";
frontend/public/index.html:3022:        const reportSection = document.getElementById("reportJourneySection");
frontend/public/index.html:3024:        if (!reportSection || !simulationSection) return;
frontend/public/index.html:3025:        if (reportSection.previousElementSibling === simulationSection) return;
frontend/public/index.html:3026:        simulationSection.insertAdjacentElement("afterend", reportSection);
frontend/public/index.html:3210:          "reportActionStatus",
frontend/public/index.html:3231:        const el = document.getElementById("reportActionStatus");

### 10. Accessibility/mobile indicators
frontend/public/index.html:5:    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
frontend/public/index.html:214:      .mobile-next-step-bar {
frontend/public/index.html:229:      .mobile-next-step-bar.show {
frontend/public/index.html:234:      .mobile-next-step-inner {
frontend/public/index.html:238:      .mobile-next-step-label {
frontend/public/index.html:245:      .mobile-next-step-summary {
frontend/public/index.html:250:      .mobile-next-step-btn {
frontend/public/index.html:1465:        .mobile-next-step-visible .wrap {
frontend/public/index.html:1468:        .mobile-next-step-visible .homeowner-toast {
frontend/public/index.html:1473:        .mobile-next-step-bar {
frontend/public/index.html:1797:                  <div class="map-basemap-toggle" role="group" aria-label="Basemap">
frontend/public/index.html:1994:      role="dialog"
frontend/public/index.html:1995:      aria-modal="true"
frontend/public/index.html:1996:      aria-hidden="true"
frontend/public/index.html:1997:      aria-labelledby="addressVerifyModalTitle"
frontend/public/index.html:1999:      <div class="modal-card" role="document">
frontend/public/index.html:2031:            <div id="verifyLocationMap" aria-label="Address verification map"></div>
frontend/public/index.html:2086:    <div id="mobileNextStepBar" class="mobile-next-step-bar" aria-live="polite" hidden>
frontend/public/index.html:2087:      <div class="mobile-next-step-inner">
frontend/public/index.html:2088:        <div class="mobile-next-step-label">Next best step</div>

### 11. Environment variable usage
backend/address_resolution.py:69:    raw = os.getenv(name)
backend/address_resolution.py:290:    config_path_raw = str(os.getenv("WF_LOCATION_RESOLUTION_SOURCE_CONFIG") or "").strip()
backend/address_resolution.py:334:        raw = str(os.getenv(env_key) or "").strip()
backend/address_resolution.py:490:    min_point_ratio_raw = str(os.getenv("WF_ADDRESS_POINT_MIN_POINT_RATIO", "0.85")).strip()
backend/address_resolution.py:496:    min_complete_ratio_raw = str(os.getenv("WF_ADDRESS_POINT_MIN_COMPLETE_RATIO", "0.1")).strip()
backend/address_resolution.py:762:    min_score_raw = str(os.getenv("WF_LOCAL_ADDRESS_MATCH_MIN_SCORE", "0.76")).strip()
backend/address_resolution.py:1259:    max_features_per_source_raw = str(os.getenv("WF_ZIP_LOCALITY_SCAN_MAX_FEATURES", "60000")).strip()
backend/assessment_map.py:995:    candidate_limit_raw = str(os.getenv("WF_SELECTABLE_STRUCTURE_MAX_CANDIDATES", "80")).strip()
backend/auth.py:23:    raw = os.getenv("WILDFIRE_API_KEYS", "")
backend/building_footprints.py:60:            os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", ""),
backend/building_footprints.py:61:            os.getenv("WF_LAYER_FEMA_STRUCTURES_GEOJSON", ""),
backend/building_footprints.py:99:        raw = str(os.getenv(name, str(default))).strip()
backend/calibration.py:205:    configured_path = str(artifact_path or os.getenv("WF_PUBLIC_CALIBRATION_ARTIFACT", "")).strip()
backend/feature_bundle_cache.py:12:    raw = str(os.getenv(name, str(default))).strip().lower()
backend/feature_bundle_cache.py:32:            or os.getenv("WF_FEATURE_BUNDLE_CACHE_DIR")
backend/feature_bundle_cache.py:38:        ttl_raw = os.getenv("WF_FEATURE_BUNDLE_CACHE_TTL_SEC", str(6 * 3600))
backend/feature_enrichment.py:255:                env_path = str(os.getenv(env_name, "")).strip()
backend/geocoding.py:93:        timeout_raw = os.getenv("WF_GEOCODE_TIMEOUT_SECONDS", "8")
backend/geocoding.py:94:        runtime_env = str(os.getenv("WF_ENV") or os.getenv("APP_ENV") or "").strip().lower()
backend/geocoding.py:97:            or bool(os.getenv("PYTEST_CURRENT_TEST"))
backend/geocoding.py:98:            or str(os.getenv("WF_DEBUG_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}
backend/geocoding.py:101:        min_importance_raw = os.getenv("WF_GEOCODE_MIN_IMPORTANCE", default_min_importance)
backend/geocoding.py:102:        ambiguity_delta_raw = os.getenv("WF_GEOCODE_AMBIGUITY_DELTA", "0.0")
backend/geocoding.py:103:        max_candidates_raw = os.getenv("WF_GEOCODE_MAX_CANDIDATES", "5")
backend/geocoding.py:104:        max_query_variants_raw = os.getenv("WF_GEOCODE_MAX_QUERY_VARIANTS", "7")
backend/geocoding.py:105:        allow_precise_low_importance_raw = os.getenv("WF_GEOCODE_ALLOW_PRECISE_LOW_IMPORTANCE", "true")
backend/geocoding.py:134:        self.provider_name = str(provider_name or os.getenv("WF_GEOCODE_PROVIDER_NAME") or "OpenStreetMap Nominatim")
backend/geocoding.py:135:        configured_search_url = str(search_url or os.getenv("WF_GEOCODE_SEARCH_URL") or "").strip()
backend/geocoding_fallback_chain.py:74:        os.getenv("WF_GEOCODING_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
backend/geocoding_fallback_chain.py:441:        google_api_key = str(os.getenv("WF_GOOGLE_GEOCODE_API_KEY") or "").strip()
backend/geometry_source_registry.py:109:    env_path = str(os.getenv("WF_GEOMETRY_SOURCE_REGISTRY_PATH", "")).strip()
backend/homeowner_report.py:2090:        api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
backend/homeowner_report.py:2102:    model = str(os.getenv("WF_HOMEOWNER_EXPLANATION_MODEL") or "gpt-4o-mini").strip()
backend/main.py:210:    user_agent=os.getenv("WF_GEOCODE_SECONDARY_USER_AGENT", _nominatim_geocoder.user_agent),
backend/main.py:211:    provider_name=os.getenv("WF_GEOCODE_SECONDARY_PROVIDER_NAME", "Secondary Geocoder"),
backend/main.py:212:    search_url=os.getenv("WF_GEOCODE_SECONDARY_SEARCH_URL", ""),
backend/main.py:4219:        raw = os.getenv(name)
backend/main.py:4323:            os.getenv("WF_LAYER_BURN_PROB_VERSION"),
backend/main.py:4324:            os.getenv("WF_LAYER_BURN_PROB_DATE"),
backend/main.py:4330:            os.getenv("WF_LAYER_HAZARD_SEVERITY_VERSION"),

### 12. Logging count
      38

### 13. SQLite usage
backend/database.py:6:import sqlite3
backend/database.py:95:    def __init__(self, db_path: str = "wildfire_app.db") -> None:
backend/database.py:96:        self.db_path = Path(db_path)
backend/database.py:103:    def _connect(self) -> sqlite3.Connection:
backend/database.py:106:            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
backend/database.py:107:            conn.row_factory = sqlite3.Row
backend/database.py:119:                CREATE TABLE IF NOT EXISTS assessments (
backend/database.py:132:                CREATE TABLE IF NOT EXISTS organizations (
backend/database.py:143:                CREATE TABLE IF NOT EXISTS underwriting_rulesets (
backend/database.py:156:                CREATE TABLE IF NOT EXISTS assessment_scenarios (
backend/database.py:167:                CREATE TABLE IF NOT EXISTS assessment_improvement_snapshots (
backend/database.py:178:                CREATE TABLE IF NOT EXISTS assessment_annotations (
backend/database.py:203:                CREATE TABLE IF NOT EXISTS assessment_review_status (
backend/database.py:219:                CREATE TABLE IF NOT EXISTS assessment_workflow (
backend/database.py:232:                CREATE TABLE IF NOT EXISTS portfolio_jobs (
backend/database.py:247:                CREATE TABLE IF NOT EXISTS region_prep_jobs (
backend/database.py:288:                CREATE TABLE IF NOT EXISTS audit_events (
backend/database.py:304:    def _seed_organizations(self, conn: sqlite3.Connection) -> None:
backend/database.py:306:            "SELECT organization_id FROM organizations WHERE organization_id = ?",
backend/database.py:312:                INSERT INTO organizations (organization_id, organization_name, organization_type, created_at)
backend/database.py:318:    def _seed_rulesets(self, conn: sqlite3.Connection) -> None:
backend/database.py:321:                "SELECT ruleset_id FROM underwriting_rulesets WHERE ruleset_id = ?",
backend/database.py:327:                    INSERT INTO underwriting_rulesets
backend/database.py:345:                "SELECT organization_id FROM organizations WHERE organization_id = ?",
backend/database.py:352:                INSERT INTO organizations (organization_id, organization_name, organization_type, created_at)
backend/database.py:459:                "SELECT ruleset_id FROM underwriting_rulesets WHERE ruleset_id = ?",
backend/database.py:466:                INSERT INTO underwriting_rulesets
backend/database.py:508:                INSERT INTO assessment_scenarios (scenario_id, assessment_id, created_at, scenario_name, payload_json)
backend/database.py:577:                INSERT INTO assessment_review_status (assessment_id, organization_id, review_status, updated_at)
backend/database.py:597:                "SELECT assessment_id, organization_id, review_status, updated_at FROM assessment_review_status WHERE assessment_id = ?",

### 14. CORS/security config
backend/main.py:18:from fastapi.middleware.cors import CORSMiddleware
backend/main.py:197:app.add_middleware(
backend/main.py:198:    CORSMiddleware,
backend/main.py:199:    allow_origins=["*"],

### 15. py_compile checks
[Errno 2] No such file or directory: 'backend/report_pdf.py'
```
