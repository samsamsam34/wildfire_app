# Benchmark Low-Score Audit (Read-Only)

## Scope
- Date: 2026-04-30
- Audit target: Why fallback-heavy / missing-geometry benchmark rows can produce very low wildfire risk scores.
- Constraints followed: no backend/frontend/config/API edits; no scoring logic changes.

## Short Diagnosis
Very low `risk_score` values in fallback-heavy rows are caused by a combination of:
1. **Expected sparse-evidence scoring behavior** in the existing engine (weight suppression/omission under weak geometry and fallback-heavy evidence),
2. **Availability gating that still publishes a numeric wildfire score** when *either* site or home component is minimally usable,
3. **Benchmark fixture/harness mismatch** where `high_regional_hazard` labels are not passed into model inputs (`regional_hazard_hint` is ignored),
4. **Communication gap** where confidence and restriction fields are low/preliminary, but benchmark consumers may still read near-zero risk as a strong “low-risk” claim.

---

## Observed Benchmark Evidence
From [reports/benchmark_results.csv](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/reports/benchmark_results.csv):
- `ex_006` (`high_regional_hazard`): `risk_score=0.40`, `confidence_score=0.00`
- `ex_007` (`high_regional_hazard`): `risk_score=1.80`, `confidence_score=0.00`
- Most low-score rows also carry flags: `fallback_heavy`, `missing_geometry`, `prepared_region_unavailable`, `very_low_confidence`.

Diagnostics summary in [reports/benchmark_diagnostics.md](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/reports/benchmark_diagnostics.md):
- `high_regional_hazard` suspicious low-risk rows: **2**
- very low confidence rows: **12**
- fallback-heavy rows: **12**
- missing geometry rows: **12**

---

## `/risk/assess` Scoring Path (traced)
- Route entry: `backend/main.py:14162-14408` (`POST /risk/assess`)
- Assessment computation: `_compute_assessment(...)` then `_run_assessment(...)`
- Core scoring call: `risk_engine.score(...)` at `backend/main.py:7541`
- Readiness scoring call: `risk_engine.compute_insurance_readiness(...)` at `backend/main.py:7542`
- Blended wildfire score computation: `backend/main.py:7852-7857` calling `risk_engine.compute_blended_wildfire_score(...)`
- Score availability gating: `_apply_score_availability(...)` at `backend/main.py:6560-6622`

---

## Field Production Trace (requested fields)

### 1) `risk_score` (benchmark column)
- Benchmark mapping: `scripts/run_benchmark_properties.py:368` pulls `assessment["wildfire_risk_score"]`.
- API field assignment: `AssessmentResult.wildfire_risk_score` at `backend/main.py:8559`.
- Source value path:
  - raw blend computed at `backend/main.py:7852-7857`
  - then gated by `_apply_score_availability` (`6560-6622`), including component-only fallback (`6590-6601`).

### 2) `insurance_readiness_score`
- Benchmark mapping: `scripts/run_benchmark_properties.py:369`.
- API field assignment: `backend/main.py:8564`.
- Gated nullability: `_apply_score_availability` may set `None` if readiness eligibility is insufficient (`6613`, `6583-6586`).

### 3) `confidence_score`
- Benchmark mapping: `scripts/run_benchmark_properties.py:370`.
- API field assignment: `backend/main.py:8630`.
- Built from `_build_confidence(...)` (`7728-7739`) plus `_apply_hard_trust_guardrails(...)` (`7758-7767`, definition `5012-5105`).

### 4) `assessment_status`
- Produced by `_build_score_eligibility(...)` (`4717-4902`) and then adjusted by `_apply_preflight_specificity_gate(...)` (`4905-5009`).
- Serialized at `backend/main.py:8852`.
- Used in benchmark flags at `scripts/run_benchmark_properties.py:277-279`.

### 5) `missing_data_flags` (benchmark-only)
- Not a backend-native field.
- Produced in `scripts/run_benchmark_properties.py:_extract_missing_data_flags` (`271-311`) from:
  - `assessment_status`
  - null scores
  - `confidence_score`
  - `coverage_available`
  - `what_was_missing`
  - `fallback_weight_fraction`
  - `low_confidence_flags`

### 6) `key_drivers` (benchmark column)
- Produced in script (`236-253`) from `top_risk_drivers` (preferred) then `top_risk_drivers_detailed`.
- Backend top-driver generation in `_run_assessment` (`7819-7838`) with sparse-evidence fallback seed (`7830`) and additional safeguard substitution (`7908-7910`, `8085-8099`).

---

## Why Very Low Scores Happen

## A) Sparse evidence can collapse weighted risk contributions
In `risk_engine.score(...)`:
- If no available submodel inputs, `weighted_score(...)` returns `0.0` (`backend/risk_engine.py:159-160`).
- Missing/suppressed factors set `effective_weight=0` and are omitted from total weighting (`1473-1499`, `1499-1508`).
- Final score is recomputed from only remaining effective weights (`1546-1557`) and clamped (`1648`).

Net effect: under weak geometry + fallback-heavy evidence, the total can be driven by a small residual subset and land near zero.

## B) Weak-geometry suppression is explicit for geometry-sensitive factors
- Point/weak-geometry context can suppress or sharply reduce key geometry-sensitive factors (`1476-1488`).
- Additional availability multipliers also reduce those factors under point/parcel fallback conditions (`1360-1434`).

This can remove risk-contributing factors from numeric aggregation rather than treating them as “unknown but potentially high”.

## C) Availability gate still publishes numeric risk in partial mode
`_apply_score_availability(...)` allows wildfire score when either site or home is available (`6574`, `6614`), and may publish a component-only score (`6590-6601`).

So partially-scored runs can still emit a numeric wildfire score, even when confidence is near zero.

## D) Preflight gate prefers constrained estimate over hard null when any component remains usable
`_apply_preflight_specificity_gate(...)` can keep output as `limited_regional_estimate` instead of forcing `insufficient_data` if at least one component is still usable (`4957-4970`).

That preserves a score in low-specificity contexts.

## E) Confidence guardrails do not nullify low numeric wildfire score
`_apply_hard_trust_guardrails(...)` downgrades confidence/use restriction (`5027-5054`) but does not zero/null wildfire score.

Hence: very low confidence + very low risk can coexist.

## F) Benchmark “high_regional_hazard” labels are not model inputs
In fixture rows, `regional_hazard_hint` is present (`tests/fixtures/benchmark_properties.csv:7-8`) but is ignored by script normalization (`scripts/run_benchmark_properties.py:149-171`), and only known attribute keys are passed through.

So scenario-group labels do not force a high-hazard condition in scoring; they are test metadata only.

---

## Missing Components Treatment (requested classification)
For fallback-heavy / missing-geometry rows, missing evidence is generally:
- **Omitted/suppressed** from weighting (`1499-1508`, `1476-1499`),
- **Downweighted** via availability multipliers (`1360-1434`),
- **Renormalized over remaining effective weights** (`1546-1557`),
- **Not capped by confidence logic** at the wildfire-score level (confidence is gated separately at `5012-5105`).

---

## Is This a Bug or Expected?
Current behavior appears to be a **combination**:
- **Expected fallback behavior** in current engine design (weight suppression + partial-score publication),
- **Benchmark harness/design issue** (`high_regional_hazard` hints ignored),
- **Confidence/display communication issue** (low-confidence outputs can still look like confidently low risk if consumer ignores trust fields).

I did not find evidence of a route-level/API serialization bug in this path.

---

## Top 3 Safest Fixes To Consider Next (recommendations only)
1. **Benchmark harness guardrails (lowest risk):**
   - In benchmark reporting, explicitly classify rows as “not comparable for absolute risk” when `coverage_available=false`, `fallback_weight_fraction` high, and `confidence_score` very low.
   - Keep model untouched; improve interpretation.

2. **Benchmark fixture/input quality controls:**
   - Add a fixture validation warning when scenario intent keys (e.g., `regional_hazard_hint`) are ignored by `_normalize_optional_inputs`.
   - Prevent “high_regional_hazard” rows from being interpreted as true hazard tests unless mapped to real scoring inputs.

3. **Policy-level output handling for low-trust numeric wildfire scores (design decision):**
   - Consider a post-score presentation policy (not model change) that suppresses/emphasizes caution for numeric wildfire scores when trust thresholds fail.
   - This can be done in reporting/auditing layers first before any scoring change.

---

## Files Reviewed
- [backend/main.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/main.py)
- [backend/risk_engine.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/risk_engine.py)
- [backend/models.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/backend/models.py)
- [scripts/run_benchmark_properties.py](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/scripts/run_benchmark_properties.py)
- [tests/fixtures/benchmark_properties.csv](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/tests/fixtures/benchmark_properties.csv)
- [reports/benchmark_results.csv](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/reports/benchmark_results.csv)
- [reports/benchmark_diagnostics.md](/Users/samneitlich/Documents/wildfire_risk_app/wildfire_app/reports/benchmark_diagnostics.md)

## Change Confirmation
- No backend files changed.
- No frontend files changed.
- No config files changed.
- No API route behavior changed.
