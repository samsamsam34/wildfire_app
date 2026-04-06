# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A property-level wildfire risk and insurance readiness assessment platform. It produces three scores per property:
- `site_hazard` — environmental exposure (vegetation, slope, ember, flame contact, historic fire)
- `home_ignition_vulnerability` — structural vulnerability (structure + defensible space submodels)
- `home_hardening_readiness` — how prepared the property is with mitigations in place

The scoring engine is deterministic and explainable. Confidence and evidence quality are tracked explicitly. Missing data degrades confidence/specificity, not numeric scores.

## Running the App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export WILDFIRE_API_KEYS="dev-key-1"   # optional for local dev
uvicorn backend.main:app --reload      # http://127.0.0.1:8000
```

## Running Tests and Validation

```bash
# Unit/integration tests
pytest
pytest tests/test_homeowner_report.py -v   # single file

# Required benchmark checks after meaningful changes
python scripts/run_benchmark_suite.py
python scripts/run_confidence_benchmark_pack.py
python scripts/run_no_ground_truth_evaluation.py
```

## Architecture

**`backend/main.py`** (~14,700 lines) — Single FastAPI app defining all endpoints. Key endpoint groups:
- `POST /risk/assess` — primary homeowner assessment
- `POST /risk/simulate` — what-if mitigation scenarios
- `POST /risk/debug` — full diagnostics with confidence/evidence metadata
- `GET /report/{id}/homeowner` — homeowner-facing report
- `POST /portfolio/assess` — batch assessment
- `POST /regions/prepare` — queue region data prep job

**Core scoring pipeline:**
1. `address_resolution.py` + `geocoding.py` → resolve address to coordinates
2. `property_anchor.py` → determine authoritative property location (parcel > footprint > geocode point)
3. `wildfire_data.py` → sample GIS raster/vector layers for the property location
4. `building_footprints.py` → extract structure metrics in defensible-space rings (0-5ft, 5-30ft, 30-100ft, 100-300ft)
5. `feature_enrichment.py` → aggregate features with source tracking and fallback policies
6. `risk_engine.py` → compute 8 submodel scores → blend into final scores
7. `trust_metadata.py` → compute confidence, evidence quality, and guidance
8. `homeowner_report.py` / `homeowner_advisor.py` → generate plain-language outputs

**Configuration:**
- `config/scoring_parameters.yaml` — submodel weights, blending weights, risk bucket thresholds
- `config/source_registry.json` — data layer paths
- `config/geometry_source_registry.json` — parcel/footprint source mapping
- `config/public_outcome_calibration.json` — calibration artifact config

**Data preparation** (`backend/data_prep/`, `scripts/`): Region-level GIS layers are prepared offline and stored under `data/`. `prepare_region_from_catalog_or_sources.py` is the primary builder. Prepared regions are looked up at assessment time via `data_prep/region_lookup.py`.

**Versioning** (`backend/version.py`): Multiple semver dimensions tracked on every response — `scoring_model_version`, `rules_logic_version`, `factor_schema_version`, `calibration_version`. Comparability keys matter for audit and regression tracking.

## Known Architectural Issues (Prioritized)

These are findings from an architectural review — do not re-analyze, address them as separate scoped tasks.

### P1 — High Severity

**1. Silent exception swallowing throughout**
`main.py:2926`, `main.py:3006`, `main.py:8276`, `database.py:700` use bare `except Exception: pass/continue/return None` with no logging. Masks data corruption and runtime errors. Replace with structured logging or explicit propagation.

**2. 50+ hardcoded coefficients in `risk_engine.py`**
Scoring transformation curves (lines 212–420) use hardcoded numeric constants (e.g. `0.60`, `1.35`, `1.55`, distance thresholds 300 ft / 120 ft, weighting coefficients 14.0 / 2.5). These cannot be tuned, A/B tested, or overridden without code changes. They belong in `scoring_config.py` alongside the existing weight config.

**3. Duplicate auth/org/ruleset chain across endpoints**
`/risk/assess`, `/portfolio/assess`, `/portfolio/jobs` each inline the same sequence: `_require_role` → `_resolve_org_id` → `_enforce_org_scope` → ruleset lookup → audit log. No shared abstraction. Any security or behavior change must be applied three places.

### P2 — Medium Severity

**4. SQLite connection-per-query, no pooling**
`database.py:95–98` opens a fresh `sqlite3.connect()` on every operation. Batch jobs with 100+ assessments create 100+ connections. Extract a shared connection or use a simple pool.

**5. 18 inline env-var parse blocks in `main.py:9974–10051`**
Resolver parameters (`WF_RESOLVER_CONFLICT_DISTANCE_M`, etc.) are parsed with 18 separate try/except blocks inlined in main. Extract to a `ResolverConfig` dataclass in the config layer (mirrors the pattern in `scoring_config.py`).

**6. Inconsistent audit trail**
Two separate audit mechanisms coexist: `LOGGER.info()` (stdout) and `store.log_event()` (DB). Some operations hit both, some only one, some neither. No consistent policy. Standardize so security-relevant operations always reach the DB audit table.

**7. No async I/O in request handlers**
All FastAPI endpoints are synchronous `def`. Assessment endpoints call blocking GIS data sampling, CPU-bound scoring, and SQLite writes in the request thread. Under concurrent load a single slow assessment blocks a worker. Mark compute-heavy handlers as `async` or offload to a thread pool with `run_in_executor`.

### P3 — Low / Structural

**8. `trust_metadata.py` thresholds undocumented and unconfigurable**
Stability rating thresholds (`12.0`, `6.0`, multiplier `6.0`) and fallback-heaviness threshold (`0.45`) in `trust_metadata.py:289–354` have no documented rationale and no config override. They should either live in `scoring_config.py` or carry inline comments explaining their derivation.

**9. API key model has no org scope**
`auth.py` uses a flat comma-delimited set of keys with no per-key permissions, org binding, or usage logging. All keys have identical access. This blocks multi-tenant B2B use cases.

**10. All `PropertyAttributes` fields are Optional with no range validation**
`models.py` accepts any int for `construction_year`, any float for `defensible_space_ft`. Invalid values pass Pydantic validation silently. Add `Field(ge=..., le=...)` guards at model boundaries.

---

## Development Standards (from IMPLEMENT.md)

**Priority order when making changes:**
1. Reliability and internal consistency
2. Confidence correctness under weak/missing evidence
3. Monotonic mitigation behavior (hardening always reduces readiness score)
4. Benchmark and regression coverage

**Hard rules:**
- Keep diffs narrowly scoped — do not change multiple subsystems in one task
- Prefer deterministic and explainable logic over opaque complexity
- Never present a change as an improvement without validation evidence
- Do not rewrite architecture unless fixing a clearly identified reliability issue

**Expected work pattern:**
1. Read `IMPLEMENT.md`, `PLAN.md` (if present), and `METRICS.md` (if present)
2. Establish a baseline before changing anything
3. Make one narrowly scoped improvement
4. Re-run relevant validations
5. Compare before/after results; stop if improvement isn't clearly supported

**Stop and report instead of continuing when:** regressions appear, no clear improvement, labeled data is too weak to justify predictive optimization, or further iteration risks overfitting.

**Required output format for task completion:**
1. Baseline summary
2. What changed and why it should help
3. Validation results
4. Before/after metric deltas
5. Risks or uncertainties
6. Recommended next experiment
7. Suggested commit message
