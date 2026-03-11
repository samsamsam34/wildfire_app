# Open Data Model Upgrade Plan

## Root Causes Found (Pre-Change Audit)

1. **Neighborhood-level averaging was drowning parcel-level variation.**  
   Fuel/canopy context was heavily driven by 100m neighborhood sampling, so nearby homes often looked too similar.

2. **Near-structure ring features were often sparse or proxy-only.**  
   Missing/weak footprint matches triggered point-annulus fallback, reducing structure-specific discrimination.

3. **Submodels reused overlapping core inputs.**  
   Multiple submodels leaned on the same few environmental indices, which compressed contribution variance.

4. **Final blending still favored stable regional context over home-specific modifiers.**  
   Structural/readiness deltas were present but not always strong enough to separate materially different homes.

5. **Fallback behavior protected availability but could converge outputs.**  
   Confidence was penalized correctly, but fallback values still pushed different addresses toward similar score bands.

## Implemented Upgrade Summary

### 1) Open imagery pipeline (offline)

- Added `scripts/prepare_naip_structure_features.py`.
- Derives deterministic, explainable near-structure features from NAIP imagery for each footprint:
  - vegetation cover fraction per ring (`0-5`, `5-30`, `30-100`, `100-300` ft)
  - canopy proxy fraction per ring
  - high-fuel proxy fraction per ring
  - impervious/low-fuel proxy fraction per ring
  - vegetation continuity proxy per ring
  - nearest high-fuel patch distance proxy
- Writes prepared-region artifact: `naip_structure_features.json`.
- Computes region quantiles for local percentile context and stores them in the same artifact.

### 2) Runtime integration (deterministic)

- Runtime now loads `naip_structure_features.json` when available.
- Matched structure rows are selected by `structure_id` first, then centroid key fallback.
- Ring vegetation metrics are NAIP-blended (not replaced blindly), preserving legacy support when imagery is absent.
- New property-context fields are surfaced:
  - `near_structure_vegetation_0_5_pct`
  - `canopy_adjacency_proxy_pct`
  - `vegetation_continuity_proxy_pct`
  - `nearest_high_fuel_patch_distance_ft`
  - `imagery_local_percentiles`

### 3) Scoring discriminativeness upgrades

- Updated submodels in `backend/risk_engine.py` to use the new property-specific signals:
  - `ember_exposure_risk`: adds structure-to-structure exposure term
  - `flame_contact_risk`: adds canopy adjacency, continuity, and high-fuel distance pressure
  - `fuel_proximity_risk`: adds outer-ring continuity + high-fuel distance
  - `vegetation_intensity_risk`: adds continuity + local percentile signal
  - `defensible_space_risk`: adds close-in canopy adjacency pressure
- Rebalanced default weights to preserve parcel-level variance in final blend.

### 4) Calibration scaffolding (optional, additive)

- Added `backend/calibration.py` with optional calibration artifact application.
- Added scripts:
  - `scripts/build_public_outcome_calibration_dataset.py`
  - `scripts/fit_public_outcome_calibration.py`
- Supports transparent logistic/piecewise calibration artifacts from public outcome datasets (e.g., CAL FIRE DINS-derived datasets).
- Calibration is optional and does not replace deterministic factor scoring.

### 5) Diagnostics and QA upgrades

- `/risk/debug` score variance diagnostics now include richer parcel-level signal context and compression flags.
- Added `scripts/analyze_open_model_score_spread.py` for regression spread checks:
  - min/max/mean/stddev by top-level score
  - fallback frequency by factor
  - factor contribution variance ranking
  - parcel-signal spread summaries

## Data Sources Used

- **LANDFIRE** fuel/vegetation layers (existing runtime inputs)
- **NAIP imagery** for offline near-structure feature extraction
- **Historical wildfire perimeters/severity** (existing)
- **Open building footprints / parcel-aware structure context** (existing)
- **Public structure-impact outcomes** for optional calibration experiments

## Calibration Limitations

- Public outcome data is geographically incomplete and not equivalent to insurer claims.
- Calibration artifacts are advisory and versioned; they should be interpreted as directional risk-likelihood mapping, not underwriting truth.
- Regional transferability must be validated before broad deployment.

## How To Run

### NAIP feature prep

```bash
python scripts/prepare_naip_structure_features.py \
  --region-id winthrop_large \
  --naip-path /path/to/naip.tif \
  --overwrite \
  --update-manifest
```

### Score spread regression

```bash
python scripts/analyze_open_model_score_spread.py \
  --fixture tests/fixtures/score_variance_scenarios.json \
  --json-out /tmp/open_model_spread.json \
  --csv-out /tmp/open_model_spread.csv
```

### Public-outcome calibration workflow

```bash
python scripts/build_public_outcome_calibration_dataset.py \
  --input benchmark/event_backtest_results/latest.json \
  --output benchmark/calibration/public_outcome_calibration_dataset.json

python scripts/fit_public_outcome_calibration.py \
  --dataset benchmark/calibration/public_outcome_calibration_dataset.json \
  --output config/public_outcome_calibration.json
```

Set runtime calibration artifact (optional):

```bash
export WF_PUBLIC_CALIBRATION_ARTIFACT=config/public_outcome_calibration.json
```

## Before/After (Representative)

- **Before:** materially different properties in the same regional context could converge due to fallback-heavy ring pressure and overlapping submodel inputs.
- **After:** near-structure vegetation/canopy/continuity/high-fuel distance terms plus local percentile context produce wider, more realistic parcel-level separation while keeping deterministic explainability.
- Current fixture run (`tests/fixtures/score_variance_scenarios.json`) with `scripts/analyze_open_model_score_spread.py` produced:
  - `wildfire_risk_score`: min `16.2`, max `86.3`, stddev `23.22`
  - `site_hazard_score`: min `20.5`, max `82.9`, stddev `18.89`
  - Stronger variance concentration in parcel-sensitive contributors (`flame_contact_risk`, `defensible_space_risk`, `fuel_proximity_risk`).

## Governance / Determinism

- Runtime remains deterministic and interpretable.
- Heavy imagery work is offline only.
- Evidence ledger, coverage audit, confidence, provenance, and model governance outputs remain intact.
