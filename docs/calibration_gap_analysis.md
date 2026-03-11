# Calibration Gap Analysis

## Why the current model still lacks empirical grounding

The current wildfire model is explainable and deterministic, but it is still primarily a rules/heuristics engine. The main empirical gaps are:

1. **Heuristic score transforms dominate calibration to real damage outcomes**
- Submodel transforms and weights are hand-tuned and interpretable, but not fitted directly to public structure-loss outcomes.
- Result: directional logic is strong, but probability interpretation is weak.

2. **Limited closed-loop linkage between outcome labels and full feature vectors**
- Event backtesting exists, but public damage labels are not yet consistently joined to raw/transformed runtime feature vectors for calibration-grade analysis.
- Result: difficult to diagnose which feature families drive false positives/false negatives at scale.

3. **Fallback/default usage is not yet outcome-calibrated**
- Missing-data fallback behavior is transparent, but fallback-heavy examples are not yet explicitly reweighted using observed outcome error patterns.
- Result: records with sparse evidence can be over- or under-estimated without empirical correction.

4. **No standard benchmark artifact for discrimination + calibration quality**
- Existing backtests report useful diagnostics, but there is no single repeatable evaluation artifact covering ROC/AUC, PR behavior, confusion metrics, and calibration tables for public outcome labels.
- Result: harder to measure model competitiveness changes across releases.

5. **Calibration metadata is present but runtime calibration governance is still thin**
- `calibration_version` exists and optional calibration is supported, but artifact scope/limitations/status reporting needs to be richer and standardized for `/risk/debug` and report consumers.
- Result: users cannot always tell when calibrated proxies are in-scope vs advisory-only.

## Why this is the next highest-value step

Adding an additive, transparent public-outcome calibration layer improves product usefulness without sacrificing explainability:

- keeps deterministic core scoring intact
- adds measurable discrimination benchmarks against real structure outcomes
- enables versioned empirical likelihood proxies with explicit limitations
- improves trust by exposing when calibration is applied, disabled, or out-of-scope
