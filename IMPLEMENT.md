# IMPLEMENT.md

## Purpose
This repository powers a property-level wildfire risk and insurance readiness product.
When making model or scoring changes, prioritize reliability, trustworthiness, and evaluation discipline over speculative predictive gains.

## Mission
Improve model quality in a way that is:
- measurable
- explainable
- low-regression
- safe under sparse labeled data

## Priority Order
1. Reliability and internal consistency
2. Confidence correctness under weak or incomplete evidence
3. Monotonic mitigation behavior
4. Benchmark and regression coverage
5. Calibration and trust signaling
6. Bounded tuning
7. Predictive improvement only when supported by sufficient independent labeled data

## Data Constraints
- Do not assume labeled datasets are sufficient for predictive optimization.
- If dataset viability checks show insufficient independent samples, do not claim predictive improvement.
- In low-data regimes, prefer reliability, consistency, confidence, and calibration improvements.
- Avoid making changes that only improve a narrow benchmark artifact.

## Hard Rules
- Keep diffs narrowly scoped.
- Prefer deterministic and explainable logic over opaque complexity.
- Do not rewrite architecture unless needed to fix a clearly identified reliability issue.
- Do not auto-apply speculative tuning that weakens interpretability.
- Do not change multiple subsystems in one task unless required.
- Never present a change as an improvement without validation evidence.

## Reliability Standards
Any change should tend to improve one or more of the following:
- consistency between score, drivers, and mitigation advice
- correct confidence downgrades when evidence is weak, missing, or proxy-based
- sensible fallback behavior
- monotonicity of mitigation effects
- benchmark stability
- trustworthiness of outputs when no ground truth is available

## Evaluation Discipline
Never treat a single benchmark improvement as sufficient evidence.
A change is only credible if it holds across the relevant validation suite.

Watch for:
- overfitting to a single benchmark
- benchmark gaming
- score shifts without better explanations
- high-confidence outputs with weak evidence
- contradictions between outputs and evidence

## Required Validation After Each Meaningful Change
Run all applicable checks after each meaningful change.

### Core checks
- `pytest`
- `python scripts/run_benchmark_suite.py`
- `python scripts/run_confidence_benchmark_pack.py`
- `python scripts/run_no_ground_truth_evaluation.py`

### If relevant scripts exist and data is available
- public outcome validation
- dataset viability checks
- calibration fitting/evaluation
- any targeted regression scripts related to the subsystem changed

## Expected Work Pattern
For each task:
1. Read `IMPLEMENT.md`, `PLAN.md`, and `METRICS.md`
2. Establish a baseline
3. Identify the single highest-leverage issue
4. Make one narrowly scoped improvement
5. Add or update tests/benchmarks so the issue is reproducible
6. Re-run validations
7. Compare before/after results
8. Stop if improvement is not clearly supported

## Stop Conditions
Stop and report instead of continuing when:
- there is no clear improvement
- regressions appear
- the change requires major architectural expansion
- labeled data is too weak to justify predictive optimization
- further iteration appears likely to overfit or destabilize the system

## Required Output Format
At the end of the task, return:
1. Baseline summary
2. What changed
3. Why the change should help
4. Validation results
5. Before/after metric deltas
6. Risks or uncertainties
7. Recommended next experiment
8. Suggested commit message