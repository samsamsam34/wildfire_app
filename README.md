# WildfireRisk Advisor
A homeowner-focused wildfire risk and mitigation tool that aims to be property-specific when data supports it, and explicit about limits when it does not.

## What this is
WildfireRisk Advisor helps homeowners understand wildfire risk at their property and decide what to do next.

It uses deterministic scoring logic and open geospatial data, then returns:
- risk and readiness summaries
- plain-language drivers and limitations
- prioritized mitigation actions
- confidence and specificity signals

## Current status
### What works
- Homeowner-first flow: assess, review, improve inputs, and simulate mitigation changes.
- Action-oriented outputs: top risk drivers, prioritized actions, and improvement guidance.
- Trust framing: explicit confidence, assumptions, and data-gap limitations.
- Specificity tiers reflect evidence quality: `property_specific` (strong parcel/footprint/near-structure support), `address_level` (partial property evidence), `regional_estimate` (limited local property evidence).
- Prepared-region workflow for deterministic runtime scoring on local data snapshots.

### Known limitations
- Coverage is only as good as prepared regional data.
- Property-level quality depends heavily on parcel polygons, building footprints, and near-structure context.
- Some addresses require manual confirmation when candidates conflict or confidence is low.
- Data quality can vary by county/region, so outputs may be less specific in some locations.
- This is a decision-support tool, not a guarantee of real-world outcomes.

## Core principle
Be useful without pretending certainty.

If evidence is strong, return property-specific guidance. If evidence is weak, reduce specificity, surface uncertainty clearly, and still provide practical next steps.

## What users get
- Overall wildfire risk summary
- Home hardening readiness summary
- Top risk drivers
- Prioritized mitigation actions
- Confidence and specificity summary
- Assumptions and unknowns
- Before/after simulation feedback for mitigation scenarios

## How it works
1. User submits an address.
2. The app resolves a location and checks prepared-region coverage.
3. It pulls available property and regional context (parcel, footprint, vegetation/fuel, hazard, access, etc.).
4. Deterministic scoring computes risk/readiness outputs.
5. The app returns scores, drivers, limitations, confidence/specificity tier, and recommended actions.

## Current focus / next steps
- Improve property-specific reliability where parcel and footprint quality are inconsistent.
- Reduce false precision by tightening coordinate and candidate selection safety.
- Expand regional data quality and validation so more addresses can stay in `property_specific` mode.
- Keep homeowner guidance clear, practical, and tied to observable evidence.

## What this is not
- Not an underwriting engine.
- Not a fire spread simulator.
- Not a probabilistic loss forecast.
- Not a substitute for local fire officials, defensible space inspections, or code requirements.

## Philosophy
Trust is earned by clarity, not complexity.

The product should:
- prioritize homeowner decisions over internal platform detail
- expose assumptions instead of hiding them
- separate strong evidence from fallback inference
- make limitations visible at the same level as scores

## Bottom line
WildfireRisk Advisor is built to give homeowners actionable wildfire guidance with honest confidence boundaries. It should be specific when the data is strong, and transparent when it is not.
