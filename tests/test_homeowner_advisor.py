from __future__ import annotations

from backend.homeowner_advisor import (
    build_confidence_summary,
    build_ranked_risk_drivers,
    build_simulator_explanations,
    prioritize_mitigation_actions,
)
from backend.models import MitigationAction, SubmodelScore, WeightedContribution


def test_ranked_risk_drivers_prefers_high_contribution_components() -> None:
    submodels = {
        "vegetation_intensity_risk": SubmodelScore(
            score=82.0,
            weighted_contribution=11.4,
            explanation="",
            key_inputs={},
        ),
        "ember_exposure_risk": SubmodelScore(
            score=73.0,
            weighted_contribution=9.5,
            explanation="",
            key_inputs={},
        ),
        "structure_vulnerability_risk": SubmodelScore(
            score=28.0,
            weighted_contribution=3.1,
            explanation="",
            key_inputs={},
        ),
    }
    contributions = {
        "vegetation_intensity_risk": WeightedContribution(weight=0.13, score=82.0, contribution=10.7),
        "ember_exposure_risk": WeightedContribution(weight=0.13, score=73.0, contribution=9.4),
        "structure_vulnerability_risk": WeightedContribution(weight=0.14, score=28.0, contribution=3.9),
    }

    titles, detailed = build_ranked_risk_drivers(
        submodel_scores=submodels,
        weighted_contributions=contributions,
        limit=3,
    )

    assert titles[0] == "Dense vegetation near the home"
    assert len(detailed) == 3
    assert detailed[0].factor == "vegetation_proximity"
    assert detailed[0].impact in {"high", "medium", "low"}
    assert detailed[0].relative_contribution_pct is not None


def test_prioritized_mitigation_actions_sorts_by_impact_then_effort() -> None:
    actions = [
        MitigationAction(
            title="High impact medium effort",
            reason="",
            estimated_risk_reduction_band="high",
            priority=3,
            effort="medium",
        ),
        MitigationAction(
            title="Medium impact low effort",
            reason="",
            estimated_risk_reduction_band="medium",
            priority=1,
            effort="low",
        ),
        MitigationAction(
            title="High impact low effort",
            reason="",
            estimated_risk_reduction_band="high",
            priority=2,
            effort="low",
        ),
    ]

    ranked = prioritize_mitigation_actions(actions, limit=5)
    assert ranked[0].action == "High impact low effort"
    assert ranked[1].action == "High impact medium effort"
    assert ranked[0].timeline in {"now", "this_season", "later"}


def test_confidence_summary_surfaces_observed_estimated_missing_and_improvements() -> None:
    summary = build_confidence_summary(
        confidence_tier="moderate",
        observed_inputs={"roof_type": "class a"},
        inferred_inputs={"defensible_space_ft": 30},
        missing_inputs=["vent_type", "construction_year"],
        assumptions_used=["Fallback used for vegetation context."],
    )

    assert summary.confidence == "moderate"
    assert "roof material" in summary.observed_data
    assert "defensible space distance" in summary.estimated_data
    assert "vent protection" in summary.missing_data
    assert len(summary.accuracy_improvements) >= 1


def test_simulator_explanations_include_delta_and_resolved_outputs() -> None:
    baseline = {
        "wildfire_risk_score": 68.0,
        "top_risk_drivers_detailed": [{"factor": "vegetation_proximity"}],
        "prioritized_mitigation_actions": [{"action": "Clear vegetation"}],
    }
    simulated = {
        "wildfire_risk_score": 49.0,
        "top_risk_drivers_detailed": [{"factor": "ember_exposure"}],
        "prioritized_mitigation_actions": [{"action": "Install ember-resistant vents"}],
    }

    explainer = build_simulator_explanations(baseline=baseline, simulated=simulated)
    assert explainer["current_risk_score"] == 68.0
    assert explainer["simulated_risk_score"] == 49.0
    assert explainer["estimated_risk_reduction"] == 19.0
    assert "vegetation_proximity" in explainer["risk_drivers_removed"]
    assert isinstance(explainer["updated_top_risk_drivers"], list)
