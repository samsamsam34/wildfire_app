#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import PropertyAttributes
from backend.risk_engine import RiskEngine
from backend.scoring_config import load_scoring_config
from backend.wildfire_data import WildfireContext


def _build_context(payload: dict[str, Any]) -> WildfireContext:
    layer_status = {
        "burn_probability": "ok",
        "hazard": "ok",
        "slope": "ok",
        "fuel": "ok",
        "canopy": "ok",
        "fire_history": "ok",
    }
    ring_metrics = payload.get("structure_ring_metrics") or {}
    return WildfireContext(
        environmental_index=None,
        slope_index=float(payload.get("slope_index")) if payload.get("slope_index") is not None else None,
        aspect_index=float(payload.get("aspect_index")) if payload.get("aspect_index") is not None else None,
        fuel_index=float(payload.get("fuel_index")) if payload.get("fuel_index") is not None else None,
        moisture_index=float(payload.get("moisture_index")) if payload.get("moisture_index") is not None else None,
        canopy_index=float(payload.get("canopy_index")) if payload.get("canopy_index") is not None else None,
        wildland_distance_index=(
            float(payload.get("wildland_distance_index")) if payload.get("wildland_distance_index") is not None else None
        ),
        historic_fire_index=float(payload.get("historic_fire_index")) if payload.get("historic_fire_index") is not None else None,
        burn_probability_index=(
            float(payload.get("burn_probability_index")) if payload.get("burn_probability_index") is not None else None
        ),
        hazard_severity_index=(
            float(payload.get("hazard_severity_index")) if payload.get("hazard_severity_index") is not None else None
        ),
        access_exposure_index=float(payload.get("access_exposure_index")) if payload.get("access_exposure_index") is not None else None,
        burn_probability=float(payload.get("burn_probability")) if payload.get("burn_probability") is not None else None,
        wildfire_hazard=float(payload.get("wildfire_hazard")) if payload.get("wildfire_hazard") is not None else None,
        slope=float(payload.get("slope")) if payload.get("slope") is not None else None,
        fuel_model=float(payload.get("fuel_model")) if payload.get("fuel_model") is not None else None,
        canopy_cover=float(payload.get("canopy_cover")) if payload.get("canopy_cover") is not None else None,
        historic_fire_distance=(
            float(payload.get("historic_fire_distance")) if payload.get("historic_fire_distance") is not None else None
        ),
        wildland_distance=float(payload.get("wildland_distance")) if payload.get("wildland_distance") is not None else None,
        environmental_layer_status=layer_status,
        data_sources=["score-variance-fixture"],
        assumptions=[],
        structure_ring_metrics=ring_metrics,
        property_level_context={
            "footprint_used": bool(ring_metrics),
            "footprint_status": "used" if ring_metrics else "not_found",
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
            "feature_sampling": payload.get("feature_sampling") or {},
        },
    )


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "stddev": 0.0}
    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "stddev": round(statistics.pstdev(values), 2),
    }


def run_variance_analysis(fixture_path: Path, csv_out: Path | None = None) -> dict[str, Any]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    scenarios = list(payload.get("scenarios") or [])
    if not scenarios:
        raise ValueError(f"No scenarios found in {fixture_path}")

    engine = RiskEngine(load_scoring_config())

    rows: list[dict[str, Any]] = []
    factor_contributions: dict[str, list[float]] = {}
    fallback_counts: dict[str, int] = {}

    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or "unknown")
        attrs = PropertyAttributes.model_validate(scenario.get("attributes") or {})
        context = _build_context(scenario.get("context") or {})
        risk = engine.score(attrs, lat=0.0, lon=0.0, context=context)
        site = engine.compute_site_hazard_score(risk)
        home = engine.compute_home_ignition_vulnerability_score(risk)
        readiness = engine.compute_insurance_readiness(attrs, context, risk).insurance_readiness_score
        wildfire = engine.compute_blended_wildfire_score(site, home, readiness)

        for submodel, contribution in risk.weighted_contributions.items():
            factor_contributions.setdefault(submodel, []).append(float(contribution.get("contribution") or 0.0))
            assumptions = " ".join(risk.submodel_scores.get(submodel).assumptions if risk.submodel_scores.get(submodel) else [])
            if any(tok in assumptions.lower() for tok in ("fallback", "missing", "unavailable")):
                fallback_counts[submodel] = fallback_counts.get(submodel, 0) + 1

        ranked_contrib = sorted(
            (
                {
                    "submodel": name,
                    "contribution": round(float(data.get("contribution") or 0.0), 2),
                }
                for name, data in risk.weighted_contributions.items()
            ),
            key=lambda row: abs(float(row["contribution"])),
            reverse=True,
        )

        rows.append(
            {
                "scenario_id": scenario_id,
                "address": scenario.get("address"),
                "site_hazard_score": round(site, 2),
                "home_ignition_vulnerability_score": round(home, 2),
                "insurance_readiness_score": round(readiness, 2),
                "wildfire_risk_score": round(wildfire, 2),
                "top_contributors": [x["submodel"] for x in ranked_contrib[:3]],
                "top_contributions": [x["contribution"] for x in ranked_contrib[:3]],
            }
        )

    score_stats = {
        "site_hazard_score": _summarize([float(r["site_hazard_score"]) for r in rows]),
        "home_ignition_vulnerability_score": _summarize([float(r["home_ignition_vulnerability_score"]) for r in rows]),
        "insurance_readiness_score": _summarize([float(r["insurance_readiness_score"]) for r in rows]),
        "wildfire_risk_score": _summarize([float(r["wildfire_risk_score"]) for r in rows]),
    }

    contribution_variance = sorted(
        (
            {
                "submodel": name,
                "stddev": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 2) if values else 0.0,
                "max": round(max(values), 2) if values else 0.0,
            }
            for name, values in factor_contributions.items()
        ),
        key=lambda row: float(row["stddev"]),
        reverse=True,
    )

    fallback_frequency = {
        name: {
            "count": int(count),
            "share": round(float(count) / float(len(rows)), 3),
        }
        for name, count in sorted(fallback_counts.items())
    }

    wildfire_range = score_stats["wildfire_risk_score"]["max"] - score_stats["wildfire_risk_score"]["min"]
    clustered = wildfire_range < 15.0 or score_stats["wildfire_risk_score"]["stddev"] < 6.0

    summary = {
        "scenario_count": len(rows),
        "score_stats": score_stats,
        "factor_contribution_variance_ranked": contribution_variance,
        "fallback_frequency": fallback_frequency,
        "clustered_output_warning": clustered,
        "wildfire_score_range": round(wildfire_range, 2),
        "rows": rows,
    }

    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "scenario_id",
                    "address",
                    "site_hazard_score",
                    "home_ignition_vulnerability_score",
                    "insurance_readiness_score",
                    "wildfire_risk_score",
                    "top_contributors",
                    "top_contributions",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "top_contributors": ";".join(row.get("top_contributors") or []),
                        "top_contributions": ";".join(str(v) for v in (row.get("top_contributions") or [])),
                    }
                )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze wildfire score spread across scenario fixtures.")
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/score_variance_scenarios.json",
        help="Path to scenario fixture JSON.",
    )
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    csv_out = Path(args.csv_out) if args.csv_out else None
    summary = run_variance_analysis(fixture_path=fixture_path, csv_out=csv_out)

    print("Score spread summary")
    print(json.dumps(summary["score_stats"], indent=2))
    print("\nTop factor contribution variance")
    print(json.dumps(summary["factor_contribution_variance_ranked"][:8], indent=2))
    print("\nFallback frequency")
    print(json.dumps(summary["fallback_frequency"], indent=2))
    print(
        f"\nClustered output warning: {summary['clustered_output_warning']} | "
        f"wildfire_score_range={summary['wildfire_score_range']}"
    )
    if csv_out:
        print(f"CSV written to: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
