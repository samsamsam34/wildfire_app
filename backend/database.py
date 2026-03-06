from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.models import AssessmentResult
from backend.version import LEGACY_MODEL_VERSION


class AssessmentStore:
    def __init__(self, db_path: str = "wildfire_app.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(assessments)").fetchall()}
            if "model_version" not in cols:
                conn.execute("ALTER TABLE assessments ADD COLUMN model_version TEXT NOT NULL DEFAULT '1.0.0'")

    def save(self, result: AssessmentResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assessments (assessment_id, created_at, payload_json, model_version)
                VALUES (?, ?, ?, ?)
                """,
                (
                    result.assessment_id,
                    datetime.now(tz=timezone.utc).isoformat(),
                    result.model_dump_json(),
                    result.model_version,
                ),
            )

    def _upgrade_mitigation_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        upgraded = []
        for item in items:
            title = item.get("title") or item.get("action") or "Mitigation action"
            reason = item.get("reason") or item.get("impact_statement") or "Risk reduction action"
            upgraded.append(
                {
                    "title": title,
                    "reason": reason,
                    "impacted_submodels": item.get("impacted_submodels", []),
                    "impacted_readiness_factors": item.get("impacted_readiness_factors", []),
                    "estimated_risk_reduction_band": item.get("estimated_risk_reduction_band", "low"),
                    "estimated_readiness_improvement_band": item.get(
                        "estimated_readiness_improvement_band", "low"
                    ),
                    "priority": item.get("priority", 5),
                    "insurer_relevance": item.get("insurer_relevance", "recommended"),
                    "action": item.get("action", title),
                    "related_factor": item.get("related_factor"),
                    "impact_statement": item.get("impact_statement", reason),
                    "estimated_risk_reduction": item.get("estimated_risk_reduction"),
                    "effort": item.get("effort"),
                }
            )
        return upgraded

    def _upgrade_payload(self, payload: dict[str, Any], db_model_version: str) -> dict[str, Any]:
        payload.setdefault("model_version", db_model_version or LEGACY_MODEL_VERSION)

        payload.setdefault("latitude", payload.get("coordinates", {}).get("latitude", 0.0))
        payload.setdefault("longitude", payload.get("coordinates", {}).get("longitude", 0.0))
        payload.setdefault("wildfire_risk_score", payload.get("risk_scores", {}).get("wildfire_risk_score", 0.0))
        payload.setdefault("insurance_readiness_score", payload.get("risk_scores", {}).get("insurance_readiness_score", 0.0))

        if "risk_drivers" not in payload:
            fb = payload.get("factor_breakdown", {}) or {}
            payload["risk_drivers"] = {
                "environmental": fb.get("environmental_risk", 0.0),
                "structural": fb.get("structural_risk", 0.0),
                "access_exposure": fb.get("access_risk", 0.0),
            }

        if "factor_breakdown" not in payload:
            drivers = payload.get("risk_drivers", {}) or {}
            payload["factor_breakdown"] = {
                "submodels": {},
                "environmental": {},
                "structural": {},
                "environmental_risk": drivers.get("environmental", 0.0),
                "structural_risk": drivers.get("structural", 0.0),
                "access_risk": drivers.get("access_exposure", 0.0),
                "access_risk_provisional": True,
                "access_included_in_total": False,
                "access_risk_note": "Access exposure is provisional and not included in total score until real parcel/egress inputs are integrated.",
            }
        payload.setdefault("submodel_scores", {})
        fb = payload.get("factor_breakdown", {}) or {}
        fb.setdefault("submodels", {k: (v.get("score") if isinstance(v, dict) else v) for k, v in payload["submodel_scores"].items() if isinstance(v, (dict, float, int))})
        fb.setdefault("environmental", {k: fb["submodels"].get(k, 0.0) for k in [
            "vegetation_intensity_risk",
            "fuel_proximity_risk",
            "slope_topography_risk",
            "ember_exposure_risk",
            "flame_contact_risk",
            "historic_fire_risk",
        ] if k in fb["submodels"]})
        fb.setdefault("structural", {k: fb["submodels"].get(k, 0.0) for k in ["structure_vulnerability_risk", "defensible_space_risk"] if k in fb["submodels"]})
        fb.setdefault("environmental_risk", payload.get("risk_drivers", {}).get("environmental", 0.0))
        fb.setdefault("structural_risk", payload.get("risk_drivers", {}).get("structural", 0.0))
        fb.setdefault("access_risk", payload.get("risk_drivers", {}).get("access_exposure", 0.0))
        fb.setdefault("access_risk_provisional", True)
        fb.setdefault("access_included_in_total", False)
        fb.setdefault(
            "access_risk_note",
            "Access exposure is provisional and not included in total score until real parcel/egress inputs are integrated.",
        )
        payload["factor_breakdown"] = fb
        payload.setdefault("weighted_contributions", {})
        payload.setdefault("submodel_explanations", {})

        payload.setdefault("top_risk_drivers", [])
        payload.setdefault("top_protective_factors", [])
        payload.setdefault("explanation_summary", payload.get("explanation", ""))

        payload.setdefault("observed_inputs", payload.get("assumptions", {}).get("observed_inputs", {}))
        payload.setdefault("inferred_inputs", payload.get("assumptions", {}).get("inferred_inputs", {}))
        payload.setdefault("missing_inputs", payload.get("assumptions", {}).get("missing_inputs", []))
        payload.setdefault("assumptions_used", payload.get("assumptions", {}).get("assumptions_used", []))

        if "confidence_score" not in payload:
            payload["confidence_score"] = payload.get("confidence", {}).get("confidence_score", 60.0)
        if "low_confidence_flags" not in payload:
            payload["low_confidence_flags"] = payload.get("confidence", {}).get("low_confidence_flags", [])

        payload.setdefault("data_sources", [])
        payload.setdefault("mitigation_plan", payload.get("mitigation_recommendations", []))
        payload["mitigation_plan"] = self._upgrade_mitigation_items(payload.get("mitigation_plan", []))

        payload.setdefault("readiness_factors", [])
        payload.setdefault("readiness_blockers", [])
        payload.setdefault("readiness_penalties", {})
        payload.setdefault("readiness_summary", "Legacy row: readiness detail unavailable in this version.")

        payload.setdefault("scoring_notes", ["Access risk is provisional and not included in total scoring."])

        payload.setdefault("coordinates", {"latitude": payload["latitude"], "longitude": payload["longitude"]})
        payload.setdefault(
            "risk_scores",
            {
                "wildfire_risk_score": payload["wildfire_risk_score"],
                "insurance_readiness_score": payload["insurance_readiness_score"],
            },
        )
        payload.setdefault(
            "assumptions",
            {
                "observed_inputs": payload["observed_inputs"],
                "inferred_inputs": payload["inferred_inputs"],
                "missing_inputs": payload["missing_inputs"],
                "assumptions_used": payload["assumptions_used"],
            },
        )
        payload.setdefault(
            "confidence",
            {
                "confidence_score": payload["confidence_score"],
                "data_completeness_score": 50.0,
                "assumption_count": len(payload["assumptions_used"]),
                "low_confidence_flags": payload["low_confidence_flags"],
                "requires_user_verification": True,
            },
        )
        payload.setdefault("mitigation_recommendations", payload["mitigation_plan"])

        return payload

    def get(self, assessment_id: str) -> Optional[AssessmentResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, model_version FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()

        if not row:
            return None

        payload = json.loads(row["payload_json"])
        upgraded = self._upgrade_payload(payload, row["model_version"] or LEGACY_MODEL_VERSION)
        return AssessmentResult.model_validate(upgraded)
