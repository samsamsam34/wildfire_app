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
                conn.execute(
                    "ALTER TABLE assessments ADD COLUMN model_version TEXT NOT NULL DEFAULT '1.0.0'"
                )

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

    def _upgrade_payload(self, payload: dict[str, Any], db_model_version: str) -> dict[str, Any]:
        if "model_version" not in payload:
            payload["model_version"] = db_model_version or LEGACY_MODEL_VERSION

        if "coordinates" not in payload:
            payload["coordinates"] = {
                "latitude": payload.get("latitude", 0.0),
                "longitude": payload.get("longitude", 0.0),
            }

        if "risk_scores" not in payload:
            payload["risk_scores"] = {
                "wildfire_risk_score": payload.get("wildfire_risk_score", 0.0),
                "insurance_readiness_score": payload.get("insurance_readiness_score", 0.0),
            }

        if "factor_breakdown" not in payload:
            drivers = payload.get("risk_drivers", {}) or {}
            payload["factor_breakdown"] = {
                "environmental_risk": drivers.get("environmental", 0.0),
                "structural_risk": drivers.get("structural", 0.0),
                "access_risk": drivers.get("access_exposure", 0.0),
            }

        payload.setdefault("mitigation_recommendations", payload.get("mitigation_plan", []))

        if "assumptions" not in payload:
            assumptions_used = payload.get("assumptions_used", [])
            payload["assumptions"] = {
                "observed_inputs": {},
                "inferred_inputs": {},
                "missing_inputs": [],
                "assumptions_used": assumptions_used,
            }

        if "confidence" not in payload:
            payload["confidence"] = {
                "confidence_score": 60.0,
                "data_completeness_score": 50.0,
                "assumption_count": len(payload.get("assumptions_used", [])),
                "low_confidence_flags": payload.get("assumptions_used", []),
                "requires_user_verification": True,
            }

        payload.setdefault("top_risk_drivers", [])
        payload.setdefault("top_protective_factors", [])
        payload.setdefault("explanation_summary", payload.get("explanation", ""))

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
