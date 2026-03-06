from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.models import AssessmentResult


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

    def save(self, result: AssessmentResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assessments (assessment_id, created_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    result.assessment_id,
                    datetime.now(tz=timezone.utc).isoformat(),
                    result.model_dump_json(),
                ),
            )

    def get(self, assessment_id: str) -> Optional[AssessmentResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        if not row:
            return None
        return AssessmentResult.model_validate_json(row["payload_json"])
