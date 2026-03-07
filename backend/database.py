from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from backend.models import (
    AssessmentAnnotation,
    AssessmentListItem,
    AssessmentResult,
    AssessmentReviewStatus,
    PortfolioSummary,
    ReviewStatus,
    SimulationScenarioItem,
)
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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    assessment_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    assessment_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author_role TEXT NOT NULL,
                    note TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            ann_cols = {row["name"] for row in conn.execute("PRAGMA table_info(assessment_annotations)").fetchall()}
            if "review_status" not in ann_cols:
                conn.execute(
                    "ALTER TABLE assessment_annotations ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'"
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_review_status (
                    assessment_id TEXT PRIMARY KEY,
                    review_status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
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

    def save_simulation(self, assessment_id: str, scenario_name: str, payload: dict[str, Any]) -> str:
        scenario_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_scenarios (scenario_id, assessment_id, created_at, scenario_name, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    scenario_id,
                    assessment_id,
                    datetime.now(tz=timezone.utc).isoformat(),
                    scenario_name,
                    json.dumps(payload, default=str),
                ),
            )
        return scenario_id

    def set_review_status(self, assessment_id: str, review_status: ReviewStatus) -> AssessmentReviewStatus:
        updated_at = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_review_status (assessment_id, review_status, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    review_status=excluded.review_status,
                    updated_at=excluded.updated_at
                """,
                (assessment_id, review_status, updated_at),
            )

        return AssessmentReviewStatus(
            assessment_id=assessment_id,
            review_status=review_status,
            updated_at=updated_at,
        )

    def get_review_status(self, assessment_id: str) -> AssessmentReviewStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT assessment_id, review_status, updated_at FROM assessment_review_status WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()

        if not row:
            return None

        return AssessmentReviewStatus(
            assessment_id=row["assessment_id"],
            review_status=row["review_status"],
            updated_at=row["updated_at"],
        )

    def save_annotation(
        self,
        assessment_id: str,
        author_role: str,
        note: str,
        tags: list[str],
        visibility: str,
        review_status: ReviewStatus | None = None,
    ) -> AssessmentAnnotation:
        annotation_id = str(uuid4())
        created_at = datetime.now(tz=timezone.utc).isoformat()
        payload_tags = sorted(set([t.strip() for t in tags if t and t.strip()]))
        current_review = review_status or (self.get_review_status(assessment_id).review_status if self.get_review_status(assessment_id) else "pending")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_annotations (
                    annotation_id, assessment_id, created_at, author_role, note, visibility, tags_json, review_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    assessment_id,
                    created_at,
                    author_role,
                    note,
                    visibility,
                    json.dumps(payload_tags),
                    current_review,
                ),
            )

        if review_status is not None:
            self.set_review_status(assessment_id, review_status)

        return AssessmentAnnotation(
            annotation_id=annotation_id,
            assessment_id=assessment_id,
            created_at=created_at,
            author_role=author_role,
            note=note,
            tags=payload_tags,
            visibility=visibility,
            review_status=current_review,
        )

    def list_annotations(self, assessment_id: str, limit: int = 100) -> list[AssessmentAnnotation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT annotation_id, assessment_id, created_at, author_role, note, visibility, tags_json, review_status
                FROM assessment_annotations
                WHERE assessment_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (assessment_id, limit),
            ).fetchall()

        items: list[AssessmentAnnotation] = []
        for row in rows:
            try:
                tags = json.loads(row["tags_json"]) if row["tags_json"] else []
            except Exception:
                tags = []
            items.append(
                AssessmentAnnotation(
                    annotation_id=row["annotation_id"],
                    assessment_id=row["assessment_id"],
                    created_at=row["created_at"],
                    author_role=row["author_role"],
                    note=row["note"],
                    tags=tags,
                    visibility=row["visibility"],
                    review_status=row["review_status"],
                )
            )
        return items

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

    def _upgrade_payload(
        self,
        payload: dict[str, Any],
        db_model_version: str,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        payload.setdefault("model_version", db_model_version or LEGACY_MODEL_VERSION)

        payload.setdefault("audience", "homeowner")
        payload.setdefault("report_audience", None)
        payload.setdefault("audience_highlights", [])
        payload.setdefault("portfolio_name", None)
        payload.setdefault("tags", [])
        payload.setdefault("review_status", "pending")
        payload.setdefault("property_facts", {})
        payload.setdefault("confirmed_fields", [])

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
        fb.setdefault(
            "submodels",
            {
                k: (v.get("score") if isinstance(v, dict) else v)
                for k, v in payload["submodel_scores"].items()
                if isinstance(v, (dict, float, int))
            },
        )
        fb.setdefault(
            "environmental",
            {
                k: fb["submodels"].get(k, 0.0)
                for k in [
                    "vegetation_intensity_risk",
                    "fuel_proximity_risk",
                    "slope_topography_risk",
                    "ember_exposure_risk",
                    "flame_contact_risk",
                    "historic_fire_risk",
                ]
                if k in fb["submodels"]
            },
        )
        fb.setdefault(
            "structural",
            {
                k: fb["submodels"].get(k, 0.0)
                for k in ["structure_vulnerability_risk", "defensible_space_risk"]
                if k in fb["submodels"]
            },
        )
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

        payload.setdefault("confirmed_inputs", payload.get("assumptions", {}).get("confirmed_inputs", {}))
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

        payload.setdefault("generated_at", created_at or datetime.now(tz=timezone.utc).isoformat())
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
                "confirmed_inputs": payload["confirmed_inputs"],
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

    def _parse_ts(self, raw: str) -> datetime:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    def get(self, assessment_id: str) -> Optional[AssessmentResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, model_version, created_at FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()

        if not row:
            return None

        payload = json.loads(row["payload_json"])
        upgraded = self._upgrade_payload(payload, row["model_version"] or LEGACY_MODEL_VERSION, row["created_at"])
        review = self.get_review_status(assessment_id)
        if review:
            upgraded["review_status"] = review.review_status
        return AssessmentResult.model_validate(upgraded)

    def _load_assessment_records(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT assessment_id, created_at, payload_json, model_version FROM assessments"
            ).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            upgraded = self._upgrade_payload(payload, row["model_version"] or LEGACY_MODEL_VERSION, row["created_at"])
            review = self.get_review_status(upgraded["assessment_id"])
            review_status = review.review_status if review else upgraded.get("review_status", "pending")
            records.append(
                {
                    "assessment_id": upgraded["assessment_id"],
                    "created_at": row["created_at"],
                    "created_dt": self._parse_ts(row["created_at"]),
                    "address": upgraded.get("address", ""),
                    "audience": upgraded.get("audience", "homeowner"),
                    "wildfire_risk_score": float(upgraded.get("wildfire_risk_score", 0.0)),
                    "insurance_readiness_score": float(upgraded.get("insurance_readiness_score", 0.0)),
                    "model_version": upgraded.get("model_version", LEGACY_MODEL_VERSION),
                    "confidence_score": float(upgraded.get("confidence_score", 0.0)),
                    "readiness_blockers": list(upgraded.get("readiness_blockers", [])),
                    "tags": list(upgraded.get("tags", [])),
                    "review_status": review_status,
                }
            )
        return records

    def _build_summary(self, records: list[dict[str, Any]]) -> PortfolioSummary:
        if not records:
            return PortfolioSummary(
                total_count=0,
                high_risk_count=0,
                blocker_count=0,
                avg_wildfire_risk=0.0,
                avg_insurance_readiness=0.0,
            )

        total = len(records)
        avg_risk = round(sum(r["wildfire_risk_score"] for r in records) / total, 1)
        avg_readiness = round(sum(r["insurance_readiness_score"] for r in records) / total, 1)
        high_risk_count = sum(1 for r in records if r["wildfire_risk_score"] >= 70.0)
        blocker_count = sum(1 for r in records if r["readiness_blockers"])

        return PortfolioSummary(
            total_count=total,
            high_risk_count=high_risk_count,
            blocker_count=blocker_count,
            avg_wildfire_risk=avg_risk,
            avg_insurance_readiness=avg_readiness,
        )

    def query_assessments(
        self,
        *,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
        min_risk: float | None = None,
        max_risk: float | None = None,
        min_readiness: float | None = None,
        max_readiness: float | None = None,
        readiness_blocker: str | None = None,
        confidence_min: float | None = None,
        audience: str | None = None,
        tag: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        recent_days: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[AssessmentListItem], int, PortfolioSummary]:
        records = self._load_assessment_records()

        after_dt = self._parse_ts(created_after) if created_after else None
        before_dt = self._parse_ts(created_before) if created_before else None
        recent_cutoff = None
        if recent_days is not None:
            recent_cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max(0, recent_days))

        if min_risk is not None:
            records = [r for r in records if r["wildfire_risk_score"] >= min_risk]
        if max_risk is not None:
            records = [r for r in records if r["wildfire_risk_score"] <= max_risk]
        if min_readiness is not None:
            records = [r for r in records if r["insurance_readiness_score"] >= min_readiness]
        if max_readiness is not None:
            records = [r for r in records if r["insurance_readiness_score"] <= max_readiness]
        if confidence_min is not None:
            records = [r for r in records if r["confidence_score"] >= confidence_min]
        if audience is not None:
            records = [r for r in records if r["audience"] == audience]
        if readiness_blocker:
            needle = readiness_blocker.lower()
            records = [r for r in records if any(needle in b.lower() for b in r["readiness_blockers"])]
        if tag:
            needle = tag.lower()
            records = [r for r in records if any(needle == str(t).lower() for t in r["tags"])]
        if after_dt is not None:
            records = [r for r in records if r["created_dt"] >= after_dt]
        if before_dt is not None:
            records = [r for r in records if r["created_dt"] <= before_dt]
        if recent_cutoff is not None:
            records = [r for r in records if r["created_dt"] >= recent_cutoff]

        summary = self._build_summary(records)
        total = len(records)

        sort_map = {
            "created_at": "created_dt",
            "wildfire_risk_score": "wildfire_risk_score",
            "insurance_readiness_score": "insurance_readiness_score",
            "confidence_score": "confidence_score",
            "address": "address",
        }
        sort_key = sort_map.get(sort_by, "created_dt")
        reverse = sort_dir.lower() != "asc"
        records.sort(key=lambda r: r[sort_key], reverse=reverse)

        paged = records[offset : offset + limit]
        items = [
            AssessmentListItem(
                assessment_id=r["assessment_id"],
                created_at=r["created_at"],
                address=r["address"],
                audience=r["audience"],
                wildfire_risk_score=r["wildfire_risk_score"],
                insurance_readiness_score=r["insurance_readiness_score"],
                model_version=r["model_version"],
                confidence_score=r["confidence_score"],
                readiness_blockers=r["readiness_blockers"],
                tags=r["tags"],
                review_status=r["review_status"],
            )
            for r in paged
        ]

        return items, total, summary

    def summary_assessments(
        self,
        *,
        min_risk: float | None = None,
        max_risk: float | None = None,
        min_readiness: float | None = None,
        max_readiness: float | None = None,
        readiness_blocker: str | None = None,
        confidence_min: float | None = None,
        audience: str | None = None,
        tag: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        recent_days: int | None = None,
    ) -> PortfolioSummary:
        _, _, summary = self.query_assessments(
            sort_by="created_at",
            sort_dir="desc",
            min_risk=min_risk,
            max_risk=max_risk,
            min_readiness=min_readiness,
            max_readiness=max_readiness,
            readiness_blocker=readiness_blocker,
            confidence_min=confidence_min,
            audience=audience,
            tag=tag,
            created_after=created_after,
            created_before=created_before,
            recent_days=recent_days,
            limit=1_000_000,
            offset=0,
        )
        return summary

    def list_assessments(
        self,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
        min_risk: float | None = None,
        max_risk: float | None = None,
        min_readiness: float | None = None,
        max_readiness: float | None = None,
        readiness_blocker: str | None = None,
        confidence_min: float | None = None,
        audience: str | None = None,
        tag: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        recent_days: int | None = None,
    ) -> list[AssessmentListItem]:
        items, _, _ = self.query_assessments(
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            min_risk=min_risk,
            max_risk=max_risk,
            min_readiness=min_readiness,
            max_readiness=max_readiness,
            readiness_blocker=readiness_blocker,
            confidence_min=confidence_min,
            audience=audience,
            tag=tag,
            created_after=created_after,
            created_before=created_before,
            recent_days=recent_days,
        )
        return items

    def list_scenarios(self, assessment_id: str, limit: int = 20) -> list[SimulationScenarioItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT scenario_id, assessment_id, scenario_name, created_at, payload_json
                FROM assessment_scenarios
                WHERE assessment_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (assessment_id, limit),
            ).fetchall()

        items: list[SimulationScenarioItem] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            delta = payload.get("delta", {}) if isinstance(payload, dict) else {}
            items.append(
                SimulationScenarioItem(
                    scenario_id=row["scenario_id"],
                    assessment_id=row["assessment_id"],
                    scenario_name=row["scenario_name"],
                    created_at=row["created_at"],
                    wildfire_risk_score_delta=float(delta.get("wildfire_risk_score_delta", 0.0)),
                    insurance_readiness_score_delta=float(delta.get("insurance_readiness_score_delta", 0.0)),
                )
            )

        return items
