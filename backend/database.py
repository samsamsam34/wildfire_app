from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from backend.models import (
    AdminSummary,
    AssessmentAnnotation,
    AssessmentListItem,
    AssessmentResult,
    AssessmentReviewStatus,
    AssessmentWorkflowInfo,
    AuditEvent,
    Organization,
    OrganizationCreate,
    PortfolioJobsSummary,
    PortfolioSummary,
    ReviewStatus,
    SimulationScenarioItem,
    UnderwritingRuleset,
    UnderwritingRulesetCreate,
    WorkflowState,
)
from backend.version import LEGACY_MODEL_VERSION

DEFAULT_ORG_ID = "default_org"
DEFAULT_ORG_NAME = "Default Demo Organization"

DEFAULT_RULESETS: list[UnderwritingRuleset] = [
    UnderwritingRuleset(
        ruleset_id="default",
        ruleset_name="Default Carrier Profile",
        ruleset_version="1.0",
        ruleset_description="Baseline underwriting-oriented adjustments with moderate thresholds.",
        config={
            "penalty_multiplier": 1.0,
            "risk_blocker_threshold": 85.0,
            "inspection_missing_threshold": 5,
            "mitigation_required_priority_boost": 0,
            "insurer_emphasis_level": "standard",
        },
    ),
    UnderwritingRuleset(
        ruleset_id="strict_carrier_demo",
        ruleset_name="Strict Carrier Demo",
        ruleset_version="1.0",
        ruleset_description="Lower blocker thresholds and stronger readiness penalty scaling.",
        config={
            "penalty_multiplier": 1.2,
            "risk_blocker_threshold": 75.0,
            "inspection_missing_threshold": 3,
            "mitigation_required_priority_boost": 1,
            "insurer_emphasis_level": "high",
        },
    ),
    UnderwritingRuleset(
        ruleset_id="inspection_first_demo",
        ruleset_name="Inspection First Demo",
        ruleset_version="1.0",
        ruleset_description="Conservative profile that pushes inspections before final underwriting readiness.",
        config={
            "penalty_multiplier": 1.1,
            "risk_blocker_threshold": 80.0,
            "inspection_missing_threshold": 2,
            "mitigation_required_priority_boost": 0,
            "insurer_emphasis_level": "inspection_first",
        },
    ),
]


class AssessmentStore:
    def __init__(self, db_path: str = "wildfire_app.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _now(self) -> str:
        return datetime.now(tz=timezone.utc).isoformat()

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
                CREATE TABLE IF NOT EXISTS organizations (
                    organization_id TEXT PRIMARY KEY,
                    organization_name TEXT NOT NULL,
                    organization_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS underwriting_rulesets (
                    ruleset_id TEXT PRIMARY KEY,
                    ruleset_name TEXT NOT NULL,
                    ruleset_version TEXT NOT NULL,
                    ruleset_description TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

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
                    organization_id TEXT NOT NULL DEFAULT 'default_org',
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
            if "organization_id" not in ann_cols:
                conn.execute(
                    "ALTER TABLE assessment_annotations ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'default_org'"
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_review_status (
                    assessment_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL DEFAULT 'default_org',
                    review_status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            review_cols = {row["name"] for row in conn.execute("PRAGMA table_info(assessment_review_status)").fetchall()}
            if "organization_id" not in review_cols:
                conn.execute(
                    "ALTER TABLE assessment_review_status ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'default_org'"
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_workflow (
                    assessment_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL DEFAULT 'default_org',
                    workflow_state TEXT NOT NULL DEFAULT 'new',
                    assigned_reviewer TEXT,
                    assigned_role TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_jobs (
                    job_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_summary TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    audit_event_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            self._seed_organizations(conn)
            self._seed_rulesets(conn)

    def _seed_organizations(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT organization_id FROM organizations WHERE organization_id = ?",
            (DEFAULT_ORG_ID,),
        ).fetchone()
        if not row:
            conn.execute(
                """
                INSERT INTO organizations (organization_id, organization_name, organization_type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (DEFAULT_ORG_ID, DEFAULT_ORG_NAME, "demo", self._now()),
            )

    def _seed_rulesets(self, conn: sqlite3.Connection) -> None:
        for ruleset in DEFAULT_RULESETS:
            row = conn.execute(
                "SELECT ruleset_id FROM underwriting_rulesets WHERE ruleset_id = ?",
                (ruleset.ruleset_id,),
            ).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO underwriting_rulesets
                    (ruleset_id, ruleset_name, ruleset_version, ruleset_description, config_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ruleset.ruleset_id,
                        ruleset.ruleset_name,
                        ruleset.ruleset_version,
                        ruleset.ruleset_description,
                        json.dumps(ruleset.config),
                        self._now(),
                    ),
                )

    def create_organization(self, payload: OrganizationCreate) -> Organization:
        created_at = self._now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT organization_id FROM organizations WHERE organization_id = ?",
                (payload.organization_id,),
            ).fetchone()
            if existing:
                raise ValueError("Organization already exists")
            conn.execute(
                """
                INSERT INTO organizations (organization_id, organization_name, organization_type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (payload.organization_id, payload.organization_name, payload.organization_type, created_at),
            )
        return Organization(
            organization_id=payload.organization_id,
            organization_name=payload.organization_name,
            organization_type=payload.organization_type,
            created_at=created_at,
        )

    def list_organizations(self) -> list[Organization]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT organization_id, organization_name, organization_type, created_at
                FROM organizations
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [
            Organization(
                organization_id=row["organization_id"],
                organization_name=row["organization_name"],
                organization_type=row["organization_type"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_organization(self, organization_id: str) -> Optional[Organization]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT organization_id, organization_name, organization_type, created_at
                FROM organizations
                WHERE organization_id = ?
                """,
                (organization_id,),
            ).fetchone()
        if not row:
            return None
        return Organization(
            organization_id=row["organization_id"],
            organization_name=row["organization_name"],
            organization_type=row["organization_type"],
            created_at=row["created_at"],
        )

    def list_rulesets(self) -> list[UnderwritingRuleset]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ruleset_id, ruleset_name, ruleset_version, ruleset_description, config_json
                FROM underwriting_rulesets
                ORDER BY ruleset_id ASC
                """
            ).fetchall()

        rulesets: list[UnderwritingRuleset] = []
        for row in rows:
            try:
                config = json.loads(row["config_json"]) if row["config_json"] else {}
            except Exception:
                config = {}
            rulesets.append(
                UnderwritingRuleset(
                    ruleset_id=row["ruleset_id"],
                    ruleset_name=row["ruleset_name"],
                    ruleset_version=row["ruleset_version"],
                    ruleset_description=row["ruleset_description"],
                    config=config,
                )
            )
        return rulesets

    def get_ruleset(self, ruleset_id: str) -> Optional[UnderwritingRuleset]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT ruleset_id, ruleset_name, ruleset_version, ruleset_description, config_json
                FROM underwriting_rulesets
                WHERE ruleset_id = ?
                """,
                (ruleset_id,),
            ).fetchone()

        if not row:
            return None

        try:
            config = json.loads(row["config_json"]) if row["config_json"] else {}
        except Exception:
            config = {}

        return UnderwritingRuleset(
            ruleset_id=row["ruleset_id"],
            ruleset_name=row["ruleset_name"],
            ruleset_version=row["ruleset_version"],
            ruleset_description=row["ruleset_description"],
            config=config,
        )

    def create_ruleset(self, payload: UnderwritingRulesetCreate) -> UnderwritingRuleset:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT ruleset_id FROM underwriting_rulesets WHERE ruleset_id = ?",
                (payload.ruleset_id,),
            ).fetchone()
            if existing:
                raise ValueError("Ruleset already exists")
            conn.execute(
                """
                INSERT INTO underwriting_rulesets
                (ruleset_id, ruleset_name, ruleset_version, ruleset_description, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.ruleset_id,
                    payload.ruleset_name,
                    payload.ruleset_version,
                    payload.ruleset_description,
                    json.dumps(payload.config),
                    self._now(),
                ),
            )

        return UnderwritingRuleset(
            ruleset_id=payload.ruleset_id,
            ruleset_name=payload.ruleset_name,
            ruleset_version=payload.ruleset_version,
            ruleset_description=payload.ruleset_description,
            config=payload.config,
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
                    self._now(),
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
                    self._now(),
                    scenario_name,
                    json.dumps(payload, default=str),
                ),
            )
        return scenario_id

    def set_review_status(
        self,
        assessment_id: str,
        organization_id: str,
        review_status: ReviewStatus,
    ) -> AssessmentReviewStatus:
        updated_at = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_review_status (assessment_id, organization_id, review_status, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    organization_id=excluded.organization_id,
                    review_status=excluded.review_status,
                    updated_at=excluded.updated_at
                """,
                (assessment_id, organization_id, review_status, updated_at),
            )

        return AssessmentReviewStatus(
            assessment_id=assessment_id,
            organization_id=organization_id,
            review_status=review_status,
            updated_at=updated_at,
        )

    def get_review_status(self, assessment_id: str) -> AssessmentReviewStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT assessment_id, organization_id, review_status, updated_at FROM assessment_review_status WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()

        if not row:
            return None

        return AssessmentReviewStatus(
            assessment_id=row["assessment_id"],
            organization_id=row["organization_id"],
            review_status=row["review_status"],
            updated_at=row["updated_at"],
        )

    def set_workflow(
        self,
        assessment_id: str,
        organization_id: str,
        workflow_state: WorkflowState,
        assigned_reviewer: str | None = None,
        assigned_role: str | None = None,
    ) -> AssessmentWorkflowInfo:
        current = self.get_workflow(assessment_id)
        updated_at = self._now()

        reviewer = assigned_reviewer if assigned_reviewer is not None else (current.assigned_reviewer if current else None)
        role = assigned_role if assigned_role is not None else (current.assigned_role if current else None)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_workflow
                (assessment_id, organization_id, workflow_state, assigned_reviewer, assigned_role, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    organization_id=excluded.organization_id,
                    workflow_state=excluded.workflow_state,
                    assigned_reviewer=excluded.assigned_reviewer,
                    assigned_role=excluded.assigned_role,
                    updated_at=excluded.updated_at
                """,
                (assessment_id, organization_id, workflow_state, reviewer, role, updated_at),
            )

        return AssessmentWorkflowInfo(
            assessment_id=assessment_id,
            organization_id=organization_id,
            workflow_state=workflow_state,
            assigned_reviewer=reviewer,
            assigned_role=role,
            updated_at=updated_at,
        )

    def set_assignment(
        self,
        assessment_id: str,
        organization_id: str,
        assigned_reviewer: str | None,
        assigned_role: str | None,
    ) -> AssessmentWorkflowInfo:
        current = self.get_workflow(assessment_id)
        workflow_state = current.workflow_state if current else "new"
        return self.set_workflow(
            assessment_id=assessment_id,
            organization_id=organization_id,
            workflow_state=workflow_state,
            assigned_reviewer=assigned_reviewer,
            assigned_role=assigned_role,
        )

    def get_workflow(self, assessment_id: str) -> AssessmentWorkflowInfo | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT assessment_id, organization_id, workflow_state, assigned_reviewer, assigned_role, updated_at
                FROM assessment_workflow
                WHERE assessment_id = ?
                """,
                (assessment_id,),
            ).fetchone()

        if not row:
            return None

        return AssessmentWorkflowInfo(
            assessment_id=row["assessment_id"],
            organization_id=row["organization_id"],
            workflow_state=row["workflow_state"],
            assigned_reviewer=row["assigned_reviewer"],
            assigned_role=row["assigned_role"],
            updated_at=row["updated_at"],
        )

    def save_annotation(
        self,
        assessment_id: str,
        organization_id: str,
        author_role: str,
        note: str,
        tags: list[str],
        visibility: str,
        review_status: ReviewStatus | None = None,
    ) -> AssessmentAnnotation:
        annotation_id = str(uuid4())
        created_at = self._now()
        payload_tags = sorted(set([t.strip() for t in tags if t and t.strip()]))
        current_review = review_status
        if current_review is None:
            existing = self.get_review_status(assessment_id)
            current_review = existing.review_status if existing else "pending"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assessment_annotations (
                    annotation_id, assessment_id, organization_id, created_at, author_role, note, visibility, tags_json, review_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    assessment_id,
                    organization_id,
                    created_at,
                    author_role,
                    note,
                    visibility,
                    json.dumps(payload_tags),
                    current_review,
                ),
            )

        if review_status is not None:
            self.set_review_status(assessment_id, organization_id, review_status)

        return AssessmentAnnotation(
            annotation_id=annotation_id,
            assessment_id=assessment_id,
            organization_id=organization_id,
            created_at=created_at,
            author_role=author_role,
            note=note,
            tags=payload_tags,
            visibility=visibility,
            review_status=current_review,
        )

    def list_annotations(
        self,
        assessment_id: str,
        organization_id: str | None = None,
        limit: int = 100,
    ) -> list[AssessmentAnnotation]:
        sql = (
            "SELECT annotation_id, assessment_id, organization_id, created_at, author_role, note, visibility, tags_json, review_status "
            "FROM assessment_annotations WHERE assessment_id = ?"
        )
        params: list[Any] = [assessment_id]
        if organization_id:
            sql += " AND organization_id = ?"
            params.append(organization_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

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
                    organization_id=row["organization_id"],
                    created_at=row["created_at"],
                    author_role=row["author_role"],
                    note=row["note"],
                    tags=tags,
                    visibility=row["visibility"],
                    review_status=row["review_status"],
                )
            )
        return items

    def create_portfolio_job(
        self,
        organization_id: str,
        payload: dict[str, Any],
        status: str = "queued",
    ) -> str:
        job_id = str(uuid4())
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_jobs
                (job_id, organization_id, created_at, updated_at, status, request_json, result_json, error_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, organization_id, now, now, status, json.dumps(payload, default=str), None, None),
            )
        return job_id

    def update_portfolio_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error_summary: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE portfolio_jobs
                SET updated_at = ?, status = ?, result_json = ?, error_summary = ?
                WHERE job_id = ?
                """,
                (
                    self._now(),
                    status,
                    json.dumps(result, default=str) if result is not None else None,
                    error_summary,
                    job_id,
                ),
            )

    def get_portfolio_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, organization_id, created_at, updated_at, status, request_json, result_json, error_summary
                FROM portfolio_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        if not row:
            return None

        try:
            request_payload = json.loads(row["request_json"]) if row["request_json"] else {}
        except Exception:
            request_payload = {}
        try:
            result_payload = json.loads(row["result_json"]) if row["result_json"] else None
        except Exception:
            result_payload = None

        return {
            "job_id": row["job_id"],
            "organization_id": row["organization_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "request": request_payload,
            "result": result_payload,
            "error_summary": row["error_summary"],
        }

    def list_portfolio_jobs(
        self,
        organization_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT job_id FROM portfolio_jobs "
        )
        params: list[Any] = []
        if organization_id:
            sql += "WHERE organization_id = ? "
            params.append(organization_id)
        sql += "ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            loaded = self.get_portfolio_job(row["job_id"])
            if loaded:
                items.append(loaded)
        return items

    def summarize_portfolio_jobs(self, organization_id: str | None = None) -> PortfolioJobsSummary:
        jobs = self.list_portfolio_jobs(organization_id=organization_id, limit=1_000_000, offset=0)
        total = len(jobs)
        queued = sum(1 for j in jobs if j["status"] == "queued")
        running = sum(1 for j in jobs if j["status"] == "running")
        completed = sum(1 for j in jobs if j["status"] == "completed")
        failed = sum(1 for j in jobs if j["status"] == "failed")
        partial = sum(1 for j in jobs if j["status"] == "partial")
        failure_rate = round((failed / total) * 100.0, 1) if total else 0.0

        return PortfolioJobsSummary(
            total_jobs=total,
            queued_count=queued,
            running_count=running,
            completed_count=completed,
            failed_count=failed,
            partial_count=partial,
            failure_rate=failure_rate,
        )

    def log_event(
        self,
        *,
        entity_type: str,
        entity_id: str,
        organization_id: str,
        user_role: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        audit_event_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events
                (audit_event_id, entity_type, entity_id, organization_id, user_role, action, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_event_id,
                    entity_type,
                    entity_id,
                    organization_id,
                    user_role,
                    action,
                    json.dumps(metadata or {}, default=str),
                    self._now(),
                ),
            )
        return audit_event_id

    def list_audit_events(
        self,
        *,
        organization_id: str | None = None,
        entity_type: str | None = None,
        action: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[AuditEvent]:
        sql = (
            "SELECT audit_event_id, entity_type, entity_id, organization_id, user_role, action, metadata_json, created_at "
            "FROM audit_events WHERE 1=1"
        )
        params: list[Any] = []
        if organization_id:
            sql += " AND organization_id = ?"
            params.append(organization_id)
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        if action:
            sql += " AND action = ?"
            params.append(action)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        events: list[AuditEvent] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                metadata = {}
            events.append(
                AuditEvent(
                    audit_event_id=row["audit_event_id"],
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    organization_id=row["organization_id"],
                    user_role=row["user_role"],
                    action=row["action"],
                    metadata=metadata,
                    created_at=row["created_at"],
                )
            )
        return events

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

        payload.setdefault("organization_id", DEFAULT_ORG_ID)
        payload.setdefault("audience", "homeowner")
        payload.setdefault("report_audience", None)
        payload.setdefault("audience_highlights", [])
        payload.setdefault("portfolio_name", None)
        payload.setdefault("tags", [])

        payload.setdefault("ruleset_id", "default")
        payload.setdefault("ruleset_name", "Default Carrier Profile")
        payload.setdefault("ruleset_version", "1.0")
        payload.setdefault("ruleset_description", "Default underwriting-oriented readiness adjustments")

        payload.setdefault("review_status", "pending")
        payload.setdefault("workflow_state", "new")
        payload.setdefault("assigned_reviewer", None)
        payload.setdefault("assigned_role", None)

        payload.setdefault("property_facts", {})
        payload.setdefault("confirmed_fields", [])

        payload.setdefault("latitude", payload.get("coordinates", {}).get("latitude", 0.0))
        payload.setdefault("longitude", payload.get("coordinates", {}).get("longitude", 0.0))
        payload.setdefault("wildfire_risk_score", payload.get("risk_scores", {}).get("wildfire_risk_score", 0.0))
        payload.setdefault("legacy_weighted_wildfire_risk_score", payload.get("wildfire_risk_score", 0.0))
        payload.setdefault("site_hazard_score", payload.get("risk_drivers", {}).get("environmental", 0.0))
        payload.setdefault("home_ignition_vulnerability_score", payload.get("risk_drivers", {}).get("structural", 0.0))
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
        payload.setdefault("property_findings", [])
        payload.setdefault("explanation_summary", payload.get("explanation", ""))

        payload.setdefault("confirmed_inputs", payload.get("assumptions", {}).get("confirmed_inputs", {}))
        payload.setdefault("observed_inputs", payload.get("assumptions", {}).get("observed_inputs", {}))
        payload.setdefault("inferred_inputs", payload.get("assumptions", {}).get("inferred_inputs", {}))
        payload.setdefault("missing_inputs", payload.get("assumptions", {}).get("missing_inputs", []))
        payload.setdefault("assumptions_used", payload.get("assumptions", {}).get("assumptions_used", []))

        if "confidence_score" not in payload:
            payload["confidence_score"] = payload.get("confidence", {}).get("confidence_score", 60.0)
        if "data_completeness_score" not in payload:
            payload["data_completeness_score"] = payload.get("confidence", {}).get("data_completeness_score", 50.0)
        if "environmental_data_completeness_score" not in payload:
            payload["environmental_data_completeness_score"] = payload.get("confidence", {}).get(
                "environmental_data_completeness_score",
                0.0,
            )
        payload.setdefault("confidence_tier", payload.get("confidence", {}).get("confidence_tier", "preliminary"))
        payload.setdefault(
            "use_restriction",
            payload.get("confidence", {}).get("use_restriction", "not_for_underwriting_or_binding"),
        )
        if "low_confidence_flags" not in payload:
            payload["low_confidence_flags"] = payload.get("confidence", {}).get("low_confidence_flags", [])

        payload.setdefault("data_sources", [])
        payload.setdefault(
            "property_level_context",
            {
                "footprint_used": False,
                "footprint_status": "not_found",
                "fallback_mode": "point_based",
                "ring_metrics": None,
            },
        )
        payload.setdefault(
            "environmental_layer_status",
            {
                "burn_probability": "missing",
                "hazard": "missing",
                "slope": "missing",
                "fuel": "missing",
                "canopy": "missing",
                "fire_history": "missing",
            },
        )
        for layer in ["burn_probability", "hazard", "slope", "fuel", "canopy", "fire_history"]:
            payload["environmental_layer_status"].setdefault(layer, "missing")
        if isinstance(payload.get("property_level_context"), dict):
            plc = payload["property_level_context"]
            plc.setdefault("footprint_used", False)
            plc.setdefault("footprint_status", "not_found")
            if plc.get("footprint_status") == "source_unavailable":
                plc["footprint_status"] = "provider_unavailable"
            plc.setdefault("fallback_mode", "footprint" if plc.get("footprint_used") else "point_based")
            plc.setdefault("ring_metrics", None)
        payload.setdefault("mitigation_plan", payload.get("mitigation_recommendations", []))
        payload["mitigation_plan"] = self._upgrade_mitigation_items(payload.get("mitigation_plan", []))

        payload.setdefault("readiness_factors", [])
        payload.setdefault("readiness_blockers", [])
        payload.setdefault("readiness_penalties", {})
        payload.setdefault("readiness_summary", "Legacy row: readiness detail unavailable in this version.")
        payload.setdefault(
            "site_hazard_section",
            {
                "label": "Site Hazard",
                "score": payload.get("site_hazard_score", 0.0),
                "summary": "",
                "explanation": "What the landscape is doing around your property.",
                "top_drivers": payload.get("top_risk_drivers", [])[:3],
                "key_drivers": payload.get("top_risk_drivers", [])[:3],
                "protective_factors": payload.get("top_protective_factors", [])[:3],
                "top_next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
                "next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
            },
        )
        payload.setdefault(
            "home_ignition_vulnerability_section",
            {
                "label": "Home Ignition Vulnerability",
                "score": payload.get("home_ignition_vulnerability_score", 0.0),
                "summary": "",
                "explanation": "What the home and immediate surroundings are contributing.",
                "top_drivers": payload.get("property_findings", [])[:3] or payload.get("top_risk_drivers", [])[:3],
                "key_drivers": payload.get("property_findings", [])[:3] or payload.get("top_risk_drivers", [])[:3],
                "protective_factors": payload.get("top_protective_factors", [])[:3],
                "top_next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
                "next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
            },
        )
        payload.setdefault(
            "insurance_readiness_section",
            {
                "label": "Insurance Readiness",
                "score": payload.get("insurance_readiness_score", 0.0),
                "summary": payload.get("readiness_summary", ""),
                "explanation": "What an insurer is likely to care about next.",
                "top_drivers": payload.get("readiness_blockers", [])[:3],
                "key_drivers": payload.get("readiness_blockers", [])[:3],
                "protective_factors": payload.get("top_protective_factors", [])[:3],
                "top_next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
                "next_actions": [m.get("title", "") for m in payload.get("mitigation_plan", [])[:3]],
            },
        )
        payload["site_hazard_section"].setdefault("top_drivers", payload["site_hazard_section"].get("key_drivers", []))
        payload["home_ignition_vulnerability_section"].setdefault(
            "top_drivers",
            payload["home_ignition_vulnerability_section"].get("key_drivers", []),
        )
        payload["insurance_readiness_section"].setdefault(
            "top_drivers",
            payload["insurance_readiness_section"].get("key_drivers", []),
        )
        payload.setdefault(
            "score_summaries",
            {
                "site_hazard": payload.get("site_hazard_section", {}),
                "home_ignition_vulnerability": payload.get("home_ignition_vulnerability_section", {}),
                "insurance_readiness": payload.get("insurance_readiness_section", {}),
            },
        )

        payload.setdefault("generated_at", created_at or self._now())
        payload.setdefault("scoring_notes", ["Access risk is provisional and not included in total scoring."])

        payload.setdefault("coordinates", {"latitude": payload["latitude"], "longitude": payload["longitude"]})
        payload.setdefault(
            "risk_scores",
            {
                "site_hazard_score": payload["site_hazard_score"],
                "home_ignition_vulnerability_score": payload["home_ignition_vulnerability_score"],
                "wildfire_risk_score": payload["wildfire_risk_score"],
                "insurance_readiness_score": payload["insurance_readiness_score"],
            },
        )
        payload["risk_scores"].setdefault("site_hazard_score", payload["site_hazard_score"])
        payload["risk_scores"].setdefault(
            "home_ignition_vulnerability_score",
            payload["home_ignition_vulnerability_score"],
        )
        payload["risk_scores"].setdefault("wildfire_risk_score", payload["wildfire_risk_score"])
        payload["risk_scores"].setdefault("insurance_readiness_score", payload["insurance_readiness_score"])
        payload["score_summaries"].setdefault("site_hazard", payload.get("site_hazard_section", {}))
        payload["score_summaries"].setdefault(
            "home_ignition_vulnerability",
            payload.get("home_ignition_vulnerability_section", {}),
        )
        payload["score_summaries"].setdefault(
            "insurance_readiness",
            payload.get("insurance_readiness_section", {}),
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
                "data_completeness_score": payload["data_completeness_score"],
                "environmental_data_completeness_score": payload["environmental_data_completeness_score"],
                "confidence_tier": payload["confidence_tier"],
                "use_restriction": payload["use_restriction"],
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

        workflow = self.get_workflow(assessment_id)
        if workflow:
            upgraded["workflow_state"] = workflow.workflow_state
            upgraded["assigned_reviewer"] = workflow.assigned_reviewer
            upgraded["assigned_role"] = workflow.assigned_role

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
            workflow = self.get_workflow(upgraded["assessment_id"])

            records.append(
                {
                    "assessment_id": upgraded["assessment_id"],
                    "created_at": row["created_at"],
                    "created_dt": self._parse_ts(row["created_at"]),
                    "address": upgraded.get("address", ""),
                    "organization_id": upgraded.get("organization_id", DEFAULT_ORG_ID),
                    "audience": upgraded.get("audience", "homeowner"),
                    "wildfire_risk_score": float(upgraded.get("wildfire_risk_score", 0.0)),
                    "insurance_readiness_score": float(upgraded.get("insurance_readiness_score", 0.0)),
                    "model_version": upgraded.get("model_version", LEGACY_MODEL_VERSION),
                    "confidence_score": float(upgraded.get("confidence_score", 0.0)),
                    "readiness_blockers": list(upgraded.get("readiness_blockers", [])),
                    "tags": list(upgraded.get("tags", [])),
                    "portfolio_name": upgraded.get("portfolio_name"),
                    "review_status": review.review_status if review else upgraded.get("review_status", "pending"),
                    "workflow_state": workflow.workflow_state if workflow else upgraded.get("workflow_state", "new"),
                    "assigned_reviewer": workflow.assigned_reviewer if workflow else upgraded.get("assigned_reviewer"),
                    "assigned_role": workflow.assigned_role if workflow else upgraded.get("assigned_role"),
                    "ruleset_id": upgraded.get("ruleset_id", "default"),
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
        organization_id: str | None = None,
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
        portfolio_name: str | None = None,
        workflow_state: str | None = None,
        assigned_reviewer: str | None = None,
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

        if organization_id:
            records = [r for r in records if r["organization_id"] == organization_id]
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
        if portfolio_name:
            records = [r for r in records if r["portfolio_name"] == portfolio_name]
        if workflow_state:
            records = [r for r in records if r["workflow_state"] == workflow_state]
        if assigned_reviewer:
            records = [r for r in records if r["assigned_reviewer"] == assigned_reviewer]
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
                organization_id=r["organization_id"],
                audience=r["audience"],
                wildfire_risk_score=r["wildfire_risk_score"],
                insurance_readiness_score=r["insurance_readiness_score"],
                model_version=r["model_version"],
                confidence_score=r["confidence_score"],
                readiness_blockers=r["readiness_blockers"],
                tags=r["tags"],
                review_status=r["review_status"],
                workflow_state=r["workflow_state"],
                assigned_reviewer=r["assigned_reviewer"],
                assigned_role=r["assigned_role"],
                ruleset_id=r["ruleset_id"],
            )
            for r in paged
        ]

        return items, total, summary

    def summary_assessments(
        self,
        *,
        organization_id: str | None = None,
        min_risk: float | None = None,
        max_risk: float | None = None,
        min_readiness: float | None = None,
        max_readiness: float | None = None,
        readiness_blocker: str | None = None,
        confidence_min: float | None = None,
        audience: str | None = None,
        tag: str | None = None,
        portfolio_name: str | None = None,
        workflow_state: str | None = None,
        assigned_reviewer: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        recent_days: int | None = None,
    ) -> PortfolioSummary:
        _, _, summary = self.query_assessments(
            organization_id=organization_id,
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
            portfolio_name=portfolio_name,
            workflow_state=workflow_state,
            assigned_reviewer=assigned_reviewer,
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
        organization_id: str | None = None,
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
        portfolio_name: str | None = None,
        workflow_state: str | None = None,
        assigned_reviewer: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        recent_days: int | None = None,
    ) -> list[AssessmentListItem]:
        items, _, _ = self.query_assessments(
            limit=limit,
            offset=offset,
            organization_id=organization_id,
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
            portfolio_name=portfolio_name,
            workflow_state=workflow_state,
            assigned_reviewer=assigned_reviewer,
            created_after=created_after,
            created_before=created_before,
            recent_days=recent_days,
        )
        return items

    def list_assessments_by_portfolio(
        self,
        *,
        portfolio_name: str,
        organization_id: str | None = None,
        limit: int = 5000,
    ) -> list[AssessmentListItem]:
        return self.list_assessments(
            limit=limit,
            offset=0,
            organization_id=organization_id,
            portfolio_name=portfolio_name,
            sort_by="created_at",
            sort_dir="desc",
        )

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

    def build_admin_summary(self, organization_id: str | None = None, recent_days: int = 30) -> AdminSummary:
        recent_cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=max(0, recent_days))).isoformat()
        records = self.list_assessments(
            limit=1_000_000,
            offset=0,
            organization_id=organization_id,
            recent_days=recent_days,
        )

        high_risk = sum(1 for r in records if r.wildfire_risk_score >= 70.0)
        blocker = sum(1 for r in records if r.readiness_blockers)
        pending_review = sum(1 for r in records if r.review_status == "pending")
        needs_insp = sum(1 for r in records if r.workflow_state == "needs_inspection")
        ready_for_review = sum(1 for r in records if r.workflow_state == "ready_for_review")
        approved = sum(1 for r in records if r.workflow_state == "approved")
        declined = sum(1 for r in records if r.workflow_state == "declined")
        escalated = sum(1 for r in records if r.workflow_state == "escalated")

        avg_risk = round(sum(r.wildfire_risk_score for r in records) / len(records), 1) if records else 0.0
        avg_readiness = round(sum(r.insurance_readiness_score for r in records) / len(records), 1) if records else 0.0

        jobs_summary = self.summarize_portfolio_jobs(organization_id=organization_id)

        return AdminSummary(
            organization_id=organization_id,
            assessments_created_recently=len(records),
            high_risk_count=high_risk,
            blocker_count=blocker,
            pending_review_count=pending_review,
            needs_inspection_count=needs_insp,
            ready_for_review_count=ready_for_review,
            approved_count=approved,
            declined_count=declined,
            escalated_count=escalated,
            avg_wildfire_risk=avg_risk,
            avg_insurance_readiness=avg_readiness,
            jobs_summary=jobs_summary,
        )
