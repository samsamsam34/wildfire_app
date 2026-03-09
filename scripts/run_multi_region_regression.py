#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore

DEFAULT_FIXTURE_PATH = Path("tests") / "fixtures" / "multi_region_sample_properties.json"
DEFAULT_OUTPUT_DIR = Path("benchmark") / "multi_region_runtime"


def _now_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_fixture(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Fixture must be an object with a 'samples' list.")
    return payload


def _fallback_notes(scoring_notes: list[str]) -> list[str]:
    flagged: list[str] = []
    for note in scoring_notes:
        text = str(note).lower()
        if any(token in text for token in ["fallback", "unavailable", "missing", "partial data"]):
            flagged.append(str(note))
    return flagged


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@contextmanager
def _patched_runtime(samples: list[dict[str, Any]]):
    original_geocode = app_main.geocoder.geocode
    original_store = app_main.store
    original_api_keys = set(auth.API_KEYS)

    geocode_map = {
        str(row["address"]): (
            float(row["latitude"]),
            float(row["longitude"]),
            "regression-fixture",
        )
        for row in samples
    }

    def _fixture_geocode(address: str):
        key = str(address)
        if key not in geocode_map:
            raise RuntimeError(f"Address not found in regression fixture: {key}")
        return geocode_map[key]

    auth.API_KEYS = set()
    app_main.geocoder.geocode = _fixture_geocode  # type: ignore[assignment]
    with tempfile.TemporaryDirectory(prefix="wf_multi_region_regression_") as tmp:
        app_main.store = AssessmentStore(str(Path(tmp) / "regression.db"))
        try:
            yield
        finally:
            app_main.geocoder.geocode = original_geocode  # type: ignore[assignment]
            app_main.store = original_store
            auth.API_KEYS = original_api_keys


def _evaluate_case(
    client: TestClient,
    sample: dict[str, Any],
    *,
    excessive_fallback_threshold: int,
) -> dict[str, Any]:
    address = str(sample["address"])
    lat = float(sample["latitude"])
    lon = float(sample["longitude"])
    expected_prefix = sample.get("expected_region_prefix")
    expected_coverage = bool(sample.get("expected_coverage", True))

    coverage_resp = client.post(
        "/regions/coverage-check",
        json={"latitude": lat, "longitude": lon},
        headers={"X-User-Role": "admin", "X-Organization-Id": "default_org", "X-User-Id": "regression"},
    )
    coverage_payload = coverage_resp.json() if coverage_resp.status_code == 200 else {"error": coverage_resp.text}

    assess_payload = {
        "address": address,
        "attributes": sample.get("attributes") or {},
        "confirmed_fields": sample.get("confirmed_fields") or [],
        "audience": "homeowner",
        "tags": ["multi_region_regression"],
    }
    assess_resp = client.post(
        "/risk/assess",
        json=assess_payload,
        headers={"X-User-Role": "admin", "X-Organization-Id": "default_org", "X-User-Id": "regression"},
    )

    row: dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "address": address,
        "city": sample.get("city"),
        "expected_region_prefix": expected_prefix,
        "expected_coverage": expected_coverage,
        "coverage_check": coverage_payload,
        "assessment_http_status": assess_resp.status_code,
        "issues": [],
    }

    if assess_resp.status_code == 200:
        body = assess_resp.json()
        region_resolution = body.get("region_resolution") or {}
        resolved_region_id = region_resolution.get("resolved_region_id") or (body.get("property_level_context") or {}).get("region_id")
        scoring_notes = list(body.get("scoring_notes") or [])
        fallback_lines = _fallback_notes(scoring_notes)

        row.update(
            {
                "assessment_status": body.get("assessment_status"),
                "region_resolution": region_resolution,
                "resolved_region_id": resolved_region_id,
                "wildfire_risk_score": body.get("wildfire_risk_score"),
                "insurance_readiness_score": body.get("insurance_readiness_score"),
                "top_risk_drivers": list(body.get("top_risk_drivers") or [])[:3],
                "top_mitigation_recommendations": [
                    rec.get("title")
                    for rec in (body.get("mitigation_plan") or [])[:3]
                    if isinstance(rec, dict)
                ],
                "warnings_or_fallbacks": fallback_lines[:8],
                "property_level_context": body.get("property_level_context") or {},
            }
        )

        if expected_coverage and not bool(region_resolution.get("coverage_available", False)):
            row["issues"].append("expected_covered_but_resolution_uncovered")
        if expected_prefix and (not resolved_region_id or not str(resolved_region_id).startswith(str(expected_prefix))):
            row["issues"].append("resolved_region_prefix_mismatch")
        if expected_coverage:
            if body.get("wildfire_risk_score") is None or body.get("wildfire_risk_score_available") is False:
                row["issues"].append("missing_wildfire_score")
            if body.get("insurance_readiness_score") is None or body.get("insurance_readiness_score_available") is False:
                row["issues"].append("missing_readiness_score")
        if len(fallback_lines) > excessive_fallback_threshold and expected_coverage:
            row["issues"].append("excessive_fallback_behavior")
        if expected_coverage and str((body.get("property_level_context") or {}).get("region_manifest_path") or "").strip() == "":
            row["issues"].append("missing_region_manifest_path")
        if expected_coverage and any("legacy direct layer" in str(note).lower() for note in scoring_notes):
            row["issues"].append("legacy_fallback_detected")

    elif assess_resp.status_code == 409:
        detail = assess_resp.json().get("detail") if isinstance(assess_resp.json(), dict) else {}
        row.update(
            {
                "assessment_status": "region_not_ready",
                "region_resolution": detail,
                "resolved_region_id": detail.get("resolved_region_id"),
                "wildfire_risk_score": None,
                "insurance_readiness_score": None,
                "top_risk_drivers": [],
                "top_mitigation_recommendations": [],
                "warnings_or_fallbacks": list(detail.get("diagnostics") or [])[:8],
            }
        )
        if expected_coverage:
            row["issues"].append("expected_covered_but_region_not_ready")
        if detail.get("reason") != "no_prepared_region_for_location":
            row["issues"].append("unexpected_409_reason")

    else:
        row.update(
            {
                "assessment_status": "hard_failure",
                "region_resolution": {"error": assess_resp.text},
                "resolved_region_id": None,
                "wildfire_risk_score": None,
                "insurance_readiness_score": None,
                "top_risk_drivers": [],
                "top_mitigation_recommendations": [],
                "warnings_or_fallbacks": [],
            }
        )
        row["issues"].append("unexpected_hard_failure")

    return row


def _find_identical_output_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[tuple[Any, Any], list[str]] = {}
    for row in rows:
        if row.get("assessment_http_status") != 200:
            continue
        if not row.get("expected_coverage", True):
            continue
        if row.get("wildfire_risk_score") is None or row.get("insurance_readiness_score") is None:
            continue
        key = (
            round(_as_float(row.get("wildfire_risk_score")) or -1.0, 2),
            round(_as_float(row.get("insurance_readiness_score")) or -1.0, 2),
        )
        clusters.setdefault(key, []).append(str(row.get("sample_id")))

    suspicious: list[dict[str, Any]] = []
    for (risk, readiness), sample_ids in clusters.items():
        if len(sample_ids) >= 3:
            suspicious.append(
                {
                    "wildfire_risk_score": risk,
                    "insurance_readiness_score": readiness,
                    "sample_ids": sample_ids,
                    "note": "Three or more distinct homes produced identical top-level scores.",
                }
            )
    return suspicious


def run_regression(
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    excessive_fallback_threshold: int = 5,
) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    samples: list[dict[str, Any]] = list(fixture.get("samples") or [])

    with _patched_runtime(samples):
        client = TestClient(app_main.app)
        rows = [
            _evaluate_case(client, sample, excessive_fallback_threshold=excessive_fallback_threshold)
            for sample in samples
        ]

    identical_clusters = _find_identical_output_clusters(rows)
    if identical_clusters:
        for row in rows:
            cluster_samples = {
                sid for cluster in identical_clusters for sid in cluster.get("sample_ids", [])
            }
            if str(row.get("sample_id")) in cluster_samples:
                row.setdefault("issues", []).append("identical_output_cluster")

    total_issues = sum(len(row.get("issues") or []) for row in rows)
    hard_failures = [row for row in rows if "unexpected_hard_failure" in (row.get("issues") or [])]
    unresolved_covered = [
        row for row in rows if row.get("expected_coverage") and any(issue in (row.get("issues") or []) for issue in [
            "expected_covered_but_resolution_uncovered",
            "resolved_region_prefix_mismatch",
            "expected_covered_but_region_not_ready",
        ])
    ]

    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_id": fixture.get("dataset_id"),
        "sample_count": len(rows),
        "covered_sample_count": sum(1 for row in rows if row.get("expected_coverage")),
        "uncovered_sample_count": sum(1 for row in rows if not row.get("expected_coverage")),
        "successful_assessments": sum(1 for row in rows if row.get("assessment_http_status") == 200),
        "region_not_ready_responses": sum(1 for row in rows if row.get("assessment_http_status") == 409),
        "issue_count": total_issues,
        "unexpected_hard_failures": len(hard_failures),
        "suspicious_resolution_count": len(unresolved_covered),
        "identical_output_cluster_count": len(identical_clusters),
        "blocking_issue_count": sum(
            1
            for row in rows
            for issue in (row.get("issues") or [])
            if issue
            in {
                "missing_wildfire_score",
                "missing_readiness_score",
                "expected_covered_but_resolution_uncovered",
                "resolved_region_prefix_mismatch",
                "expected_covered_but_region_not_ready",
                "unexpected_hard_failure",
                "legacy_fallback_detected",
            }
        ),
        "ready": False,
    }
    summary["ready"] = (
        len(hard_failures) == 0
        and len(unresolved_covered) == 0
        and int(summary["blocking_issue_count"]) == 0
    )

    artifact = {
        "summary": summary,
        "identical_output_clusters": identical_clusters,
        "samples": rows,
    }

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    json_path = out_dir / f"multi_region_regression_{stamp}.json"
    md_path = out_dir / f"multi_region_regression_{stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# Multi-Region Runtime Regression",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Successful assessments: `{summary['successful_assessments']}`",
        f"- Region-not-ready responses: `{summary['region_not_ready_responses']}`",
        f"- Total issues: `{summary['issue_count']}`",
        f"- Ready: `{summary['ready']}`",
        "",
        "## Samples",
    ]

    for row in rows:
        md_lines.append(
            f"- `{row.get('sample_id')}` ({row.get('city')}): "
            f"region=`{row.get('resolved_region_id')}` "
            f"risk=`{row.get('wildfire_risk_score')}` "
            f"readiness=`{row.get('insurance_readiness_score')}` "
            f"issues={row.get('issues') or []}"
        )

    if identical_clusters:
        md_lines.extend(["", "## Identical Output Clusters"])
        for cluster in identical_clusters:
            md_lines.append(
                f"- risk={cluster['wildfire_risk_score']} readiness={cluster['insurance_readiness_score']} samples={cluster['sample_ids']}"
            )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    artifact["artifact_path"] = str(json_path)
    artifact["markdown_summary_path"] = str(md_path)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-region address regression checks using prepared-region runtime resolution."
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-fallback-lines", type=int, default=5)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args(argv)

    artifact = run_regression(
        fixture_path=Path(args.fixture).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        excessive_fallback_threshold=max(1, int(args.max_fallback_lines)),
    )

    print(
        json.dumps(
            {
                "artifact_path": artifact.get("artifact_path"),
                "markdown_summary_path": artifact.get("markdown_summary_path"),
                "summary": artifact.get("summary"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.fail_on_issues and int((artifact.get("summary") or {}).get("issue_count") or 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
