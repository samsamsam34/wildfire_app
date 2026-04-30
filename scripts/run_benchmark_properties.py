#!/usr/bin/env python3
"""Read-only benchmark scaffold for comparing risk scores across fixture properties.

This benchmark harness validates fixture rows and evaluates each property
against the existing `/risk/assess` API pipeline through FastAPI TestClient.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import sys
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "benchmark_properties.csv"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "benchmark_results.csv"
DIAGNOSTICS_PATH = REPORTS_DIR / "benchmark_diagnostics.md"

REQUIRED_COLUMNS = [
    "id",
    "address",
    "city",
    "state",
    "latitude",
    "longitude",
    "scenario_group",
    "expected_relative_risk_notes",
    "optional_inputs_json",
]

RESULT_COLUMNS = [
    "id",
    "address",
    "scenario_group",
    "risk_score",
    "insurance_readiness_score",
    "confidence_score",
    "key_drivers",
    "missing_data_flags",
    "notes",
]

DEFAULT_HEADERS = {
    "X-User-Role": "admin",
    "X-Organization-Id": "default_org",
    "X-User-Id": "benchmark_runner",
}

# Heuristic diagnostics thresholds for benchmark reporting only.
VERY_LOW_CONFIDENCE_THRESHOLD = 0.30
SUSPICIOUS_LOW_RISK_THRESHOLD_HIGH_HAZARD = 35.0
OPTIONAL_EFFECT_MIN_DELTA = 1.0
FALLBACK_HEAVY_THRESHOLD = 0.40

PROPERTY_ATTRIBUTE_FIELDS = {
    "roof_type",
    "vent_type",
    "siding_type",
    "window_type",
    "defensible_space_ft",
    "vegetation_condition",
    "driveway_access_notes",
    "construction_year",
    "inspection_notes",
}

DEFENSIBLE_SPACE_BUCKET_TO_FEET = {
    "lt5": 5.0,
    "lt_5": 5.0,
    "0_5": 5.0,
    "10_30": 20.0,
    "5_30": 20.0,
    "30_100": 60.0,
    "gt100": 120.0,
    "gt_100": 120.0,
}


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_confidence(value: Any) -> float | None:
    raw = _as_float(value)
    if raw is None:
        return None
    # API may provide 0-1 or 0-100 across contexts; normalize to 0-1 for diagnostics.
    if raw > 1.0 and raw <= 100.0:
        return raw / 100.0
    return raw


def _load_fixture_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(
                "benchmark_properties.csv is missing required columns: "
                + ", ".join(missing)
            )
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _parse_optional_inputs(raw: str) -> tuple[dict[str, Any], str | None]:
    text = (raw or "").strip()
    if not text:
        return {}, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON ({exc.msg})"
    if not isinstance(parsed, dict):
        return {}, "optional_inputs_json must be a JSON object"
    return parsed, None


def _normalize_optional_inputs(
    optional_inputs: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    attrs: dict[str, Any] = {}
    ignored_keys: list[str] = []
    notes: list[str] = []
    confirmed_fields: list[str] = []

    for key, value in optional_inputs.items():
        k = str(key).strip()
        if k in PROPERTY_ATTRIBUTE_FIELDS:
            attrs[k] = value
            confirmed_fields.append(k)
            continue

        if k == "defensible_space":
            bucket = str(value or "").strip().lower()
            mapped = DEFENSIBLE_SPACE_BUCKET_TO_FEET.get(bucket)
            if mapped is not None:
                attrs["defensible_space_ft"] = mapped
                confirmed_fields.append("defensible_space_ft")
            else:
                notes.append(f"unmapped defensible_space bucket '{value}'")
            continue

        if k == "home_details_provided":
            # Benchmark hint only; not part of scoring attributes.
            continue

        ignored_keys.append(k)

    if "construction_year" in attrs:
        try:
            attrs["construction_year"] = int(attrs["construction_year"])
        except (TypeError, ValueError):
            notes.append("invalid construction_year ignored")
            attrs.pop("construction_year", None)
            if "construction_year" in confirmed_fields:
                confirmed_fields.remove("construction_year")

    deduped_confirmed = sorted(set(confirmed_fields))
    return attrs, ignored_keys, notes


def _full_address(row: dict[str, str]) -> str:
    address = (row.get("address") or "").strip()
    city = (row.get("city") or "").strip()
    state = (row.get("state") or "").strip()
    parts = [part for part in [address, city, state] if part]
    return ", ".join(parts) if parts else address


def _build_payload(
    row: dict[str, str],
    *,
    attributes: dict[str, Any],
    confirmed_fields: list[str],
) -> dict[str, Any]:
    lat = _as_float(row.get("latitude"))
    lon = _as_float(row.get("longitude"))
    if lat is None or lon is None:
        raise ValueError("Missing/invalid latitude or longitude")
    scenario_group = (row.get("scenario_group") or "").strip() or "benchmark"
    return {
        "address": _full_address(row),
        "attributes": attributes,
        "confirmed_fields": confirmed_fields,
        "audience": "homeowner",
        "property_anchor_point": {"latitude": lat, "longitude": lon},
        "user_selected_point": {"latitude": lat, "longitude": lon},
        "tags": ["benchmark_properties", scenario_group],
    }


def _call_assessment(client: TestClient, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = client.post("/risk/assess", json=payload, headers=DEFAULT_HEADERS)
    except Exception as exc:  # pragma: no cover
        return None, f"request_exception: {exc}"

    if response.status_code == 200:
        try:
            return response.json(), None
        except Exception as exc:  # pragma: no cover
            return None, f"invalid_json_response: {exc}"

    detail: Any
    try:
        payload_json = response.json()
        detail = payload_json.get("detail", payload_json)
    except Exception:
        detail = response.text
    return None, f"http_{response.status_code}: {detail}"


def _extract_key_drivers(assessment: dict[str, Any]) -> list[str]:
    lines = [str(line).strip() for line in (assessment.get("top_risk_drivers") or []) if str(line).strip()]
    if lines:
        return lines[:3]
    detailed = assessment.get("top_risk_drivers_detailed") or []
    parsed: list[str] = []
    for row in detailed:
        if not isinstance(row, dict):
            continue
        text = (
            row.get("description")
            or row.get("explanation")
            or row.get("reason")
            or row.get("factor")
        )
        if text is not None and str(text).strip():
            parsed.append(str(text).strip())
    return parsed[:3]


def _extract_actions(assessment: dict[str, Any]) -> list[str]:
    direct = [str(v).strip() for v in (assessment.get("top_recommended_actions") or []) if str(v).strip()]
    if direct:
        return direct[:5]
    plan = assessment.get("mitigation_plan") or []
    parsed: list[str] = []
    for row in plan:
        if not isinstance(row, dict):
            continue
        text = row.get("title") or row.get("action") or row.get("description")
        if text is not None and str(text).strip():
            parsed.append(str(text).strip())
    return parsed[:5]


def _extract_missing_data_flags(
    assessment: dict[str, Any],
    *,
    scenario_group: str,
) -> list[str]:
    flags: set[str] = set()
    status = str(assessment.get("assessment_status") or "").strip()
    if status and status != "fully_scored":
        flags.add(f"assessment_status:{status}")

    if assessment.get("wildfire_risk_score") is None:
        flags.add("wildfire_risk_unavailable")
    if assessment.get("insurance_readiness_score") is None:
        flags.add("readiness_unavailable")

    norm_conf = _normalize_confidence(assessment.get("confidence_score"))
    if norm_conf is not None and norm_conf < VERY_LOW_CONFIDENCE_THRESHOLD:
        flags.add("very_low_confidence")

    if not bool(assessment.get("coverage_available", True)):
        flags.add("prepared_region_unavailable")

    what_missing = assessment.get("what_was_missing") or []
    if isinstance(what_missing, list) and what_missing:
        flags.add("missing_inputs_present")
        lower = " ".join(str(v).lower() for v in what_missing)
        if "parcel" in lower or "footprint" in lower or "geometry" in lower:
            flags.add("missing_geometry")

    fallback_weight_fraction = _as_float(assessment.get("fallback_weight_fraction"))
    if fallback_weight_fraction is not None and fallback_weight_fraction >= FALLBACK_HEAVY_THRESHOLD:
        flags.add("fallback_heavy")

    low_conf_flags = assessment.get("low_confidence_flags") or []
    if isinstance(low_conf_flags, list) and low_conf_flags:
        flags.add("low_confidence_flags_present")

    if scenario_group == "missing_geometry":
        flags.add("scenario:missing_geometry")

    return sorted(flags)


def _format_score(value: Any) -> str:
    num = _as_float(value)
    if num is None:
        return ""
    return f"{num:.2f}"


def _build_result_row(
    row: dict[str, str],
    *,
    assessment: dict[str, Any] | None,
    error: str | None,
    optional_error: str | None,
    optional_ignored_keys: list[str],
    optional_mapping_notes: list[str],
    optional_effect_note: str | None,
) -> dict[str, str]:
    scenario_group = (row.get("scenario_group") or "").strip()
    notes: list[str] = ["Benchmark fixture row (not ground truth)."]
    if optional_error:
        notes.append(f"optional_inputs_json issue: {optional_error}")
    if optional_ignored_keys:
        notes.append(f"ignored_optional_keys={','.join(sorted(optional_ignored_keys))}")
    if optional_mapping_notes:
        notes.extend(optional_mapping_notes)
    if optional_effect_note:
        notes.append(optional_effect_note)
    if error:
        notes.append(f"assessment_error: {error}")

    if assessment is None:
        missing_flags = ["assessment_failed"]
        if optional_error:
            missing_flags.append("invalid_optional_inputs_json")
        return {
            "id": (row.get("id") or "").strip(),
            "address": _full_address(row),
            "scenario_group": scenario_group,
            "risk_score": "",
            "insurance_readiness_score": "",
            "confidence_score": "",
            "key_drivers": "",
            "missing_data_flags": "; ".join(missing_flags),
            "notes": " ".join(notes),
        }

    key_drivers = _extract_key_drivers(assessment)
    flags = _extract_missing_data_flags(assessment, scenario_group=scenario_group)
    if optional_error:
        flags.append("invalid_optional_inputs_json")
    return {
        "id": (row.get("id") or "").strip(),
        "address": _full_address(row),
        "scenario_group": scenario_group,
        "risk_score": _format_score(assessment.get("wildfire_risk_score")),
        "insurance_readiness_score": _format_score(assessment.get("insurance_readiness_score")),
        "confidence_score": _format_score(assessment.get("confidence_score")),
        "key_drivers": " | ".join(key_drivers),
        "missing_data_flags": "; ".join(sorted(set(flags))),
        "notes": " ".join(notes),
    }


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_diagnostics(
    path: Path,
    fixture_path: Path,
    total_rows: int,
    scenario_counts: Counter[str],
    invalid_json_rows: list[str],
    scored_rows: int,
    failed_rows: list[tuple[str, str]],
    very_low_confidence_rows: list[str],
    suspicious_low_risk_rows: list[str],
    optional_no_effect_rows: list[str],
    fallback_heavy_rows: list[str],
    missing_geometry_rows: list[str],
    unmapped_optional_rows: list[str],
) -> None:
    sorted_groups = sorted(scenario_counts.items(), key=lambda kv: kv[0])
    invalid_display = ", ".join(invalid_json_rows) if invalid_json_rows else "None"

    lines = [
        "# Benchmark Scaffold Diagnostics",
        "",
        "This run uses the existing `POST /risk/assess` pipeline through FastAPI TestClient.",
        "No backend scoring logic was modified.",
        "",
        "## Input Summary",
        f"- Fixture path: `{fixture_path}`",
        f"- Total properties loaded: **{total_rows}**",
        f"- Successful assessments: **{scored_rows}**",
        f"- Failed assessments: **{len(failed_rows)}**",
        "- Scenario groups present:",
    ]
    for group, count in sorted_groups:
        lines.append(f"  - `{group}`: {count}")

    lines.extend(
        [
            f"- Rows with invalid `optional_inputs_json`: **{len(invalid_json_rows)}**",
            f"  - IDs: {invalid_display}",
            f"- Very low confidence rows (< {VERY_LOW_CONFIDENCE_THRESHOLD:.2f} normalized): **{len(very_low_confidence_rows)}**",
            f"  - IDs: {', '.join(very_low_confidence_rows) if very_low_confidence_rows else 'None'}",
            "- Suspiciously low risk in `high_regional_hazard` rows "
            f"(risk < {SUSPICIOUS_LOW_RISK_THRESHOLD_HIGH_HAZARD:.1f}): "
            f"**{len(suspicious_low_risk_rows)}**",
            f"  - IDs: {', '.join(suspicious_low_risk_rows) if suspicious_low_risk_rows else 'None'}",
            f"- Optional inputs with little/no observable score effect: **{len(optional_no_effect_rows)}**",
            f"  - IDs: {', '.join(optional_no_effect_rows) if optional_no_effect_rows else 'None'}",
            f"- Missing geometry indicators: **{len(missing_geometry_rows)}**",
            f"  - IDs: {', '.join(missing_geometry_rows) if missing_geometry_rows else 'None'}",
            f"- Fallback-heavy indicators: **{len(fallback_heavy_rows)}**",
            f"  - IDs: {', '.join(fallback_heavy_rows) if fallback_heavy_rows else 'None'}",
            f"- Rows with optional inputs that could not map to scoring attributes: **{len(unmapped_optional_rows)}**",
            f"  - IDs: {', '.join(unmapped_optional_rows) if unmapped_optional_rows else 'None'}",
            "",
            "## Failed Rows",
            *(
                [f"- `{row_id}`: {error}" for row_id, error in failed_rows]
                if failed_rows
                else ["- None"]
            ),
            "",
            "## TODOs for Scoring Integration",
            "- TODO: Calibrate anomaly thresholds using observed benchmark distributions over multiple runs.",
            "- TODO: Add commit hash + environment metadata to make regressions easier to compare.",
            "- TODO: Optionally add paired baseline-vs-override deltas for every row as separate report columns.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_repo_on_path()
    # Benchmark rows include explicit coordinates, so geocode backoff query retries
    # add latency without improving the point-anchored assessment path.
    os.environ.setdefault("WF_GEOCODE_ENABLE_PROVIDER_BACKOFF_QUERY", "0")
    import backend.main as app_main  # local import to keep script startup lightweight

    client = TestClient(app_main.app)
    rows, _fieldnames = _load_fixture_rows(FIXTURE_PATH)

    scenario_counts: Counter[str] = Counter()
    invalid_json_rows: list[str] = []
    result_rows: list[dict[str, str]] = []
    failed_rows: list[tuple[str, str]] = []
    very_low_confidence_rows: list[str] = []
    suspicious_low_risk_rows: list[str] = []
    optional_no_effect_rows: list[str] = []
    fallback_heavy_rows: list[str] = []
    missing_geometry_rows: list[str] = []
    unmapped_optional_rows: list[str] = []
    scored_rows = 0
    comparison_pair_scores: dict[str, list[tuple[str, float | None, float | None]]] = {}

    for row in rows:
        row_id = (row.get("id") or "(missing_id)").strip()
        scenario = (row.get("scenario_group") or "").strip() or "(missing)"
        scenario_counts[scenario] += 1

        parsed_inputs, parse_error = _parse_optional_inputs(row.get("optional_inputs_json") or "")
        if parse_error:
            invalid_json_rows.append(row_id)
        normalized_attrs, ignored_optional_keys, mapping_notes = _normalize_optional_inputs(parsed_inputs)
        if ignored_optional_keys:
            unmapped_optional_rows.append(row_id)

        assessment_payload = _build_payload(
            row,
            attributes=normalized_attrs,
            confirmed_fields=sorted(normalized_attrs.keys()),
        )
        assessment, assessment_error = _call_assessment(client, assessment_payload)

        optional_effect_note: str | None = None

        result_rows.append(
            _build_result_row(
                row,
                assessment=assessment,
                error=assessment_error,
                optional_error=parse_error,
                optional_ignored_keys=ignored_optional_keys,
                optional_mapping_notes=mapping_notes,
                optional_effect_note=optional_effect_note,
            )
        )

        if assessment is None:
            failed_rows.append((row_id, assessment_error or "unknown assessment failure"))
            continue

        scored_rows += 1
        comparison_pair = str(parsed_inputs.get("comparison_pair") or "").strip()
        if comparison_pair:
            comparison_pair_scores.setdefault(comparison_pair, []).append(
                (
                    row_id,
                    _as_float(assessment.get("wildfire_risk_score")),
                    _as_float(assessment.get("insurance_readiness_score")),
                )
            )
        norm_conf = _normalize_confidence(assessment.get("confidence_score"))
        if norm_conf is not None and norm_conf < VERY_LOW_CONFIDENCE_THRESHOLD:
            very_low_confidence_rows.append(row_id)
        risk_score = _as_float(assessment.get("wildfire_risk_score"))
        if (
            scenario == "high_regional_hazard"
            and risk_score is not None
            and risk_score < SUSPICIOUS_LOW_RISK_THRESHOLD_HIGH_HAZARD
        ):
            suspicious_low_risk_rows.append(row_id)
        fallback_weight_fraction = _as_float(assessment.get("fallback_weight_fraction"))
        if fallback_weight_fraction is not None and fallback_weight_fraction >= FALLBACK_HEAVY_THRESHOLD:
            fallback_heavy_rows.append(row_id)
        missing_lines = " ".join(str(v).lower() for v in (assessment.get("what_was_missing") or []))
        if "parcel" in missing_lines or "footprint" in missing_lines or "geometry" in missing_lines:
            missing_geometry_rows.append(row_id)

    # Optional-input effect diagnostics (pair-based) to avoid redundant reruns.
    for _pair, rows_for_pair in comparison_pair_scores.items():
        if len(rows_for_pair) < 2:
            continue
        risks = [v[1] for v in rows_for_pair if v[1] is not None]
        readiness = [v[2] for v in rows_for_pair if v[2] is not None]
        deltas: list[float] = []
        if len(risks) >= 2:
            deltas.append(max(risks) - min(risks))
        if len(readiness) >= 2:
            deltas.append(max(readiness) - min(readiness))
        if deltas and max(deltas) < OPTIONAL_EFFECT_MIN_DELTA:
            optional_no_effect_rows.extend([row_id for row_id, _risk, _read in rows_for_pair])

    _write_results(RESULTS_PATH, result_rows)
    _write_diagnostics(
        DIAGNOSTICS_PATH,
        FIXTURE_PATH,
        total_rows=len(rows),
        scenario_counts=scenario_counts,
        invalid_json_rows=invalid_json_rows,
        scored_rows=scored_rows,
        failed_rows=failed_rows,
        very_low_confidence_rows=sorted(set(very_low_confidence_rows)),
        suspicious_low_risk_rows=sorted(set(suspicious_low_risk_rows)),
        optional_no_effect_rows=sorted(set(optional_no_effect_rows)),
        fallback_heavy_rows=sorted(set(fallback_heavy_rows)),
        missing_geometry_rows=sorted(set(missing_geometry_rows)),
        unmapped_optional_rows=sorted(set(unmapped_optional_rows)),
    )

    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {DIAGNOSTICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
