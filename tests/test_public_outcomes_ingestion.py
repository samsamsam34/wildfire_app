from __future__ import annotations

import json
from pathlib import Path

from scripts.ingest_public_outcomes import (
    OutcomeSourceSpec,
    _discover_input_files,
    _load_source_specs_from_manifest,
    run_public_outcomes_ingestion,
)


def _write_source_csv(path: Path, rows: list[str]) -> None:
    header = (
        "record_id,event_id,incident_name,inspection_date,address,city,state,zip,"
        "latitude,longitude,damage_state,label_confidence"
    )
    path.write_text("\n".join([header] + rows), encoding="utf-8")


def test_label_normalization_and_schema_fields(tmp_path: Path) -> None:
    src = tmp_path / "source_a.csv"
    _write_source_csv(
        src,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Destroyed,0.95",
            "2,evt-a,Event A,2021-08-01,101 Main St,Town,CA,90001,39.1002,-120.1002,Minor Damage,0.80",
            "3,evt-a,Event A,2021-08-01,102 Main St,Town,CA,90001,39.1003,-120.1003,Unknown Label,0.30",
        ],
    )
    result = run_public_outcomes_ingestion(
        sources=[OutcomeSourceSpec(path=src, source_name="fixture_a")],
        output_root=tmp_path / "out",
        run_id="fixed_run",
        overwrite=True,
    )
    payload = json.loads(
        (Path(result["run_dir"]) / "normalized_outcomes.json").read_text(encoding="utf-8")
    )
    rows = payload["records"]
    assert len(rows) == 3
    required_keys = {
        "source_name",
        "event_name",
        "event_year",
        "latitude",
        "longitude",
        "address_text",
        "damage_label",
        "damage_severity_class",
        "adverse_outcome_binary",
        "adverse_outcome_label",
        "source_native_label",
        "match_confidence",
    }
    assert required_keys.issubset(set(rows[0].keys()))
    by_label = {row["source_native_label"]: row for row in rows}
    assert by_label["Destroyed"]["damage_severity_class"] == "destroyed"
    assert by_label["Destroyed"]["adverse_outcome_binary"] is True
    assert by_label["Minor Damage"]["damage_severity_class"] == "minor"
    assert by_label["Minor Damage"]["adverse_outcome_binary"] is False
    assert by_label["Unknown Label"]["damage_severity_class"] == "unknown"
    assert by_label["Unknown Label"]["adverse_outcome_binary"] is None


def test_deduplication_prefers_higher_quality_record(tmp_path: Path) -> None:
    src_a = tmp_path / "source_a.csv"
    src_b = tmp_path / "source_b.csv"
    # Duplicate location/event with weaker confidence + unknown label in source_a.
    _write_source_csv(
        src_a,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Unknown Label,0.20",
        ],
    )
    # Same event/location with stronger label in source_b; should win dedupe.
    _write_source_csv(
        src_b,
        [
            "9,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Major Damage,0.90",
        ],
    )
    result = run_public_outcomes_ingestion(
        sources=[
            OutcomeSourceSpec(path=src_a, source_name="fixture_a"),
            OutcomeSourceSpec(path=src_b, source_name="fixture_b"),
        ],
        output_root=tmp_path / "out",
        run_id="dedupe_run",
        overwrite=True,
    )
    rows = json.loads((Path(result["run_dir"]) / "normalized_outcomes.json").read_text(encoding="utf-8"))[
        "records"
    ]
    assert len(rows) == 1
    assert rows[0]["damage_label"] == "major_damage"
    assert rows[0]["source_name"] == "fixture_b"
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert int((manifest.get("summary") or {}).get("deduplicated_record_count") or 0) == 1


def test_deterministic_output_with_fixed_run_id_and_missing_source_graceful(tmp_path: Path) -> None:
    src = tmp_path / "source_a.csv"
    _write_source_csv(
        src,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Destroyed,0.95",
            "2,evt-a,Event A,2021-08-01,101 Main St,Town,CA,90001,39.1002,-120.1002,No Damage,0.70",
        ],
    )
    missing = tmp_path / "does_not_exist.csv"
    out_root = tmp_path / "out"
    first = run_public_outcomes_ingestion(
        sources=[
            OutcomeSourceSpec(path=src, source_name="fixture_a"),
            OutcomeSourceSpec(path=missing, source_name="missing_source"),
        ],
        output_root=out_root,
        run_id="stable_run",
        overwrite=True,
    )
    second = run_public_outcomes_ingestion(
        sources=[
            OutcomeSourceSpec(path=src, source_name="fixture_a"),
            OutcomeSourceSpec(path=missing, source_name="missing_source"),
        ],
        output_root=out_root,
        run_id="stable_run",
        overwrite=True,
    )
    assert first["manifest_path"] == second["manifest_path"]
    manifest_1 = Path(first["manifest_path"]).read_text(encoding="utf-8")
    manifest_2 = Path(second["manifest_path"]).read_text(encoding="utf-8")
    assert manifest_1 == manifest_2
    manifest = json.loads(manifest_1)
    excluded = (manifest.get("summary") or {}).get("excluded_sources") or []
    assert excluded and excluded[0]["status"] == "not_found"


def test_directory_discovery_and_coverage_counts(tmp_path: Path) -> None:
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_a = src_dir / "event_a.csv"
    src_b = src_dir / "event_b.csv"
    _write_source_csv(
        src_a,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Destroyed,0.95",
            "2,evt-a,Event A,2021-08-01,101 Main St,Town,CA,90001,39.1002,-120.1002,No Damage,0.70",
        ],
    )
    _write_source_csv(
        src_b,
        [
            "3,evt-b,Event B,2022-09-04,200 Oak St,Town,CA,90002,38.2001,-121.2001,Major Damage,0.88",
        ],
    )
    discovered = _discover_input_files(input_dirs=[str(src_dir)], input_glob="**/*")
    assert {path.name for path in discovered} == {"event_a.csv", "event_b.csv"}

    result = run_public_outcomes_ingestion(
        sources=[OutcomeSourceSpec(path=path) for path in discovered],
        output_root=tmp_path / "out",
        run_id="discovery_run",
        overwrite=True,
    )
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest.get("summary") or {}
    assert summary.get("event_count") == 2
    assert summary.get("total_dataset_size") == 3
    assert (summary.get("records_per_event") or {}).get("evt-a") == 2
    assert (summary.get("records_per_event") or {}).get("evt-b") == 1
    assert summary.get("configured_source_count") == 2
    assert summary.get("included_source_count") == 2


def test_source_manifest_expands_multiple_sources(tmp_path: Path) -> None:
    src_a = tmp_path / "event_a.csv"
    src_b = tmp_path / "event_b.csv"
    _write_source_csv(
        src_a,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.1001,-120.1001,Destroyed,0.95",
        ],
    )
    _write_source_csv(
        src_b,
        [
            "3,evt-b,Event B,2022-09-04,200 Oak St,Town,CA,90002,38.2001,-121.2001,Major Damage,0.88",
        ],
    )
    manifest_path = tmp_path / "sources_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(src_a), "source_name": "a_feed"},
                    {"path": str(src_b), "source_name": "b_feed", "enabled": True},
                    {"path": str(tmp_path / "disabled.csv"), "enabled": False},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    specs = _load_source_specs_from_manifest(manifest_path, default_state="CA")
    assert len(specs) == 2
    assert {spec.source_name for spec in specs} == {"a_feed", "b_feed"}

    result = run_public_outcomes_ingestion(
        sources=specs,
        output_root=tmp_path / "out",
        run_id="manifest_run",
        overwrite=True,
    )
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest.get("summary") or {}
    assert summary.get("configured_source_count") == 2
    assert summary.get("included_source_count") == 2
    assert summary.get("event_count") == 2


def test_deduplication_does_not_collapse_distinct_addresses_same_coordinate(tmp_path: Path) -> None:
    src_a = tmp_path / "source_a.csv"
    src_b = tmp_path / "source_b.csv"
    _write_source_csv(
        src_a,
        [
            "1,evt-a,Event A,2021-08-01,100 Main St,Town,CA,90001,39.100100,-120.100100,Destroyed,0.95",
        ],
    )
    _write_source_csv(
        src_b,
        [
            "2,evt-a,Event A,2021-08-01,200 Main St,Town,CA,90001,39.100100,-120.100100,Major Damage,0.90",
        ],
    )
    result = run_public_outcomes_ingestion(
        sources=[
            OutcomeSourceSpec(path=src_a, source_name="fixture_a"),
            OutcomeSourceSpec(path=src_b, source_name="fixture_b"),
        ],
        output_root=tmp_path / "out",
        run_id="non_collapse_run",
        overwrite=True,
    )
    payload = json.loads((Path(result["run_dir"]) / "normalized_outcomes.json").read_text(encoding="utf-8"))
    rows = payload.get("records") or []
    assert len(rows) == 2
