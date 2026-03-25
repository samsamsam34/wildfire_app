from __future__ import annotations

import json
import os
from pathlib import Path

from backend.public_outcome_governance import (
    list_public_outcome_runs,
    resolve_baseline_run_id,
)


def _write_manifest(run_dir: Path, *, generated_at: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"generated_at": generated_at}, indent=2),
        encoding="utf-8",
    )


def test_list_public_outcome_runs_handles_missing_artifact_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_validation_root"
    result = list_public_outcome_runs(artifact_root=missing_root)
    assert result.get("available") is False
    assert result.get("run_directory_count") == 0
    assert result.get("latest_run_id") is None
    assert str(missing_root) in str(result.get("message") or "")


def test_list_public_outcome_runs_sorts_compact_and_iso_timestamps(tmp_path: Path) -> None:
    artifact_root = tmp_path / "validation_runs"
    artifact_root.mkdir(parents=True, exist_ok=True)

    run_iso = artifact_root / "run_iso"
    _write_manifest(run_iso, generated_at="2026-03-25T18:01:50Z")

    run_compact = artifact_root / "run_compact"
    _write_manifest(run_compact, generated_at="20260325T180151Z")

    run_mtime = artifact_root / "run_mtime_only"
    run_mtime.mkdir(parents=True, exist_ok=True)
    # Ensure fallback mtime is older than manifest-derived timestamps.
    os.utime(run_mtime, (1_700_000_000, 1_700_000_000))

    result = list_public_outcome_runs(artifact_root=artifact_root)
    runs = result.get("runs") if isinstance(result.get("runs"), list) else []
    run_ids = [str(row.get("run_id")) for row in runs if isinstance(row, dict)]

    assert result.get("available") is True
    assert result.get("latest_run_id") == "run_compact"
    assert run_ids[:3] == ["run_compact", "run_iso", "run_mtime_only"]
    assert all("sort_timestamp" not in row for row in runs if isinstance(row, dict))


def test_resolve_baseline_run_id_behavior() -> None:
    ordered = ["run_latest", "run_previous", "run_oldest"]
    assert (
        resolve_baseline_run_id(
            ordered_run_ids=ordered,
            current_run_id="run_latest",
            baseline_run_id=None,
        )
        == "run_previous"
    )
    assert (
        resolve_baseline_run_id(
            ordered_run_ids=ordered,
            current_run_id="run_latest",
            baseline_run_id="run_oldest",
        )
        == "run_oldest"
    )
    assert (
        resolve_baseline_run_id(
            ordered_run_ids=ordered,
            current_run_id="run_latest",
            baseline_run_id="run_latest",
        )
        is None
    )
