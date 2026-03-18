from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_NO_GROUND_TRUTH_FIXTURE_PATH = (
    repository_root() / "benchmark" / "fixtures" / "no_ground_truth" / "scenario_pack_v1.json"
)

DEFAULT_NO_GROUND_TRUTH_ARTIFACT_ROOT = (
    repository_root() / "benchmark" / "no_ground_truth_evaluation"
)


def resolve_no_ground_truth_artifact_root(path_hint: str | Path | None = None) -> Path:
    hint = str(path_hint or os.getenv("WF_NO_GROUND_TRUTH_EVAL_DIR") or "").strip()
    if hint:
        return Path(hint).expanduser()
    return DEFAULT_NO_GROUND_TRUTH_ARTIFACT_ROOT

