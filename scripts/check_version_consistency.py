#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.version import (
    API_VERSION,
    BENCHMARK_PACK_VERSION,
    CALIBRATION_VERSION,
    FACTOR_SCHEMA_VERSION,
    PRODUCT_VERSION,
    RULESET_LOGIC_VERSION,
    SCORING_MODEL_VERSION,
)

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate version/governance metadata consistency across the repo."
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to changelog markdown file.",
    )
    parser.add_argument(
        "--benchmark-pack",
        default="benchmark/scenario_pack_v1.json",
        help="Path to benchmark scenario pack JSON.",
    )
    return parser.parse_args()


def _is_semver(value: str) -> bool:
    return bool(SEMVER_RE.match(value))


def _check_changelog(path: Path, product_version: str) -> str | None:
    if not path.exists():
        return f"Missing changelog file: {path}"
    text = path.read_text(encoding="utf-8")
    if f"[{product_version}]" not in text:
        return f"CHANGELOG does not include current product version heading [{product_version}]"
    return None


def _check_benchmark_pack(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return [f"Missing benchmark pack: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Unable to parse benchmark pack JSON: {exc}"]

    pack_version = str(payload.get("benchmark_pack_version") or "")
    pack_factor_schema = str(payload.get("factor_schema_version") or "")
    if pack_version != BENCHMARK_PACK_VERSION:
        failures.append(
            f"benchmark_pack_version mismatch: file={pack_version} code={BENCHMARK_PACK_VERSION}"
        )
    if pack_factor_schema != FACTOR_SCHEMA_VERSION:
        failures.append(
            f"factor_schema_version mismatch in benchmark pack: file={pack_factor_schema} code={FACTOR_SCHEMA_VERSION}"
        )
    return failures


def main() -> int:
    args = parse_args()

    failures: list[str] = []
    checks = {
        "PRODUCT_VERSION": PRODUCT_VERSION,
        "API_VERSION": API_VERSION,
        "SCORING_MODEL_VERSION": SCORING_MODEL_VERSION,
        "RULESET_LOGIC_VERSION": RULESET_LOGIC_VERSION,
        "FACTOR_SCHEMA_VERSION": FACTOR_SCHEMA_VERSION,
        "BENCHMARK_PACK_VERSION": BENCHMARK_PACK_VERSION,
        "CALIBRATION_VERSION": CALIBRATION_VERSION,
    }
    for key, value in checks.items():
        if not _is_semver(value):
            failures.append(f"{key} must be semantic version (x.y.z), got: {value}")

    changelog_failure = _check_changelog(Path(args.changelog), PRODUCT_VERSION)
    if changelog_failure:
        failures.append(changelog_failure)

    failures.extend(_check_benchmark_pack(Path(args.benchmark_pack)))

    if failures:
        print("Version consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Version consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
