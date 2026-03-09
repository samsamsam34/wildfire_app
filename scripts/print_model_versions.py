#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.version import (
    BENCHMARK_PACK_VERSION,
    DEFAULT_RULESET_VERSION,
    build_model_governance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print current product/API/scoring governance versions."
    )
    parser.add_argument(
        "--ruleset-version",
        default=DEFAULT_RULESET_VERSION,
        help="Ruleset version to include in governance output.",
    )
    parser.add_argument("--region-data-version", default=None, help="Optional region data version override.")
    parser.add_argument("--data-bundle-version", default=None, help="Optional data bundle version override.")
    parser.add_argument(
        "--benchmark-pack-version",
        default=BENCHMARK_PACK_VERSION,
        help="Benchmark pack version override.",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    governance = build_model_governance(
        ruleset_version=args.ruleset_version,
        benchmark_pack_version=args.benchmark_pack_version,
        region_data_version=args.region_data_version,
        data_bundle_version=args.data_bundle_version,
    )
    if args.compact:
        print(json.dumps(governance, sort_keys=True))
    else:
        print(json.dumps(governance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
