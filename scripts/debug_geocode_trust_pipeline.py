#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import _build_geocode_debug_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run address inputs through the geocode/trust/region pipeline and print structured diagnostics."
    )
    parser.add_argument(
        "--address",
        action="append",
        dest="addresses",
        help="Address to evaluate (repeat flag for multiple addresses).",
    )
    parser.add_argument(
        "--addresses-file",
        type=Path,
        help="Optional text file with one address per line.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def load_addresses(args: argparse.Namespace) -> list[str]:
    entries = list(args.addresses or [])
    if args.addresses_file:
        if not args.addresses_file.exists():
            raise FileNotFoundError(f"Address file not found: {args.addresses_file}")
        entries.extend(
            line.strip()
            for line in args.addresses_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = " ".join(entry.split())
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def main() -> int:
    args = parse_args()
    addresses = load_addresses(args)
    if not addresses:
        print("Provide at least one --address or --addresses-file.", file=sys.stderr)
        return 2

    # This harness is intended for local debugging.
    os.environ.setdefault("WF_ENV", "development")
    os.environ.setdefault("WF_DEBUG_MODE", "1")

    for idx, address in enumerate(addresses, start=1):
        payload = _build_geocode_debug_payload(address)
        compact = {
            "address": address,
            "geocode_status": payload.get("geocode_status"),
            "geocode_outcome": payload.get("geocode_outcome"),
            "trust_status": (payload.get("trust") or {}).get("trusted_match_status"),
            "trust_failure_reason": (payload.get("trust") or {}).get("trusted_match_failure_reason"),
            "resolved_latitude": payload.get("resolved_latitude"),
            "resolved_longitude": payload.get("resolved_longitude"),
            "selected_region_id": payload.get("selected_region_id")
            or ((payload.get("region_resolution") or {}).get("resolved_region_id")),
            "coverage_available": ((payload.get("region_resolution") or {}).get("coverage_available")),
            "region_reason": ((payload.get("region_resolution") or {}).get("reason")),
            "candidate_count": payload.get("match_count"),
            "candidate_summaries": payload.get("candidate_summaries"),
        }
        if args.pretty:
            print(f"# Address {idx}")
            print(json.dumps(compact, indent=2, sort_keys=True))
        else:
            print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
