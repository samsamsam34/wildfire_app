#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import _build_geocode_debug_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-debug the shared address-resolution pipeline.")
    parser.add_argument("--address", action="append", dest="addresses", help="Address to resolve. Repeat flag for many.")
    parser.add_argument("--addresses-file", type=Path, help="Text file with one address per line.")
    parser.add_argument("--csv", type=Path, help="CSV file containing an `address` column (or first column fallback).")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print each result.")
    return parser.parse_args()


def load_addresses(args: argparse.Namespace) -> list[str]:
    rows = list(args.addresses or [])
    if args.addresses_file:
        lines = args.addresses_file.read_text(encoding="utf-8").splitlines()
        rows.extend(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))
    if args.csv:
        with args.csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "address" in [name.strip().lower() for name in reader.fieldnames]:
                address_key = next(name for name in reader.fieldnames if name.strip().lower() == "address")
                rows.extend(str(row.get(address_key) or "").strip() for row in reader)
            else:
                handle.seek(0)
                raw_reader = csv.reader(handle)
                for csv_row in raw_reader:
                    if csv_row:
                        rows.append(str(csv_row[0]).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = " ".join(row.split())
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _compact(payload: dict[str, object], address: str) -> dict[str, object]:
    region = payload.get("region_resolution") or {}
    resolver_candidates = list(payload.get("resolver_candidates") or [])
    score_margin = None
    if len(resolver_candidates) >= 2:
        try:
            score_margin = round(
                float(resolver_candidates[0].get("rank_score") or 0.0)
                - float(resolver_candidates[1].get("rank_score") or 0.0),
                4,
            )
        except Exception:
            score_margin = None
    return {
        "address": address,
        "normalized_address": payload.get("normalized_address"),
        "error_class": payload.get("error_class"),
        "final_status": payload.get("final_status"),
        "address_exists": payload.get("address_exists"),
        "address_confidence": payload.get("address_confidence"),
        "address_validation_sources": payload.get("address_validation_sources"),
        "geocode_status": payload.get("geocode_status"),
        "geocode_outcome": payload.get("geocode_outcome"),
        "resolution_status": payload.get("resolution_status"),
        "resolution_method": payload.get("resolution_method"),
        "fallback_used": payload.get("fallback_used"),
        "final_location_confidence": payload.get("final_location_confidence"),
        "coordinate_confidence": payload.get("coordinate_confidence"),
        "resolved_latitude": payload.get("resolved_latitude"),
        "resolved_longitude": payload.get("resolved_longitude"),
        "selected_region_id": payload.get("selected_region_id") or region.get("resolved_region_id"),
        "coverage_available": region.get("coverage_available"),
        "region_error_class": region.get("error_class"),
        "region_reason": region.get("reason"),
        "coordinate_source": payload.get("coordinate_source"),
        "final_coordinate_source": payload.get("final_coordinate_source"),
        "match_confidence": payload.get("match_confidence"),
        "match_method": payload.get("match_method"),
        "score_margin_over_second_candidate": score_margin,
        "final_candidate_selected": payload.get("final_candidate_selected"),
        "final_acceptance_decision": payload.get("final_acceptance_decision"),
        "failure_reason": payload.get("failure_reason"),
        "map_confirmation_required": bool(payload.get("needs_user_confirmation")),
        "needs_user_confirmation": payload.get("needs_user_confirmation"),
        "candidate_sources_attempted": payload.get("candidate_sources_attempted"),
        "candidates_found": payload.get("candidates_found"),
        "provider_statuses": payload.get("provider_statuses"),
        "resolver_candidates": payload.get("resolver_candidates"),
        "candidate_disagreement_distances": payload.get("candidate_disagreement_distances"),
    }


def main() -> int:
    args = parse_args()
    addresses = load_addresses(args)
    if not addresses:
        print("Provide at least one --address or --addresses-file", file=sys.stderr)
        return 2

    os.environ.setdefault("WF_ENV", "development")
    os.environ.setdefault("WF_DEBUG_MODE", "1")

    for idx, address in enumerate(addresses, start=1):
        payload = _build_geocode_debug_payload(address)
        compact = _compact(payload, address)
        if args.pretty:
            print(f"# Address {idx}")
            print(json.dumps(compact, indent=2, sort_keys=True))
        else:
            print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
