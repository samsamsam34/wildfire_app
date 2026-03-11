from __future__ import annotations

import json
from pathlib import Path

from backend.address_resolution import (
    normalize_address_for_matching,
    resolve_local_address_candidate,
    validate_local_fallback_records,
)


def test_normalize_address_for_matching_handles_common_variants() -> None:
    assert normalize_address_for_matching("6 Pineview Road, Winthrop, Washington 98862") == (
        "6 pineview rd winthrop wa 98862"
    )
    assert normalize_address_for_matching("6   Pineview Rd., Winthrop, WA 98862") == (
        "6 pineview rd winthrop wa 98862"
    )


def test_resolve_local_address_candidate_matches_alias_file(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(
        json.dumps(
            {
                "addresses": [
                    {
                        "address": "6 Pineview Rd, Winthrop, WA 98862",
                        "latitude": 48.4772,
                        "longitude": -120.1864,
                        "region_id": "winthrop_pilot",
                        "source_name": "test_alias",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = resolve_local_address_candidate(
        address="6 Pineview Road, Winthrop, Washington 98862",
        regions_root=str(tmp_path / "regions"),
        alias_path=str(alias_path),
    )
    assert result["matched"] is True
    assert result["best_match"]["region_id"] == "winthrop_pilot"
    assert result["best_match"]["latitude"] == 48.4772
    assert result["best_match"]["longitude"] == -120.1864


def test_resolve_local_address_candidate_reports_no_match(tmp_path: Path) -> None:
    result = resolve_local_address_candidate(
        address="999 Totally Invalid Route, Unknown, ZZ 00000",
        regions_root=str(tmp_path / "regions"),
        alias_path=str(tmp_path / "missing_aliases.json"),
    )
    assert result["matched"] is False
    assert result["candidate_count"] == 0
    assert result["best_match"] is None


def test_resolve_local_address_candidate_rejects_street_only_fallback(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(
        json.dumps(
            {
                "addresses": [
                    {
                        "address": "Pineview Rd, Winthrop, WA 98862",
                        "latitude": 48.4772,
                        "longitude": -120.1864,
                        "region_id": "winthrop_pilot",
                        "source_name": "street_only_alias",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = resolve_local_address_candidate(
        address="6 Pineview Rd, Winthrop, WA 98862",
        regions_root=str(tmp_path / "regions"),
        alias_path=str(alias_path),
        include_authoritative_sources=False,
        include_alias_sources=True,
    )
    assert result["matched"] is False
    assert result["candidate_count"] == 0
    assert result["best_match"] is None
    assert any("street/locality-only" in note for note in result["diagnostics"])


def test_validate_local_fallback_records_flags_conflicting_duplicates(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(
        json.dumps(
            {
                "addresses": [
                    {
                        "address": "6 Pineview Rd, Winthrop, WA 98862",
                        "latitude": 48.4772,
                        "longitude": -120.1864,
                    },
                    {
                        "address": "6 Pineview Road, Winthrop, Washington 98862",
                        "latitude": 48.4800,
                        "longitude": -120.1500,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = validate_local_fallback_records(alias_path)
    assert report["valid"] is False
    assert report["errors"]
