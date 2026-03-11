from __future__ import annotations

from pathlib import Path

from scripts.ingest_public_structure_damage import normalize_public_damage_rows


def test_normalize_public_damage_rows_csv(tmp_path: Path):
    raw_path = tmp_path / "dins_sample.csv"
    raw_path.write_text(
        "\n".join(
            [
                "record_id,incident_name,inspection_date,address,city,state,zip,latitude,longitude,damage_state",
                "1,Sample Fire,2021-08-01,100 Main St,Winthrop,CA,90001,39.10,-120.12,Destroyed",
                "2,Sample Fire,2021-08-01,101 Main St,Winthrop,CA,90001,39.11,-120.13,Major Damage",
                "3,Sample Fire,2021-08-01,102 Main St,Winthrop,CA,90001,39.12,-120.14,No Damage",
                "4,Sample Fire,2021-08-01,103 Main St,Winthrop,CA,90001,,,Minor Damage",
            ]
        ),
        encoding="utf-8",
    )

    payload = normalize_public_damage_rows(input_path=raw_path, source_name="calfire_dins")
    assert payload["record_count"] == 3
    assert payload["dropped_missing_or_invalid_coordinates"] == 1

    labels = [row["damage_label"] for row in payload["records"]]
    assert labels == ["destroyed", "major_damage", "no_damage"]
    binaries = [row["structure_loss_or_major_damage"] for row in payload["records"]]
    assert binaries == [1, 1, 0]
