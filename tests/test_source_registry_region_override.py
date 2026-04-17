from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_region_from_catalog_or_sources import _load_source_config


def test_load_source_config_applies_region_layer_override(tmp_path: Path) -> None:
    source_registry_path = tmp_path / "source_registry.json"
    payload = {
        "layers": {
            "parcel_polygons": {
                "provider_type": "arcgis_feature_service",
                "source_endpoint": "https://example.com/default/parcels",
            }
        },
        "regions": {
            "missoula_pilot": {
                "layers": {
                    "parcel_polygons": {
                        "source_endpoint": "https://example.com/missoula/parcels",
                    }
                }
            }
        },
    }
    source_registry_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg_default, meta_default = _load_source_config(str(source_registry_path), region_id=None)
    cfg_missoula, meta_missoula = _load_source_config(str(source_registry_path), region_id="missoula_pilot")

    assert cfg_default["parcel_polygons"]["source_endpoint"] == "https://example.com/default/parcels"
    assert cfg_missoula["parcel_polygons"]["source_endpoint"] == "https://example.com/missoula/parcels"
    assert meta_default["region_override_applied"] is False
    assert meta_missoula["region_override_applied"] is True
