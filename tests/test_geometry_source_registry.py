from __future__ import annotations

from pathlib import Path

from backend.geometry_source_registry import build_region_geometry_source_manifest
from backend.wildfire_data import WildfireDataClient


def test_geometry_source_manifest_applies_region_override_and_normalization() -> None:
    registry = {
        "version": 3,
        "defaults": {
            "source_order": {
                "parcel_sources": ["parcel_polygons", "nearest_parcel_fallback"],
                "footprint_sources": ["building_footprints", "fema_structures"],
            },
            "source_definitions": {
                "parcel_sources": {
                    "parcel_polygons": {
                        "display_name": "Prepared parcels",
                        "layer_keys": ["parcel_polygons"],
                    },
                    "nearest_parcel_fallback": {
                        "display_name": "Nearest fallback",
                        "layer_keys": [],
                        "fallback_only": True,
                        "explicit_downgrade": True,
                    },
                },
                "footprint_sources": {
                    "building_footprints": {
                        "display_name": "Prepared footprints",
                        "layer_keys": ["building_footprints"],
                    },
                    "fema_structures": {
                        "display_name": "FEMA backup",
                        "layer_keys": ["fema_structures"],
                    },
                },
            },
            "schema_normalization_rules": {
                "parcel_sources": {"parcel_id_fields": ["PARCEL_ID", "parcelid"]},
                "footprint_sources": {"structure_id_fields": ["OBJECTID", "building_id"]},
            },
            "confidence_weights": {
                "parcel_sources": {"parcel_polygons": 0.91, "nearest_parcel_fallback": 0.25},
                "footprint_sources": {"building_footprints": 0.83, "fema_structures": 0.58},
            },
            "known_limitations": ["Default limitation."],
        },
        "regions": {
            "region_a": {
                "source_order": {
                    "parcel_sources": ["parcel_polygons_override", "parcel_polygons", "nearest_parcel_fallback"],
                },
                "source_definitions": {
                    "parcel_sources": {
                        "parcel_polygons_override": {
                            "display_name": "Region parcel override",
                            "layer_keys": ["parcel_polygons_override"],
                        }
                    }
                },
                "schema_normalization_rules": {
                    "parcel_sources": {"parcel_id_fields": ["APN_OVERRIDE", "PARCEL_ID"]},
                },
                "confidence_weights": {
                    "parcel_sources": {"parcel_polygons_override": 0.97},
                },
            }
        },
    }
    files = {
        "building_footprints": "building_footprints.geojson",
        "parcel_polygons": "parcel_polygons.geojson",
        "parcel_polygons_override": "parcel_polygons_override.geojson",
    }
    layers_meta = {
        "building_footprints": {"dataset_version": "footprint-v2"},
        "parcel_polygons": {"dataset_version": "parcel-v1"},
        "parcel_polygons_override": {"dataset_version": "parcel-override-v3"},
    }

    manifest = build_region_geometry_source_manifest(
        region_id="region_a",
        files=files,
        layers_meta=layers_meta,
        registry=registry,
    )

    assert manifest["version"] == 3
    assert manifest["region_id"] == "region_a"
    assert manifest["default_source_order"]["parcel_sources"][0] == "parcel_polygons_override"
    assert manifest["schema_normalization_rules"]["parcel_sources"]["parcel_id_fields"][0] == "APN_OVERRIDE"

    first_parcel = manifest["parcel_sources"][0]
    assert first_parcel["source_id"] == "parcel_polygons_override"
    assert first_parcel["available"] is True
    assert first_parcel["selected_layer_key"] == "parcel_polygons_override"
    assert first_parcel["confidence_weight"] == 0.97

    assert manifest["source_versions"]["parcel_polygons_override"] == "parcel-override-v3"
    assert "nearest_parcel_fallback" in [entry["source_id"] for entry in manifest["parcel_sources"]]
    assert any("explicit confidence downgrade" in item.lower() for item in manifest["known_limitations"])


def test_runtime_uses_geometry_source_manifest_precedence_for_buildings_and_parcels() -> None:
    client = WildfireDataClient()
    runtime_paths = {
        "footprints": "/tmp/prepared_local.geojson",
        "footprints_overture": "/tmp/overture.geojson",
        "footprints_microsoft": "/tmp/microsoft.geojson",
        "fema_structures": "/tmp/fema.geojson",
        "parcels": "/tmp/parcels.geojson",
        "parcels_override": "/tmp/parcels_override.geojson",
    }
    region_context = {
        "geometry_source_manifest": {
            "default_source_order": {
                "footprint_sources": [
                    "building_footprints_microsoft",
                    "building_footprints",
                    "building_footprints_overture",
                    "fema_structures",
                ],
                "parcel_sources": [
                    "parcel_polygons_override",
                    "parcel_polygons",
                    "nearest_parcel_fallback",
                ],
            }
        }
    }

    building_paths = client._resolve_building_source_paths(runtime_paths, region_context)
    assert building_paths[:4] == [
        "/tmp/microsoft.geojson",
        "/tmp/prepared_local.geojson",
        "/tmp/overture.geojson",
        "/tmp/fema.geojson",
    ]

    parcel_paths = client._resolve_parcel_source_paths(runtime_paths, region_context=region_context)
    assert parcel_paths[:2] == [
        str(Path("/tmp/parcels_override.geojson").resolve()),
        str(Path("/tmp/parcels.geojson").resolve()),
    ]
