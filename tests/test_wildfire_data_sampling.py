from __future__ import annotations

from types import SimpleNamespace

from backend.wildfire_data import WildfireDataClient


def test_sample_layer_value_detailed_uses_nearby_sample_when_point_is_nodata(monkeypatch):
    client = WildfireDataClient()

    class DummyDataset:
        nodata = -9999.0
        bounds = SimpleNamespace(left=0.0, right=10.0, bottom=0.0, top=10.0)

        def sample(self, _coords):
            yield [-9999.0]

    monkeypatch.setattr(client, "_file_exists", lambda _path: True)
    monkeypatch.setattr(client, "_open_raster", lambda _path: DummyDataset())
    monkeypatch.setattr(client, "_to_dataset_crs", lambda _ds, _lon, _lat: (5.0, 5.0))
    monkeypatch.setattr(client, "_sample_raster_nearby", lambda _path, _lat, _lon: (42.0, 60.0))

    value, status, reason = client._sample_layer_value_detailed("dummy.tif", 46.87, -113.99)

    assert value == 42.0
    assert status == "ok_nearby"
    assert reason is not None and "nearest valid sample" in reason.lower()


def test_sample_layer_value_treats_nearby_status_as_observed(monkeypatch):
    client = WildfireDataClient()
    monkeypatch.setattr(
        client,
        "_sample_layer_value_detailed",
        lambda _path, _lat, _lon: (17.5, "ok_nearby", "sampled nearby"),
    )

    value, status = client._sample_layer_value("dummy.tif", 46.87, -113.99)

    assert value == 17.5
    assert status == "ok"


def test_naip_enrichment_can_match_nearest_centroid_when_exact_key_missing(monkeypatch):
    client = WildfireDataClient()
    monkeypatch.setattr(client, "_resolve_naip_feature_artifact_path", lambda **_kwargs: "/tmp/naip_structure_features.json")
    monkeypatch.setattr(
        "backend.wildfire_data.load_naip_feature_artifact",
        lambda _path: {
            "features_by_key": {
                "centroid:46.872300,-113.994100": {
                    "ring_metrics": {
                        "ring_0_5_ft": {"vegetation_density_proxy": 40.0},
                    },
                    "near_structure_vegetation_0_5_pct": 47.0,
                    "canopy_adjacency_proxy_pct": 33.0,
                    "vegetation_continuity_proxy_pct": 29.0,
                    "nearest_high_fuel_patch_distance_ft": 24.0,
                },
                "centroid:46.900000,-114.100000": {
                    "ring_metrics": {
                        "ring_0_5_ft": {"vegetation_density_proxy": 90.0},
                    },
                    "near_structure_vegetation_0_5_pct": 88.0,
                },
            },
            "keys_by_structure_id": {},
            "quantiles": {},
        },
    )

    ring_context = {
        "matched_structure_id": "missing-id",
        "footprint_centroid": {"latitude": 46.8722, "longitude": -113.9940},
        "ring_metrics": {"ring_0_5_ft": {"vegetation_density": 38.0}},
    }
    enriched, assumptions, sources = client._apply_naip_feature_enrichment(
        ring_context=ring_context,
        runtime_paths={},
        region_context={},
    )

    assert enriched["naip_feature_match_method"] == "nearest_centroid"
    assert float(enriched["naip_feature_match_distance_m"]) < 30.0
    assert enriched["near_structure_vegetation_0_5_pct"] == 47.0
    assert enriched["ring_metrics"]["ring_0_5_ft"]["basis"] == "footprint_naip_blended"
    assert any("nearest centroid" in note.lower() for note in assumptions)
    assert any("naip imagery-derived ring features" in src.lower() for src in sources)


def test_naip_near_structure_features_differ_for_nearby_homes_with_different_patterns(monkeypatch):
    client = WildfireDataClient()
    monkeypatch.setattr(client, "_resolve_naip_feature_artifact_path", lambda **_kwargs: "/tmp/naip_structure_features.json")
    monkeypatch.setattr(
        "backend.wildfire_data.load_naip_feature_artifact",
        lambda _path: {
            "features_by_key": {
                "centroid:46.872300,-113.994100": {
                    "ring_metrics": {
                        "ring_0_5_ft": {
                            "vegetation_density_proxy": 85.0,
                            "canopy_proxy_pct": 76.0,
                            "impervious_low_fuel_pct": 10.0,
                        },
                        "ring_5_30_ft": {
                            "vegetation_density_proxy": 78.0,
                        },
                    },
                    "near_structure_vegetation_0_5_pct": 87.0,
                    "near_structure_vegetation_5_30_pct": 79.0,
                    "canopy_adjacency_proxy_pct": 74.0,
                },
                "centroid:46.872320,-113.994120": {
                    "ring_metrics": {
                        "ring_0_5_ft": {
                            "vegetation_density_proxy": 18.0,
                            "canopy_proxy_pct": 12.0,
                            "impervious_low_fuel_pct": 72.0,
                        },
                        "ring_5_30_ft": {
                            "vegetation_density_proxy": 22.0,
                        },
                    },
                    "near_structure_vegetation_0_5_pct": 20.0,
                    "near_structure_vegetation_5_30_pct": 24.0,
                    "canopy_adjacency_proxy_pct": 14.0,
                },
            },
            "keys_by_structure_id": {},
            "quantiles": {},
        },
    )

    dense_context = {
        "matched_structure_id": "dense-home",
        "footprint_used": True,
        "ring_metrics": {
            "geometry_type": "footprint",
            "precision_flag": "footprint_relative",
            "ring_0_5_ft": {"vegetation_density": 70.0},
            "ring_5_30_ft": {"vegetation_density": 65.0},
        },
        "footprint_centroid": {"latitude": 46.872300, "longitude": -113.994100},
    }
    clear_context = {
        "matched_structure_id": "clear-home",
        "footprint_used": True,
        "ring_metrics": {
            "geometry_type": "footprint",
            "precision_flag": "footprint_relative",
            "ring_0_5_ft": {"vegetation_density": 28.0},
            "ring_5_30_ft": {"vegetation_density": 30.0},
        },
        "footprint_centroid": {"latitude": 46.872320, "longitude": -113.994120},
    }

    dense_enriched, _dense_assumptions, _dense_sources = client._apply_naip_feature_enrichment(
        ring_context=dense_context,
        runtime_paths={},
        region_context={},
    )
    clear_enriched, _clear_assumptions, _clear_sources = client._apply_naip_feature_enrichment(
        ring_context=clear_context,
        runtime_paths={},
        region_context={},
    )

    dense_features = dense_enriched.get("near_structure_features") or {}
    clear_features = clear_enriched.get("near_structure_features") or {}

    assert dense_features["veg_density_0_5"] > clear_features["veg_density_0_5"]
    assert dense_features["veg_density_5_30"] > clear_features["veg_density_5_30"]
    assert dense_features["canopy_overlap"] > clear_features["canopy_overlap"]
    assert dense_features["hardscape_ratio"] < clear_features["hardscape_ratio"]
    assert dense_features["confidence_flag"] in {"high", "moderate"}
    assert clear_features["confidence_flag"] in {"high", "moderate"}
