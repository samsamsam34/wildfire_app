from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.building_footprints import BuildingFootprintResult
from backend import wildfire_data as wildfire_data_module
from backend.wildfire_data import WildfireDataClient

try:
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover - geospatial deps may be unavailable in minimal test envs
    Polygon = None


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


def test_structure_relative_slope_differs_for_nearby_homes_with_micro_topography(monkeypatch):
    client = WildfireDataClient()

    class StubFootprints:
        def get_building_footprint(self, _lat, _lon, **_kwargs):
            return BuildingFootprintResult(
                found=False,
                source="stub",
                confidence=0.0,
                assumptions=["no footprint match in fixture"],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return None

    client.footprints = StubFootprints()

    def _point_proxy_ring_metrics(**_kwargs):
        return {
            "ring_0_5_ft": {"vegetation_density": 42.0},
            "ring_5_30_ft": {"vegetation_density": 48.0},
            "ring_30_100_ft": {"vegetation_density": 52.0},
            "ring_100_300_ft": {"vegetation_density": 36.0},
            "geometry_type": "point",
            "precision_flag": "fallback_point_proxy",
        }

    def _structure_features(*, origin_lat: float, **_kwargs):
        if origin_lat < 46.872315:
            sector_slopes = {"north": 12.0, "east": 19.0, "south": 10.0, "west": 14.0}
        else:
            sector_slopes = {"north": 29.0, "east": 36.0, "south": 24.0, "west": 31.0}
        return {
            "near_structure_vegetation_0_5_pct": 44.0,
            "near_structure_vegetation_5_30_pct": 47.0,
            "vegetation_edge_directional_concentration_pct": 18.0,
            "canopy_dense_fuel_asymmetry_pct": 10.0,
            "nearest_continuous_vegetation_distance_ft": 22.0,
            "vegetation_directional_sectors": {
                direction: {"slope_deg": value}
                for direction, value in sector_slopes.items()
            },
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.35,
            "vegetation_directional_basis": "point_proxy_relative",
            "directional_risk": {},
        }

    def _sample_raster_point(_path: str, lat: float, _lon: float):
        return 11.0 if lat < 46.872315 else 27.0

    def _sample_circle(_path: str, lat: float, _lon: float, **_kwargs):
        base = 12.0 if lat < 46.872315 else 26.0
        return [base, base + 2.0, base + 4.0]

    monkeypatch.setattr(client, "_build_point_proxy_ring_metrics", _point_proxy_ring_metrics)
    monkeypatch.setattr(client, "_compute_structure_aware_vegetation_features", _structure_features)
    monkeypatch.setattr(client, "_sample_raster_point", _sample_raster_point)
    monkeypatch.setattr(client, "_sample_circle", _sample_circle)

    lower_home, _assumptions_a, _sources_a = client._compute_structure_ring_metrics(
        46.872300,
        -113.994100,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )
    steeper_home, _assumptions_b, _sources_b = client._compute_structure_ring_metrics(
        46.872330,
        -113.994100,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )

    slope_a = lower_home.get("structure_relative_slope") or {}
    slope_b = steeper_home.get("structure_relative_slope") or {}

    assert slope_a.get("precision_flag") == "fallback_point_proxy"
    assert slope_b.get("precision_flag") == "fallback_point_proxy"
    assert slope_a.get("local_slope") != slope_b.get("local_slope")
    assert slope_a.get("slope_within_30_ft") != slope_b.get("slope_within_30_ft")
    assert slope_a.get("uphill_exposure") != slope_b.get("uphill_exposure")
    assert slope_a.get("downhill_buffer") != slope_b.get("downhill_buffer")


@pytest.mark.skipif(Polygon is None, reason="Structure attribute inference test requires shapely")
def test_structure_attributes_differ_for_distinct_footprints(monkeypatch):
    client = WildfireDataClient()
    compact = Polygon(
        [
            (-113.994120, 46.872290),
            (-113.994080, 46.872290),
            (-113.994080, 46.872320),
            (-113.994120, 46.872320),
        ]
    )
    irregular = Polygon(
        [
            (-113.994210, 46.872300),
            (-113.994120, 46.872300),
            (-113.994120, 46.872330),
            (-113.994170, 46.872330),
            (-113.994170, 46.872360),
            (-113.994210, 46.872360),
        ]
    )

    class StubFootprints:
        def get_building_footprint(self, lat, _lon, **_kwargs):
            geom = compact if lat < 46.87233 else irregular
            centroid = geom.centroid
            return BuildingFootprintResult(
                found=True,
                footprint=geom,
                centroid=(float(centroid.y), float(centroid.x)),
                source="stub.geojson",
                confidence=0.91,
                match_status="matched",
                match_method="point_in_footprint",
                matched_structure_id=f"home-{1 if geom is compact else 2}",
                match_distance_m=0.0,
                candidate_count=1,
                candidate_summaries=[],
                assumptions=[],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return {
                "nearby_structure_count_100_ft": 2,
                "nearby_structure_count_300_ft": 7,
                "nearest_structure_distance_ft": 18.0,
            }

    client.footprints = StubFootprints()
    monkeypatch.setattr(
        client,
        "_summarize_ring_canopy",
        lambda *_args, **_kwargs: {"canopy_mean": 35.0, "canopy_max": 42.0, "vegetation_density": 38.0, "coverage_pct": 36.0},
    )
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", lambda *_args, **_kwargs: 34.0)
    monkeypatch.setattr(
        client,
        "_compute_structure_aware_vegetation_features",
        lambda **_kwargs: {
            "near_structure_vegetation_0_5_pct": 40.0,
            "near_structure_vegetation_5_30_pct": 44.0,
            "vegetation_edge_directional_concentration_pct": 19.0,
            "canopy_dense_fuel_asymmetry_pct": 11.0,
            "nearest_continuous_vegetation_distance_ft": 20.0,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "footprint_relative",
            "vegetation_directional_precision_score": 0.82,
            "vegetation_directional_basis": "footprint_relative",
            "directional_risk": {},
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_relative_slope",
        lambda **_kwargs: {
            "local_slope": 14.0,
            "uphill_exposure": 24.0,
            "downhill_buffer": 76.0,
            "precision_flag": "footprint_relative",
        },
    )

    compact_ctx, _a, _b = client._compute_structure_ring_metrics(
        46.872300,
        -113.994100,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )
    irregular_ctx, _c, _d = client._compute_structure_ring_metrics(
        46.872360,
        -113.994160,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )

    compact_attrs = compact_ctx.get("structure_attributes") or {}
    irregular_attrs = irregular_ctx.get("structure_attributes") or {}
    assert compact_attrs.get("area", {}).get("sqft") != irregular_attrs.get("area", {}).get("sqft")
    assert compact_attrs.get("shape_complexity", {}).get("index") != irregular_attrs.get("shape_complexity", {}).get("index")
    assert compact_attrs.get("provenance", {}).get("area") == "observed"
    assert irregular_attrs.get("provenance", {}).get("shape_complexity") == "inferred"


def test_structure_attributes_fallback_when_footprint_missing(monkeypatch):
    client = WildfireDataClient()

    class StubFootprints:
        def get_building_footprint(self, _lat, _lon, **_kwargs):
            return BuildingFootprintResult(
                found=False,
                source="stub.geojson",
                confidence=0.0,
                match_status="none",
                assumptions=["no footprint match in fixture"],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return {
                "nearby_structure_count_100_ft": 1,
                "nearby_structure_count_300_ft": 5,
                "nearest_structure_distance_ft": 42.0,
            }

    client.footprints = StubFootprints()
    monkeypatch.setattr(
        client,
        "_build_point_proxy_ring_metrics",
        lambda **_kwargs: {
            "ring_0_5_ft": {"vegetation_density": 32.0},
            "ring_5_30_ft": {"vegetation_density": 46.0},
            "ring_30_100_ft": {"vegetation_density": 41.0},
            "ring_100_300_ft": {"vegetation_density": 30.0},
            "geometry_type": "point",
            "precision_flag": "fallback_point_proxy",
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_aware_vegetation_features",
        lambda **_kwargs: {
            "near_structure_vegetation_0_5_pct": 32.0,
            "near_structure_vegetation_5_30_pct": 46.0,
            "vegetation_edge_directional_concentration_pct": 14.0,
            "canopy_dense_fuel_asymmetry_pct": 8.0,
            "nearest_continuous_vegetation_distance_ft": 26.0,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.31,
            "vegetation_directional_basis": "point_proxy_relative",
            "directional_risk": {},
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_relative_slope",
        lambda **_kwargs: {
            "local_slope": 9.0,
            "precision_flag": "fallback_point_proxy",
            "source": "sampled",
        },
    )

    context, _assumptions, _sources = client._compute_structure_ring_metrics(
        46.872300,
        -113.994100,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
    )

    attrs = context.get("structure_attributes") or {}
    assert context.get("footprint_used") is False
    assert attrs.get("area", {}).get("sqft") is None
    assert attrs.get("shape_complexity", {}).get("index") is None
    assert attrs.get("density_context", {}).get("index") is not None
    assert attrs.get("estimated_age_proxy", {}).get("proxy_year") is not None
    assert attrs.get("provenance", {}).get("area") == "unavailable"
    assert attrs.get("provenance", {}).get("density_context") == "inferred"


def test_structure_attributes_public_record_enrichment_marks_observed_provenance(monkeypatch):
    client = WildfireDataClient()

    class StubFootprints:
        def get_building_footprint(self, _lat, _lon, **_kwargs):
            return BuildingFootprintResult(
                found=False,
                source="stub.geojson",
                confidence=0.0,
                match_status="none",
                assumptions=["no footprint match in fixture"],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return {
                "nearby_structure_count_100_ft": 1,
                "nearby_structure_count_300_ft": 4,
                "nearest_structure_distance_ft": 36.0,
            }

    client.footprints = StubFootprints()
    monkeypatch.setattr(
        client,
        "_build_point_proxy_ring_metrics",
        lambda **_kwargs: {
            "ring_0_5_ft": {"vegetation_density": 30.0},
            "ring_5_30_ft": {"vegetation_density": 45.0},
            "ring_30_100_ft": {"vegetation_density": 40.0},
            "ring_100_300_ft": {"vegetation_density": 28.0},
            "geometry_type": "point",
            "precision_flag": "fallback_point_proxy",
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_aware_vegetation_features",
        lambda **_kwargs: {
            "near_structure_vegetation_0_5_pct": 30.0,
            "near_structure_vegetation_5_30_pct": 45.0,
            "vegetation_edge_directional_concentration_pct": 12.0,
            "canopy_dense_fuel_asymmetry_pct": 7.0,
            "nearest_continuous_vegetation_distance_ft": 24.0,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.34,
            "vegetation_directional_basis": "point_proxy_relative",
            "directional_risk": {},
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_relative_slope",
        lambda **_kwargs: {
            "local_slope": 8.0,
            "precision_flag": "fallback_point_proxy",
            "source": "sampled",
        },
    )

    context, _assumptions, _sources = client._compute_structure_ring_metrics(
        46.872300,
        -113.994100,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
        parcel_properties={
            "year_built": 1992,
            "gross_building_area": 2260,
            "land_use": "single_family_residential",
            "roof_material": "tile",
        },
    )

    attrs = context.get("structure_attributes") or {}
    assert attrs.get("year_built") == 1992
    assert attrs.get("building_area_sqft") == 2260.0
    assert attrs.get("land_use_class") == "single_family_residential"
    assert attrs.get("roof_material_public_record") == "tile"
    provenance = attrs.get("attribute_provenance") or {}
    assert provenance.get("year_built") == "observed_public_record"
    assert provenance.get("building_area_sqft") == "observed_public_record"
    assert provenance.get("land_use_class") == "observed_public_record"
    assert provenance.get("roof_material_public_record") == "observed_public_record"


@pytest.mark.skipif(Polygon is None, reason="Parcel-boundary sampling test requires shapely")
def test_neighboring_parcels_produce_distinct_parcel_based_metrics(monkeypatch):
    if wildfire_data_module.Transformer is None:
        pytest.skip("Parcel-boundary sampling test requires pyproj transformer support")

    client = WildfireDataClient()

    class StubFootprints:
        def get_building_footprint(self, _lat, _lon, **_kwargs):
            return BuildingFootprintResult(
                found=False,
                source="stub.geojson",
                confidence=0.0,
                match_status="none",
                assumptions=["no footprint match in fixture"],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return None

    client.footprints = StubFootprints()

    def _canopy_summary(ring_geometry, canopy_path: str):  # noqa: ARG001
        cx = float(ring_geometry.centroid.x)
        vegetation = 82.0 if cx < -113.99412 else 24.0
        return {
            "canopy_mean": vegetation,
            "canopy_max": vegetation,
            "coverage_pct": vegetation,
            "vegetation_density": vegetation,
        }

    def _fuel_summary(ring_geometry, fuel_path: str):  # noqa: ARG001
        cx = float(ring_geometry.centroid.x)
        return 74.0 if cx < -113.99412 else 20.0

    monkeypatch.setattr(client, "_summarize_ring_canopy", _canopy_summary)
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", _fuel_summary)
    monkeypatch.setattr(
        client,
        "_compute_structure_aware_vegetation_features",
        lambda **_kwargs: {
            "near_structure_vegetation_0_5_pct": 48.0,
            "near_structure_vegetation_5_30_pct": 52.0,
            "vegetation_edge_directional_concentration_pct": 14.0,
            "canopy_dense_fuel_asymmetry_pct": 9.0,
            "nearest_continuous_vegetation_distance_ft": 20.0,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "point_proxy",
            "vegetation_directional_precision_score": 0.35,
            "vegetation_directional_basis": "point_proxy_relative",
            "directional_risk": {},
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_relative_slope",
        lambda **_kwargs: {
            "local_slope": 11.0,
            "precision_flag": "fallback_point_proxy",
            "source": "sampled",
        },
    )

    west_parcel = Polygon(
        [
            (-113.994260, 46.872250),
            (-113.994125, 46.872250),
            (-113.994125, 46.872390),
            (-113.994260, 46.872390),
        ]
    )
    east_parcel = Polygon(
        [
            (-113.994115, 46.872250),
            (-113.993980, 46.872250),
            (-113.993980, 46.872390),
            (-113.994115, 46.872390),
        ]
    )

    west_ctx, _west_assumptions, _west_sources = client._compute_structure_ring_metrics(
        46.872320,
        -113.994120,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
        parcel_polygon=west_parcel,
    )
    east_ctx, _east_assumptions, _east_sources = client._compute_structure_ring_metrics(
        46.872320,
        -113.994120,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
        parcel_polygon=east_parcel,
    )

    assert west_ctx.get("fallback_mode") == "point_based"
    assert east_ctx.get("fallback_mode") == "point_based"
    assert (west_ctx.get("ring_metrics") or {}).get("_meta", {}).get("ring_generation_mode") == "point_annulus_parcel_clipped"
    assert (east_ctx.get("ring_metrics") or {}).get("_meta", {}).get("ring_generation_mode") == "point_annulus_parcel_clipped"
    assert (west_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("sampling_boundary") == "parcel_clipped"
    assert (east_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("sampling_boundary") == "parcel_clipped"

    west_parcel_metrics = west_ctx.get("parcel_based_metrics") or {}
    east_parcel_metrics = east_ctx.get("parcel_based_metrics") or {}
    assert west_parcel_metrics.get("vegetation_within_parcel") != east_parcel_metrics.get("vegetation_within_parcel")
    assert west_parcel_metrics.get("cleared_area_ratio") != east_parcel_metrics.get("cleared_area_ratio")
    assert west_parcel_metrics.get("edge_exposure") != east_parcel_metrics.get("edge_exposure")


@pytest.mark.skipif(Polygon is None, reason="Parcel-boundary sampling test requires shapely")
def test_matched_footprint_ring_sampling_is_clipped_to_parcel_boundary(monkeypatch):
    if wildfire_data_module.Transformer is None:
        pytest.skip("Parcel-boundary sampling test requires pyproj transformer support")

    client = WildfireDataClient()
    shared_footprint = Polygon(
        [
            (-113.994180, 46.872300),
            (-113.994050, 46.872300),
            (-113.994050, 46.872360),
            (-113.994180, 46.872360),
        ]
    )
    centroid = shared_footprint.centroid

    class StubFootprints:
        def get_building_footprint(self, _lat, _lon, **_kwargs):
            return BuildingFootprintResult(
                found=True,
                footprint=shared_footprint,
                centroid=(float(centroid.y), float(centroid.x)),
                source="stub-footprints.geojson",
                confidence=0.9,
                match_status="matched",
                match_method="point_in_footprint",
                matched_structure_id="stub-1",
                match_distance_m=0.0,
                candidate_count=1,
                candidate_summaries=[],
                assumptions=[],
            )

        def get_neighbor_structure_metrics(self, **_kwargs):
            return {
                "nearby_structure_count_100_ft": 2,
                "nearby_structure_count_300_ft": 7,
                "nearest_structure_distance_ft": 20.0,
            }

    client.footprints = StubFootprints()

    def _canopy_summary(ring_geometry, canopy_path: str):  # noqa: ARG001
        cx = float(ring_geometry.centroid.x)
        vegetation = 78.0 if cx < -113.99412 else 18.0
        return {
            "canopy_mean": vegetation,
            "canopy_max": vegetation,
            "coverage_pct": vegetation,
            "vegetation_density": vegetation,
        }

    def _fuel_summary(ring_geometry, fuel_path: str):  # noqa: ARG001
        cx = float(ring_geometry.centroid.x)
        return 70.0 if cx < -113.99412 else 16.0

    monkeypatch.setattr(client, "_summarize_ring_canopy", _canopy_summary)
    monkeypatch.setattr(client, "_summarize_ring_fuel_presence", _fuel_summary)
    monkeypatch.setattr(
        client,
        "_compute_structure_aware_vegetation_features",
        lambda **_kwargs: {
            "near_structure_vegetation_0_5_pct": 55.0,
            "near_structure_vegetation_5_30_pct": 58.0,
            "vegetation_edge_directional_concentration_pct": 12.0,
            "canopy_dense_fuel_asymmetry_pct": 8.0,
            "nearest_continuous_vegetation_distance_ft": 18.0,
            "vegetation_directional_sectors": {},
            "vegetation_directional_precision": "footprint_relative",
            "vegetation_directional_precision_score": 0.9,
            "vegetation_directional_basis": "footprint_relative",
            "directional_risk": {},
        },
    )
    monkeypatch.setattr(
        client,
        "_compute_structure_relative_slope",
        lambda **_kwargs: {
            "local_slope": 14.0,
            "uphill_exposure": 22.0,
            "downhill_buffer": 78.0,
            "precision_flag": "footprint_relative",
        },
    )

    west_parcel = Polygon(
        [
            (-113.994260, 46.872250),
            (-113.994125, 46.872250),
            (-113.994125, 46.872390),
            (-113.994260, 46.872390),
        ]
    )
    east_parcel = Polygon(
        [
            (-113.994115, 46.872250),
            (-113.993980, 46.872250),
            (-113.993980, 46.872390),
            (-113.994115, 46.872390),
        ]
    )

    west_ctx, _west_assumptions, _west_sources = client._compute_structure_ring_metrics(
        46.872330,
        -113.994120,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
        parcel_polygon=west_parcel,
    )
    east_ctx, _east_assumptions, _east_sources = client._compute_structure_ring_metrics(
        46.872330,
        -113.994120,
        canopy_path="canopy.tif",
        fuel_path="fuel.tif",
        slope_path="slope.tif",
        parcel_polygon=east_parcel,
    )

    assert west_ctx.get("footprint_used") is True
    assert east_ctx.get("footprint_used") is True
    assert (west_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("sampling_boundary") == "parcel_clipped"
    assert (east_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("sampling_boundary") == "parcel_clipped"
    assert (west_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("vegetation_density") != (
        (east_ctx.get("ring_metrics") or {}).get("ring_0_5_ft", {}).get("vegetation_density")
    )

    west_parcel_metrics = west_ctx.get("parcel_based_metrics") or {}
    east_parcel_metrics = east_ctx.get("parcel_based_metrics") or {}
    assert west_parcel_metrics.get("vegetation_within_parcel") != east_parcel_metrics.get("vegetation_within_parcel")
    assert west_parcel_metrics.get("edge_exposure") != east_parcel_metrics.get("edge_exposure")
