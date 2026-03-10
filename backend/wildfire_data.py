from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.building_footprints import BuildingFootprintClient, compute_structure_rings
from backend.layer_diagnostics import (
    initialize_layer_audit,
    summarize_layer_audit,
    update_layer_audit,
)
from backend.open_data_adapters import (
    GridMETAdapter,
    MTBSAdapter,
    OSMRoadAdapter,
    WHPAdapter,
)
from backend.data_prep.region_lookup import find_region_for_point as lookup_region_for_point
from backend.region_registry import (
    resolve_region_file,
    validate_region_files,
)

try:
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Point, mapping, shape
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover - graceful runtime fallback when geo deps are missing
    np = None
    rasterio = None
    Transformer = None
    Point = None
    mapping = None
    shape = None
    shapely_transform = None


@dataclass
class WildfireContext:
    environmental_index: Optional[float]
    slope_index: Optional[float]
    aspect_index: Optional[float]
    fuel_index: Optional[float]
    moisture_index: Optional[float]
    canopy_index: Optional[float]
    wildland_distance_index: Optional[float]
    historic_fire_index: Optional[float]
    burn_probability_index: Optional[float]
    hazard_severity_index: Optional[float]
    access_exposure_index: Optional[float] = None
    burn_probability: Optional[float] = None
    wildfire_hazard: Optional[float] = None
    slope: Optional[float] = None
    fuel_model: Optional[float] = None
    canopy_cover: Optional[float] = None
    historic_fire_distance: Optional[float] = None
    wildland_distance: Optional[float] = None
    environmental_layer_status: Dict[str, str] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    structure_ring_metrics: Dict[str, Dict[str, float | None]] = field(default_factory=dict)
    property_level_context: Dict[str, Any] = field(default_factory=dict)
    region_context: Dict[str, Any] = field(default_factory=dict)
    hazard_context: Dict[str, Any] = field(default_factory=dict)
    moisture_context: Dict[str, Any] = field(default_factory=dict)
    historical_fire_context: Dict[str, Any] = field(default_factory=dict)
    access_context: Dict[str, Any] = field(default_factory=dict)
    layer_coverage_audit: List[Dict[str, Any]] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)


def compute_environmental_data_completeness(context: WildfireContext) -> float:
    status = context.environmental_layer_status or {}
    if not status:
        return 0.0

    tracked_layers = [
        "burn_probability",
        "hazard",
        "slope",
        "fuel",
        "canopy",
        "fire_history",
    ]
    available = sum(1 for layer in tracked_layers if status.get(layer) == "ok")
    return round((available / float(len(tracked_layers))) * 100.0, 1)


class WildfireDataClient:
    """Layer-backed wildfire context provider.

    Expected dataset environment variables (GeoTIFF/GeoJSON):
    - WF_LAYER_BURN_PROB_TIF
    - WF_LAYER_HAZARD_SEVERITY_TIF
    - WF_LAYER_SLOPE_TIF (optional when DEM is provided)
    - WF_LAYER_ASPECT_TIF (optional when DEM is provided)
    - WF_LAYER_DEM_TIF
    - WF_LAYER_FUEL_TIF
    - WF_LAYER_CANOPY_TIF
    - WF_LAYER_MOISTURE_TIF (optional; higher values should represent drier fuels)
    - WF_LAYER_FIRE_PERIMETERS_GEOJSON
    """

    def __init__(self) -> None:
        self.base_paths = {
            "burn_prob": os.getenv("WF_LAYER_BURN_PROB_TIF", ""),
            "hazard": os.getenv("WF_LAYER_HAZARD_SEVERITY_TIF", ""),
            "whp": os.getenv("WF_LAYER_WHP_TIF", ""),
            "slope": os.getenv("WF_LAYER_SLOPE_TIF", ""),
            "aspect": os.getenv("WF_LAYER_ASPECT_TIF", ""),
            "dem": os.getenv("WF_LAYER_DEM_TIF", ""),
            "fuel": os.getenv("WF_LAYER_FUEL_TIF", ""),
            "canopy": os.getenv("WF_LAYER_CANOPY_TIF", ""),
            "moisture": os.getenv("WF_LAYER_MOISTURE_TIF", ""),
            "gridmet_dryness": os.getenv("WF_LAYER_GRIDMET_DRYNESS_TIF", ""),
            "perimeters": os.getenv("WF_LAYER_FIRE_PERIMETERS_GEOJSON", ""),
            "mtbs_severity": os.getenv("WF_LAYER_MTBS_SEVERITY_TIF", ""),
            "footprints": os.getenv("WF_LAYER_BUILDING_FOOTPRINTS_GEOJSON", ""),
            "fema_structures": os.getenv("WF_LAYER_FEMA_STRUCTURES_GEOJSON", ""),
            "roads": os.getenv("WF_LAYER_OSM_ROADS_GEOJSON", ""),
        }
        # Active runtime paths for the most recent collect_context() call.
        self.paths = dict(self.base_paths)
        self.region_data_dir = os.getenv("WF_REGION_DATA_DIR", str(Path("data") / "regions"))
        self.use_prepared_regions = os.getenv("WF_USE_PREPARED_REGIONS", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.allow_legacy_layer_fallback = os.getenv("WF_ALLOW_LEGACY_LAYER_FALLBACK", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.footprints = BuildingFootprintClient()
        self.whp_adapter = WHPAdapter()
        self.gridmet_adapter = GridMETAdapter()
        self.mtbs_adapter = MTBSAdapter()
        self.osm_adapter = OSMRoadAdapter()

    @staticmethod
    def _to_index(value: float, src_min: float, src_max: float) -> float:
        if src_max <= src_min:
            return 50.0
        v = max(src_min, min(src_max, value))
        return round(100.0 * (v - src_min) / (src_max - src_min), 1)

    @staticmethod
    def _meters_to_lat_deg(meters: float) -> float:
        return meters / 111_320.0

    @staticmethod
    def _meters_to_lon_deg(meters: float, lat: float) -> float:
        denom = 111_320.0 * max(0.1, math.cos(math.radians(lat)))
        return meters / denom

    @staticmethod
    def _file_exists(path: str) -> bool:
        return bool(path) and Path(path).exists()

    def _legacy_layer_configured(self, configured_paths: dict[str, str]) -> bool:
        configured = [
            configured_paths.get("dem"),
            configured_paths.get("slope"),
            configured_paths.get("fuel"),
            configured_paths.get("canopy"),
            configured_paths.get("perimeters"),
            configured_paths.get("footprints"),
            configured_paths.get("whp"),
            configured_paths.get("gridmet_dryness"),
            configured_paths.get("roads"),
        ]
        return any(self._file_exists(path or "") for path in configured)

    def _resolve_runtime_layer_paths(
        self, lat: float, lon: float
    ) -> tuple[dict[str, str], dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []
        configured_paths = dict(self.base_paths)
        if self.paths and self.paths != self.base_paths:
            configured_paths.update(self.paths)
        runtime_paths = dict(configured_paths)
        region_context: dict[str, Any] = {
            "region_status": "legacy_fallback",
            "region_id": None,
            "region_display_name": None,
            "manifest_path": None,
        }

        region_manifest: dict[str, Any] | None = None
        if self.use_prepared_regions:
            lookup = lookup_region_for_point(lat=lat, lon=lon, regions_root=self.region_data_dir)
            if lookup.get("covered"):
                manifest = lookup.get("manifest")
                if isinstance(manifest, dict):
                    region_manifest = manifest
            else:
                diagnostics = list(lookup.get("diagnostics") or [])
                assumptions.extend(diagnostics[:3])

        if region_manifest:
            valid, missing = validate_region_files(region_manifest, base_dir=self.region_data_dir)
            if valid:
                region_context = {
                    "region_status": "prepared",
                    "region_id": region_manifest.get("region_id"),
                    "region_display_name": region_manifest.get("display_name"),
                    "manifest_path": region_manifest.get("_manifest_path"),
                }
                sources.append(f"Prepared region: {region_manifest.get('region_id')}")
                layer_key_map = {
                    "burn_prob": ("burn_probability", "burn_prob"),
                    "hazard": ("wildfire_hazard", "hazard"),
                    "whp": ("whp", "wildfire_hazard_potential"),
                    "slope": ("slope",),
                    "aspect": ("aspect",),
                    "dem": ("dem",),
                    "fuel": ("fuel",),
                    "canopy": ("canopy",),
                    "moisture": ("moisture",),
                    "gridmet_dryness": ("gridmet_dryness", "gridmet"),
                    "perimeters": ("fire_perimeters", "perimeters"),
                    "mtbs_severity": ("mtbs_severity", "burn_severity"),
                    "footprints": ("building_footprints", "footprints"),
                    "fema_structures": ("fema_structures",),
                    "roads": ("roads", "osm_roads", "road_network"),
                }
                for runtime_key, manifest_keys in layer_key_map.items():
                    resolved: str | None = None
                    for manifest_key in manifest_keys:
                        resolved = resolve_region_file(region_manifest, manifest_key, base_dir=self.region_data_dir)
                        if resolved:
                            break
                    if resolved:
                        runtime_paths[runtime_key] = resolved
                return runtime_paths, region_context, assumptions, sources

            region_context = {
                "region_status": "invalid_manifest",
                "region_id": region_manifest.get("region_id"),
                "region_display_name": region_manifest.get("display_name"),
                "manifest_path": region_manifest.get("_manifest_path"),
            }
            assumptions.append("Prepared region manifest is missing required files for this area.")
            assumptions.extend([f"Region file validation: {reason}" for reason in missing[:5]])

        if self.allow_legacy_layer_fallback and self._legacy_layer_configured(configured_paths):
            if not region_manifest:
                assumptions.append("Prepared region not found for location; using legacy direct layer paths.")
                region_context = {
                    "region_status": "legacy_fallback",
                    "region_id": None,
                    "region_display_name": None,
                    "manifest_path": None,
                }
            sources.append("Legacy direct layer paths")
            return runtime_paths, region_context, assumptions, sources

        assumptions.append(
            "Region not prepared for this location; initialize prepared regional layers before high-confidence scoring."
        )
        return (
            {k: "" for k in runtime_paths.keys()},
            {
                "region_status": "region_not_prepared",
                "region_id": None,
                "region_display_name": None,
                "manifest_path": None,
            },
            assumptions,
            sources,
        )

    @lru_cache(maxsize=16)
    def _open_raster(self, path: str):
        return rasterio.open(path)

    @lru_cache(maxsize=8)
    def _load_perimeters(self, path: str) -> List[Any]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", []) if isinstance(payload, dict) else []
        geoms = []
        for feat in features:
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom:
                geoms.append(shape(geom))
        return geoms

    def _to_dataset_crs(self, ds, lon: float, lat: float) -> tuple[float, float]:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return lon, lat
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return transformer.transform(lon, lat)

    def _to_geom_crs(self, geom: Any, ds) -> Any:
        if ds.crs is None or str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return geom
        if not (Transformer and shapely_transform):
            return geom
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        return shapely_transform(transformer.transform, geom)

    def _polygon_raster_values(self, path: str, polygon: Any) -> list[float]:
        if not (rasterio and np is not None and mapping and self._file_exists(path)):
            return []

        ds = self._open_raster(path)
        try:
            geom_ds = self._to_geom_crs(polygon, ds)
            window = rasterio.features.geometry_window(ds, [mapping(geom_ds)])
            arr = ds.read(1, window=window, masked=True)
            inside_mask = rasterio.features.geometry_mask(
                [mapping(geom_ds)],
                transform=ds.window_transform(window),
                out_shape=arr.shape,
                invert=True,
                all_touched=True,
            )
            vals = arr[inside_mask]
            if hasattr(vals, "compressed"):
                vals = vals.compressed()
            return [float(v) for v in vals.tolist()]
        except Exception:
            return []

    def _summarize_ring_canopy(self, ring_geometry: Any, canopy_path: str) -> dict[str, float] | None:
        canopy_vals = self._polygon_raster_values(canopy_path, ring_geometry)
        if not canopy_vals:
            return None

        canopy_mean = float(sum(canopy_vals) / len(canopy_vals))
        canopy_max = float(max(canopy_vals))
        coverage_pct = float(sum(1 for v in canopy_vals if v >= 20.0) / len(canopy_vals) * 100.0)
        vegetation_density = float(self._to_index(canopy_mean, 0.0, 100.0))

        return {
            "canopy_mean": round(canopy_mean, 1),
            "canopy_max": round(canopy_max, 1),
            "coverage_pct": round(coverage_pct, 1),
            "vegetation_density": round(vegetation_density, 1),
        }

    def _summarize_ring_fuel_presence(self, ring_geometry: Any, fuel_path: str) -> float | None:
        fuel_vals = self._polygon_raster_values(fuel_path, ring_geometry)
        return self._fuel_presence_proxy_from_values(fuel_vals)

    def _fuel_presence_proxy_from_values(self, fuel_vals: list[float]) -> float | None:
        if not fuel_vals:
            return None
        wildland_cells = 0
        for raw in fuel_vals:
            code = int(round(raw))
            if (1 <= code <= 13) or (101 <= code <= 149):
                wildland_cells += 1

        return round((wildland_cells / len(fuel_vals)) * 100.0, 1)

    def _sample_annulus(
        self,
        path: str,
        lat: float,
        lon: float,
        *,
        inner_radius_m: float,
        outer_radius_m: float,
        radial_step_m: float = 12.0,
        angular_step_m: float = 12.0,
    ) -> list[float]:
        if not (rasterio and self._file_exists(path)):
            return []
        if outer_radius_m <= 0 or outer_radius_m <= inner_radius_m:
            return []

        values: list[float] = []
        radial = max(inner_radius_m + radial_step_m, inner_radius_m)
        while radial <= outer_radius_m + 1e-6:
            points = max(12, int(2 * math.pi * radial / max(1.0, angular_step_m)))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = self._meters_to_lat_deg(radial * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radial * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    values.append(sample)
            radial += radial_step_m
        return values

    def _build_point_proxy_ring_metrics(
        self,
        *,
        lat: float,
        lon: float,
        canopy_path: str,
        fuel_path: str,
    ) -> dict[str, dict[str, float | None]]:
        zone_bounds_ft = {
            "ring_0_5_ft": (0.0, 5.0),
            "ring_5_30_ft": (5.0, 30.0),
            "ring_30_100_ft": (30.0, 100.0),
            "ring_100_300_ft": (100.0, 300.0),
        }
        zone_aliases = {
            "ring_0_5_ft": "zone_0_5_ft",
            "ring_5_30_ft": "zone_5_30_ft",
            "ring_30_100_ft": "zone_30_100_ft",
            "ring_100_300_ft": "zone_100_300_ft",
        }
        metrics: dict[str, dict[str, float | None]] = {}
        for ring_key, (inner_ft, outer_ft) in zone_bounds_ft.items():
            inner_m = inner_ft * 0.3048
            outer_m = outer_ft * 0.3048
            canopy_vals = self._sample_annulus(
                canopy_path,
                lat,
                lon,
                inner_radius_m=inner_m,
                outer_radius_m=outer_m,
            )
            fuel_vals = self._sample_annulus(
                fuel_path,
                lat,
                lon,
                inner_radius_m=inner_m,
                outer_radius_m=outer_m,
            )
            canopy_stats: dict[str, float] | None = None
            if canopy_vals:
                canopy_mean = float(sum(canopy_vals) / len(canopy_vals))
                canopy_max = float(max(canopy_vals))
                coverage_pct = float(sum(1 for v in canopy_vals if v >= 20.0) / len(canopy_vals) * 100.0)
                canopy_stats = {
                    "canopy_mean": round(canopy_mean, 1),
                    "canopy_max": round(canopy_max, 1),
                    "coverage_pct": round(coverage_pct, 1),
                    "vegetation_density": round(float(self._to_index(canopy_mean, 0.0, 100.0)), 1),
                }

            fuel_presence = self._fuel_presence_proxy_from_values(fuel_vals)
            if canopy_stats is None and fuel_presence is None:
                zone = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": None,
                    "coverage_pct": None,
                    "fuel_presence_proxy": None,
                    "basis": "point_proxy",
                }
            elif canopy_stats is None:
                zone = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": fuel_presence,
                    "coverage_pct": fuel_presence,
                    "fuel_presence_proxy": fuel_presence,
                    "basis": "point_proxy",
                }
            else:
                vegetation_density = canopy_stats["vegetation_density"]
                if fuel_presence is not None:
                    vegetation_density = round((vegetation_density + fuel_presence) / 2.0, 1)
                zone = {
                    "canopy_mean": canopy_stats["canopy_mean"],
                    "canopy_max": canopy_stats["canopy_max"],
                    "vegetation_density": vegetation_density,
                    "coverage_pct": canopy_stats["coverage_pct"],
                    "fuel_presence_proxy": fuel_presence,
                    "basis": "point_proxy",
                }
            metrics[ring_key] = zone
            metrics[zone_aliases[ring_key]] = dict(zone)
        return metrics

    def _compute_structure_ring_metrics(
        self,
        lat: float,
        lon: float,
        *,
        canopy_path: str,
        fuel_path: str,
        footprint_path: str | None = None,
        fallback_footprint_path: str | None = None,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        assumptions: list[str] = []
        sources: list[str] = []
        footprint_client = self.footprints
        source_paths = [p for p in [footprint_path, fallback_footprint_path] if p]
        if source_paths:
            footprint_client = BuildingFootprintClient(path=source_paths[0], extra_paths=source_paths[1:])

        try:
            result = footprint_client.get_building_footprint(lat, lon)
        except Exception as exc:  # pragma: no cover - defensive guard for malformed sources
            assumptions.append(f"Building footprint lookup failed: {exc}")
            assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
            return {
                "footprint_used": False,
                "footprint_found": False,
                "footprint_status": "error",
                "footprint_source": None,
                "footprint_confidence": 0.0,
                "fallback_mode": "point_based",
                "ring_metrics": None,
                "nearest_vegetation_distance_ft": None,
                "neighboring_structure_metrics": None,
            }, assumptions, sources

        assumptions.extend(result.assumptions)

        if not result.found or result.footprint is None:
            assumptions.append("Building footprint analysis unavailable; using point-based vegetation context.")
            point_proxy_metrics = self._build_point_proxy_ring_metrics(
                lat=lat,
                lon=lon,
                canopy_path=canopy_path,
                fuel_path=fuel_path,
            )
            nearest_vegetation_distance_ft: float | None = None
            for ring_key, approx_ft in [
                ("ring_0_5_ft", 3.0),
                ("ring_5_30_ft", 18.0),
                ("ring_30_100_ft", 65.0),
                ("ring_100_300_ft", 180.0),
            ]:
                density = (point_proxy_metrics.get(ring_key) or {}).get("vegetation_density")
                if density is not None and float(density) >= 40.0:
                    nearest_vegetation_distance_ft = approx_ft
                    break
            if any(
                (point_proxy_metrics.get(zone) or {}).get("vegetation_density") is not None
                for zone in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft")
            ):
                assumptions.append("Structure ring metrics were approximated from point-based annulus sampling.")
                sources.append("Point-proxy ring vegetation summaries")
            status = "not_found"
            assumptions_blob = " ".join(result.assumptions).lower()
            if "not configured" in assumptions_blob or "missing" in assumptions_blob:
                status = "provider_unavailable"
            return {
                "footprint_used": False,
                "footprint_found": False,
                "footprint_status": status,
                "footprint_source": result.source,
                "footprint_confidence": result.confidence,
                "fallback_mode": "point_based",
                "ring_metrics": point_proxy_metrics if point_proxy_metrics else None,
                "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
                "neighboring_structure_metrics": None,
            }, assumptions, sources

        sources.append("Building footprint source")
        rings, ring_assumptions = compute_structure_rings(result.footprint)
        assumptions.extend(ring_assumptions)

        ring_metrics: dict[str, dict[str, float | None]] = {}
        zone_aliases = {
            "ring_0_5_ft": "zone_0_5_ft",
            "ring_5_30_ft": "zone_5_30_ft",
            "ring_30_100_ft": "zone_30_100_ft",
            "ring_100_300_ft": "zone_100_300_ft",
        }
        for ring_key in ("ring_0_5_ft", "ring_5_30_ft", "ring_30_100_ft", "ring_100_300_ft"):
            ring_geom = rings.get(ring_key)
            if ring_geom is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": None,
                    "coverage_pct": None,
                    "fuel_presence_proxy": None,
                }
                ring_metrics[ring_key] = metrics
                ring_metrics[zone_aliases[ring_key]] = dict(metrics)
                continue

            canopy_stats = self._summarize_ring_canopy(ring_geom, canopy_path=canopy_path)
            fuel_presence = self._summarize_ring_fuel_presence(ring_geom, fuel_path=fuel_path)

            if canopy_stats is None:
                metrics = {
                    "canopy_mean": None,
                    "canopy_max": None,
                    "vegetation_density": fuel_presence,
                    "coverage_pct": fuel_presence,
                    "fuel_presence_proxy": fuel_presence,
                }
            else:
                vegetation_density = canopy_stats["vegetation_density"]
                if fuel_presence is not None:
                    vegetation_density = round((vegetation_density + fuel_presence) / 2.0, 1)
                metrics = {
                    "canopy_mean": canopy_stats["canopy_mean"],
                    "canopy_max": canopy_stats["canopy_max"],
                    "vegetation_density": vegetation_density,
                    "coverage_pct": canopy_stats["coverage_pct"],
                    "fuel_presence_proxy": fuel_presence,
                }
            ring_metrics[ring_key] = metrics
            ring_metrics[zone_aliases[ring_key]] = dict(metrics)

        nearest_vegetation_distance_ft: float | None = None
        for ring_key, approx_ft in [
            ("ring_0_5_ft", 3.0),
            ("ring_5_30_ft", 18.0),
            ("ring_30_100_ft", 65.0),
            ("ring_100_300_ft", 180.0),
        ]:
            density = (ring_metrics.get(ring_key) or {}).get("vegetation_density")
            if density is not None and float(density) >= 40.0:
                nearest_vegetation_distance_ft = approx_ft
                break

        neighbor_metrics = footprint_client.get_neighbor_structure_metrics(
            lat=lat,
            lon=lon,
            subject_footprint=result.footprint,
            source_path=result.source,
            radius_m=300.0 * 0.3048,
        )

        if ring_metrics:
            sources.append("Structure ring vegetation summaries")

        return {
            "footprint_used": bool(ring_metrics),
            "footprint_found": result.found,
            "footprint_status": "used" if ring_metrics else "error",
            "footprint_source": result.source,
            "footprint_confidence": result.confidence,
            "fallback_mode": "footprint" if ring_metrics else "point_based",
            "ring_metrics": ring_metrics,
            "nearest_vegetation_distance_ft": nearest_vegetation_distance_ft,
            "neighboring_structure_metrics": neighbor_metrics,
            "footprint_centroid": {
                "latitude": result.centroid[0] if result.centroid else None,
                "longitude": result.centroid[1] if result.centroid else None,
            },
        }, assumptions, sources

    def _sample_raster_point_raw(self, path: str, lat: float, lon: float) -> float | None:
        if not (rasterio and self._file_exists(path)):
            return None
        ds = self._open_raster(path)
        x, y = self._to_dataset_crs(ds, lon, lat)
        sample = next(ds.sample([(x, y)]))[0]
        nodata = ds.nodata
        if nodata is not None and float(sample) == float(nodata):
            return None
        return float(sample)

    def _sample_raster_point(self, path: str, lat: float, lon: float) -> float | None:
        try:
            return self._sample_raster_point_raw(path, lat, lon)
        except Exception:
            return None

    def _sample_layer_value_detailed(self, path: str, lat: float, lon: float) -> tuple[float | None, str, str | None]:
        if not path:
            return None, "not_configured", "Layer path is not configured."
        if not self._file_exists(path):
            return None, "missing_file", "Configured layer file is missing."
        if not rasterio:
            return None, "sampling_failed", "Raster dependencies unavailable for sampling."
        try:
            ds = self._open_raster(path)
            x, y = self._to_dataset_crs(ds, lon, lat)
            bounds = ds.bounds
            if x < bounds.left or x > bounds.right or y < bounds.bottom or y > bounds.top:
                return None, "outside_extent", "Property point is outside layer extent."
            sample = next(ds.sample([(x, y)]))[0]
            nodata = ds.nodata
            if nodata is not None and float(sample) == float(nodata):
                return None, "outside_extent", "Layer sampled nodata at property location."
            if np is not None and hasattr(np, "isnan") and np.isnan(sample):
                return None, "outside_extent", "Layer sampled nodata at property location."
            return float(sample), "ok", None
        except Exception as exc:
            return None, "sampling_failed", str(exc)

    def _sample_layer_value(self, path: str, lat: float, lon: float) -> Tuple[float | None, str]:
        if not self._file_exists(path):
            return None, "missing"
        try:
            value = self._sample_raster_point_raw(path, lat, lon)
        except Exception:
            return None, "error"
        if value is None:
            return None, "missing"
        return value, "ok"

    def _sample_circle(self, path: str, lat: float, lon: float, radius_m: float, step_m: float = 30.0) -> List[float]:
        if not (rasterio and self._file_exists(path)):
            return []

        values: List[float] = []
        rings = max(1, int(radius_m / step_m))
        for ring in range(1, rings + 1):
            r = ring * step_m
            points = max(8, int(2 * math.pi * r / step_m))
            for i in range(points):
                theta = 2.0 * math.pi * i / points
                d_lat = self._meters_to_lat_deg(r * math.sin(theta))
                d_lon = self._meters_to_lon_deg(r * math.cos(theta), lat)
                sample = self._sample_raster_point(path, lat + d_lat, lon + d_lon)
                if sample is not None:
                    values.append(sample)
        return values

    def _derive_slope_aspect_from_dem(self, dem_path: str, lat: float, lon: float, cell_m: float = 30.0) -> tuple[float | None, float | None]:
        center = self._sample_raster_point(dem_path, lat, lon)
        if center is None:
            return None, None

        d_lat = self._meters_to_lat_deg(cell_m)
        d_lon = self._meters_to_lon_deg(cell_m, lat)

        n = self._sample_raster_point(dem_path, lat + d_lat, lon)
        s = self._sample_raster_point(dem_path, lat - d_lat, lon)
        e = self._sample_raster_point(dem_path, lat, lon + d_lon)
        w = self._sample_raster_point(dem_path, lat, lon - d_lon)

        if None in {n, s, e, w}:
            return None, None

        dzdx = (e - w) / (2 * cell_m)
        dzdy = (n - s) / (2 * cell_m)
        slope_rad = math.atan(math.sqrt(dzdx * dzdx + dzdy * dzdy))
        slope_deg = math.degrees(slope_rad)

        aspect_rad = math.atan2(dzdx, -dzdy)
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0

        return slope_deg, aspect_deg

    def _fuel_combustibility_index(self, fuel_values: Iterable[float]) -> float:
        vals = [v for v in fuel_values if v is not None]
        if not vals:
            return 50.0

        weighted = []
        for raw in vals:
            code = int(round(raw))
            if 1 <= code <= 3:
                weighted.append(25.0)
            elif 4 <= code <= 9:
                weighted.append(45.0)
            elif 10 <= code <= 13:
                weighted.append(65.0)
            elif 101 <= code <= 109:
                weighted.append(75.0)
            elif 121 <= code <= 124:
                weighted.append(85.0)
            elif 141 <= code <= 149:
                weighted.append(70.0)
            else:
                weighted.append(50.0)
        return round(sum(weighted) / len(weighted), 1)

    def _wildland_distance_metrics(
        self,
        lat: float,
        lon: float,
        fuel_path: str,
        canopy_path: str,
    ) -> Tuple[float | None, float | None]:
        if not (rasterio and (self._file_exists(fuel_path) or self._file_exists(canopy_path))):
            return None, None

        for radius in (30, 60, 120, 250, 500, 1000, 2000):
            samples = max(8, int(2 * math.pi * radius / 30))
            for i in range(samples):
                theta = 2.0 * math.pi * i / samples
                d_lat = self._meters_to_lat_deg(radius * math.sin(theta))
                d_lon = self._meters_to_lon_deg(radius * math.cos(theta), lat)
                p_lat, p_lon = lat + d_lat, lon + d_lon
                fuel = self._sample_raster_point(fuel_path, p_lat, p_lon) if self._file_exists(fuel_path) else None
                canopy = self._sample_raster_point(canopy_path, p_lat, p_lon) if self._file_exists(canopy_path) else None

                is_wildland = False
                if fuel is not None:
                    fuel_code = int(round(fuel))
                    is_wildland = (1 <= fuel_code <= 13) or (101 <= fuel_code <= 149)
                if canopy is not None and canopy >= 20:
                    is_wildland = True

                if is_wildland:
                    index = round(max(0.0, min(100.0, 100.0 - (radius / 2000.0) * 100.0)), 1)
                    return float(radius), index

        # Layer exists and no contiguous wildland found within search radius.
        return 2000.0, 0.0

    def _historical_fire_metrics(self, lat: float, lon: float, perimeter_path: str) -> Tuple[float | None, float | None, str]:
        if not (shape and Point and self._file_exists(perimeter_path)):
            return None, None, "missing"

        try:
            geoms = self._load_perimeters(perimeter_path)
        except Exception:
            return None, None, "error"

        if not geoms:
            return None, 0.0, "ok"

        pt = Point(lon, lat)
        near_1km = 0
        near_5km = 0
        nearest_km: float | None = None

        lat_1 = self._meters_to_lat_deg(1000)
        lon_1 = self._meters_to_lon_deg(1000, lat)
        lat_5 = self._meters_to_lat_deg(5000)
        lon_5 = self._meters_to_lon_deg(5000, lat)

        for g in geoms:
            minx, miny, maxx, maxy = g.bounds
            if not (maxx < lon - lon_5 or minx > lon + lon_5 or maxy < lat - lat_5 or miny > lat + lat_5):
                near_5km += 1
            if not (maxx < lon - lon_1 or minx > lon + lon_1 or maxy < lat - lat_1 or miny > lat + lat_1):
                near_1km += 1
            if g.contains(pt):
                near_1km = max(near_1km, 2)
                near_5km = max(near_5km, 3)

            try:
                # Shapely distance here is in degrees for EPSG:4326; convert approximately to km.
                km = float(pt.distance(g)) * 111.32
                nearest_km = km if nearest_km is None else min(nearest_km, km)
            except Exception:
                pass

        score = near_1km * 22.0 + near_5km * 6.0
        return nearest_km, round(max(0.0, min(100.0, score)), 1), "ok"

    def collect_context(self, lat: float, lon: float) -> WildfireContext:
        assumptions: List[str] = []
        sources: List[str] = []
        runtime_paths, region_context, runtime_assumptions, runtime_sources = self._resolve_runtime_layer_paths(lat, lon)
        assumptions.extend(runtime_assumptions)
        sources.extend(runtime_sources)
        self.paths = dict(runtime_paths)
        layer_audit = initialize_layer_audit(runtime_paths, region_context)
        environmental_layer_status: dict[str, str] = {
            "burn_probability": "missing",
            "hazard": "missing",
            "slope": "missing",
            "fuel": "missing",
            "canopy": "missing",
            "fire_history": "missing",
        }
        property_level_context: dict[str, Any] = {
            "footprint_used": False,
            "footprint_found": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": None,
            "hazard_context": {},
            "moisture_context": {},
            "historical_fire_context": {},
            "access_context": {},
            "region_status": region_context.get("region_status"),
            "region_id": region_context.get("region_id"),
            "region_display_name": region_context.get("region_display_name"),
            "region_manifest_path": region_context.get("manifest_path"),
        }
        structure_ring_metrics: dict[str, dict[str, float | None]] = {}

        def _status_for_env(sample_status: str) -> str:
            if sample_status == "ok":
                return "ok"
            if sample_status == "sampling_failed":
                return "error"
            return "missing"

        if not (rasterio and np is not None and Transformer is not None):
            assumptions.append("Geospatial stack unavailable; install rasterio/numpy/pyproj/shapely.")
            for lk in ["dem", "slope", "fuel", "canopy", "whp", "mtbs_severity", "gridmet_dryness", "roads"]:
                update_layer_audit(
                    layer_audit,
                    lk,
                    sample_attempted=True,
                    sample_succeeded=False,
                    coverage_status="sampling_failed",
                    failure_reason="Geospatial stack unavailable.",
                )
            layer_audit_rows = [layer_audit[k] for k in sorted(layer_audit.keys())]
            coverage_summary = summarize_layer_audit(layer_audit_rows)
            return WildfireContext(
                environmental_index=None,
                slope_index=None,
                aspect_index=None,
                fuel_index=None,
                moisture_index=None,
                canopy_index=None,
                wildland_distance_index=None,
                historic_fire_index=None,
                burn_probability_index=None,
                hazard_severity_index=None,
                access_exposure_index=None,
                burn_probability=None,
                wildfire_hazard=None,
                slope=None,
                fuel_model=None,
                canopy_cover=None,
                historic_fire_distance=None,
                wildland_distance=None,
                environmental_layer_status=environmental_layer_status,
                data_sources=sources,
                assumptions=assumptions,
                structure_ring_metrics=structure_ring_metrics,
                property_level_context=property_level_context,
                region_context=region_context,
                hazard_context={},
                moisture_context={},
                historical_fire_context={},
                access_context={},
                layer_coverage_audit=layer_audit_rows,
                coverage_summary=coverage_summary,
            )

        ring_context, ring_assumptions, ring_sources = self._compute_structure_ring_metrics(
            lat,
            lon,
            canopy_path=runtime_paths.get("canopy", ""),
            fuel_path=runtime_paths.get("fuel", ""),
            footprint_path=runtime_paths.get("footprints"),
            fallback_footprint_path=runtime_paths.get("fema_structures"),
        )
        property_level_context = ring_context
        property_level_context.update(
            {
                "region_status": region_context.get("region_status"),
                "region_id": region_context.get("region_id"),
                "region_display_name": region_context.get("region_display_name"),
                "region_manifest_path": region_context.get("manifest_path"),
            }
        )
        footprint_status = str(ring_context.get("footprint_status") or "not_found")
        update_layer_audit(
            layer_audit,
            "building_footprints",
            sample_attempted=True,
            sample_succeeded=bool(ring_context.get("footprint_used")),
            coverage_status=(
                "observed"
                if ring_context.get("footprint_used")
                else ("not_configured" if footprint_status == "provider_unavailable" else "fallback_used")
            ),
            failure_reason=None if ring_context.get("footprint_used") else f"footprint_status={footprint_status}",
        )
        neighbor_metrics = ring_context.get("neighboring_structure_metrics")
        update_layer_audit(
            layer_audit,
            "neighbor_structures",
            sample_attempted=True,
            sample_succeeded=isinstance(neighbor_metrics, dict) and neighbor_metrics.get("neighbor_count") is not None,
            coverage_status=(
                "observed"
                if isinstance(neighbor_metrics, dict) and neighbor_metrics.get("neighbor_count") is not None
                else "fallback_used"
            ),
            raw_value_preview=neighbor_metrics if isinstance(neighbor_metrics, dict) else None,
            failure_reason=None if isinstance(neighbor_metrics, dict) else "Neighbor structure metrics unavailable.",
        )
        structure_ring_metrics = ring_context.get("ring_metrics", {}) or {}
        assumptions.extend(ring_assumptions)
        sources.extend(ring_sources)

        burn_prob, burn_status_detail, burn_reason = self._sample_layer_value_detailed(runtime_paths["burn_prob"], lat, lon)
        hazard, hazard_status_detail, hazard_reason = self._sample_layer_value_detailed(runtime_paths["hazard"], lat, lon)
        update_layer_audit(
            layer_audit,
            "dem",
            sample_attempted=True,
            sample_succeeded=self._file_exists(runtime_paths.get("dem", "")),
            coverage_status="observed" if self._file_exists(runtime_paths.get("dem", "")) else "missing_file",
        )

        hazard_context: dict[str, Any] = {
            "source": None,
            "status": "missing",
            "raw_value": None,
            "hazard_class": None,
        }
        if burn_prob is None or hazard is None:
            whp_obs = self.whp_adapter.sample(lat=lat, lon=lon, whp_path=runtime_paths.get("whp"))
            update_layer_audit(
                layer_audit,
                "whp",
                sample_attempted=True,
                sample_succeeded=whp_obs.status == "ok",
                coverage_status=(
                    "observed"
                    if whp_obs.status == "ok"
                    else ("missing_file" if runtime_paths.get("whp") else "not_configured")
                ),
                raw_value_preview=whp_obs.raw_value,
                failure_reason=None if whp_obs.status == "ok" else "; ".join(whp_obs.notes[:2]),
            )
            if whp_obs.status == "ok":
                if burn_prob is None:
                    burn_prob = whp_obs.raw_value
                    burn_status_detail = "ok"
                if hazard is None:
                    hazard = whp_obs.raw_value
                    hazard_status_detail = "ok"
                hazard_context = {
                    "source": whp_obs.source_dataset,
                    "status": "observed",
                    "raw_value": whp_obs.raw_value,
                    "hazard_class": whp_obs.hazard_class,
                    "burn_probability_index": whp_obs.burn_probability_index,
                    "hazard_severity_index": whp_obs.hazard_severity_index,
                    "notes": whp_obs.notes,
                }
                sources.append("USFS WHP raster")
            else:
                assumptions.extend(whp_obs.notes)

        environmental_layer_status["burn_probability"] = _status_for_env(burn_status_detail)
        update_layer_audit(
            layer_audit,
            "whp",
            note=burn_reason if burn_reason else None,
        )
        update_layer_audit(
            layer_audit,
            "whp",
            note=hazard_reason if hazard_reason else None,
        )
        if burn_prob is None:
            assumptions.append("Burn probability layer unavailable at property location.")
            burn_probability_index = None
        elif hazard_context.get("burn_probability_index") is not None:
            burn_probability_index = float(hazard_context["burn_probability_index"])
        else:
            burn_probability_index = self._to_index(burn_prob, 0.0, 1.0 if burn_prob <= 1.0 else 100.0)
            sources.append("Burn probability raster")

        environmental_layer_status["hazard"] = _status_for_env(hazard_status_detail)
        if hazard is None:
            assumptions.append("Wildfire hazard severity layer unavailable at property location.")
            hazard_severity_index = None
        elif hazard_context.get("hazard_severity_index") is not None:
            hazard_severity_index = float(hazard_context["hazard_severity_index"])
        else:
            hazard_severity_index = self._to_index(hazard, 0.0, 5.0 if hazard <= 5.0 else 100.0)
            sources.append("Wildfire hazard severity raster")

        slope, slope_status_detail, slope_reason = self._sample_layer_value_detailed(runtime_paths["slope"], lat, lon)
        aspect, aspect_status_detail, _aspect_reason = self._sample_layer_value_detailed(runtime_paths["aspect"], lat, lon)
        if slope is None or aspect is None:
            dem_path = runtime_paths["dem"]
            if self._file_exists(dem_path):
                derived_slope, derived_aspect = self._derive_slope_aspect_from_dem(dem_path, lat, lon)
                if slope is None and derived_slope is not None:
                    slope = derived_slope
                    slope_status_detail = "ok"
                    update_layer_audit(
                        layer_audit,
                        "slope",
                        sample_attempted=True,
                        sample_succeeded=True,
                        coverage_status="observed",
                        raw_value_preview=round(float(derived_slope), 2),
                        note="Slope derived from DEM fallback.",
                    )
                if aspect is None and derived_aspect is not None:
                    aspect = derived_aspect
                    aspect_status_detail = "ok"
                if derived_slope is not None and derived_aspect is not None:
                    sources.append("DEM-derived slope/aspect")
                else:
                    assumptions.append("Could not derive slope/aspect from DEM at property location.")
            else:
                assumptions.append("Slope/aspect rasters missing and DEM not configured.")

        if slope_status_detail != "ok":
            update_layer_audit(
                layer_audit,
                "slope",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "outside_extent"
                    if slope_status_detail == "outside_extent"
                    else ("missing_file" if slope_status_detail == "missing_file" else "not_configured")
                    if slope_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                failure_reason=slope_reason,
            )
        elif layer_audit["slope"]["coverage_status"] != "observed":
            update_layer_audit(
                layer_audit,
                "slope",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(float(slope), 2) if slope is not None else None,
            )
        environmental_layer_status["slope"] = _status_for_env(slope_status_detail)
        slope_index = None if slope is None else self._to_index(slope, 0.0, 45.0)
        if slope is not None and "DEM-derived slope/aspect" not in sources:
            sources.append("Slope raster")

        if aspect is None:
            aspect_index = None
        else:
            a = float(aspect) % 360.0
            if 180.0 <= a <= 315.0:
                aspect_index = 75.0
            elif 135.0 <= a < 180.0 or 315.0 < a <= 360.0:
                aspect_index = 60.0
            else:
                aspect_index = 40.0
            if "DEM-derived slope/aspect" not in sources:
                sources.append("Aspect raster")

        fuel_path = runtime_paths["fuel"]
        fuel_center, fuel_status_detail, fuel_reason = self._sample_layer_value_detailed(fuel_path, lat, lon)
        fuel_samples = self._sample_circle(fuel_path, lat, lon, radius_m=100.0) if self._file_exists(fuel_path) else []
        if fuel_samples:
            fuel_index = self._fuel_combustibility_index(fuel_samples)
            fuel_model = round(sum(fuel_samples) / len(fuel_samples), 2)
            environmental_layer_status["fuel"] = "ok"
            sources.append("Fuel model raster")
            update_layer_audit(
                layer_audit,
                "fuel",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(fuel_model, 2),
            )
        else:
            fuel_index = None
            fuel_model = None
            environmental_layer_status["fuel"] = _status_for_env(fuel_status_detail)
            update_layer_audit(
                layer_audit,
                "fuel",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "outside_extent"
                    if fuel_status_detail == "outside_extent"
                    else ("missing_file" if fuel_status_detail == "missing_file" else "not_configured")
                    if fuel_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                raw_value_preview=fuel_center,
                failure_reason=fuel_reason,
            )
            assumptions.append("Fuel model unavailable within 100m neighborhood.")

        canopy_path = runtime_paths["canopy"]
        canopy_center, canopy_status_detail, canopy_reason = self._sample_layer_value_detailed(canopy_path, lat, lon)
        canopy_samples = self._sample_circle(canopy_path, lat, lon, radius_m=100.0) if self._file_exists(canopy_path) else []
        if canopy_samples:
            canopy_mean = sum(canopy_samples) / len(canopy_samples)
            canopy_index = self._to_index(canopy_mean, 0.0, 100.0)
            canopy_cover = round(canopy_mean, 2)
            environmental_layer_status["canopy"] = "ok"
            sources.append("Canopy density raster")
            update_layer_audit(
                layer_audit,
                "canopy",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(canopy_cover, 2),
            )
        else:
            canopy_index = None
            canopy_cover = None
            environmental_layer_status["canopy"] = _status_for_env(canopy_status_detail)
            update_layer_audit(
                layer_audit,
                "canopy",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "outside_extent"
                    if canopy_status_detail == "outside_extent"
                    else ("missing_file" if canopy_status_detail == "missing_file" else "not_configured")
                    if canopy_status_detail in {"missing_file", "not_configured"}
                    else "sampling_failed"
                ),
                raw_value_preview=canopy_center,
                failure_reason=canopy_reason,
            )
            assumptions.append("Canopy density unavailable within 100m neighborhood.")

        moisture_context: dict[str, Any] = {
            "source": None,
            "status": "missing",
            "raw_value": None,
            "dryness_index": None,
        }
        moisture, _moisture_status = self._sample_layer_value(runtime_paths["moisture"], lat, lon)
        if moisture is None:
            gridmet_obs = self.gridmet_adapter.sample_dryness(
                lat=lat,
                lon=lon,
                dryness_raster_path=runtime_paths.get("gridmet_dryness"),
            )
            update_layer_audit(
                layer_audit,
                "gridmet_dryness",
                sample_attempted=True,
                sample_succeeded=gridmet_obs.status == "ok",
                coverage_status=(
                    "observed"
                    if gridmet_obs.status == "ok"
                    else ("missing_file" if runtime_paths.get("gridmet_dryness") else "not_configured")
                ),
                raw_value_preview=gridmet_obs.raw_value,
                failure_reason=None if gridmet_obs.status == "ok" else "; ".join(gridmet_obs.notes[:2]),
            )
            if gridmet_obs.status == "ok":
                moisture = gridmet_obs.raw_value
                moisture_index = gridmet_obs.dryness_index
                moisture_context = {
                    "source": gridmet_obs.source_dataset,
                    "status": "observed",
                    "raw_value": gridmet_obs.raw_value,
                    "dryness_index": gridmet_obs.dryness_index,
                    "rolling_window_days": gridmet_obs.rolling_window_days,
                    "notes": gridmet_obs.notes,
                }
                sources.append("gridMET dryness proxy")
            else:
                moisture_index = None
                assumptions.extend(gridmet_obs.notes)
                assumptions.append("Moisture/fuel dryness context unavailable from configured sources.")
        else:
            moisture_index = self._to_index(moisture, 0.0, 100.0)
            moisture_context = {
                "source": "Moisture/fuel dryness raster",
                "status": "observed",
                "raw_value": moisture,
                "dryness_index": moisture_index,
            }
            sources.append("Moisture/fuel dryness raster")
            update_layer_audit(
                layer_audit,
                "gridmet_dryness",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=round(float(moisture), 2),
                note="Primary moisture layer sampled successfully.",
            )

        wildland_distance, wildland_distance_index = self._wildland_distance_metrics(lat, lon, fuel_path, canopy_path)
        if wildland_distance is not None:
            sources.append("Distance to wildland vegetation (derived)")
        else:
            assumptions.append("Fuel/canopy rasters missing; wildland distance unavailable.")

        mtbs_summary = self.mtbs_adapter.summarize(
            lat=lat,
            lon=lon,
            perimeter_path=runtime_paths.get("perimeters"),
            burn_severity_path=runtime_paths.get("mtbs_severity"),
        )
        historical_fire_context: dict[str, Any] = {
            "source": mtbs_summary.source_dataset,
            "status": mtbs_summary.status,
            "nearest_perimeter_km": mtbs_summary.nearest_perimeter_km,
            "intersects_prior_burn": mtbs_summary.intersects_prior_burn,
            "nearby_high_severity": mtbs_summary.nearby_high_severity,
            "notes": mtbs_summary.notes,
        }
        if mtbs_summary.status == "ok":
            historic_fire_distance = mtbs_summary.nearest_perimeter_km
            historic_fire_index = mtbs_summary.fire_history_index
            fire_history_status = "ok"
            sources.append("MTBS historical fire context")
            update_layer_audit(
                layer_audit,
                "fire_perimeters",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=historic_fire_distance,
            )
            update_layer_audit(
                layer_audit,
                "mtbs_severity",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=mtbs_summary.nearby_high_severity,
            )
        else:
            historic_fire_distance, historic_fire_index, fire_history_status = self._historical_fire_metrics(
                lat, lon, runtime_paths["perimeters"]
            )
            if fire_history_status == "ok":
                sources.append("Historical fire perimeter recurrence")
                update_layer_audit(
                    layer_audit,
                    "fire_perimeters",
                    sample_attempted=True,
                    sample_succeeded=True,
                    coverage_status="observed",
                    raw_value_preview=historic_fire_distance,
                )
            else:
                assumptions.extend(mtbs_summary.notes)
                assumptions.append("Historical perimeter layer unavailable; recurrence unavailable.")
                update_layer_audit(
                    layer_audit,
                    "fire_perimeters",
                    sample_attempted=True,
                    sample_succeeded=False,
                    coverage_status=(
                        "missing_file" if runtime_paths.get("perimeters") else "not_configured"
                    ),
                    failure_reason="Historical perimeter sampling unavailable.",
                )
            update_layer_audit(
                layer_audit,
                "mtbs_severity",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "fallback_used" if runtime_paths.get("mtbs_severity") else "not_configured"
                ),
                failure_reason="MTBS severity context unavailable; perimeter fallback used.",
            )
        environmental_layer_status["fire_history"] = fire_history_status

        access_summary = self.osm_adapter.summarize(
            lat=lat,
            lon=lon,
            roads_path=runtime_paths.get("roads"),
        )
        access_context: dict[str, Any] = {
            "source": access_summary.source_dataset,
            "status": access_summary.status,
            "distance_to_nearest_road_m": access_summary.distance_to_nearest_road_m,
            "road_segments_within_300m": access_summary.road_segments_within_300m,
            "intersections_within_300m": access_summary.intersections_within_300m,
            "dead_end_indicator": access_summary.dead_end_indicator,
            "notes": access_summary.notes,
        }
        access_exposure_index = access_summary.access_exposure_index
        if access_summary.status == "ok" and access_exposure_index is not None:
            sources.append("OSM road-network access metrics")
            update_layer_audit(
                layer_audit,
                "roads",
                sample_attempted=True,
                sample_succeeded=True,
                coverage_status="observed",
                raw_value_preview=access_exposure_index,
            )
        else:
            assumptions.extend(access_summary.notes)
            update_layer_audit(
                layer_audit,
                "roads",
                sample_attempted=True,
                sample_succeeded=False,
                coverage_status=(
                    "missing_file"
                    if runtime_paths.get("roads") and not self._file_exists(runtime_paths.get("roads", ""))
                    else ("not_configured" if not runtime_paths.get("roads") else "sampling_failed")
                ),
                failure_reason="; ".join(access_summary.notes[:2]) if access_summary.notes else "Road sampling unavailable.",
            )

        property_level_context.update(
            {
                "hazard_context": hazard_context,
                "moisture_context": moisture_context,
                "historical_fire_context": historical_fire_context,
                "access_context": access_context,
            }
        )
        layer_audit_rows = [layer_audit[k] for k in sorted(layer_audit.keys())]
        coverage_summary = summarize_layer_audit(layer_audit_rows)
        property_level_context.update(
            {
                "layer_coverage_audit": layer_audit_rows,
                "coverage_summary": coverage_summary,
                "runtime_paths": {k: v for k, v in runtime_paths.items() if v},
            }
        )

        weighted_terms = [
            (0.20, burn_probability_index),
            (0.16, hazard_severity_index),
            (0.12, slope_index),
            (0.06, aspect_index),
            (0.12, fuel_index),
            (0.10, moisture_index),
            (0.10, canopy_index),
            (0.08, wildland_distance_index),
            (0.06, historic_fire_index),
        ]
        available_terms = [(w, v) for (w, v) in weighted_terms if v is not None]
        environmental = None
        if available_terms:
            numerator = sum(w * float(v) for w, v in available_terms)
            denominator = sum(w for w, _ in available_terms)
            environmental = round(numerator / denominator, 1)

        return WildfireContext(
            environmental_index=environmental,
            slope_index=None if slope_index is None else round(slope_index, 1),
            aspect_index=None if aspect_index is None else round(aspect_index, 1),
            fuel_index=None if fuel_index is None else round(fuel_index, 1),
            moisture_index=None if moisture_index is None else round(moisture_index, 1),
            canopy_index=None if canopy_index is None else round(canopy_index, 1),
            wildland_distance_index=None if wildland_distance_index is None else round(wildland_distance_index, 1),
            historic_fire_index=None if historic_fire_index is None else round(historic_fire_index, 1),
            burn_probability_index=None if burn_probability_index is None else round(burn_probability_index, 1),
            hazard_severity_index=None if hazard_severity_index is None else round(hazard_severity_index, 1),
            access_exposure_index=None if access_exposure_index is None else round(access_exposure_index, 1),
            burn_probability=burn_prob,
            wildfire_hazard=hazard,
            slope=slope,
            fuel_model=fuel_model,
            canopy_cover=canopy_cover,
            historic_fire_distance=None if historic_fire_distance is None else round(historic_fire_distance, 2),
            wildland_distance=wildland_distance,
            environmental_layer_status=environmental_layer_status,
            data_sources=sorted(set(sources)),
            assumptions=assumptions,
            structure_ring_metrics=structure_ring_metrics,
            property_level_context=property_level_context,
            region_context=region_context,
            hazard_context=hazard_context,
            moisture_context=moisture_context,
            historical_fire_context=historical_fire_context,
            access_context=access_context,
            layer_coverage_audit=layer_audit_rows,
            coverage_summary=coverage_summary,
        )
