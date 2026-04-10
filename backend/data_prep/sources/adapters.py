from __future__ import annotations

import csv
import io
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


BoundingBox = dict[str, float]


@dataclass
class SourceAsset:
    url: str
    dataset_name: str
    dataset_version: str | None
    dataset_provider: str
    layer_key: str
    layer_type: str
    expected_format: str
    tile_id: str | None = None
    discovered: bool = True
    discovery_notes: str | None = None


class SourceAdapter(Protocol):
    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        ...


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    import json

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON payload for {url}")
    return data


def _fetch_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


class USGS3DEPAdapter:
    def __init__(self) -> None:
        self.api_url = os.getenv("WF_USGS_3DEP_API_URL", "https://tnmaccess.nationalmap.gov/api/v1/products")
        self.dataset_name = os.getenv(
            "WF_USGS_3DEP_DATASET",
            "National Elevation Dataset (NED) 1/3 arc-second Current",
        )
        self.timeout = float(os.getenv("WF_SOURCE_DISCOVERY_TIMEOUT", "30"))

    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        params = {
            "datasets": self.dataset_name,
            "bbox": f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
            "prodFormats": "GeoTIFF",
        }
        query = urllib.parse.urlencode(params)
        data = _fetch_json(f"{self.api_url}?{query}", timeout=self.timeout)
        items = data.get("items", []) if isinstance(data, dict) else []
        assets: list[SourceAsset] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("downloadURL") or item.get("downloadUrl")
            if not url:
                continue
            assets.append(
                SourceAsset(
                    url=str(url),
                    dataset_name=str(item.get("title") or self.dataset_name),
                    dataset_version=str(item.get("lastUpdated") or data.get("dateCreated") or "") or None,
                    dataset_provider="USGS 3DEP/The National Map",
                    layer_key="dem",
                    layer_type="raster",
                    expected_format="tif",
                    tile_id=str(item.get("sourceId") or item.get("id") or "") or None,
                )
            )
        return assets


class NIFCFirePerimeterAdapter:
    def __init__(self) -> None:
        self.url = os.getenv("WF_NIFC_FIRE_PERIMETERS_URL", "")

    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        if not self.url:
            return []
        return [
            SourceAsset(
                url=self.url,
                dataset_name="NIFC Interagency Fire Perimeter History",
                dataset_version=os.getenv("WF_NIFC_FIRE_PERIMETERS_VERSION"),
                dataset_provider="NIFC",
                layer_key="fire_perimeters",
                layer_type="vector",
                expected_format="geojson",
            )
        ]


def _lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (2**zoom))


def _lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    return int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)


def _tile_to_quadkey(tile_x: int, tile_y: int, zoom: int) -> str:
    quadkey = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tile_x & mask:
            digit += 1
        if tile_y & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


def _bbox_quadkeys(bbox: BoundingBox, zoom: int) -> set[str]:
    min_x = _lon_to_tile_x(bbox["min_lon"], zoom)
    max_x = _lon_to_tile_x(bbox["max_lon"], zoom)
    min_y = _lat_to_tile_y(bbox["max_lat"], zoom)
    max_y = _lat_to_tile_y(bbox["min_lat"], zoom)
    keys: set[str] = set()
    for tx in range(min_x, max_x + 1):
        for ty in range(min_y, max_y + 1):
            keys.add(_tile_to_quadkey(tx, ty, zoom))
    return keys


class MicrosoftBuildingFootprintAdapter:
    # Public catalog CSV published by Microsoft for the US Building Footprints
    # dataset.  Override with WF_MS_BUILDINGS_INDEX_URL if a mirror or newer
    # version is preferred.
    _DEFAULT_INDEX_URL = (
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    )

    def __init__(self) -> None:
        self.index_url = os.getenv("WF_MS_BUILDINGS_INDEX_URL", self._DEFAULT_INDEX_URL)
        self.zoom = int(os.getenv("WF_MS_BUILDINGS_TILE_ZOOM", "9"))
        self.timeout = float(os.getenv("WF_SOURCE_DISCOVERY_TIMEOUT", "30"))

    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        if not self.index_url:
            return []
        text = _fetch_text(self.index_url, timeout=self.timeout)
        reader = csv.DictReader(io.StringIO(text))
        needed = _bbox_quadkeys(bbox, self.zoom)
        assets: list[SourceAsset] = []
        for row in reader:
            quad = (row.get("QuadKey") or row.get("quadkey") or "").strip()
            url = (row.get("Url") or row.get("url") or "").strip()
            if not quad or not url:
                continue
            if quad in needed:
                assets.append(
                    SourceAsset(
                        url=url,
                        dataset_name="Microsoft US Building Footprints",
                        dataset_version=os.getenv("WF_MS_BUILDINGS_VERSION"),
                        dataset_provider="Microsoft",
                        layer_key="building_footprints",
                        layer_type="vector",
                        expected_format="geojson",
                        tile_id=quad,
                    )
                )
        return sorted(assets, key=lambda a: a.tile_id or "")


class OvertureBuildingFootprintAdapter:
    def __init__(self) -> None:
        self.default_url = os.getenv("WF_OVERTURE_BUILDINGS_URL", "")
        self.template = os.getenv("WF_OVERTURE_BUILDINGS_URL_TEMPLATE", "")

    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        url = self.default_url
        if not url and self.template:
            url = self.template.format(
                min_lon=bbox["min_lon"],
                min_lat=bbox["min_lat"],
                max_lon=bbox["max_lon"],
                max_lat=bbox["max_lat"],
                bbox=f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
            )
        if not url:
            return []
        return [
            SourceAsset(
                url=url,
                dataset_name="Overture Maps Global Buildings",
                dataset_version=os.getenv("WF_OVERTURE_BUILDINGS_VERSION"),
                dataset_provider="Overture Maps Foundation",
                layer_key="building_footprints_overture",
                layer_type="vector",
                expected_format="geojson",
                discovery_notes="Overture building subset source resolved from configured URL/template.",
            )
        ]


class LANDFIREFuelAdapter:
    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        return _resolve_landfire_layer(
            bbox=bbox,
            layer_key="fuel",
            name="LANDFIRE Fuel",
            default_url_env="WF_LANDFIRE_FUEL_URL",
            template_env="WF_LANDFIRE_FUEL_URL_TEMPLATE",
        )


class LANDFIRECanopyAdapter:
    def resolve_sources(self, bbox: BoundingBox) -> list[SourceAsset]:
        return _resolve_landfire_layer(
            bbox=bbox,
            layer_key="canopy",
            name="LANDFIRE Canopy",
            default_url_env="WF_LANDFIRE_CANOPY_URL",
            template_env="WF_LANDFIRE_CANOPY_URL_TEMPLATE",
        )


def _resolve_landfire_layer(
    *,
    bbox: BoundingBox,
    layer_key: str,
    name: str,
    default_url_env: str,
    template_env: str,
) -> list[SourceAsset]:
    default_url = os.getenv(default_url_env, "")
    template = os.getenv(template_env, "")
    url = default_url
    if not url and template:
        url = template.format(
            min_lon=bbox["min_lon"],
            min_lat=bbox["min_lat"],
            max_lon=bbox["max_lon"],
            max_lat=bbox["max_lat"],
            bbox=f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        )
    if not url:
        return []
    return [
        SourceAsset(
            url=url,
            dataset_name=name,
            dataset_version=os.getenv("WF_LANDFIRE_VERSION"),
            dataset_provider="LANDFIRE",
            layer_key=layer_key,
            layer_type="raster",
            expected_format="tif",
            discovery_notes="LANDFIRE auto-discovery is template-based in this pilot.",
        )
    ]


def discover_wildfire_sources(bbox: BoundingBox) -> dict[str, list[SourceAsset]]:
    adapters: list[SourceAdapter] = [
        USGS3DEPAdapter(),
        NIFCFirePerimeterAdapter(),
        OvertureBuildingFootprintAdapter(),
        MicrosoftBuildingFootprintAdapter(),
        LANDFIREFuelAdapter(),
        LANDFIRECanopyAdapter(),
    ]
    discovered: dict[str, list[SourceAsset]] = {}
    for adapter in adapters:
        try:
            assets = adapter.resolve_sources(bbox)
        except Exception:
            assets = []
        for asset in assets:
            discovered.setdefault(asset.layer_key, []).append(asset)
    return discovered
