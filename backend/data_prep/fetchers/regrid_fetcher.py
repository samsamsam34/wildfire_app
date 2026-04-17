from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from backend.data_prep.fetchers.common import (
    BoundingBox,
    ParcelFetchResult,
    clip_and_filter_polygon_features,
    normalize_bounds,
    normalize_parcel_feature,
    write_parcel_geojson,
)


class RegridParcelFetcher:
    """Fetch parcel polygons from Regrid /api/v1/parcels with pagination."""

    DEFAULT_BASE_URL = "https://app.regrid.com/api/v1/parcels"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_key = str(api_key or os.getenv("WF_REGRID_API_KEY") or "").strip()
        self.base_url = str(
            base_url
            or os.getenv("WF_REGRID_PARCELS_ENDPOINT")
            or self.DEFAULT_BASE_URL
        ).strip()
        self.timeout_seconds = float(timeout_seconds)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = (
            payload.get("results"),
            payload.get("data"),
            payload.get("features"),
            ((payload.get("parcels") or {}).get("results") if isinstance(payload.get("parcels"), dict) else None),
        )
        for row in candidates:
            if isinstance(row, list):
                return [item for item in row if isinstance(item, dict)]
        return []

    @staticmethod
    def _next_token(payload: dict[str, Any]) -> str | None:
        next_cursor = payload.get("next_cursor") or payload.get("nextCursor")
        if next_cursor is not None and str(next_cursor).strip():
            return str(next_cursor).strip()
        pagination = payload.get("pagination")
        if isinstance(pagination, dict):
            cursor = pagination.get("next_cursor") or pagination.get("next")
            if cursor is not None and str(cursor).strip():
                return str(cursor).strip()
        links = payload.get("links")
        if isinstance(links, dict) and links.get("next"):
            return str(links.get("next")).strip()
        return None

    @staticmethod
    def _row_to_feature(row: dict[str, Any]) -> dict[str, Any] | None:
        geometry = row.get("geometry")
        props = row.get("properties")
        if geometry is None and isinstance(row.get("feature"), dict):
            geometry = (row.get("feature") or {}).get("geometry")
            props = (row.get("feature") or {}).get("properties")
        if not isinstance(geometry, dict):
            return None
        if str(geometry.get("type") or "") not in {"Polygon", "MultiPolygon"}:
            return None
        properties = dict(props or {})
        for key in ("id", "parcel_id", "parcelid", "apn"):
            value = row.get(key)
            if value is not None and str(value).strip() and key not in properties:
                properties[key] = value
        return {
            "type": "Feature",
            "properties": properties,
            "geometry": geometry,
        }

    def fetch(
        self,
        *,
        bounds: BoundingBox,
        region_dir: Path,
        page_size: int = 500,
        max_pages: int = 200,
    ) -> ParcelFetchResult:
        if not self.api_key:
            return ParcelFetchResult(
                source_id="regrid",
                success=False,
                message="WF_REGRID_API_KEY is not set.",
            )
        if not self.base_url:
            return ParcelFetchResult(
                source_id="regrid",
                success=False,
                message="Regrid endpoint is not configured.",
            )

        normalized_bounds = normalize_bounds(bounds)
        warnings: list[str] = []
        all_rows: list[dict[str, Any]] = []
        cursor: str | None = None
        fetched_pages = 0
        last_error: str | None = None

        for page_index in range(max(1, int(max_pages))):
            params = {
                "bbox": (
                    f"{normalized_bounds['min_lon']},"
                    f"{normalized_bounds['min_lat']},"
                    f"{normalized_bounds['max_lon']},"
                    f"{normalized_bounds['max_lat']}"
                ),
                "limit": str(max(1, int(page_size))),
            }
            if cursor:
                params["cursor"] = cursor
            request_url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            fetched_pages = page_index + 1
            try:
                req = urllib.request.Request(request_url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last_error = str(exc)
                break

            rows = self._extract_rows(payload if isinstance(payload, dict) else {})
            all_rows.extend(rows)

            next_token = self._next_token(payload if isinstance(payload, dict) else {})
            if not next_token:
                if len(rows) < int(page_size):
                    break
                cursor = str(page_index + 2)
            else:
                cursor = next_token

            if len(rows) == 0:
                break

        raw_features = [feature for row in all_rows if (feature := self._row_to_feature(row)) is not None]
        clipped = clip_and_filter_polygon_features(features=raw_features, bounds=normalized_bounds)
        normalized_features = [
            normalize_parcel_feature(
                geometry=dict(feature.get("geometry") or {}),
                properties=dict(feature.get("properties") or {}),
                source="regrid",
            )
            for feature in clipped
        ]

        if not normalized_features:
            detail = f"; last_error={last_error}" if last_error else ""
            return ParcelFetchResult(
                source_id="regrid",
                success=False,
                record_count=0,
                warnings=warnings,
                message=f"Regrid returned no parcel polygons for bbox{detail}",
                diagnostics={
                    "pages_fetched": fetched_pages,
                    "rows_seen": len(all_rows),
                },
            )

        output_path = write_parcel_geojson(
            features=normalized_features,
            region_dir=region_dir,
            source_id="regrid",
            bounds=normalized_bounds,
        )
        return ParcelFetchResult(
            source_id="regrid",
            success=True,
            output_path=str(output_path),
            record_count=len(normalized_features),
            warnings=warnings,
            diagnostics={
                "pages_fetched": fetched_pages,
                "rows_seen": len(all_rows),
            },
        )
