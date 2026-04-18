"""Structure classification: primary, accessory, and neighbouring structures.

Given a set of building footprints within a property's ring zones, this module
classifies each footprint relative to the subject (primary) structure:

* **primary** — the matched subject structure (passed through unchanged).
* **accessory** — on-parcel footprints that are NOT the primary structure.
  Sub-classified by area thresholds into shed / garage / adu / barn / unknown.
* **neighbors** — off-parcel footprints within ring zones.

When no parcel polygon is available, all non-subject footprints are labeled as
``neighbors`` and ``classification_basis`` is set to ``"heuristic"``.

Accepts Shapely geometry objects or raw GeoJSON geometry dicts interchangeably.
Raw dicts are converted internally via ``shapely.geometry.shape``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger("wildfire_app.structure_classifier")

try:
    from pyproj import Transformer
    from shapely.geometry import shape as _shape
    from shapely.ops import transform as _transform
    _GEO_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback
    Transformer = None  # type: ignore[misc,assignment]
    _shape = None  # type: ignore[assignment]
    _transform = None  # type: ignore[assignment]
    _GEO_AVAILABLE = False

# Area thresholds (m²) for accessory structure type inference.
_SHED_MAX_M2 = 30.0
_GARAGE_MAX_M2 = 80.0
_ADU_MIN_M2 = 80.0
_BARN_MIN_M2 = 200.0


@dataclass
class AccessoryDetail:
    """Classification detail for a single accessory structure.

    Attributes
    ----------
    geometry:
        Shapely geometry of the accessory structure.
    area_m2:
        Floor-area proxy in square metres (computed in EPSG:3857).
    inferred_type:
        Inferred accessory type — one of ``"shed"``, ``"garage"``,
        ``"adu"``, ``"barn"``, or ``"unknown"``.
    distance_from_primary_m:
        Edge-to-edge distance from the primary structure in metres.
    """

    geometry: Any
    area_m2: float
    inferred_type: str
    distance_from_primary_m: float


@dataclass
class StructureClassification:
    """Result of classifying structures on and around a property.

    Attributes
    ----------
    primary:
        The subject (matched) structure geometry — passed through unchanged.
    accessory:
        On-parcel structures that are NOT the primary structure (Shapely geoms).
    neighbors:
        Off-parcel structures within ring zones (Shapely geoms).
    classification_basis:
        ``"parcel_derived"`` when a parcel polygon was used to split on/off-parcel,
        ``"heuristic"`` when no parcel was available.
    on_parcel_count:
        Total number of on-parcel structures (primary + accessory).
    off_parcel_count:
        Number of neighboring (off-parcel) structures.
    accessory_details:
        Typed detail records for each accessory structure, ordered by distance
        from the primary structure (nearest first).
    """

    primary: Any
    accessory: list = field(default_factory=list)
    neighbors: list = field(default_factory=list)
    classification_basis: str = "heuristic"
    on_parcel_count: int = 0
    off_parcel_count: int = 0
    accessory_details: list[AccessoryDetail] = field(default_factory=list)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_shapely(geom: Any) -> Any | None:
    """Return a Shapely geometry from a Shapely object or a GeoJSON dict."""
    if geom is None:
        return None
    if isinstance(geom, dict):
        if _shape is None:
            return None
        try:
            return _shape(geom)
        except Exception:
            return None
    # Assume it is already a Shapely geometry.
    return geom


def _area_m2(geom: Any) -> float:
    """Return the area of *geom* in square metres (EPSG:3857 projection)."""
    if not _GEO_AVAILABLE or Transformer is None or _transform is None:
        return 0.0
    try:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        return float(max(0.0, _transform(to_3857, geom).area))
    except Exception:
        return 0.0


def _distance_m(geom_a: Any, geom_b: Any) -> float:
    """Edge-to-edge distance between two geometries in metres."""
    if not _GEO_AVAILABLE or Transformer is None or _transform is None:
        return 0.0
    try:
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        a_m = _transform(to_3857, geom_a)
        b_m = _transform(to_3857, geom_b)
        return float(max(0.0, a_m.distance(b_m)))
    except Exception:
        return 0.0


def _infer_accessory_type(area_m2: float, building_class: str | None) -> str:
    """Infer the type of an accessory structure from its area and optional class.

    Parameters
    ----------
    area_m2:
        Footprint area in square metres.
    building_class:
        Overture/source building class string, or ``None`` if unavailable.

    Returns
    -------
    One of ``"shed"``, ``"garage"``, ``"adu"``, ``"barn"``, ``"unknown"``.
    """
    cls = str(building_class or "").strip().lower()

    if area_m2 < _SHED_MAX_M2:
        return "shed"
    if area_m2 < _GARAGE_MAX_M2:
        return "garage"
    # area_m2 >= 80
    if area_m2 >= _BARN_MIN_M2 and cls and cls not in {
        "residential", "house", "detached", "apartments", "adu"
    }:
        return "barn"
    if area_m2 >= _ADU_MIN_M2 and cls in {"residential", "house", "detached", "apartments", "adu", ""}:
        return "adu"
    return "unknown"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def classify_structures(
    parcel_polygon: Any,
    all_footprints: list,
    subject_footprint: Any,
) -> StructureClassification:
    """Classify building footprints relative to the primary subject structure.

    Parameters
    ----------
    parcel_polygon:
        Shapely Polygon / MultiPolygon representing the matched parcel, or
        ``None`` if no parcel data is available.
    all_footprints:
        List of Shapely geometries or GeoJSON geometry dicts within ring zones.
        May include the subject footprint — it is identified and excluded from
        the accessory/neighbor lists.
    subject_footprint:
        The primary (matched) structure geometry.  Passed through as
        ``StructureClassification.primary``.

    Returns
    -------
    :class:`StructureClassification` with primary, accessory, and neighbor lists
    populated, plus typed :class:`AccessoryDetail` records for accessory structures.
    """
    subject_geom = _to_shapely(subject_footprint)
    parcel_geom = _to_shapely(parcel_polygon)

    # No parcel → heuristic classification: all non-primary are neighbors.
    if parcel_geom is None or getattr(parcel_geom, "is_empty", True):
        neighbors = []
        for fp in all_footprints:
            geom = _to_shapely(fp)
            if geom is None or geom.is_empty:
                continue
            if subject_geom is not None and not geom.is_empty:
                try:
                    if _distance_m(geom, subject_geom) < 0.5:
                        continue  # skip the subject itself
                except Exception:
                    pass
            neighbors.append(geom)
        return StructureClassification(
            primary=subject_footprint,
            accessory=[],
            neighbors=neighbors,
            classification_basis="heuristic",
            on_parcel_count=1 if subject_geom is not None else 0,
            off_parcel_count=len(neighbors),
            accessory_details=[],
        )

    # Parcel available → classify on-parcel vs off-parcel.
    on_parcel: list[Any] = []
    off_parcel: list[Any] = []
    for fp in all_footprints:
        geom = _to_shapely(fp)
        if geom is None or geom.is_empty:
            continue
        try:
            on = bool(parcel_geom.intersects(geom))
        except Exception:
            on = False
        if on:
            on_parcel.append(geom)
        else:
            off_parcel.append(geom)

    # Identify subject within on_parcel (exclude it from accessory list).
    accessory_geoms: list[Any] = []
    for geom in on_parcel:
        if subject_geom is not None:
            try:
                if _distance_m(geom, subject_geom) < 0.5:
                    continue  # this is the primary
            except Exception:
                pass
        accessory_geoms.append(geom)

    # Build typed detail records for each accessory.
    accessory_details: list[AccessoryDetail] = []
    for geom in accessory_geoms:
        area = _area_m2(geom)
        dist = _distance_m(geom, subject_geom) if subject_geom is not None else 0.0
        # Extract building_class from properties if geom carries it (e.g. from
        # a raw dict that was passed in and then shapely-converted).  Since we
        # already converted to shapely, the class is not accessible here.
        # Callers that want class-aware typing should pass structured dicts and
        # pre-extract the class before calling this function.
        inferred = _infer_accessory_type(area, building_class=None)
        accessory_details.append(
            AccessoryDetail(
                geometry=geom,
                area_m2=round(area, 2),
                inferred_type=inferred,
                distance_from_primary_m=round(dist, 2),
            )
        )

    # Sort accessory details nearest-first.
    accessory_details.sort(key=lambda d: d.distance_from_primary_m)

    on_parcel_count = 1 + len(accessory_geoms)  # primary counts as 1
    return StructureClassification(
        primary=subject_footprint,
        accessory=accessory_geoms,
        neighbors=off_parcel,
        classification_basis="parcel_derived",
        on_parcel_count=on_parcel_count,
        off_parcel_count=len(off_parcel),
        accessory_details=accessory_details,
    )
