from __future__ import annotations

# Source-backed core layers required to build a usable prepared region.
REQUIRED_CORE_RASTER_LAYERS = ("dem", "fuel", "canopy")
REQUIRED_CORE_VECTOR_LAYERS = ("fire_perimeters", "building_footprints")

# Derived layers produced during region assembly.
DERIVED_RASTER_LAYERS = ("slope",)

# Optional enrichment layers; absence should warn, not hard-fail.
OPTIONAL_LAYERS = (
    "building_footprints_overture",
    "roads",
    "whp",
    "mtbs_severity",
    "gridmet_dryness",
    "parcel_polygons",
    "parcel_address_points",
)
