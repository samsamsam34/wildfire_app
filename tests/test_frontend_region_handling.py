from __future__ import annotations

from pathlib import Path


def _frontend_html() -> str:
    path = Path("frontend/public/index.html")
    return path.read_text(encoding="utf-8")


def test_frontend_has_uncovered_location_renderer() -> None:
    html = _frontend_html()
    assert "function renderUncoveredLocationState(detail)" in html
    assert "Location not yet prepared" in html
    assert "no_prepared_region_for_location" in html


def test_frontend_handles_structured_region_not_ready_errors() -> None:
    html = _frontend_html()
    assert "detail.region_not_ready === true" in html
    assert "detail.coverage_available === false" in html
    assert "fetchCoverageForAddress(" in html
    assert "renderUncoveredLocationState(enriched);" in html


def test_frontend_does_not_require_manual_region_selection() -> None:
    html = _frontend_html()
    assert "/risk/assess" in html
    assert "fetchCoverageForAddress(" in html
    assert "resolved_region_id" in html
    assert 'id="region_id"' not in html
    assert "outside the currently prepared region set" in html


def test_frontend_trims_address_and_probes_coverage_on_geocode_failures() -> None:
    html = _frontend_html()
    assert 'address: String(document.getElementById("address").value || "").trim()' in html
    assert "fetchCoverageForAddress(document.getElementById(\"address\")?.value || \"\")" in html
    assert "String(coverage.geocode_status || \"\").toLowerCase() === \"accepted\"" in html


def test_frontend_region_debug_metadata_is_dev_mode_only() -> None:
    html = _frontend_html()
    assert "window.WILDFIRE_DEBUG_MODE" in html
    assert 'id="regionDebugInline"' in html
    assert "Region: unsupported_location" in html
    assert "Region: ${regionText}" in html
    assert 'id="regionResolutionText"' in html
    assert 'id="geocodeDebugText"' in html
    assert "Geocode: status=" in html
    assert 'style="display:none;"' in html


def test_frontend_renders_geocode_failure_debug_state_in_dev_mode() -> None:
    html = _frontend_html()
    assert "function renderGeocodeFailureState(detail, submittedAddress)" in html
    assert 'detail.error === "geocoding_failed"' in html
    assert "Region: unresolved" in html


def test_frontend_surfaces_defensible_space_zone_outputs() -> None:
    html = _frontend_html()
    assert "top_near_structure_risk_drivers" in html
    assert "prioritized_vegetation_actions" in html
    assert "defensible_space_analysis" in html
    assert "defensible_space_limitations_summary" in html


def test_frontend_includes_assessment_map_panel_and_layer_controls() -> None:
    html = _frontend_html()
    assert 'id="assessmentMap"' in html
    assert 'id="mapLayerToggles"' in html
    assert 'id="mapLimitationsText"' in html
    assert 'id="mapDebugPanel"' in html
    assert "function renderAssessmentMap(assessmentId)" in html
    assert "/report/${assessmentId}/map" in html
    assert "defensible_space_rings" in html
    assert "geocoded_address_point" in html
    assert "matched_structure_centroid" in html
    assert "display_point_source" in html
    assert "structure_match_status" in html
    assert "geocode_precision" in html
    assert "candidate_summaries" in html
    assert "property_anchor_point" in html
    assert "parcel_polygon" in html
    assert "selectable_structure_footprints" in html
    assert "auto_detected_structure" in html
    assert "user_selected_structure" in html


def test_frontend_includes_home_confirmation_step_and_selection_payload_fields() -> None:
    html = _frontend_html()
    assert "We detected this building as your home. Is this correct?" in html
    assert "Click one of the highlighted building polygons to select your home." in html
    assert "Yes — Continue" in html
    assert "No — Select My Home" in html
    assert "Can’t find your house? Click directly on your home" in html
    assert "structure_geometry_source" in html
    assert "selection_mode" in html
    assert "user_selected_point" in html
    assert "selected_structure_id" in html
    assert "selected_structure_geometry" in html
    assert "Basemap outlines are not clickable" in html
    assert 'createPane("selectableStructuresPane")' in html
    assert 'pane: "selectableStructuresPane"' in html
    assert "interactive_layer_loaded" in html


def test_frontend_selectable_structure_layer_has_click_and_selection_logic() -> None:
    html = _frontend_html()
    assert 'layerKey === "selectable_structure_footprints"' in html
    assert "const isSelected = !!selectedStructureId" in html
    assert 'layer.on("click", async () => {' in html
    assert "selectedStructureId = structureId;" in html
    assert 'source: "user_selected"' in html
    assert "await runAssessment(false);" in html
    assert "Selection mode is on. Click one of the highlighted building polygons to select your home." in html


def test_frontend_point_selection_fallback_places_user_selected_point_and_reassesses() -> None:
    html = _frontend_html()
    assert 'mode: "point"' in html
    assert "Point selection mode enabled. Click directly on your home location in the map." in html
    assert "Using your selected map point. Updating assessment..." in html
    assert "renderManualSelectedPointMarker" in html
    assert 'source: "user_selected_point"' in html


def test_frontend_map_degrades_gracefully_for_uncovered_or_geocode_failure() -> None:
    html = _frontend_html()
    assert "clearMapLayers();" in html
    assert "setMapStatus(\"Map unavailable until this location has prepared-region coverage.\");" in html
    assert "setMapStatus(\"Map unavailable until geocoding succeeds.\");" in html
    assert "renderMapDebugPanel(null);" in html


def test_frontend_map_uses_geojson_directly_without_axis_reordering() -> None:
    html = _frontend_html()
    assert "L.geoJSON(fc" in html
    assert "pointToLayer: (_feature, latlng) => L.circleMarker(latlng" in html
    assert "coordinates[0]" not in html
    assert "coordinates[1]" not in html
