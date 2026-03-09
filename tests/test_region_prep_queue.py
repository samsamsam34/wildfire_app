from __future__ import annotations

import argparse
from pathlib import Path

from fastapi.testclient import TestClient

import backend.auth as auth
import backend.main as app_main
from backend.database import AssessmentStore
from backend.wildfire_data import WildfireContext
from scripts import run_region_prep_worker


client = TestClient(app_main.app)


def _context() -> WildfireContext:
    return WildfireContext(
        environmental_index=55.0,
        slope_index=55.0,
        aspect_index=50.0,
        fuel_index=60.0,
        moisture_index=52.0,
        canopy_index=58.0,
        wildland_distance_index=40.0,
        historic_fire_index=30.0,
        burn_probability_index=56.0,
        hazard_severity_index=54.0,
        burn_probability=56.0,
        wildfire_hazard=54.0,
        slope=15.0,
        fuel_model=45.0,
        canopy_cover=52.0,
        historic_fire_distance=2.0,
        wildland_distance=120.0,
        environmental_layer_status={
            "burn_probability": "ok",
            "hazard": "ok",
            "slope": "ok",
            "fuel": "ok",
            "canopy": "ok",
            "fire_history": "ok",
        },
        data_sources=["test"],
        assumptions=[],
        structure_ring_metrics={},
        property_level_context={
            "footprint_used": False,
            "footprint_status": "not_found",
            "fallback_mode": "point_based",
            "ring_metrics": None,
        },
    )


def _payload() -> dict:
    return {
        "address": "123 Test Ave, Bozeman, MT",
        "attributes": {"roof_type": "wood_shake", "vent_type": "standard"},
        "confirmed_fields": ["roof_type"],
        "audience": "homeowner",
        "tags": [],
    }


def test_uncovered_assess_enqueues_region_prep_job(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "queue.db")))
    monkeypatch.setenv("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", "true")
    monkeypatch.setenv("WF_REGION_DATA_DIR", str(tmp_path / "regions"))
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {
            "covered": False,
            "diagnostics": ["No prepared region bounds contain point."],
        },
    )

    response = client.post("/risk/assess", json=_payload())
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["region_not_ready"] is True
    assert detail["prep_job_id"]
    assert detail["prep_job_status"] in {"queued", "running", "completed"}
    assert "requested_bbox" in detail

    jobs = app_main.store.list_region_prep_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0]["requested_address"] == _payload()["address"]


def test_covered_assess_does_not_enqueue_region_prep_job(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "covered.db")))
    monkeypatch.setenv("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", "true")
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (45.67, -111.04, "test-geocoder"))
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {"covered": True, "region_id": "prepared_demo"},
    )
    monkeypatch.setattr(app_main.wildfire_data, "collect_context", lambda _lat, _lon: _context())

    response = client.post("/risk/assess", json=_payload())
    assert response.status_code == 200
    jobs = app_main.store.list_region_prep_jobs(limit=10)
    assert jobs == []


def test_regions_prepare_endpoint_dedupes_jobs(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "dedupe.db")))

    req = {
        "region_id": "bozeman_auto",
        "display_name": "Bozeman Auto",
        "bbox": {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8},
        "validate": True,
    }
    first = client.post("/regions/prepare", json=req)
    second = client.post("/regions/prepare", json=req)
    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["job_id"] == second_body["job_id"]
    assert first_body["reused_existing_job"] is False
    assert second_body["reused_existing_job"] is True

    loaded = client.get(f"/regions/prepare/{first_body['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["job_id"] == first_body["job_id"]


def test_regions_coverage_check_endpoint_reports_uncovered(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "coverage.db")))
    monkeypatch.setenv("WF_REGION_DATA_DIR", str(tmp_path / "regions"))
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {"covered": False, "diagnostics": ["No covering region"]},
    )

    response = client.post("/regions/coverage-check", json={"address": "123 Test Ave, Denver, CO"})
    assert response.status_code == 200
    body = response.json()
    assert body["covered"] is False
    assert body["coverage_available"] is False
    assert body["resolved_region_id"] is None
    assert body["reason"] == "no_prepared_region_for_location"
    assert body["recommended_action"]
    assert body["message"]
    assert body["diagnostics"]


def test_assess_requires_prepared_region_when_enabled(monkeypatch, tmp_path: Path) -> None:
    auth.API_KEYS = set()
    monkeypatch.setattr(app_main, "store", AssessmentStore(str(tmp_path / "require_prepared.db")))
    monkeypatch.setenv("WF_REQUIRE_PREPARED_REGION_COVERAGE", "true")
    monkeypatch.setenv("WF_AUTO_QUEUE_REGION_PREP_ON_MISS", "false")
    monkeypatch.setattr(app_main.geocoder, "geocode", lambda _addr: (39.7392, -104.9903, "test-geocoder"))
    monkeypatch.setattr(
        app_main,
        "lookup_region_for_point",
        lambda lat, lon, regions_root=None: {"covered": False, "diagnostics": ["No prepared region"]},
    )

    response = client.post("/risk/assess", json=_payload())
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["region_not_ready"] is True
    assert detail["coverage_available"] is False
    assert detail["resolved_region_id"] is None
    assert detail["reason"] == "no_prepared_region_for_location"
    assert detail["recommended_action"]
    assert detail["prep_job_id"] is None
    assert detail["requested_bbox"]
    assert "offline region-prep" in detail["message"].lower()
    assert app_main.store.list_region_prep_jobs(limit=10) == []


def test_region_prep_worker_processes_queued_job(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "worker.db"
    store = AssessmentStore(str(db_path))
    bbox = {"min_lon": -111.2, "min_lat": 45.5, "max_lon": -110.9, "max_lat": 45.8}
    request_payload = {
        "region_id": "worker_region",
        "display_name": "Worker Region",
        "bbox": bbox,
        "validate": False,
        "skip_optional_layers": True,
    }
    created, reused = store.create_or_get_region_prep_job(
        region_id="worker_region",
        display_name="Worker Region",
        requested_bbox=bbox,
        request_payload=request_payload,
    )
    assert reused is False

    manifest_path = str(tmp_path / "regions" / "worker_region" / "manifest.json")

    def _fake_prepare(**kwargs):
        return {
            "mode": "executed",
            "region_id": kwargs["region_id"],
            "manifest_path": manifest_path,
        }

    monkeypatch.setattr(run_region_prep_worker, "prepare_region_from_catalog_or_sources", _fake_prepare)
    args = argparse.Namespace(
        db_path=str(db_path),
        poll_interval=0.01,
        once=True,
        max_jobs=1,
        catalog_root=None,
        regions_root=None,
        cache_root=None,
        source_config=None,
        target_resolution=None,
        download_timeout=60.0,
        download_retries=2,
        retry_backoff_seconds=1.5,
    )

    code = run_region_prep_worker.run_worker(args)
    assert code == 0
    updated = store.get_region_prep_job(created["job_id"])
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["manifest_path"] == manifest_path
