from __future__ import annotations

from backend import open_data_adapters as oda


class _FakeAxis:
    def __init__(self, name: str, direction: str) -> None:
        self.name = name
        self.direction = direction
        self.unit_name = "degree"


class _FakeCRS:
    is_geographic = True
    axis_info = [_FakeAxis("Geodetic latitude", "north"), _FakeAxis("Geodetic longitude", "east")]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "EPSG:4326"


class _FakeDataset:
    def __init__(self) -> None:
        self.crs = _FakeCRS()
        # Chosen so default (lon, lat) is out-of-bounds while swapped (lat, lon) is in-bounds.
        self.bounds = (0.0, -200.0, 100.0, -50.0)
        self.nodata = None
        self.sample_calls: list[tuple[float, float]] = []

    def sample(self, coords):
        for coord in coords:
            self.sample_calls.append((float(coord[0]), float(coord[1])))
            yield [12.5]


def test_sample_raster_point_detailed_swaps_when_lat_first_axis(monkeypatch):
    ds = _FakeDataset()
    monkeypatch.setattr(oda, "_file_exists", lambda _path: True)
    monkeypatch.setattr(oda, "_open_raster", lambda _path: ds)

    result = oda._sample_raster_point_detailed("fake.tif", lat=46.87, lon=-114.01)

    assert result["status"] == "ok"
    assert result["sample_coords"] == (46.87, -114.01)
    assert ds.sample_calls[0] == (46.87, -114.01)


def test_whp_adapter_returns_error_on_sampling_exception(monkeypatch):
    monkeypatch.setattr(oda, "_file_exists", lambda _path: True)
    monkeypatch.setattr(
        oda,
        "_sample_raster_point_detailed",
        lambda _path, _lat, _lon: {
            "status": "error",
            "reason": "forced test failure",
            "value": None,
            "axis_info": [],
            "sample_coords": None,
        },
    )

    obs = oda.WHPAdapter().sample(lat=46.87, lon=-114.01, whp_path="fake.tif")
    assert obs.status == "error"
    assert "forced test failure" in " ".join(obs.notes)


def test_gridmet_adapter_returns_error_on_sampling_exception(monkeypatch):
    monkeypatch.setattr(oda, "_file_exists", lambda _path: True)
    monkeypatch.setattr(
        oda,
        "_sample_raster_circle_detailed",
        lambda _path, _lat, _lon, radius_m, step_m=120.0: {
            "status": "error",
            "reason": "forced gridmet failure",
            "values": [],
            "axis_info": [],
            "sample_coords": None,
        },
    )

    obs = oda.GridMETAdapter().sample_dryness(lat=46.87, lon=-114.01, dryness_raster_path="fake.tif")
    assert obs.status == "error"
    assert "forced gridmet failure" in " ".join(obs.notes)
