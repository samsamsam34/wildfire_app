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
