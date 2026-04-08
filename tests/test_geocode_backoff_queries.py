from __future__ import annotations

from backend import main as app_main


def test_backoff_queries_include_unit_stripped_and_house_stripped_variants(monkeypatch) -> None:
    monkeypatch.setenv("WF_GEOCODE_BACKOFF_QUERY_LIMIT", "8")
    queries = app_main._build_provider_backoff_queries("12 Twin Lakes Rd Apt 3, Winthrop, WA 98862")
    assert queries
    assert queries[0] == "12 Twin Lakes Rd Apt 3, Winthrop, WA 98862"
    assert any("apt 3" not in query.lower() and "twin lakes rd, winthrop, wa 98862" in query.lower() for query in queries)
    assert any(query.lower().startswith("twin lakes rd") for query in queries)


def test_backoff_queries_strip_suite_and_hash_noise(monkeypatch) -> None:
    monkeypatch.setenv("WF_GEOCODE_BACKOFF_QUERY_LIMIT", "8")
    queries = app_main._build_provider_backoff_queries("44 Main St Ste 200 #12, Winthrop, WA 98862")
    assert queries
    assert any("ste 200" not in query.lower() for query in queries[1:])
    assert any("#12" not in query for query in queries[1:])
