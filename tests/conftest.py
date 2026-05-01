from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """
    Reset the in-memory rate limiter storage before every test.

    Root cause: the default rate limit for /risk/assess is 10/minute and
    100/day.  The full test suite makes 140+ calls to that endpoint across
    many test files.  Without resetting between tests, the daily counter
    accumulates and eventually starts returning 429 for tests beyond call
    #100, causing spurious assertion failures (status_code 429 != 200).

    This fixture resets the counter before each test so every test starts
    with a clean slate.  test_rate_limiting.py also calls
    app_main.limiter.limiter.storage.reset() inside its own fixture, so
    this autouse fixture is harmlessly redundant for those tests — they
    continue to work correctly because they set their own rate limit env
    vars and reset the storage themselves.
    """
    try:
        import backend.main as app_main
        app_main.limiter.limiter.storage.reset()
    except Exception:
        # If the app hasn't been imported or the limiter isn't available,
        # skip silently — this fixture is best-effort.
        pass
    yield
