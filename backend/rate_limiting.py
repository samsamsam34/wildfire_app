"""
Rate limiting configuration for the WildfireRisk Advisor API.

Limits are intentionally conservative for the assessment endpoints because
each assessment triggers multiple external API calls (Census, Regrid, LANDFIRE,
NLCD, MTBS). A single client hammering /assess could exhaust third-party
rate limits for all users.

Limits are per IP address and configurable via environment variables.
"""

from __future__ import annotations

import os
import uuid

from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limit strings follow the limits library format: "N/period"
# where period is second, minute, hour, day
DEFAULT_ASSESS_LIMIT = "10/minute"
DEFAULT_ASSESS_DAILY_LIMIT = "100/day"
DEFAULT_SIMULATE_LIMIT = "20/minute"
DEFAULT_GENERAL_LIMIT = "60/minute"


def get_assess_limit() -> str:
    return os.environ.get("WF_RATE_LIMIT_ASSESS", DEFAULT_ASSESS_LIMIT)


def get_assess_daily_limit() -> str:
    return os.environ.get("WF_RATE_LIMIT_ASSESS_DAILY", DEFAULT_ASSESS_DAILY_LIMIT)


def get_simulate_limit() -> str:
    return os.environ.get("WF_RATE_LIMIT_SIMULATE", DEFAULT_SIMULATE_LIMIT)


def get_assess_combined_limit() -> str:
    return f"{get_assess_limit()};{get_assess_daily_limit()}"


def get_storage_uri() -> str:
    """Rate limiter backend URI; defaults to in-memory storage."""
    return os.environ.get("WF_REDIS_URL", "memory://")


def _bypass_keys() -> set[str]:
    raw = str(os.environ.get("WF_RATE_LIMIT_BYPASS_KEYS", "") or "").strip()
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def should_bypass_rate_limit_header(request) -> bool:
    key = str(request.headers.get("X-API-Key", "") or "").strip()
    if not key:
        return False
    return key in _bypass_keys()


def rate_limit_key_func(request) -> str:
    """
    Resolve the rate-limit key for a request.

    Bypass keys are intentionally mapped to per-request unique buckets so they
    do not consume shared per-IP quota for consumer traffic.
    """
    if should_bypass_rate_limit_header(request):
        key = str(request.headers.get("X-API-Key", "") or "").strip()
        return f"bypass:{key}:{uuid.uuid4().hex}"
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key_func, storage_uri=get_storage_uri())
