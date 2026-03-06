from __future__ import annotations

import os
from typing import Set

from fastapi import Header, HTTPException, status


def _load_api_keys() -> Set[str]:
    raw = os.getenv("WILDFIRE_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


API_KEYS = _load_api_keys()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # Local dev can run without auth if no keys are configured.
    if not API_KEYS:
        return
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
