from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Set

from fastapi import Header, HTTPException, status

_AUTH_LOGGER = logging.getLogger("wildfire_app.auth")


def _load_api_keys() -> Dict[str, Optional[str]]:
    """Parse WILDFIRE_API_KEYS into a mapping of key → org_id (or None).

    Supports two formats, freely mixed in the same env-var value:
      - ``key``               — key with no org restriction (any org allowed)
      - ``key:org_id``        — key bound to a specific organization

    Example::

        WILDFIRE_API_KEYS="dev-key-1,prod-key-2:acme_corp,admin-key-3:admin_org"
    """
    raw = os.getenv("WILDFIRE_API_KEYS", "")
    result: Dict[str, Optional[str]] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            key, _, org = token.partition(":")
            key = key.strip()
            org = org.strip()
            if key:
                result[key] = org or None
        else:
            result[token] = None
    return result


API_KEYS: Dict[str, Optional[str]] = _load_api_keys()


def get_key_org(api_key: str) -> Optional[str]:
    """Return the org_id bound to *api_key*, or None if the key is unscoped."""
    return API_KEYS.get(api_key)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Validate the X-Api-Key header.

    - If no keys are configured (WILDFIRE_API_KEYS is empty) the check is
      skipped so local dev works without configuration.
    - If keys are configured, the header must be present and match a known key.
    - Access denial is logged at WARNING level (stdout); the DB audit trail for
      role/org denials is handled separately in main.py.
    """
    if not API_KEYS:
        return  # Local dev: no keys configured → open access
    if not x_api_key or x_api_key not in API_KEYS:
        _AUTH_LOGGER.warning("api_key_rejected key_present=%s", bool(x_api_key))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
