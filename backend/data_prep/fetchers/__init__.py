from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.data_prep.fetchers.common import BoundingBox, ParcelFetchResult
from backend.data_prep.fetchers.regrid_fetcher import RegridParcelFetcher

if TYPE_CHECKING:  # pragma: no cover
    from backend.data_prep.fetchers.overture_fetcher import OvertureParcelFetcher

__all__ = [
    "BoundingBox",
    "ParcelFetchResult",
    "RegridParcelFetcher",
    "OvertureParcelFetcher",
]


def __getattr__(name: str) -> Any:
    if name == "OvertureParcelFetcher":
        from backend.data_prep.fetchers.overture_fetcher import OvertureParcelFetcher

        return OvertureParcelFetcher
    raise AttributeError(name)
