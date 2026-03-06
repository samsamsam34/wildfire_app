from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

from backend.models import PropertyAttributes, RiskDrivers
from backend.wildfire_data import WildfireContext


@dataclass
class RiskComputation:
    total_score: float
    drivers: RiskDrivers
    assumptions: List[str]


class RiskEngine:
    def geocode_stub(self, address: str) -> Tuple[float, float]:
        digest = hashlib.md5(address.strip().lower().encode("utf-8")).hexdigest()
        lat_seed = int(digest[:8], 16) / 0xFFFFFFFF
        lon_seed = int(digest[8:16], 16) / 0xFFFFFFFF
        latitude = 32.5 + (lat_seed * 9.5)
        longitude = -124.3 + (lon_seed * 10.0)
        return round(latitude, 6), round(longitude, 6)

    def score(
        self,
        attrs: PropertyAttributes,
        lat: float,
        lon: float,
        context: WildfireContext,
    ) -> RiskComputation:
        assumptions: List[str] = list(context.assumptions)

        environmental = context.environmental_index

        structural = 35.0
        if attrs.roof_type is None:
            assumptions.append("Roof type missing; assumed composition shingle baseline.")
        elif attrs.roof_type.lower() in {"wood", "untreated wood shake"}:
            structural += 30
        elif attrs.roof_type.lower() in {"class a", "metal", "tile", "composite"}:
            structural -= 15

        if attrs.vent_type is None:
            assumptions.append("Vent type missing; assumed standard non-ember-resistant vents.")
        elif "ember" in attrs.vent_type.lower():
            structural -= 10
        else:
            structural += 8

        if attrs.defensible_space_ft is None:
            assumptions.append("Defensible space unknown; assumed 15 ft effective clearance.")
            defensible_space_ft = 15
        else:
            defensible_space_ft = attrs.defensible_space_ft

        if defensible_space_ft < 5:
            structural += 20
        elif defensible_space_ft < 30:
            structural += 8
        else:
            structural -= 10

        if attrs.construction_year is None:
            assumptions.append("Construction year missing; assumed pre-2008 vent/siding standards.")
        elif attrs.construction_year >= 2015:
            structural -= 8

        structural = max(0, min(100, structural))

        # Access/exposure now depends on layer-derived proximity and recurrence, not coordinate hash proxies.
        access_exposure = round(
            0.65 * context.wildland_distance_index + 0.35 * context.historic_fire_index,
            1,
        )

        total = 0.55 * environmental + 0.30 * structural + 0.15 * access_exposure

        return RiskComputation(
            total_score=round(total, 1),
            drivers=RiskDrivers(
                environmental=round(environmental, 1),
                structural=round(structural, 1),
                access_exposure=round(access_exposure, 1),
            ),
            assumptions=assumptions,
        )
