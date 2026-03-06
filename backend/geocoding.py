from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Tuple


class Geocoder:
    def __init__(self, user_agent: str = "WildfireRiskAdvisor/0.1") -> None:
        self.user_agent = user_agent

    def geocode(self, address: str) -> Tuple[float, float, str]:
        query = urllib.parse.urlencode({"q": address, "format": "json", "limit": 1})
        url = f"https://nominatim.openstreetmap.org/search?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not payload:
            raise ValueError("No geocoding result")

        lat = float(payload[0]["lat"])
        lon = float(payload[0]["lon"])
        return lat, lon, "OpenStreetMap Nominatim"
