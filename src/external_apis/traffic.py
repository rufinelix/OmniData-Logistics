"""External traffic and congestion API connector.

Connects to TomTom Routing & Traffic API to fetch real-time commercial truck congestion delays
and traffic severity indices. Gracefully falls back to heuristic corridor traffic estimations
when API credentials are not supplied or external service is unreachable.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class APIConfigurationError(Exception):
    """Raised when an external API credential is invalid or unconfigured."""


def _is_placeholder_key(val: str | None) -> bool:
    """Check if environment variable key is absent or contains placeholder string."""
    if not val:
        return True
    cleaned = val.strip().lower().strip("'\"")
    return not cleaned or "your_" in cleaned or "here" in cleaned or cleaned == "none"


def _estimate_fallback_traffic(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> dict[str, float]:
    """Calculate deterministic heuristic congestion delay and severity for fallback."""
    # Haversine distance
    r = 6371.0
    phi1, phi2 = np.radians(origin_lat), np.radians(dest_lat)
    dphi = np.radians(dest_lat - origin_lat)
    dlambda = np.radians(dest_lon - origin_lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    dist_km = float(r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)) * 1.25)

    # Heuristic: base truck speed 75 km/h, free-flow time in minutes
    free_flow_mins = (dist_km / 75.0) * 60.0

    # Deterministic pseudo-random variation based on coordinate coordinates hash
    coord_hash = abs(hash((round(origin_lat, 2), round(dest_lon, 2))))
    congestion_factor = 0.08 + (coord_hash % 25) / 100.0  # 8% to 32% congestion delay
    delay_minutes = round(free_flow_mins * congestion_factor, 1)

    # Traffic severity index scale (1.0 = normal flow, 1.5 = moderate delay, 2.0+ = severe congestion)
    severity_index = round(1.0 + (delay_minutes / (free_flow_mins + 1e-5)) * 2.5, 2)

    return {
        "traffic_delay_minutes": delay_minutes,
        "traffic_severity_index": severity_index,
        "free_flow_travel_time_minutes": round(free_flow_mins, 1),
        "current_travel_time_minutes": round(free_flow_mins + delay_minutes, 1),
    }


def fetch_traffic_incidents_and_delay(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> dict[str, float]:
    """Fetch live commercial truck route traffic delay and severity via TomTom Routing API.

    Falls back to deterministic corridor traffic modeling if TomTom API key is not supplied.

    Args:
        origin_lat: Origin latitude coordinate.
        origin_lon: Origin longitude coordinate.
        dest_lat: Destination latitude coordinate.
        dest_lon: Destination longitude coordinate.

    Returns:
        dict[str, float]: Traffic delay metrics including delay minutes, severity index,
                          free flow time, and current travel time.
    """
    api_key = os.getenv("TOMTOM_TRAFFIC_API_KEY", "").strip().strip("'\"")

    if not _is_placeholder_key(api_key):
        try:
            # TomTom Routing API with truck travelMode
            endpoint = (
                f"https://api.tomtom.com/routing/1/calculateRoute/"
                f"{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
            )
            params = {
                "key": api_key,
                "travelMode": "truck",
                "traffic": "true",
                "routeType": "fastest",
            }
            resp = requests.get(endpoint, params=params, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if routes:
                    summary = routes[0].get("summary", {})
                    traffic_delay_secs = summary.get("trafficDelayInSeconds", 0)
                    travel_time_secs = summary.get("travelTimeInSeconds", 0)
                    free_flow_secs = summary.get("noTrafficTravelTimeInSeconds", travel_time_secs - traffic_delay_secs)

                    delay_mins = round(traffic_delay_secs / 60.0, 1)
                    free_mins = round(free_flow_secs / 60.0, 1)
                    curr_mins = round(travel_time_secs / 60.0, 1)

                    # Traffic severity calculation
                    severity = round(1.0 + (delay_mins / max(1.0, free_mins)) * 3.0, 2)
                    logger.info("TomTom live truck traffic retrieved: %s mins delay.", delay_mins)
                    return {
                        "traffic_delay_minutes": delay_mins,
                        "traffic_severity_index": severity,
                        "free_flow_travel_time_minutes": free_mins,
                        "current_travel_time_minutes": curr_mins,
                    }
            else:
                logger.warning("TomTom API returned HTTP %d: %s. Using fallback.", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("TomTom Traffic API request failed (%s). Using fallback traffic calculation.", exc)
    else:
        logger.info("TOMTOM_TRAFFIC_API_KEY not configured or placeholder. Using fallback traffic estimation.")

    return _estimate_fallback_traffic(origin_lat, origin_lon, dest_lat, dest_lon)
