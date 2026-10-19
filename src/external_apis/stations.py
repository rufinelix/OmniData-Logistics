"""External stations API connector for alternative fuel infrastructure.

Connects to:
1. NREL AFDC (Alternative Fuels Data Center) REST API for LNG, Bio-LNG, and Hydrogen stations.
2. OpenChargeMap API for commercial high-power DC fast-charging (>150 kW) stations.
Provides seamless fallback to the local corridor database if API keys are missing or unreachable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOCAL_STATIONS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "refueling_stations.csv"


class APIConfigurationError(Exception):
    """Raised when an external API credential is misconfigured or unauthorized."""


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in km between two GPS coordinates."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _is_placeholder_key(val: str | None) -> bool:
    """Check if environment variable key is absent or contains placeholder string."""
    if not val:
        return True
    cleaned = val.strip().lower().strip("'\"")
    return not cleaned or "your_" in cleaned or "here" in cleaned or cleaned == "none"


def fetch_real_lng_stations(
    latitude: float,
    longitude: float,
    radius_km: float = 100.0,
) -> pd.DataFrame:
    """Fetch operational LNG/CNG/Hydrogen stations within radius via NREL AFDC API.

    Falls back to local refueling database if NREL API key is missing or offline.

    Args:
        latitude: Target center coordinate latitude.
        longitude: Target center coordinate longitude.
        radius_km: Search radius in kilometers (default: 100 km).

    Returns:
        pd.DataFrame: DataFrame containing station records.
    """
    api_key = os.getenv("NREL_AFDC_API_KEY", "").strip().strip("'\"")

    if not _is_placeholder_key(api_key):
        try:
            radius_miles = radius_km * 0.621371
            url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.json"
            params = {
                "api_key": api_key,
                "latitude": latitude,
                "longitude": longitude,
                "radius": radius_miles,
                "fuel_type": "LNG,CNG,HY",
                "status": "E",
                "limit": 25,
            }
            resp = requests.get(url, params=params, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                fuel_stations = data.get("fuel_stations", [])
                if fuel_stations:
                    records = []
                    for s in fuel_stations:
                        st_lat = float(s.get("latitude", 0.0))
                        st_lon = float(s.get("longitude", 0.0))
                        records.append({
                            "station_id": str(s.get("id", f"NREL-{len(records)+1}")),
                            "station_name": s.get("station_name", "NREL Station"),
                            "corridor_name": f"{s.get('city', 'Regional')}-Corridor",
                            "latitude": st_lat,
                            "longitude": st_lon,
                            "fuel_type": "LNG" if s.get("fuel_type_code") in ["LNG", "CNG"] else "Hydrogen",
                            "charger_power_kw": 0.0,
                            "avg_detour_km": round(_haversine(latitude, longitude, st_lat, st_lon), 1),
                            "source": "NREL_AFDC_API",
                        })
                    logger.info("Retrieved %d stations from NREL AFDC API.", len(records))
                    return pd.DataFrame(records)
            else:
                logger.warning("NREL API returned HTTP status %d: %s. Using local fallback.", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("Failed to reach NREL API (%s). Falling back to local stations dataset.", exc)
    else:
        logger.info("NREL_AFDC_API_KEY not configured or placeholder. Using local station catalog.")

    # Local fallback query
    if not LOCAL_STATIONS_FILE.exists():
        logger.warning("Local stations file %s does not exist.", LOCAL_STATIONS_FILE)
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude", "fuel_type", "avg_detour_km", "source"])

    df_local = pd.read_csv(LOCAL_STATIONS_FILE)
    df_gas = df_local[df_local["fuel_type"].isin(["LNG", "Bio-LNG", "Hydrogen", "HVO"])].copy()
    if df_gas.empty:
        return pd.DataFrame()

    df_gas["dist_km"] = df_gas.apply(
        lambda r: _haversine(latitude, longitude, float(r["latitude"]), float(r["longitude"])), axis=1
    )
    df_matches = df_gas[df_gas["dist_km"] <= radius_km].copy()
    if df_matches.empty:
        # Fallback to nearest 3 stations if none within strict radius
        df_matches = df_gas.nsmallest(3, "dist_km").copy()

    df_matches["avg_detour_km"] = df_matches["dist_km"].round(1)
    df_matches["source"] = "LOCAL_DATABASE"
    return df_matches[["station_id", "station_name", "corridor_name", "latitude", "longitude", "fuel_type", "charger_power_kw", "avg_detour_km", "source"]]


def fetch_real_electric_charging(
    latitude: float,
    longitude: float,
    distance_km: float = 50.0,
) -> pd.DataFrame:
    """Fetch commercial high-power (>= 150 kW) EV charging stations via OpenChargeMap API.

    Falls back to local refueling database if OpenChargeMap API key is missing or offline.

    Args:
        latitude: Target center coordinate latitude.
        longitude: Target center coordinate longitude.
        distance_km: Search radius in kilometers (default: 50 km).

    Returns:
        pd.DataFrame: DataFrame containing charging station records.
    """
    api_key = os.getenv("OPENCHARGE_MAP_API_KEY", "").strip().strip("'\"")

    if not _is_placeholder_key(api_key):
        try:
            url = "https://api.openchargemap.io/v3/poi/"
            params = {
                "output": "json",
                "latitude": latitude,
                "longitude": longitude,
                "distance": distance_km,
                "distanceunit": "KM",
                "minpowerkw": 150,
                "maxresults": 20,
                "key": api_key,
            }
            resp = requests.get(url, params=params, timeout=6.0)
            if resp.status_code == 200:
                pois = resp.json()
                if pois:
                    records = []
                    for p in pois:
                        addr = p.get("AddressInfo", {})
                        p_lat = float(addr.get("Latitude", 0.0))
                        p_lon = float(addr.get("Longitude", 0.0))
                        # Find max charger power
                        connections = p.get("Connections", [])
                        max_kw = max([c.get("PowerKW", 150.0) or 150.0 for c in connections], default=150.0)
                        records.append({
                            "station_id": str(p.get("ID", f"OCM-{len(records)+1}")),
                            "station_name": addr.get("Title", "Commercial DC Fast Charger"),
                            "corridor_name": f"{addr.get('Town', 'Regional')}-EV-Corridor",
                            "latitude": p_lat,
                            "longitude": p_lon,
                            "fuel_type": "Electric",
                            "charger_power_kw": float(max_kw),
                            "avg_detour_km": round(_haversine(latitude, longitude, p_lat, p_lon), 1),
                            "source": "OPENCHARGE_MAP_API",
                        })
                    logger.info("Retrieved %d charging stations from OpenChargeMap API.", len(records))
                    return pd.DataFrame(records)
            else:
                logger.warning("OpenChargeMap returned status %d. Using local fallback.", resp.status_code)
        except Exception as exc:
            logger.warning("OpenChargeMap request failed (%s). Using local fallback.", exc)
    else:
        logger.info("OPENCHARGE_MAP_API_KEY not configured or placeholder. Using local station catalog.")

    # Local fallback query
    if not LOCAL_STATIONS_FILE.exists():
        logger.warning("Local stations file %s not found.", LOCAL_STATIONS_FILE)
        return pd.DataFrame(columns=["station_id", "station_name", "latitude", "longitude", "fuel_type", "charger_power_kw", "avg_detour_km", "source"])

    df_local = pd.read_csv(LOCAL_STATIONS_FILE)
    df_ev = df_local[df_local["fuel_type"] == "Electric"].copy()
    if df_ev.empty:
        return pd.DataFrame()

    df_ev["dist_km"] = df_ev.apply(
        lambda r: _haversine(latitude, longitude, float(r["latitude"]), float(r["longitude"])), axis=1
    )
    df_matches = df_ev[df_ev["dist_km"] <= distance_km].copy()
    if df_matches.empty:
        df_matches = df_ev.nsmallest(3, "dist_km").copy()

    df_matches["avg_detour_km"] = df_matches["dist_km"].round(1)
    df_matches["source"] = "LOCAL_DATABASE"
    return df_matches[["station_id", "station_name", "corridor_name", "latitude", "longitude", "fuel_type", "charger_power_kw", "avg_detour_km", "source"]]
