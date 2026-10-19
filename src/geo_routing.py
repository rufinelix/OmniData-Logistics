"""Geocoded weather routing and interactive mapping module.

Integrates Open-Meteo API for real-time weather telemetry along transport corridors
and builds interactive Folium route maps with corridor waypoints and refueling hubs.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import folium
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _is_placeholder_key(val: str | None) -> bool:
    """Check if environment variable key is absent or placeholder."""
    if not val:
        return True
    cleaned = val.strip().lower().strip("'\"")
    return not cleaned or "your_" in cleaned or "here" in cleaned or cleaned == "none"


def _compute_midpoint(coord1: tuple[float, float], coord2: tuple[float, float]) -> tuple[float, float]:
    """Calculate geographic midpoint between two coordinates."""
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    bx = np.cos(lat2) * np.cos(lon2 - lon1)
    by = np.cos(lat2) * np.sin(lon2 - lon1)
    mid_lat = np.arctan2(np.sin(lat1) + np.sin(lat2), np.sqrt((np.cos(lat1) + bx) ** 2 + by ** 2))
    mid_lon = lon1 + np.arctan2(by, np.cos(lat1) + bx)
    return (float(np.degrees(mid_lat)), float(np.degrees(mid_lon)))


def fetch_waypoint_weather(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
) -> dict[str, Any]:
    """Query Open-Meteo API for real-time weather along corridor waypoints.

    Computes precipitation, wind speed, ambient temperature, and derived weather risk index.

    Args:
        origin_coords: (latitude, longitude) of origin.
        dest_coords: (latitude, longitude) of destination.

    Returns:
        dict[str, Any]: Weather intelligence data including risk index and alert flags.
    """
    mid_coords = _compute_midpoint(origin_coords, dest_coords)
    api_key = os.getenv("OPEN_METEO_API_KEY", "").strip().strip("'\"")

    waypoints = [
        ("origin", origin_coords[0], origin_coords[1]),
        ("midpoint", mid_coords[0], mid_coords[1]),
        ("destination", dest_coords[0], dest_coords[1]),
    ]

    weather_reports: dict[str, dict[str, float]] = {}
    max_precip = 0.0
    max_wind = 0.0
    min_temp = 20.0

    for name, lat, lon in waypoints:
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
        }
        if not _is_placeholder_key(api_key):
            params["apikey"] = api_key

        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=5.0)
            if resp.status_code == 200:
                current = resp.json().get("current", {})
                t = float(current.get("temperature_2m", 15.0))
                p = float(current.get("precipitation", 0.0))
                w = float(current.get("wind_speed_10m", 12.0))
                code = int(current.get("weather_code", 0))

                weather_reports[name] = {
                    "temperature_c": t,
                    "precipitation_mm": p,
                    "wind_speed_kmh": w,
                    "weather_code": code,
                }
                max_precip = max(max_precip, p)
                max_wind = max(max_wind, w)
                min_temp = min(min_temp, t)
            else:
                # Deterministic synthetic fallback for waypoint
                weather_reports[name] = {"temperature_c": 14.5, "precipitation_mm": 0.2, "wind_speed_kmh": 15.0, "weather_code": 1}
        except Exception as exc:
            logger.warning("Open-Meteo weather query failed for %s (%s). Using fallback data.", name, exc)
            weather_reports[name] = {"temperature_c": 14.0, "precipitation_mm": 0.0, "wind_speed_kmh": 12.0, "weather_code": 0}

    # Weather risk index calculation (scale 0.0 to 3.0+)
    # Base = 0.2, rain adds risk, wind > 40 km/h adds risk, freezing temp adds ice risk
    risk_index = 0.20
    if max_precip > 5.0:
        risk_index += 0.60
    elif max_precip > 1.0:
        risk_index += 0.25

    if max_wind > 55.0:
        risk_index += 0.80
    elif max_wind > 35.0:
        risk_index += 0.30

    if min_temp < 0.0:
        risk_index += 0.60  # Black ice risk

    is_adverse = risk_index >= 1.0

    return {
        "waypoints": weather_reports,
        "max_precipitation_mm": round(max_precip, 1),
        "max_wind_speed_kmh": round(max_wind, 1),
        "min_temperature_c": round(min_temp, 1),
        "weather_risk_index": round(risk_index, 2),
        "is_adverse_weather": is_adverse,
        "summary": "Adverse Weather Alert: High wind / precipitation detected" if is_adverse else "Clear Weather: Optimal transit conditions",
    }


def render_route_map(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    origin_name: str = "Origin Hub",
    dest_name: str = "Destination Hub",
    weather_data: dict[str, Any] | None = None,
    refueling_stations: pd.DataFrame | None = None,
) -> folium.Map:
    """Render an interactive Folium route map with weather styling and corridor stations.

    Args:
        origin_coords: (latitude, longitude) of origin.
        dest_coords: (latitude, longitude) of destination.
        origin_name: Display label for origin.
        dest_name: Display label for destination.
        weather_data: Optional weather payload to determine route color.
        refueling_stations: Optional DataFrame of stations to plot.

    Returns:
        folium.Map: Configured interactive Folium map object.
    """
    center_lat = (origin_coords[0] + dest_coords[0]) / 2.0
    center_lon = (origin_coords[1] + dest_coords[1]) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    # Determine route line color
    is_adverse = weather_data.get("is_adverse_weather", False) if weather_data else False
    line_color = "#ef4444" if is_adverse else "#10b981"  # Red if adverse, Green if clear

    # Draw Route Corridor PolyLine
    route_points = [origin_coords, dest_coords]
    folium.PolyLine(
        route_points,
        color=line_color,
        weight=5,
        opacity=0.85,
        tooltip=f"Freight Corridor: {origin_name} to {dest_name}",
    ).add_to(m)

    # Origin Marker
    folium.Marker(
        location=origin_coords,
        popup=f"<b>Origin:</b> {origin_name}<br>Lat: {origin_coords[0]:.4f}, Lon: {origin_coords[1]:.4f}",
        tooltip=f"Origin: {origin_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Destination Marker
    folium.Marker(
        location=dest_coords,
        popup=f"<b>Destination:</b> {dest_name}<br>Lat: {dest_coords[0]:.4f}, Lon: {dest_coords[1]:.4f}",
        tooltip=f"Destination: {dest_name}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    # Plot Refueling & Recharging Stations if provided
    if refueling_stations is not None and not refueling_stations.empty:
        for _, station in refueling_stations.iterrows():
            st_lat = float(station.get("latitude", 0.0))
            st_lon = float(station.get("longitude", 0.0))
            fuel = str(station.get("fuel_type", "Electric"))
            name = str(station.get("station_name", "Station"))
            detour = float(station.get("avg_detour_km", 0.0))

            icon_name = "bolt" if fuel == "Electric" else "tint"
            icon_color = "blue" if fuel == "Electric" else "purple"

            folium.CircleMarker(
                location=[st_lat, st_lon],
                radius=6,
                color=icon_color,
                fill=True,
                fill_color=icon_color,
                fill_opacity=0.8,
                tooltip=f"<b>{name}</b> ({fuel}) - Detour: {detour} km",
                popup=f"<b>{name}</b><br>Type: {fuel}<br>Avg Detour: {detour} km",
            ).add_to(m)

    return m
