"""Unit and integration tests for external API connectors.

Validates:
1. NREL AFDC station connector and local corridor database fallback.
2. OpenChargeMap commercial charger connector and fallback.
3. TomTom truck traffic delay computation and heuristic fallback.
4. Open-Meteo waypoint weather querying and risk index calculation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.external_apis.stations import fetch_real_electric_charging, fetch_real_lng_stations
from src.external_apis.traffic import fetch_traffic_incidents_and_delay
from src.geo_routing import fetch_waypoint_weather


class TestExternalStationConnectors:
    """Tests for alternative fuel and charging station connectors."""

    def test_fetch_lng_stations_fallback(self) -> None:
        # Query near Madrid
        df_lng = fetch_real_lng_stations(latitude=40.4168, longitude=-3.7038, radius_km=150.0)
        assert isinstance(df_lng, pd.DataFrame)
        assert not df_lng.empty
        expected_cols = {"station_id", "station_name", "latitude", "longitude", "fuel_type", "avg_detour_km", "source"}
        assert expected_cols.issubset(set(df_lng.columns))
        assert df_lng["fuel_type"].iloc[0] in ["LNG", "Bio-LNG", "Hydrogen", "HVO"]
        assert df_lng["source"].iloc[0] in ["NREL_AFDC_API", "LOCAL_DATABASE"]

    def test_fetch_electric_charging_fallback(self) -> None:
        # Query near Milan
        df_ev = fetch_real_electric_charging(latitude=45.4642, longitude=9.1900, distance_km=100.0)
        assert isinstance(df_ev, pd.DataFrame)
        assert not df_ev.empty
        assert "charger_power_kw" in df_ev.columns
        assert df_ev["fuel_type"].iloc[0] == "Electric"
        assert df_ev["source"].iloc[0] in ["OPENCHARGE_MAP_API", "LOCAL_DATABASE"]


class TestTrafficConnector:
    """Tests for TomTom traffic connector and heuristic fallback."""

    def test_fetch_traffic_delay_metrics(self) -> None:
        origin_lat, origin_lon = 40.4168, -3.7038  # Madrid
        dest_lat, dest_lon = 41.3879, 2.1699      # Barcelona

        res = fetch_traffic_incidents_and_delay(origin_lat, origin_lon, dest_lat, dest_lon)
        assert isinstance(res, dict)
        for key in ["traffic_delay_minutes", "traffic_severity_index", "free_flow_travel_time_minutes", "current_travel_time_minutes"]:
            assert key in res
            assert isinstance(res[key], (float, int))
            assert res[key] >= 0.0


class TestWeatherConnector:
    """Tests for Open-Meteo waypoint weather connector."""

    def test_fetch_waypoint_weather(self) -> None:
        origin_coords = (40.4168, -3.7038)
        dest_coords = (48.8566, 2.3522)

        res = fetch_waypoint_weather(origin_coords, dest_coords)
        assert isinstance(res, dict)
        assert "weather_risk_index" in res
        assert isinstance(res["weather_risk_index"], float)
        assert res["weather_risk_index"] >= 0.0
        assert "is_adverse_weather" in res
        assert "waypoints" in res
        assert "origin" in res["waypoints"]
        assert "destination" in res["waypoints"]
