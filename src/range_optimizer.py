"""Alternative fuel range and refueling constraints optimizer.

Evaluates vehicle operational range against route distance, applying battery safety buffers,
live alternative fuel station queries (NREL AFDC and OpenChargeMap), and enterprise constraint
disqualification rules defined in RouteConfigManager.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from src.external_apis.stations import fetch_real_electric_charging, fetch_real_lng_stations

logger = logging.getLogger(__name__)


def validate_route_autonomy(
    carrier_row: pd.Series | dict[str, Any],
    route_distance_km: float,
    fuel_type: str,
    config: dict[str, Any],
    origin_coords: tuple[float, float] = (40.4168, -3.7038),
) -> dict[str, Any]:
    """Validate carrier autonomy and evaluate en-route refueling feasibility.

    Args:
        carrier_row: Carrier attributes including 'effective_range_km', 'base_rate_usd', etc.
        route_distance_km: Transit route distance in kilometers.
        fuel_type: Energy source / propulsion type (e.g. 'Electric', 'LNG', 'Diesel').
        config: Constraints dictionary loaded from RouteConfigManager.
        origin_coords: (lat, lon) reference coordinates for corridor station queries.

    Returns:
        dict[str, Any]: Validation results with keys:
            - 'is_viable': bool
            - 'status_flag': str
            - 'added_cost_usd': float
            - 'added_time_mins': float
            - 'detour_km': float
            - 'usable_range_km': float
            - 'stops_required': int
    """
    row = carrier_row.to_dict() if isinstance(carrier_row, pd.Series) else dict(carrier_row)

    effective_range = float(row.get("effective_range_km", 800.0))
    buffer_pct = float(config.get("safety_battery_buffer_pct", 0.15))
    usable_range_km = round(effective_range * (1.0 - buffer_pct), 1)

    base_rate = float(row.get("base_rate_usd", row.get("base_rate", 180.0)))
    rate_km = float(row.get("rate_per_km", 1.45))
    surcharge = float(row.get("fuel_surcharge_pct", 0.10))
    nominal_cost = base_rate + (route_distance_km * rate_km * (1.0 + surcharge))

    max_detour_allowed = float(config.get("max_detour_km_allowed", 30.0))
    max_delay_allowed = float(config.get("max_refueling_delay_minutes", 45.0))
    green_premium_limit = float(config.get("green_premium_pct_limit", 0.15))

    # Case 1: Route is within single-charge / single-tank usable autonomy
    if route_distance_km <= usable_range_km:
        return {
            "is_viable": True,
            "status_flag": "Viable: Direct Route (Within Autonomy)",
            "added_cost_usd": 0.0,
            "added_time_mins": 0.0,
            "detour_km": 0.0,
            "usable_range_km": usable_range_km,
            "stops_required": 0,
        }

    # Case 2: Route distance exceeds usable range -> Refueling / Charging stop required
    stops_required = max(1, int(np.ceil(route_distance_km / usable_range_km)) - 1)

    # Fetch corridor infrastructure
    is_electric = fuel_type.lower() == "electric"
    stations_df = (
        fetch_real_electric_charging(origin_coords[0], origin_coords[1], distance_km=150.0)
        if is_electric
        else fetch_real_lng_stations(origin_coords[0], origin_coords[1], radius_km=150.0)
    )

    if not stations_df.empty and "avg_detour_km" in stations_df.columns:
        best_station_detour = float(stations_df["avg_detour_km"].min())
    else:
        best_station_detour = 12.0  # Conservative standard corridor detour

    total_detour_km = round(best_station_detour * stops_required, 1)

    # Calculate operational delay minutes
    if is_electric:
        # Commercial Megawatt / DC fast charging: 35 min per session on 350kW+
        charger_power = float(stations_df["charger_power_kw"].max()) if not stations_df.empty and "charger_power_kw" in stations_df.columns else 300.0
        charger_power = max(150.0, charger_power)
        # 300 kWh replenishment to 80% SOC
        charging_mins_per_stop = (250.0 / charger_power) * 60.0
    else:
        # Gas / Fuel high-flow refueling: ~18 mins per stop
        charging_mins_per_stop = 18.0

    # Driving time for detour (avg 65 km/h on access roads)
    detour_driving_mins = (total_detour_km / 65.0) * 60.0
    added_time_mins = round((charging_mins_per_stop * stops_required) + detour_driving_mins, 1)

    # Cost of extra detour distance
    added_cost_usd = round(total_detour_km * rate_km * (1.0 + surcharge), 2)
    total_cost_with_detours = nominal_cost + added_cost_usd
    cost_increase_pct = (added_cost_usd / nominal_cost) if nominal_cost > 0 else 0.0

    # Disqualification Rule Evaluation against RouteConfigManager
    if total_detour_km > max_detour_allowed:
        return {
            "is_viable": False,
            "status_flag": "Disqualified: Refueling Detour Exceeds Limit",
            "added_cost_usd": added_cost_usd,
            "added_time_mins": added_time_mins,
            "detour_km": total_detour_km,
            "usable_range_km": usable_range_km,
            "stops_required": stops_required,
        }

    if added_time_mins > max_delay_allowed:
        return {
            "is_viable": False,
            "status_flag": "Disqualified: Charging Delay Exceeds Limit",
            "added_cost_usd": added_cost_usd,
            "added_time_mins": added_time_mins,
            "detour_km": total_detour_km,
            "usable_range_km": usable_range_km,
            "stops_required": stops_required,
        }

    if cost_increase_pct > green_premium_limit:
        return {
            "is_viable": False,
            "status_flag": "Disqualified: Green Premium Exceeded",
            "added_cost_usd": added_cost_usd,
            "added_time_mins": added_time_mins,
            "detour_km": total_detour_km,
            "usable_range_km": usable_range_km,
            "stops_required": stops_required,
        }

    return {
        "is_viable": True,
        "status_flag": "Viable: Refueling Stop Feasible",
        "added_cost_usd": added_cost_usd,
        "added_time_mins": added_time_mins,
        "detour_km": total_detour_km,
        "usable_range_km": usable_range_km,
        "stops_required": stops_required,
    }
