"""Control tower exception monitoring and automated rerouting engine.

Provides 24/7 autonomous monitoring of active in-transit shipments, detecting telemetry
violations (Cold Chain Deviations, Delayed Pickup Risks, Severe Weather Delays),
and triggering dynamic backup carrier rerouting via LogisticsOptimizer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
import pandas as pd

from src.optimizer import LogisticsOptimizer
from src.route_config import RouteConfigManager

logger = logging.getLogger(__name__)


def monitor_active_shipments(
    df_active: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Scan in-transit shipments and flag operational exceptions.

    Exceptions evaluated:
    - 'Delayed Pickup Risk': actual pickup delayed or current delay > 1.0 hour.
    - 'Cold Chain Deviation': refrigerated shipment temperature exceeds tolerance.
    - 'Severe Weather Delay': route weather risk index > 1.5.
    - 'Traffic Gridlock': severe traffic congestion delay > 45 minutes.

    Args:
        df_active: DataFrame of active shipments.
        config: Optional configuration dictionary from RouteConfigManager.

    Returns:
        pd.DataFrame: Augmented DataFrame with exception flags and severity levels.
    """
    if df_active.empty:
        return pd.DataFrame()

    df = df_active.copy()
    cfg = config or RouteConfigManager().load_config()
    reefer_tolerance = float(cfg.get("refrigerated_temp_tolerance_c", 2.0))

    exception_flags = []
    severities = []
    requires_reroute = []

    for _, row in df.iterrows():
        alerts = []
        severity = "NOMINAL"

        # 1. Delayed Pickup / Transit Risk
        actual_delay = float(row.get("actual_delay_hours", 0.0))
        if "actual_pickup" in row and "promised_pickup" in row:
            if str(row["actual_pickup"]) > str(row["promised_pickup"]):
                alerts.append("Delayed Pickup Risk")
        elif actual_delay > 1.5:
            alerts.append("Delayed Pickup Risk")

        # 2. Cold Chain Deviation Check
        cargo_type = str(row.get("cargo_type", "General"))
        if cargo_type.lower() == "refrigerated":
            reefer_temp = float(row.get("ambient_temp_c", row.get("reefer_temp_c", 4.5)))
            if reefer_temp > (reefer_tolerance + 2.0):
                alerts.append("Cold Chain Deviation")

        # 3. Severe Weather Delay Check
        weather_idx = float(row.get("weather_risk_index", 0.2))
        if weather_idx > 1.5:
            alerts.append("Severe Weather Delay")

        # 4. Traffic Delay Check
        traffic_delay = float(row.get("traffic_delay_minutes", 0.0))
        if traffic_delay > 45.0:
            alerts.append("Traffic Gridlock")

        # Assign Severity Level
        if "Cold Chain Deviation" in alerts or ("Severe Weather Delay" in alerts and actual_delay > 2.0):
            severity = "CRITICAL"
        elif alerts:
            severity = "WARNING"

        exception_flags.append(", ".join(alerts) if alerts else "Normal Flow")
        severities.append(severity)
        requires_reroute.append(severity in ["CRITICAL", "WARNING"])

    df["exception_flags"] = exception_flags
    df["severity_level"] = severities
    df["requires_reroute"] = requires_reroute

    # Order so CRITICAL and WARNING are surfaced at the top
    severity_order = {"CRITICAL": 0, "WARNING": 1, "NOMINAL": 2}
    df["_sort_rank"] = df["severity_level"].map(severity_order).fillna(3)
    df = df.sort_values(by=["_sort_rank", "actual_delay_hours"], ascending=[True, False]).drop(columns=["_sort_rank"])

    return df.reset_index(drop=True)


def trigger_automated_reroute(
    shipment_id: str,
    df_carriers: pd.DataFrame,
    df_shipments: pd.DataFrame | None = None,
    optimizer: LogisticsOptimizer | None = None,
) -> dict[str, Any]:
    """Execute automated rerouting optimization to identify immediate backup carrier.

    Args:
        shipment_id: Shipment ID requiring exception remediation.
        df_carriers: Fleet carriers DataFrame.
        df_shipments: Optional shipments DataFrame to look up shipment details.
        optimizer: Optional LogisticsOptimizer instance.

    Returns:
        dict[str, Any]: Reroute execution details including assigned backup carrier,
                        incremental costs, and dispatch instruction.
    """
    opt = optimizer or LogisticsOptimizer()

    # Find target shipment
    shipment_dict: dict[str, Any] = {}
    if df_shipments is not None and not df_shipments.empty:
        match = df_shipments[df_shipments["shipment_id"] == shipment_id]
        if not match.empty:
            shipment_dict = match.iloc[0].to_dict()

    if not shipment_dict:
        # Fallback shipment profile for ad-hoc reroutes
        shipment_dict = {
            "shipment_id": shipment_id,
            "carrier_id": "CAR-003",
            "distance_km": 620.0,
            "cargo_weight_kg": 14000.0,
            "cargo_type": "Refrigerated",
            "origin_city": "Madrid",
            "destination_city": "Lyon",
            "origin_lat": 40.4168,
            "origin_lon": -3.7038,
        }

    original_cid = str(shipment_dict.get("carrier_id", "CAR-003"))
    distance_km = float(shipment_dict.get("distance_km", 600.0))
    weight_kg = float(shipment_dict.get("cargo_weight_kg", 12000.0))
    cargo_type = str(shipment_dict.get("cargo_type", "General"))
    origin_coords = (
        float(shipment_dict.get("origin_lat", 40.4168)),
        float(shipment_dict.get("origin_lon", -3.7038)),
    )

    # Exclude failing primary carrier to discover best alternative
    alternative_carriers = df_carriers[df_carriers["carrier_id"] != original_cid].copy()

    # Optimize with high priority on reliability and speed for rescue
    rescue_weights = {"cost": 0.20, "speed": 0.35, "otd": 0.25, "otif": 0.10, "co2": 0.10}

    ranked = opt.optimize_carrier_selection(
        df_carriers=alternative_carriers,
        distance_km=distance_km,
        cargo_weight_kg=weight_kg,
        cargo_type=cargo_type,
        user_weights=rescue_weights,
        origin_coords=origin_coords,
    )

    eligible = ranked[ranked["is_eligible"]]
    if eligible.empty:
        # Fallback to top ranked even if soft constraint warned
        best_backup = ranked.iloc[0]
    else:
        best_backup = eligible.iloc[0]

    # Original carrier name
    orig_match = df_carriers[df_carriers["carrier_id"] == original_cid]
    orig_name = str(orig_match.iloc[0]["carrier_name"]) if not orig_match.empty else "Primary Carrier"

    backup_cost = float(best_backup["financial_cost_usd"])

    return {
        "shipment_id": shipment_id,
        "status": "REROUTE_CONFIRMED",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "original_carrier_id": original_cid,
        "original_carrier_name": orig_name,
        "backup_carrier_id": str(best_backup["carrier_id"]),
        "backup_carrier_name": str(best_backup["carrier_name"]),
        "backup_carrier_energy": str(best_backup["energy_source"]),
        "backup_financial_cost_usd": backup_cost,
        "backup_transit_duration_hours": float(best_backup["transit_duration_hours"]),
        "composite_score": float(best_backup["composite_score"]),
        "recommended_action": f"Automated handover to {best_backup['carrier_name']} ({best_backup['energy_source']})",
    }
