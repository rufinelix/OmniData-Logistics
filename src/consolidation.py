"""LTL (Less-Than-Truckload) freight consolidation engine.

Detects multi-order consolidation opportunities for sub-5,000 kg shipments sharing
corridors within a 48-hour pickup window, computing financial savings and Scope 3 CO2 reductions.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_FTL_PAYLOAD_KG = 24000.0
MAX_LTL_WEIGHT_KG = 5000.0
CONSOLIDATION_WINDOW_HOURS = 48.0

# Baseline rate parameters
LTL_BASE_RATE_USD = 220.0
LTL_RATE_PER_KM = 2.10
FTL_BASE_RATE_USD = 200.0
FTL_RATE_PER_KM = 1.45
FTL_STOP_SURCHARGE_USD = 60.0

# Emission factors (kg CO2 / vehicle-km)
LTL_VEHICLE_CO2_PER_KM = 0.72
FTL_VEHICLE_CO2_PER_KM = 0.84


def detect_ltl_consolidation_opportunities(df_shipments: pd.DataFrame) -> pd.DataFrame:
    """Scan shipments and identify multi-order consolidation clusters into FTL movements.

    Args:
        df_shipments: Historical or active shipments DataFrame.

    Returns:
        pd.DataFrame: Identified consolidation candidates with financial and CO2 savings.
    """
    if df_shipments.empty:
        return pd.DataFrame()

    df = df_shipments.copy()

    # Ensure datetime format
    if "creation_timestamp" in df.columns:
        df["creation_timestamp"] = pd.to_datetime(df["creation_timestamp"], errors="coerce")
    else:
        df["creation_timestamp"] = pd.Timestamp.now()

    # Filter for LTL candidate orders
    ltl_mask = df["cargo_weight_kg"] < MAX_LTL_WEIGHT_KG
    df_ltl = df[ltl_mask].sort_values("creation_timestamp").copy()

    if df_ltl.empty:
        return pd.DataFrame()

    consolidation_records: list[dict[str, Any]] = []
    cluster_id = 1

    # Group by corridor (origin_city, destination_city)
    grouped = df_ltl.groupby(["origin_city", "destination_city"])

    for (origin, dest), group in grouped:
        if len(group) < 2:
            continue

        orders = group.to_dict("records")
        current_cluster: list[dict[str, Any]] = []
        cluster_start_time = None
        cluster_weight = 0.0

        for order in orders:
            order_time = order["creation_timestamp"]
            order_weight = float(order["cargo_weight_kg"])

            if not current_cluster:
                current_cluster.append(order)
                cluster_start_time = order_time
                cluster_weight = order_weight
                continue

            time_diff_hours = (order_time - cluster_start_time).total_seconds() / 3600.0

            # Check 48h window and payload capacity constraints
            if time_diff_hours <= CONSOLIDATION_WINDOW_HOURS and (cluster_weight + order_weight) <= MAX_FTL_PAYLOAD_KG:
                current_cluster.append(order)
                cluster_weight += order_weight
            else:
                # Process completed cluster if >= 2 orders
                if len(current_cluster) >= 2:
                    consolidation_records.append(_build_consolidation_record(cluster_id, origin, dest, current_cluster))
                    cluster_id += 1

                # Start new cluster with current order
                current_cluster = [order]
                cluster_start_time = order_time
                cluster_weight = order_weight

        # Check trailing cluster
        if len(current_cluster) >= 2:
            consolidation_records.append(_build_consolidation_record(cluster_id, origin, dest, current_cluster))
            cluster_id += 1

    if not consolidation_records:
        return pd.DataFrame()

    df_result = pd.DataFrame(consolidation_records)
    return df_result.sort_values("financial_savings_usd", ascending=False).reset_index(drop=True)


def _build_consolidation_record(
    cluster_id: int,
    origin: str,
    dest: str,
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    """Helper to compute economics and emissions for a consolidation bundle."""
    num_orders = len(orders)
    shipment_ids = [str(o["shipment_id"]) for o in orders]
    total_weight_kg = sum(float(o["cargo_weight_kg"]) for o in orders)
    avg_distance_km = float(np.mean([float(o["distance_km"]) for o in orders]))

    # Individual LTL baseline economics
    individual_ltl_costs = [
        LTL_BASE_RATE_USD + (float(o["distance_km"]) * LTL_RATE_PER_KM)
        for o in orders
    ]
    baseline_ltl_total_cost = sum(individual_ltl_costs)

    # Consolidated FTL economics (single truck + multi-stop surcharges)
    ftl_cost = FTL_BASE_RATE_USD + (avg_distance_km * FTL_RATE_PER_KM) + ((num_orders - 1) * FTL_STOP_SURCHARGE_USD)
    financial_savings_usd = max(0.0, baseline_ltl_total_cost - ftl_cost)

    # Emissions baseline vs consolidated
    # Baseline: num_orders separate trucks
    baseline_co2_kg = num_orders * (avg_distance_km * LTL_VEHICLE_CO2_PER_KM)
    # Consolidated: 1 FTL truck with modest stop detours (1.10x distance factor)
    ftl_co2_kg = (avg_distance_km * 1.10) * FTL_VEHICLE_CO2_PER_KM
    co2_reduction_kg = max(0.0, baseline_co2_kg - ftl_co2_kg)
    co2_reduction_pct = (co2_reduction_kg / baseline_co2_kg * 100.0) if baseline_co2_kg > 0 else 0.0

    return {
        "cluster_id": f"CNS-{cluster_id:04d}",
        "corridor": f"{origin} -> {dest}",
        "origin_city": origin,
        "destination_city": dest,
        "shipment_count": num_orders,
        "shipment_ids": ", ".join(shipment_ids[:4]) + (f" (+{num_orders-4} more)" if num_orders > 4 else ""),
        "total_weight_kg": round(total_weight_kg, 1),
        "distance_km": round(avg_distance_km, 1),
        "baseline_ltl_cost_usd": round(baseline_ltl_total_cost, 2),
        "consolidated_ftl_cost_usd": round(ftl_cost, 2),
        "financial_savings_usd": round(financial_savings_usd, 2),
        "co2_reduction_kg": round(co2_reduction_kg, 2),
        "co2_reduction_pct": round(co2_reduction_pct, 1),
        "time_window_start": str(orders[0]["creation_timestamp"]),
        "time_window_end": str(orders[-1]["creation_timestamp"]),
    }
