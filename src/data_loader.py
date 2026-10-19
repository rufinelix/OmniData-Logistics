"""Data ingestion and carrier onboarding module.

Provides resilient dataset loading functions and the dynamic carrier onboarding
pipeline for OmniData-Logistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CARRIER_RATES_FILE = DEFAULT_DATA_DIR / "carrier_rates.csv"
REFUELING_STATIONS_FILE = DEFAULT_DATA_DIR / "refueling_stations.csv"
SYNTHETIC_SHIPMENTS_FILE = DEFAULT_DATA_DIR / "synthetic_shipments.csv"

REQUIRED_CARRIER_FIELDS = [
    "carrier_id",
    "carrier_name",
    "base_rate_usd",
    "rate_per_km",
    "fuel_surcharge_pct",
    "otd_historical",
    "in_full_historical",
    "co2_g_per_ton_km",
    "energy_source",
    "effective_range_km",
    "fleet_capacity_trucks",
    "noise_pollution_db",
]


def load_carrier_rates(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load carrier rates and operational parameters.

    Args:
        file_path: Optional custom path to carrier_rates.csv.

    Returns:
        pd.DataFrame: DataFrame containing carrier records.
    """
    path = Path(file_path) if file_path else CARRIER_RATES_FILE
    if not path.exists():
        logger.warning("Carrier rates file %s not found. Returning empty DataFrame.", path)
        return pd.DataFrame(columns=REQUIRED_CARRIER_FIELDS)

    df = pd.read_csv(path)
    # Ensure numerical types
    numeric_cols = [
        "base_rate_usd", "rate_per_km", "fuel_surcharge_pct",
        "otd_historical", "in_full_historical", "co2_g_per_ton_km",
        "effective_range_km", "noise_pollution_db"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fleet_capacity_trucks" in df.columns:
        df["fleet_capacity_trucks"] = pd.to_numeric(df["fleet_capacity_trucks"], errors="coerce").fillna(0).astype(int)

    return df


def load_refueling_stations(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load corridor refueling and recharging stations.

    Args:
        file_path: Optional custom path to refueling_stations.csv.

    Returns:
        pd.DataFrame: DataFrame of corridor stations.
    """
    path = Path(file_path) if file_path else REFUELING_STATIONS_FILE
    if not path.exists():
        logger.warning("Refueling stations file %s not found. Returning empty DataFrame.", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in ["latitude", "longitude", "charger_power_kw", "avg_detour_km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_shipments(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load historical shipment log records.

    Args:
        file_path: Optional custom path to synthetic_shipments.csv.

    Returns:
        pd.DataFrame: DataFrame of shipment records.
    """
    path = Path(file_path) if file_path else SYNTHETIC_SHIPMENTS_FILE
    if not path.exists():
        logger.warning("Shipments log file %s not found. Returning empty DataFrame.", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Parse timestamp columns
    for dt_col in ["creation_timestamp", "promised_delivery_timestamp", "actual_delivery_timestamp"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # Cast numeric columns
    numeric_cols = [
        "origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance_km",
        "cargo_weight_kg", "ordered_qty", "delivered_qty_ok", "actual_delay_hours",
        "weather_risk_index", "traffic_delay_minutes"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def append_new_carrier(carrier_data: dict[str, Any], file_path: Path | str | None = None) -> None:
    """Append a validated new carrier onboarding entry to carrier_rates.csv cleanly.

    Args:
        carrier_data: Dictionary containing carrier profile fields.
        file_path: Optional path to target CSV file.

    Raises:
        ValueError: If mandatory fields are missing.
    """
    path = Path(file_path) if file_path else CARRIER_RATES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate mandatory fields
    missing_fields = [f for f in REQUIRED_CARRIER_FIELDS if f not in carrier_data]
    if missing_fields:
        raise ValueError(f"Missing mandatory carrier fields: {missing_fields}")

    # Build clean row aligned to schema
    new_row = {
        "carrier_id": str(carrier_data["carrier_id"]),
        "carrier_name": str(carrier_data["carrier_name"]),
        "base_rate_usd": float(carrier_data["base_rate_usd"]),
        "rate_per_km": float(carrier_data["rate_per_km"]),
        "fuel_surcharge_pct": float(carrier_data["fuel_surcharge_pct"]),
        "otd_historical": float(carrier_data["otd_historical"]),
        "in_full_historical": float(carrier_data["in_full_historical"]),
        "co2_g_per_ton_km": float(carrier_data["co2_g_per_ton_km"]),
        "energy_source": str(carrier_data["energy_source"]),
        "effective_range_km": float(carrier_data["effective_range_km"]),
        "fleet_capacity_trucks": int(carrier_data["fleet_capacity_trucks"]),
        "noise_pollution_db": float(carrier_data["noise_pollution_db"]),
    }

    if path.exists():
        df_existing = pd.read_csv(path)
        # Check for duplicate carrier_id
        if str(new_row["carrier_id"]) in df_existing["carrier_id"].astype(str).values:
            # Update existing row
            idx = df_existing.index[df_existing["carrier_id"].astype(str) == str(new_row["carrier_id"])].tolist()[0]
            for k, v in new_row.items():
                df_existing.at[idx, k] = v
            df_existing.to_csv(path, index=False)
            logger.info("Updated existing carrier %s in %s.", new_row["carrier_id"], path)
            return

        df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([new_row])

    df_updated.to_csv(path, index=False)
    logger.info("Successfully onboarded carrier %s to %s.", new_row["carrier_id"], path)
