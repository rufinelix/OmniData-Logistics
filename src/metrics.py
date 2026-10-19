"""Core logistics metrics engine.

Provides high-performance, vectorized calculation of logistics KPIs including:
- On-Time Delivery (OTD)
- In-Full Delivery
- On-Time In-Full (OTIF)
- Financial Transportation Cost
"""

from __future__ import annotations

import pandas as pd


def calculate_otd(df: pd.DataFrame) -> pd.Series:
    """Calculate vectorized On-Time Delivery (OTD) boolean series.

    A shipment is considered on-time if actual_delay_hours <= 0.

    Args:
        df: DataFrame containing 'actual_delay_hours'.

    Returns:
        pd.Series: Boolean series where True indicates on-time delivery.
    """
    if "actual_delay_hours" not in df.columns:
        raise KeyError("Column 'actual_delay_hours' is required to calculate OTD.")
    return df["actual_delay_hours"] <= 0.0


def calculate_in_full(df: pd.DataFrame) -> pd.Series:
    """Calculate vectorized In-Full Delivery boolean series.

    A shipment is delivered in full when delivered_qty_ok equals ordered_qty.

    Args:
        df: DataFrame containing 'delivered_qty_ok' and 'ordered_qty'.

    Returns:
        pd.Series: Boolean series where True indicates complete delivery.
    """
    for col in ["delivered_qty_ok", "ordered_qty"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is required to calculate In-Full.")
    return df["delivered_qty_ok"] == df["ordered_qty"]


def calculate_otif(df: pd.DataFrame) -> pd.Series:
    """Calculate vectorized On-Time In-Full (OTIF) boolean series.

    OTIF requires both OTD (actual_delay_hours <= 0) and In-Full (delivered_qty_ok == ordered_qty).

    Args:
        df: DataFrame containing required columns for OTD and In-Full.

    Returns:
        pd.Series: Boolean series where True indicates on-time in-full delivery.
    """
    return calculate_otd(df) & calculate_in_full(df)


def calculate_financial_cost(df: pd.DataFrame) -> pd.Series:
    """Calculate total financial cost per shipment.

    Formula:
        base_rate + (distance_km * rate_per_km * (1 + fuel_surcharge_pct))

    Args:
        df: DataFrame containing rate parameters.

    Returns:
        pd.Series: Series of calculated costs in USD.
    """
    # Accommodate both base_rate_usd and base_rate column namings
    base_col = "base_rate_usd" if "base_rate_usd" in df.columns else "base_rate"
    for col in [base_col, "distance_km", "rate_per_km", "fuel_surcharge_pct"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is required to calculate financial cost.")

    base_rate = df[base_col].fillna(0.0)
    distance = df["distance_km"].fillna(0.0)
    rate_km = df["rate_per_km"].fillna(0.0)
    surcharge = df["fuel_surcharge_pct"].fillna(0.0)

    total_cost = base_rate + (distance * rate_km * (1.0 + surcharge))
    return total_cost.round(2)


def compute_kpi_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute overall fleet aggregate KPI metrics.

    Args:
        df: DataFrame of shipment records.

    Returns:
        dict[str, float]: Dictionary of aggregate percentages and counts.
    """
    if df.empty:
        return {
            "total_shipments": 0,
            "otd_pct": 0.0,
            "in_full_pct": 0.0,
            "otif_pct": 0.0,
            "avg_delay_hours": 0.0,
        }

    otd_series = calculate_otd(df)
    infull_series = calculate_in_full(df)
    otif_series = otd_series & infull_series

    return {
        "total_shipments": len(df),
        "otd_pct": round(float(otd_series.mean() * 100.0), 2),
        "in_full_pct": round(float(infull_series.mean() * 100.0), 2),
        "otif_pct": round(float(otif_series.mean() * 100.0), 2),
        "avg_delay_hours": round(float(df["actual_delay_hours"].mean()), 2) if "actual_delay_hours" in df.columns else 0.0,
    }
