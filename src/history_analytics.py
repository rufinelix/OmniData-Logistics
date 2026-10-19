"""Historical analytics and business intelligence engine.

Processes historical logistics logs, computes aggregate operational KPIs,
and generates interactive Plotly BI visualizations:
- OTIF % vs Cost Time-Series Trend
- Cost vs Transit Risk Scatter Matrix
- Carbon Emissions by Propulsion Vector Donut Chart
- Delay Variance Distribution Histogram
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.esg_reporting import calculate_iso14083_emissions
from src.metrics import calculate_financial_cost, calculate_in_full, calculate_otd, calculate_otif

logger = logging.getLogger(__name__)

# Modern enterprise color palette
CHART_THEME = "plotly_white"
COLOR_PRIMARY = "#2563eb"
COLOR_SUCCESS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#ef4444"


def filter_historical_shipments(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    carriers: list[str] | None = None,
    cargo_types: list[str] | None = None,
    energy_sources: list[str] | None = None,
    df_carriers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter historical shipments by date range, carriers, cargo types, and energy sources.

    Args:
        df: Historical shipments DataFrame.
        start_date: ISO date string for start boundary (e.g. '2026-01-01').
        end_date: ISO date string for end boundary (e.g. '2026-06-30').
        carriers: List of allowed carrier_id or carrier_name values.
        cargo_types: List of allowed cargo types.
        energy_sources: List of allowed energy sources.
        df_carriers: Optional carrier metadata DataFrame to join energy_source.

    Returns:
        pd.DataFrame: Filtered shipment records.
    """
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # Ensure creation_timestamp is datetime
    if "creation_timestamp" in filtered.columns:
        filtered["creation_timestamp"] = pd.to_datetime(filtered["creation_timestamp"], errors="coerce")

    # Join energy source if available and needed
    if "energy_source" not in filtered.columns and df_carriers is not None and not df_carriers.empty:
        carrier_map = df_carriers.set_index("carrier_id")["energy_source"].to_dict()
        filtered["energy_source"] = filtered["carrier_id"].map(carrier_map).fillna("Diesel")

    # 1. Date filter
    if start_date and "creation_timestamp" in filtered.columns:
        filtered = filtered[filtered["creation_timestamp"] >= pd.to_datetime(start_date)]
    if end_date and "creation_timestamp" in filtered.columns:
        filtered = filtered[filtered["creation_timestamp"] <= pd.to_datetime(end_date)]

    # 2. Carrier filter
    if carriers:
        filtered = filtered[filtered["carrier_id"].isin(carriers) | filtered.get("carrier_name", pd.Series(dtype=str)).isin(carriers)]

    # 3. Cargo type filter
    if cargo_types and "cargo_type" in filtered.columns:
        filtered = filtered[filtered["cargo_type"].isin(cargo_types)]

    # 4. Energy source filter
    if energy_sources and "energy_source" in filtered.columns:
        filtered = filtered[filtered["energy_source"].isin(energy_sources)]

    return filtered.reset_index(drop=True)


def compute_aggregated_metrics(
    df_filtered: pd.DataFrame,
    df_carriers: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute aggregate business intelligence logistics metrics across filtered records.

    Args:
        df_filtered: Filtered shipment records.
        df_carriers: Optional carrier DataFrame to merge rate parameters.

    Returns:
        dict[str, float]: Aggregated KPIs including total spend, avg cost/km,
                          OTD %, In-Full %, OTIF %, and total CO2 (kg).
    """
    if df_filtered.empty:
        return {
            "total_shipments": 0,
            "total_spend_usd": 0.0,
            "avg_cost_per_km": 0.0,
            "otd_pct": 0.0,
            "in_full_pct": 0.0,
            "otif_pct": 0.0,
            "total_co2_kg": 0.0,
            "total_ton_km": 0.0,
        }

    df = df_filtered.copy()

    # Calculate financial costs
    if "financial_cost_usd" in df.columns:
        costs = df["financial_cost_usd"]
    elif "base_rate_usd" in df.columns and "rate_per_km" in df.columns:
        costs = calculate_financial_cost(df)
    elif df_carriers is not None and not df_carriers.empty:
        # Merge carrier rates
        merged = df.merge(df_carriers[["carrier_id", "base_rate_usd", "rate_per_km", "fuel_surcharge_pct"]], on="carrier_id", how="left")
        costs = calculate_financial_cost(merged)
    else:
        # Standard average truck cost approximation ($1.55/km + $180 base)
        costs = 180.0 + (df["distance_km"] * 1.55 * 1.10)

    total_spend = float(costs.sum())
    total_km = float(df["distance_km"].sum())
    avg_cost_per_km = (total_spend / total_km) if total_km > 0 else 0.0

    otd_series = calculate_otd(df)
    in_full_series = calculate_in_full(df)
    otif_series = otd_series & in_full_series

    # Aggregate CO2 emissions (kg)
    total_co2_kg = 0.0
    total_ton_km = 0.0

    if "co2_emissions_kg" in df.columns:
        total_co2_kg = float(df["co2_emissions_kg"].sum())
    else:
        # Vectorized or row calculation
        for _, r in df.iterrows():
            w_tons = float(r.get("cargo_weight_kg", 15000.0)) / 1000.0
            dist = float(r.get("distance_km", 500.0))
            vector = str(r.get("energy_source", "Diesel"))
            res = calculate_iso14083_emissions(w_tons, dist, vector)
            total_co2_kg += res["wtw_co2e_kg"]
            total_ton_km += res["ton_km"]

    return {
        "total_shipments": len(df),
        "total_spend_usd": round(total_spend, 2),
        "avg_cost_per_km": round(avg_cost_per_km, 2),
        "otd_pct": round(float(otd_series.mean() * 100.0), 1),
        "in_full_pct": round(float(in_full_series.mean() * 100.0), 1),
        "otif_pct": round(float(otif_series.mean() * 100.0), 1),
        "total_co2_kg": round(total_co2_kg, 1),
        "total_ton_km": round(total_ton_km, 1),
    }


def plot_time_series_trend(df: pd.DataFrame, freq: str = "W") -> go.Figure:
    """Generate interactive dual-axis line chart for OTIF % vs Avg Cost over time."""
    if df.empty or "creation_timestamp" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Historical Time Series Data Available", template=CHART_THEME)
        return fig

    df_ts = df.copy()
    df_ts["creation_timestamp"] = pd.to_datetime(df_ts["creation_timestamp"], errors="coerce")
    df_ts = df_ts.dropna(subset=["creation_timestamp"])

    df_ts["is_otif"] = (calculate_otd(df_ts) & calculate_in_full(df_ts)).astype(int)
    if "financial_cost_usd" not in df_ts.columns:
        df_ts["financial_cost_usd"] = 180.0 + (df_ts["distance_km"] * 1.55 * 1.10)

    # Resample by specified frequency (Weekly or Monthly)
    df_ts = df_ts.set_index("creation_timestamp")
    resampled = df_ts.resample(freq).agg({
        "is_otif": "mean",
        "financial_cost_usd": "mean",
        "shipment_id": "count",
    }).reset_index()

    resampled["otif_pct"] = (resampled["is_otif"] * 100.0).round(1)
    resampled["avg_cost"] = resampled["financial_cost_usd"].round(2)

    fig = go.Figure()

    # Trace 1: OTIF %
    fig.add_trace(
        go.Scatter(
            x=resampled["creation_timestamp"],
            y=resampled["otif_pct"],
            name="OTIF %",
            mode="lines+markers",
            line=dict(color=COLOR_SUCCESS, width=3),
            marker=dict(size=6),
            yaxis="y1",
        )
    )

    # Trace 2: Average Cost ($)
    fig.add_trace(
        go.Scatter(
            x=resampled["creation_timestamp"],
            y=resampled["avg_cost"],
            name="Avg Cost ($)",
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=3, dash="dash"),
            marker=dict(size=6),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="<b>Weekly OTIF Performance (%) vs Average Shipment Cost ($)</b>",
        template=CHART_THEME,
        hovermode="x unified",
        xaxis=dict(title="Timeline"),
        yaxis=dict(
            title=dict(text="OTIF Fulfillment (%)", font=dict(color=COLOR_SUCCESS)),
            tickfont=dict(color=COLOR_SUCCESS),
            range=[70, 105],
        ),
        yaxis2=dict(
            title=dict(text="Average Shipment Cost (USD)", font=dict(color=COLOR_PRIMARY)),
            tickfont=dict(color=COLOR_PRIMARY),
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_cost_vs_risk_matrix(df: pd.DataFrame) -> go.Figure:
    """Generate Cost vs Delay scatter plot grouped by carrier."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available for Risk Matrix", template=CHART_THEME)
        return fig

    df_plot = df.copy()
    if "financial_cost_usd" not in df_plot.columns:
        df_plot["financial_cost_usd"] = 180.0 + (df_plot["distance_km"] * 1.55 * 1.10)

    fig = px.scatter(
        df_plot,
        x="financial_cost_usd",
        y="actual_delay_hours",
        color="carrier_id",
        size="cargo_weight_kg",
        hover_data=["shipment_id", "origin_city", "destination_city", "cargo_type"],
        title="<b>Carrier Cost vs Delay Variance Matrix</b>",
        labels={
            "financial_cost_usd": "Shipment Cost (USD)",
            "actual_delay_hours": "Actual Delay (Hours)",
            "carrier_id": "Carrier ID",
        },
        template=CHART_THEME,
    )

    # Reference line at delay = 0
    fig.add_hline(y=0.0, line_dash="dot", line_color=COLOR_SUCCESS, annotation_text="On-Time Threshold (0h)")

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_emissions_by_fuel(df: pd.DataFrame, df_carriers: pd.DataFrame | None = None) -> go.Figure:
    """Generate interactive Donut / Pie chart for CO2 emissions by energy source."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Emissions Data Available", template=CHART_THEME)
        return fig

    df_fuel = df.copy()
    if "energy_source" not in df_fuel.columns and df_carriers is not None and not df_carriers.empty:
        carrier_map = df_carriers.set_index("carrier_id")["energy_source"].to_dict()
        df_fuel["energy_source"] = df_fuel["carrier_id"].map(carrier_map).fillna("Diesel")
    elif "energy_source" not in df_fuel.columns:
        df_fuel["energy_source"] = "Diesel"

    # Compute emissions per record
    emissions_list = []
    for _, r in df_fuel.iterrows():
        w_tons = float(r.get("cargo_weight_kg", 15000.0)) / 1000.0
        dist = float(r.get("distance_km", 500.0))
        vector = str(r.get("energy_source", "Diesel"))
        res = calculate_iso14083_emissions(w_tons, dist, vector)
        emissions_list.append(res["wtw_co2e_kg"])

    df_fuel["co2_wtw_kg"] = emissions_list
    agg = df_fuel.groupby("energy_source")["co2_wtw_kg"].sum().reset_index()

    color_discrete_map = {
        "Electric": "#10b981",
        "Hydrogen": "#06b6d4",
        "Bio-LNG": "#84cc16",
        "HVO": "#22c55e",
        "LNG": "#f59e0b",
        "Diesel": "#64748b",
    }

    fig = px.pie(
        agg,
        names="energy_source",
        values="co2_wtw_kg",
        hole=0.45,
        title="<b>Total Well-to-Wheel (WTW) Carbon Emissions by Fuel Vector</b>",
        color="energy_source",
        color_discrete_map=color_discrete_map,
        template=CHART_THEME,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

    return fig


def plot_delay_distribution(df: pd.DataFrame) -> go.Figure:
    """Generate distribution histogram of actual delay hours with threshold markers."""
    if df.empty or "actual_delay_hours" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Delay Data Available", template=CHART_THEME)
        return fig

    delays = df["actual_delay_hours"].dropna()
    mean_delay = float(delays.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=delays,
            nbinsx=40,
            marker_color=COLOR_PRIMARY,
            opacity=0.75,
            name="Delay Distribution",
        )
    )

    # Reference lines
    fig.add_vline(x=0.0, line_width=2, line_dash="solid", line_color=COLOR_SUCCESS, annotation_text="On-Time (0h)")
    fig.add_vline(x=mean_delay, line_width=2, line_dash="dash", line_color=COLOR_WARNING, annotation_text=f"Mean ({mean_delay:.2f}h)")

    fig.update_layout(
        title="<b>Distribution of Historical Transit Delays (Hours)</b>",
        xaxis_title="Actual Delay (Hours: negative = early, positive = delayed)",
        yaxis_title="Shipment Volume Count",
        template=CHART_THEME,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
