"""OmniData-Logistics: Enterprise Green Freight Optimization, Risk Management & Control Tower Dashboard.

A unified Streamlit architecture delivering:
1. Shipment Quote & MCDM Carrier Optimizer
2. Route Rules & Dynamic Constraint Manager
3. Stochastic Monte Carlo Risk Engine
4. LTL Freight Consolidation Hub
5. 24/7 Autonomous Control Tower & Dispatch Rerouting
6. Historical Logistics BI & Analytics
7. Carrier Management & Onboarding Pipeline
8. ISO 14083 ESG Carbon Accounting & Audit PDF Generation
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Load environment variables
load_dotenv()

# OmniData internal services
from src.consolidation import detect_ltl_consolidation_opportunities
from src.control_tower import monitor_active_shipments, trigger_automated_reroute
from src.data_loader import append_new_carrier, load_carrier_rates, load_refueling_stations, load_shipments
from src.esg_reporting import calculate_iso14083_emissions, generate_pdf_esg_report
from src.external_apis.traffic import fetch_traffic_incidents_and_delay
from src.geo_routing import fetch_waypoint_weather, render_route_map
from src.history_analytics import (
    compute_aggregated_metrics,
    filter_historical_shipments,
    plot_cost_vs_risk_matrix,
    plot_delay_distribution,
    plot_emissions_by_fuel,
    plot_time_series_trend,
)
from src.ml_engine import load_delay_model, predict_delay
from src.optimizer import LogisticsOptimizer
from src.range_optimizer import validate_route_autonomy
from src.route_config import RouteConfigManager

# Streamlit Page Setup
st.set_page_config(
    page_title="OmniData-Logistics | Enterprise Green Freight Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CITIES_METADATA = {
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3879, 2.1699),
    "Valencia": (39.4699, -0.3763),
    "Zaragoza": (41.6488, -0.8891),
    "Paris": (48.8566, 2.3522),
    "Lyon": (45.7640, 4.8357),
    "Frankfurt": (50.1109, 8.6821),
    "Munich": (48.1351, 11.5820),
    "Rotterdam": (51.9244, 4.4777),
    "Milan": (45.4642, 9.1900),
}


def _calc_geo_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """Haversine distance in km with winding factor."""
    r = 6371.0
    phi1, phi2 = np.radians(c1[0]), np.radians(c2[0])
    dphi = np.radians(c2[0] - c1[0])
    dlambda = np.radians(c2[1] - c1[1])
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return round(float(r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)) * 1.25), 1)


@st.cache_resource
def get_config_manager() -> RouteConfigManager:
    """Cached RouteConfigManager instance."""
    return RouteConfigManager()


@st.cache_resource
def get_optimizer() -> LogisticsOptimizer:
    """Cached LogisticsOptimizer instance."""
    return LogisticsOptimizer(get_config_manager())


@st.cache_data(ttl=60)
def get_carrier_rates() -> pd.DataFrame:
    """Cached carrier rates table."""
    return load_carrier_rates()


@st.cache_data(ttl=60)
def get_refueling_stations() -> pd.DataFrame:
    """Cached corridor refueling stations."""
    return load_refueling_stations()


@st.cache_data(ttl=60)
def get_shipments() -> pd.DataFrame:
    """Cached historical shipments."""
    return load_shipments()


# Load shared resources
cfg_manager = get_config_manager()
optimizer = get_optimizer()
df_carriers = get_carrier_rates()
df_stations = get_refueling_stations()
df_shipments = get_shipments()

# Sidebar: System Status & Quick Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/delivery-truck.png", width=64)
    st.markdown("### OmniData-Logistics")
    st.caption("v2.4 Enterprise Production Edition")
    st.markdown("---")

    st.markdown("#### System Health & Connectors")
    st.markdown("🟢 **Database Engine:** Connected")
    st.markdown(f"📦 **Historical Logs:** {len(df_shipments):,} records")
    st.markdown(f"🚛 **Onboarded Carriers:** {len(df_carriers)} fleets")
    st.markdown(f"⚡ **Refueling Stations:** {len(df_stations)} hubs")

    has_open_meteo = bool(os.getenv("OPEN_METEO_API_KEY", "").strip())
    has_tomtom = bool(os.getenv("TOMTOM_TRAFFIC_API_KEY", "").strip())
    has_nrel = bool(os.getenv("NREL_AFDC_API_KEY", "").strip())

    st.markdown("---")
    st.markdown("#### API Integrations")
    st.markdown("🌤️ Open-Meteo: " + ("✅ Active" if has_open_meteo else "ℹ️ Active (Public Tier)"))
    st.markdown("🚦 TomTom Traffic: " + ("✅ Live Key" if has_tomtom else "ℹ️ Corridor Fallback"))
    st.markdown("🔋 NREL / OpenChargeMap: " + ("✅ Live Key" if has_nrel else "ℹ️ Corridor Catalog"))

    st.markdown("---")
    st.caption("Standard: ISO 14083:2023 Compliant\nFramework: GLEC Framework v3.0")

# Header Section
st.markdown('<div class="main-header">🚚 OmniData-Logistics Control Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous Green Freight Routing, MCDM Carrier Selection, Monte Carlo Risk & ISO 14083 ESG Analytics</div>',
    unsafe_allow_html=True,
)

# 8 Navigation Tabs
tabs = st.tabs([
    "1. 🎯 Quote & Optimizer",
    "2. ⚙️ Route Constraints",
    "3. 🎲 Monte Carlo Risk",
    "4. 📦 LTL Consolidation",
    "5. 🛰️ Control Tower",
    "6. 📊 Historical BI",
    "7. 🚛 Carrier Management",
    "8. 🌱 ISO 14083 ESG",
])

# -------------------------------------------------------------------------------------------------
# TAB 1: SHIPMENT QUOTE & CARRIER OPTIMIZER
# -------------------------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### 🎯 Dynamic Shipment Quotation & MCDM Carrier Selection")
    st.markdown(
        "Evaluate carrier trade-offs across financial cost, transit duration, OTD reliability, "
        "and ISO 14083 carbon intensity with live corridor weather and traffic integration."
    )

    col_q1, col_q2, col_q3 = st.columns([1, 1, 1])
    city_names = list(CITIES_METADATA.keys())

    with col_q1:
        origin_city = st.selectbox("Origin Commercial Hub", city_names, index=0)
        dest_city = st.selectbox("Destination Commercial Hub", city_names, index=4)
        c1_coords = CITIES_METADATA[origin_city]
        c2_coords = CITIES_METADATA[dest_city]
        default_dist = _calc_geo_distance(c1_coords, c2_coords)
        distance_km = st.number_input("Corridor Distance (km)", min_value=20.0, max_value=4000.0, value=float(default_dist), step=10.0)

    with col_q2:
        cargo_weight_kg = st.number_input("Payload Weight (kg)", min_value=500.0, max_value=25000.0, value=14500.0, step=500.0)
        cargo_type = st.selectbox("Cargo Classification", ["General", "Refrigerated", "Hazmat", "High-Value", "Bulk"], index=0)
        all_energies = ["Electric", "Hydrogen", "Bio-LNG", "HVO", "LNG", "Diesel"]
        selected_energies = st.multiselect("Permitted Energy Vectors", all_energies, default=all_energies)

    with col_q3:
        st.markdown("**User Objective Weights (Sum Normalized)**")
        w_cost = st.slider("Cost Priority", 0.0, 1.0, 0.30, 0.05)
        w_speed = st.slider("Speed / Duration Priority", 0.0, 1.0, 0.20, 0.05)
        w_otd = st.slider("OTD Reliability Priority", 0.0, 1.0, 0.20, 0.05)
        w_otif = st.slider("In-Full Quality Priority", 0.0, 1.0, 0.10, 0.05)
        w_co2 = st.slider("Decarbonization (CO2) Priority", 0.0, 1.0, 0.20, 0.05)

    with st.expander("🛡️ Hard Constraint Thresholds"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            min_otd_thresh = st.slider("Minimum Acceptable OTD %", 0.0, 100.0, 85.0, 1.0)
        with col_t2:
            min_otif_thresh = st.slider("Minimum Acceptable OTIF %", 0.0, 100.0, 80.0, 1.0)
        with col_t3:
            max_budget = st.number_input("Maximum Budget Limit (USD)", min_value=100.0, max_value=20000.0, value=5000.0, step=100.0)

    # Fetch live telemetry
    with st.spinner("Fetching corridor weather telemetry and traffic flow..."):
        weather_info = fetch_waypoint_weather(c1_coords, c2_coords)
        traffic_info = fetch_traffic_incidents_and_delay(c1_coords[0], c1_coords[1], c2_coords[0], c2_coords[1])

    # Run ML delay predictions across carriers
    pred_delays = {}
    for cid in df_carriers["carrier_id"]:
        pred_delays[cid] = predict_delay(
            carrier_id=cid,
            distance_km=distance_km,
            cargo_weight_kg=cargo_weight_kg,
            weather_risk_index=weather_info["weather_risk_index"],
            traffic_delay_minutes=traffic_info["traffic_delay_minutes"],
        )

    # Execute MCDM Optimization
    weights_dict = {"cost": w_cost, "speed": w_speed, "otd": w_otd, "otif": w_otif, "co2": w_co2}
    hard_constraints = {
        "allowed_energy_sources": selected_energies,
        "min_otd_pct": min_otd_thresh,
        "min_otif_pct": min_otif_thresh,
        "max_budget_usd": max_budget,
    }

    df_ranked = optimizer.optimize_carrier_selection(
        df_carriers=df_carriers,
        distance_km=distance_km,
        cargo_weight_kg=cargo_weight_kg,
        cargo_type=cargo_type,
        user_weights=weights_dict,
        hard_constraints=hard_constraints,
        predicted_delays=pred_delays,
        origin_coords=c1_coords,
    )

    eligible_carriers = df_ranked[df_ranked["is_eligible"]]

    if not eligible_carriers.empty:
        best = eligible_carriers.iloc[0]

        st.success(
            f"🏆 **Top Recommendation:** #{int(best['rank'])} **{best['carrier_name']}** "
            f"({best['energy_source']}) — Composite Score: **{best['composite_score']:.1f}/100**"
        )

        # KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Freight Cost", f"${best['financial_cost_usd']:,.2f}", f"+${best['shadow_carbon_cost_usd']:.2f} Carbon")
        with m2:
            st.metric("Transit Duration", f"{best['transit_duration_hours']:.1f} hrs", f"{best['stops_required']} Refuel Stops")
        with m3:
            st.metric("Carrier OTD %", f"{best['otd_pct']}%", f"{best['otif_pct']}% OTIF")
        with m4:
            st.metric("CO2 Footprint", f"{best['co2_emissions_kg']:,.1f} kg", f"{best['co2_g_per_ton_km']:.1f} g/tkm")
        with m5:
            st.metric("Weather Risk Index", f"{weather_info['weather_risk_index']}", weather_info["summary"][:20])

        col_map, col_chart = st.columns([1, 1])

        with col_map:
            st.markdown("##### 🗺️ Freight Corridor & Infrastructure Map")
            route_map = render_route_map(
                origin_coords=c1_coords,
                dest_coords=c2_coords,
                origin_name=origin_city,
                dest_name=dest_city,
                weather_data=weather_info,
                refueling_stations=df_stations,
            )
            st_folium(route_map, height=360, width=None)

        with col_chart:
            st.markdown("##### 📊 Composite Carrier Score (CCS) Comparison")
            fig_ccs = px.bar(
                df_ranked,
                x="carrier_name",
                y="composite_score",
                color="energy_source",
                text="composite_score",
                title="Carrier Ranking (0-100 Score)",
                labels={"carrier_name": "Carrier", "composite_score": "Score", "energy_source": "Propulsion"},
                template="plotly_white",
            )
            fig_ccs.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_ccs.update_layout(margin=dict(l=20, r=20, t=40, b=80), xaxis_tickangle=-45)
            st.plotly_chart(fig_ccs, use_container_width=True)

        st.markdown("##### 📋 Complete Multi-Criteria Carrier Evaluation Table")
        st.dataframe(
            df_ranked[[
                "rank", "carrier_name", "energy_source", "financial_cost_usd", "total_economic_cost_usd",
                "transit_duration_hours", "otd_pct", "otif_pct", "co2_g_per_ton_km", "composite_score",
                "is_eligible", "disqualification_reason"
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("⚠️ No carriers satisfy the active hard constraints and autonomy limits. Adjust threshold filters.")


# -------------------------------------------------------------------------------------------------
# TAB 2: ROUTE RULES & CONSTRAINT CONFIGURATION (PHASE 12)
# -------------------------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### ⚙️ Route Rules & Dynamic Constraint Configuration")
    st.markdown(
        "Manage enterprise operational boundaries, green premium limits, and shadow carbon pricing. "
        "Changes persist directly to `data/route_constraints_config.json`."
    )

    current_cfg = cfg_manager.load_config()

    with st.form("constraint_config_form"):
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            max_transit_time = st.number_input(
                "Max Transit Time (Hours)",
                min_value=6.0,
                max_value=168.0,
                value=float(current_cfg.get("max_transit_time_hours", 48.0)),
                step=2.0,
                help="Shipments exceeding this threshold are flagged for SLA breach.",
            )

            max_detour_km = st.number_input(
                "Max Refueling Detour Allowed (km)",
                min_value=5.0,
                max_value=150.0,
                value=float(current_cfg.get("max_detour_km_allowed", 30.0)),
                step=5.0,
                help="Alternative fuel stops requiring detours greater than this value disqualify carriers.",
            )

            max_refueling_delay = st.number_input(
                "Max En-Route Refueling Delay (Minutes)",
                min_value=10.0,
                max_value=240.0,
                value=float(current_cfg.get("max_refueling_delay_minutes", 45.0)),
                step=5.0,
                help="Maximum acceptable charging or refueling stoppage duration per leg.",
            )

            safety_buffer = st.slider(
                "Safety Battery / Fuel Buffer (%)",
                min_value=0.05,
                max_value=0.30,
                value=float(current_cfg.get("safety_battery_buffer_pct", 0.15)),
                step=0.01,
                help="Reserve autonomy withheld from usable range calculation (e.g. 15% SOC buffer).",
            )

        with col_c2:
            green_premium = st.slider(
                "Green Premium Cost Ceiling (%)",
                min_value=0.05,
                max_value=0.50,
                value=float(current_cfg.get("green_premium_pct_limit", 0.15)),
                step=0.01,
                help="Maximum allowable cost percentage increase for green fuel alternatives before disqualification.",
            )

            shadow_carbon_price = st.number_input(
                "Internal Shadow Carbon Price ($ / Metric Ton CO2e)",
                min_value=10.0,
                max_value=500.0,
                value=float(current_cfg.get("shadow_carbon_price_per_ton", 80.0)),
                step=5.0,
                help="Internal economic shadow price applied to monetize Scope 3 GHG emissions.",
            )

            reefer_tolerance = st.number_input(
                "Refrigerated Temperature Excursion Tolerance (°C)",
                min_value=0.5,
                max_value=5.0,
                value=float(current_cfg.get("refrigerated_temp_tolerance_c", 2.0)),
                step=0.5,
                help="Max permissible cold-chain excursion before triggering Control Tower Critical Alert.",
            )

            corridor_options = ["MAD-BCN-Tunnel", "ALP-Pass", "Brenner-Pass", "Rhine-Gorge", "Mont-Blanc-Tunnel"]
            selected_hazmat = st.multiselect(
                "Hazmat Restricted Freight Corridors",
                corridor_options,
                default=[c for c in current_cfg.get("hazmat_restricted_corridors", ["MAD-BCN-Tunnel", "ALP-Pass"]) if c in corridor_options],
            )

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            save_clicked = st.form_submit_button("💾 Save & Persist Configuration", use_container_width=True)
        with col_b2:
            reset_clicked = st.form_submit_button("🔄 Reset to Factory Defaults", use_container_width=True)

    if save_clicked:
        new_params = {
            "max_transit_time_hours": float(max_transit_time),
            "max_detour_km_allowed": float(max_detour_km),
            "max_refueling_delay_minutes": float(max_refueling_delay),
            "safety_battery_buffer_pct": float(safety_buffer),
            "green_premium_pct_limit": float(green_premium),
            "shadow_carbon_price_per_ton": float(shadow_carbon_price),
            "refrigerated_temp_tolerance_c": float(reefer_tolerance),
            "hazmat_restricted_corridors": selected_hazmat,
        }
        cfg_manager.update_config(new_params)
        st.success("✅ Configuration successfully persisted to `data/route_constraints_config.json`!")
        st.rerun()

    if reset_clicked:
        cfg_manager.reset_to_defaults()
        st.info("ℹ️ Configuration reset to factory defaults.")
        st.rerun()

    st.markdown("##### 📄 Active Configuration JSON Schema")
    st.json(cfg_manager.load_config())


# -------------------------------------------------------------------------------------------------
# TAB 3: RISK & MONTE CARLO ANALYTICS
# -------------------------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### 🎲 Predictive ML Delay & Stochastic Monte Carlo Risk Engine")
    st.markdown(
        "Quantify delivery variance and P95 transit risk using 1,000+ stochastic iterations "
        "simulating velocity deviations, weather impacts, and driver break compliance."
    )

    from src.risk_simulation import run_monte_carlo_simulation

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        carrier_choices = df_carriers["carrier_id"].tolist()
        carrier_names_map = dict(zip(df_carriers["carrier_id"], df_carriers["carrier_name"]))
        selected_cid = st.selectbox("Carrier Profile", carrier_choices, format_func=lambda c: f"{c} - {carrier_names_map.get(c, '')}")
    with col_r2:
        mc_distance = st.number_input("Simulation Distance (km)", 50.0, 3000.0, 680.0, 50.0)
        mc_weather = st.slider("Environmental Weather Risk Index", 0.0, 3.0, 0.75, 0.05)
    with col_r3:
        mc_iterations = st.select_slider("Stochastic Iterations", options=[500, 1000, 2500, 5000], value=1000)
        trigger_mc = st.button("🚀 Run Monte Carlo Simulation", use_container_width=True)

    if trigger_mc or "mc_results" not in st.session_state:
        st.session_state["mc_results"] = run_monte_carlo_simulation(
            carrier_id=selected_cid,
            distance_km=mc_distance,
            weather_index=mc_weather,
            iterations=mc_iterations,
        )

    res_mc = st.session_state["mc_results"]

    # Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("P95 Worst-Case ETA", f"{res_mc['p95_eta_hours']:.1f} hrs", "95% Confidence")
    with k2:
        st.metric("Median Transit Duration", f"{res_mc['median_eta_hours']:.1f} hrs", f"Mean: {res_mc['mean_eta_hours']:.1f}h")
    with k3:
        st.metric("SLA Compliance Probability", f"{res_mc['sla_compliance_probability']*100:.1f}%", f"Benchmark: {res_mc['nominal_sla_hours']:.1f}h")
    with k4:
        ci = res_mc["confidence_interval_90"]
        st.metric("90% Confidence Interval", f"{ci[0]:.1f}h - {ci[1]:.1f}h", "P05 to P95")

    # Histogram Visualization
    fig_hist = go.Figure()
    sim_data = res_mc["simulated_etas"]

    fig_hist.add_trace(
        go.Histogram(
            x=sim_data,
            nbinsx=40,
            marker_color="#3b82f6",
            opacity=0.75,
            name="Simulated ETA Outcomes",
        )
    )

    fig_hist.add_vline(x=res_mc["nominal_sla_hours"], line_width=3, line_dash="solid", line_color="#10b981", annotation_text="Nominal SLA Benchmark")
    fig_hist.add_vline(x=res_mc["p95_eta_hours"], line_width=3, line_dash="dash", line_color="#ef4444", annotation_text=f"P95 ETA ({res_mc['p95_eta_hours']:.1f}h)")

    fig_hist.update_layout(
        title=f"<b>Stochastic Transit Time Distribution ({mc_iterations:,} Iterations)</b>",
        xaxis_title="Total Transit Duration (Hours)",
        yaxis_title="Frequency Count",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# TAB 4: LTL CONSOLIDATION HUB
# -------------------------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### 📦 Less-Than-Truckload (LTL) Freight Consolidation Hub")
    st.markdown(
        "Autonomous grouping of sub-5,000 kg shipments sharing freight corridors within a rolling "
        "48-hour pickup window, eliminating empty backhauls and reducing Scope 3 transportation emissions."
    )

    df_consolidation = detect_ltl_consolidation_opportunities(df_shipments)

    if not df_consolidation.empty:
        total_clusters = len(df_consolidation)
        total_sav = df_consolidation["financial_savings_usd"].sum()
        total_co2_sav = df_consolidation["co2_reduction_kg"].sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Consolidation Opportunities", f"{total_clusters} Bundles", f"{df_consolidation['shipment_count'].sum()} Orders Merged")
        with c2:
            st.metric("Total Financial Savings", f"${total_sav:,.2f}", "vs Independent LTL")
        with c3:
            st.metric("Carbon Emissions Avoided", f"{total_co2_sav:,.1f} kg CO2e", "Truck Trip Elimination")

        st.markdown("##### 🔍 Identified Multi-Order Consolidation Candidates")
        st.dataframe(
            df_consolidation[[
                "cluster_id", "corridor", "shipment_count", "total_weight_kg", "distance_km",
                "baseline_ltl_cost_usd", "consolidated_ftl_cost_usd", "financial_savings_usd",
                "co2_reduction_kg", "co2_reduction_pct", "time_window_start", "time_window_end"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            selected_cluster = st.selectbox(
                "Select Consolidation Bundle to Authorize",
                df_consolidation["cluster_id"].tolist(),
                format_func=lambda cid: f"{cid}: {df_consolidation[df_consolidation['cluster_id']==cid]['corridor'].values[0]} (${df_consolidation[df_consolidation['cluster_id']==cid]['financial_savings_usd'].values[0]:,.2f} savings)",
            )
        with col_act2:
            st.write("")
            st.write("")
            if st.button("🚀 Merge & Dispatch Consolidated FTL", use_container_width=True):
                st.balloons()
                st.success(f"🎉 Bundle **{selected_cluster}** successfully authorized and dispatched as multi-stop FTL movement!")
    else:
        st.info("ℹ️ No active sub-5,000 kg orders eligible for consolidation within current 48-hour window.")


# -------------------------------------------------------------------------------------------------
# TAB 5: CONTROL TOWER & EXCEPTION MANAGEMENT
# -------------------------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 🛰️ 24/7 Autonomous Control Tower & Exception Remediation")
    st.markdown(
        "Continuous telemetry monitoring scanning for Cold Chain Deviations, Extreme Weather Bottlenecks, "
        "and Transit Delays with instant automated backup carrier rerouting."
    )

    monitored_shipments = monitor_active_shipments(df_shipments.head(40), config=cfg_manager.load_config())

    n_crit = len(monitored_shipments[monitored_shipments["severity_level"] == "CRITICAL"])
    n_warn = len(monitored_shipments[monitored_shipments["severity_level"] == "WARNING"])
    n_nom = len(monitored_shipments[monitored_shipments["severity_level"] == "NOMINAL"])

    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
    with k_c1:
        st.metric("Active Monitored Shipments", len(monitored_shipments), "Real-time Telemetry")
    with k_c2:
        st.metric("Critical Alerts", n_crit, "Immediate Handover Required", delta_color="inverse")
    with k_c3:
        st.metric("Warnings Active", n_warn, "Congestion / Delay Risk", delta_color="inverse")
    with k_c4:
        st.metric("Nominal Flow", n_nom, "On-Schedule", delta_color="normal")

    st.markdown("##### 🚨 Active In-Transit Telemetry Exceptions")
    st.dataframe(
        monitored_shipments[[
            "shipment_id", "carrier_id", "origin_city", "destination_city", "cargo_type",
            "distance_km", "actual_delay_hours", "weather_risk_index", "severity_level", "exception_flags"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### ⚡ Dynamic Automated Rerouting Engine")

    col_re1, col_re2 = st.columns([2, 1])
    with col_re1:
        candidates_to_reroute = monitored_shipments[monitored_shipments["requires_reroute"]]["shipment_id"].tolist()
        if not candidates_to_reroute:
            candidates_to_reroute = monitored_shipments["shipment_id"].tolist()
        selected_reroute_id = st.selectbox("Select Distressed Shipment ID", candidates_to_reroute)
    with col_re2:
        st.write("")
        st.write("")
        trigger_reroute = st.button("🚨 Trigger Automated Secondary Reroute", use_container_width=True)

    if trigger_reroute:
        reroute_action = trigger_automated_reroute(
            shipment_id=selected_reroute_id,
            df_carriers=df_carriers,
            df_shipments=monitored_shipments,
            optimizer=optimizer,
        )

        st.success(f"✅ Reroute Dispatch Order Confirmed for **{selected_reroute_id}**!")
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.metric("Primary Carrier (Failed)", reroute_action["original_carrier_name"], reroute_action["original_carrier_id"])
        with r_c2:
            st.metric("Assigned Backup Carrier", reroute_action["backup_carrier_name"], reroute_action["backup_carrier_energy"])
        with r_c3:
            st.metric("Backup Quoted Rate", f"${reroute_action['backup_financial_cost_usd']:,.2f}", f"Duration: {reroute_action['backup_transit_duration_hours']:.1f}h")

        st.info(f"📋 **Action Taken:** {reroute_action['recommended_action']} | Timestamp: {reroute_action['timestamp']}")


# -------------------------------------------------------------------------------------------------
# TAB 6: HISTORICAL BI & ANALYTICS
# -------------------------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 📊 Historical Analytics & Business Intelligence Dashboard")
    st.markdown("Explore aggregated logistics performance across delivery volume, spend, OTIF fulfillment, and decarbonization.")

    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        d_start = st.date_input("Start Date", datetime(2026, 1, 1))
    with col_f2:
        d_end = st.date_input("End Date", datetime(2026, 10, 31))
    with col_f3:
        all_cids = df_carriers["carrier_id"].tolist()
        filt_carriers = st.multiselect("Filter Carriers", all_cids, default=[])
    with col_f4:
        all_cargos = ["General", "Refrigerated", "Hazmat", "High-Value", "Bulk"]
        filt_cargos = st.multiselect("Filter Cargo Types", all_cargos, default=[])

    df_filtered_hist = filter_historical_shipments(
        df=df_shipments,
        start_date=str(d_start),
        end_date=str(d_end),
        carriers=filt_carriers,
        cargo_types=filt_cargos,
        df_carriers=df_carriers,
    )

    kpi_hist = compute_aggregated_metrics(df_filtered_hist, df_carriers)

    # Metric Row
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        st.metric("Total Shipments", f"{kpi_hist['total_shipments']:,}")
    with h2:
        st.metric("Total Spend", f"${kpi_hist['total_spend_usd']:,.0f}")
    with h3:
        st.metric("Avg Cost / km", f"${kpi_hist['avg_cost_per_km']:.2f}")
    with h4:
        st.metric("OTIF Fulfillment", f"{kpi_hist['otif_pct']}%", f"{kpi_hist['otd_pct']}% OTD")
    with h5:
        st.metric("In-Full Delivery", f"{kpi_hist['in_full_pct']}%")
    with h6:
        st.metric("Total CO2 Emissions", f"{kpi_hist['total_co2_kg']/1000:,.1f} tons")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(plot_time_series_trend(df_filtered_hist), use_container_width=True)
    with col_g2:
        st.plotly_chart(plot_emissions_by_fuel(df_filtered_hist, df_carriers), use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.plotly_chart(plot_cost_vs_risk_matrix(df_filtered_hist.head(200)), use_container_width=True)
    with col_g4:
        st.plotly_chart(plot_delay_distribution(df_filtered_hist), use_container_width=True)

    csv_data = df_filtered_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Shipment Records (CSV)",
        data=csv_data,
        file_name=f"omnidata_filtered_shipments_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# -------------------------------------------------------------------------------------------------
# TAB 7: CARRIER MANAGEMENT & ONBOARDING
# -------------------------------------------------------------------------------------------------
with tabs[6]:
    st.markdown("### 🚛 Carrier Fleet Management & Dynamic Onboarding")
    st.markdown("Onboard new transport providers with operational parameters and update the live fleet database.")

    with st.expander("➕ Onboard New Freight Carrier", expanded=False):
        with st.form("carrier_onboard_form"):
            co1, co2, co3 = st.columns(3)
            with co1:
                new_cid = st.text_input("Carrier ID", f"CAR-{len(df_carriers)+1:03d}")
                new_cname = st.text_input("Carrier Legal Name", "TransEuropa Clean Cargo")
                new_energy = st.selectbox("Energy Vector", ["Electric", "Hydrogen", "Bio-LNG", "HVO", "LNG", "Diesel"])
                new_base_rate = st.number_input("Base Rate (USD)", 50.0, 1000.0, 190.0, 10.0)
            with co2:
                new_rate_km = st.number_input("Rate per km (USD)", 0.5, 5.0, 1.48, 0.05)
                new_fuel_surch = st.slider("Fuel Surcharge (%)", 0.0, 0.35, 0.10, 0.01)
                new_range = st.number_input("Effective Range (km)", 100.0, 2500.0, 1200.0, 50.0)
                new_fleet_size = st.number_input("Fleet Trucks Capacity", 5, 1000, 80, 5)
            with co3:
                new_otd = st.slider("Historical OTD Rate (0.0 to 1.0)", 0.70, 1.00, 0.96, 0.01)
                new_infull = st.slider("Historical In-Full Rate (0.0 to 1.0)", 0.70, 1.00, 0.98, 0.01)
                new_co2 = st.number_input("Carbon Intensity (g CO2e/tkm)", 1.0, 120.0, 22.0, 1.0)
                new_noise = st.number_input("Noise Pollution (dB)", 40.0, 90.0, 68.0, 1.0)

            submit_carrier = st.form_submit_button("🚀 Onboard & Persist Carrier to Fleet", use_container_width=True)

        if submit_carrier:
            carrier_entry = {
                "carrier_id": new_cid,
                "carrier_name": new_cname,
                "base_rate_usd": new_base_rate,
                "rate_per_km": new_rate_km,
                "fuel_surcharge_pct": new_fuel_surch,
                "otd_historical": new_otd,
                "in_full_historical": new_infull,
                "co2_g_per_ton_km": new_co2,
                "energy_source": new_energy,
                "effective_range_km": new_range,
                "fleet_capacity_trucks": int(new_fleet_size),
                "noise_pollution_db": new_noise,
            }
            append_new_carrier(carrier_entry)
            st.success(f"✅ Carrier **{new_cname}** ({new_cid}) successfully onboarded to `carrier_rates.csv`!")
            st.rerun()

    st.markdown("##### 📋 Active Onboarded Carriers Catalog")
    st.dataframe(df_carriers, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------------------------------
# TAB 8: ESG & SUSTAINABILITY REPORTING (ISO 14083)
# -------------------------------------------------------------------------------------------------
with tabs[7]:
    st.markdown("### 🌱 ISO 14083 Carbon Accounting & Audit Certificate Generator")
    st.markdown(
        "Quantify transport chain greenhouse gas emissions conforming to the **ISO 14083:2023** standard "
        "and GLEC Framework v3.0, generating audit-ready PDF sustainability certificates."
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        esg_carrier = st.selectbox("Select Assigned Carrier", df_carriers["carrier_name"].tolist())
        selected_carrier_row = df_carriers[df_carriers["carrier_name"] == esg_carrier].iloc[0]
        esg_vector = str(selected_carrier_row["energy_source"])
        st.info(f"Assigned Energy Vector: **{esg_vector}**")
    with col_e2:
        esg_weight_tons = st.number_input("Payload Weight (Metric Tons)", 1.0, 30.0, 16.5, 0.5)
        esg_distance = st.number_input("Transport Distance (km)", 20.0, 3000.0, 720.0, 20.0)
    with col_e3:
        esg_shipment_ref = st.text_input("Shipment / Audit ID", "SHP-AUDIT-2026-EU")
        st.write("")
        generate_cert_btn = st.button("📜 Generate Official ISO 14083 PDF Certificate", use_container_width=True)

    # Compute emissions
    esg_results = calculate_iso14083_emissions(
        weight_tons=esg_weight_tons,
        distance_km=esg_distance,
        energy_vector=esg_vector,
    )

    # Metric Row
    em1, em2, em3, em4 = st.columns(4)
    with em1:
        st.metric("Total WTW Emissions", f"{esg_results['wtw_co2e_kg']:,.1f} kg CO2e", f"{esg_results['emission_intensity_g_tkm']} g/tkm")
    with em2:
        st.metric("WTT (Upstream) Emissions", f"{esg_results['wtt_co2e_kg']:,.1f} kg CO2e")
    with em3:
        st.metric("TTW (Direct Tailpipe)", f"{esg_results['ttw_co2e_kg']:,.1f} kg CO2e")
    with em4:
        st.metric("Net Avoided Carbon vs Diesel", f"-{esg_results['co2_saved_kg']:,.1f} kg", f"-{esg_results['co2_reduction_pct']}%")

    # Comparison across all energy vectors
    vectors = ["Electric", "Hydrogen", "Bio-LNG", "HVO", "LNG", "Diesel"]
    vector_comp = []
    for v in vectors:
        res = calculate_iso14083_emissions(esg_weight_tons, esg_distance, v)
        vector_comp.append({
            "Energy Vector": v,
            "Total WTW (kg CO2e)": res["wtw_co2e_kg"],
            "WTT Upstream": res["wtt_co2e_kg"],
            "TTW Direct": res["ttw_co2e_kg"],
            "Carbon Intensity (g/tkm)": res["emission_intensity_g_tkm"],
        })
    df_vector_comp = pd.DataFrame(vector_comp)

    fig_esg = px.bar(
        df_vector_comp,
        x="Energy Vector",
        y="Total WTW (kg CO2e)",
        color="Energy Vector",
        text="Total WTW (kg CO2e)",
        title="<b>ISO 14083 Well-to-Wheel (WTW) Carbon Footprint Comparison</b>",
        color_discrete_map={
            "Electric": "#10b981",
            "Hydrogen": "#06b6d4",
            "Bio-LNG": "#84cc16",
            "HVO": "#22c55e",
            "LNG": "#f59e0b",
            "Diesel": "#64748b",
        },
        template="plotly_white",
    )
    fig_esg.update_traces(texttemplate="%{text:,.1f} kg", textposition="outside")
    fig_esg.update_layout(margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig_esg, use_container_width=True)

    # PDF Certificate generation
    if generate_cert_btn or "cert_pdf_bytes" in st.session_state:
        pdf_out_path = Path("data") / f"certificate_{esg_shipment_ref}.pdf"
        cert_data = {
            "shipment_id": esg_shipment_ref,
            "carrier_name": esg_carrier,
            "origin_city": "Madrid Cargo Hub",
            "destination_city": "Frankfurt Gateway",
            "weight_tons": esg_weight_tons,
            "distance_km": esg_distance,
            "energy_source": esg_vector,
        }
        generated_file = generate_pdf_esg_report(cert_data, str(pdf_out_path))

        with open(generated_file, "rb") as f:
            pdf_bytes = f.read()

        st.session_state["cert_pdf_bytes"] = pdf_bytes
        st.success(f"✅ Official ISO 14083 Certificate generated successfully!")

        st.download_button(
            label="📥 Download Official ISO 14083 PDF Carbon Certificate",
            data=pdf_bytes,
            file_name=f"ISO14083_Certificate_{esg_shipment_ref}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
