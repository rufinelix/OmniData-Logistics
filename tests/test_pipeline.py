"""Core pipeline unit tests for OmniData-Logistics.

Validates:
1. RouteConfigManager JSON persistence, mutation, and reset.
2. Vectorized logistics metrics (OTD, In-Full, OTIF, financial cost).
3. Range optimizer vehicle autonomy and constraint disqualifications.
4. MCDM LogisticsOptimizer weight normalization and carrier ranking.
5. ISO 14083 ESG emission calculations and PDF reporting.
6. LTL consolidation bundling economics.
7. Control Tower exception detection and automated backup rerouting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.consolidation import detect_ltl_consolidation_opportunities
from src.control_tower import monitor_active_shipments, trigger_automated_reroute
from src.esg_reporting import calculate_iso14083_emissions, generate_pdf_esg_report
from src.metrics import calculate_financial_cost, calculate_in_full, calculate_otd, calculate_otif
from src.optimizer import LogisticsOptimizer
from src.range_optimizer import validate_route_autonomy
from src.route_config import FACTORY_DEFAULT_CONSTRAINTS, RouteConfigManager


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary JSON config path."""
    return tmp_path / "test_constraints.json"


@pytest.fixture
def sample_carriers_df() -> pd.DataFrame:
    """Fixture providing sample carrier profiles."""
    return pd.DataFrame([
        {
            "carrier_id": "CAR-001",
            "carrier_name": "Alpha Green Lines",
            "base_rate_usd": 150.0,
            "rate_per_km": 1.40,
            "fuel_surcharge_pct": 0.10,
            "otd_historical": 0.95,
            "in_full_historical": 0.98,
            "co2_g_per_ton_km": 15.0,
            "energy_source": "Electric",
            "effective_range_km": 400.0,
            "fleet_capacity_trucks": 50,
            "noise_pollution_db": 60.0,
        },
        {
            "carrier_id": "CAR-002",
            "carrier_name": "Beta Standard Haulage",
            "base_rate_usd": 120.0,
            "rate_per_km": 1.30,
            "fuel_surcharge_pct": 0.15,
            "otd_historical": 0.90,
            "in_full_historical": 0.94,
            "co2_g_per_ton_km": 62.0,
            "energy_source": "Diesel",
            "effective_range_km": 1600.0,
            "fleet_capacity_trucks": 150,
            "noise_pollution_db": 75.0,
        },
    ])


class TestRouteConfigManager:
    """Tests for RouteConfigManager JSON operations."""

    def test_load_defaults_when_file_missing(self, temp_config_file: Path) -> None:
        manager = RouteConfigManager(config_path=temp_config_file)
        config = manager.load_config()
        assert config["max_transit_time_hours"] == 48.0
        assert config["max_detour_km_allowed"] == 30.0
        assert config["safety_battery_buffer_pct"] == 0.15
        assert temp_config_file.exists()

    def test_update_and_persist_config(self, temp_config_file: Path) -> None:
        manager = RouteConfigManager(config_path=temp_config_file)
        manager.update_config({"max_detour_km_allowed": 42.5, "shadow_carbon_price_per_ton": 120.0})

        # Reload independently from disk
        fresh_manager = RouteConfigManager(config_path=temp_config_file)
        reloaded = fresh_manager.load_config()
        assert reloaded["max_detour_km_allowed"] == 42.5
        assert reloaded["shadow_carbon_price_per_ton"] == 120.0

    def test_reset_to_defaults(self, temp_config_file: Path) -> None:
        manager = RouteConfigManager(config_path=temp_config_file)
        manager.update_config({"max_detour_km_allowed": 99.0})
        reset_cfg = manager.reset_to_defaults()
        assert reset_cfg["max_detour_km_allowed"] == FACTORY_DEFAULT_CONSTRAINTS["max_detour_km_allowed"]


class TestMetricsEngine:
    """Tests for vectorized logistics metrics."""

    def test_vectorized_otd(self) -> None:
        df = pd.DataFrame({"actual_delay_hours": [-2.0, 0.0, 0.5, 3.2]})
        otd = calculate_otd(df)
        assert otd.tolist() == [True, True, False, False]

    def test_vectorized_in_full(self) -> None:
        df = pd.DataFrame({
            "ordered_qty": [100, 200, 50],
            "delivered_qty_ok": [100, 195, 50],
        })
        in_full = calculate_in_full(df)
        assert in_full.tolist() == [True, False, True]

    def test_vectorized_otif(self) -> None:
        df = pd.DataFrame({
            "actual_delay_hours": [-1.0, -0.5, 2.0, 0.0],
            "ordered_qty": [100, 100, 100, 100],
            "delivered_qty_ok": [100, 90, 100, 100],
        })
        otif = calculate_otif(df)
        # 0: on-time & in-full -> True
        # 1: on-time & short -> False
        # 2: late & in-full -> False
        # 3: on-time & in-full -> True
        assert otif.tolist() == [True, False, False, True]

    def test_financial_cost_calculation(self) -> None:
        df = pd.DataFrame({
            "base_rate_usd": [100.0, 200.0],
            "distance_km": [500.0, 1000.0],
            "rate_per_km": [1.50, 2.00],
            "fuel_surcharge_pct": [0.10, 0.20],
        })
        # Row 0: 100 + (500 * 1.5 * 1.10) = 100 + 825 = 925.0
        # Row 1: 200 + (1000 * 2.0 * 1.20) = 200 + 2400 = 2600.0
        costs = calculate_financial_cost(df)
        assert costs.tolist() == [925.0, 2600.0]


class TestRangeOptimizer:
    """Tests for range autonomy validation and disqualification rules."""

    def test_viable_within_autonomy(self) -> None:
        carrier = {
            "effective_range_km": 500.0,
            "base_rate_usd": 150.0,
            "rate_per_km": 1.5,
            "fuel_surcharge_pct": 0.1,
        }
        config = {"safety_battery_buffer_pct": 0.20, "max_detour_km_allowed": 30.0, "max_refueling_delay_minutes": 45.0, "green_premium_pct_limit": 0.15}
        # Usable range = 500 * 0.8 = 400 km. Route = 350 km.
        res = validate_route_autonomy(carrier, route_distance_km=350.0, fuel_type="Electric", config=config)
        assert res["is_viable"] is True
        assert res["stops_required"] == 0

    def test_disqualification_detour_limit(self) -> None:
        carrier = {
            "effective_range_km": 400.0,
            "base_rate_usd": 150.0,
            "rate_per_km": 1.5,
            "fuel_surcharge_pct": 0.1,
        }
        # Set max detour to an impossibly small number (0.1 km)
        config = {"safety_battery_buffer_pct": 0.15, "max_detour_km_allowed": 0.1, "max_refueling_delay_minutes": 60.0, "green_premium_pct_limit": 0.50}
        res = validate_route_autonomy(carrier, route_distance_km=900.0, fuel_type="Electric", config=config)
        assert res["is_viable"] is False
        assert "Disqualified: Refueling Detour Exceeds Limit" in res["status_flag"]


class TestLogisticsOptimizer:
    """Tests for MCDM normalization and carrier ranking."""

    def test_weight_normalization_and_silencing(self) -> None:
        opt = LogisticsOptimizer()
        weights = {"cost": 0.4, "speed": 0.0, "otd": 0.4, "otif": 0.2, "co2": 0.0}
        norm = opt.normalize_weights(weights)

        # Active sum = 0.4 + 0.4 + 0.2 = 1.0
        assert norm["cost"][0] == pytest.approx(0.40)
        assert norm["speed"][0] == 0.0
        assert norm["speed"][1] is True  # Silenced
        assert norm["co2"][1] is True  # Silenced

        total_active_w = sum(v[0] for v in norm.values())
        assert total_active_w == pytest.approx(1.0)

    def test_optimize_carrier_selection(self, sample_carriers_df: pd.DataFrame) -> None:
        opt = LogisticsOptimizer()
        ranked = opt.optimize_carrier_selection(
            df_carriers=sample_carriers_df,
            distance_km=300.0,
            cargo_weight_kg=10000.0,
            user_weights={"cost": 0.3, "speed": 0.2, "otd": 0.2, "otif": 0.1, "co2": 0.2},
        )
        assert len(ranked) == 2
        assert "composite_score" in ranked.columns
        assert "rank" in ranked.columns
        assert ranked.iloc[0]["composite_score"] >= ranked.iloc[1]["composite_score"]


class TestESGReporting:
    """Tests for ISO 14083 carbon accounting and PDF generation."""

    def test_iso14083_emissions(self) -> None:
        # 10 metric tons over 500 km = 5,000 ton-km
        # Electric intensity = 8.5 g/tkm -> 42.5 kg CO2e
        res = calculate_iso14083_emissions(weight_tons=10.0, distance_km=500.0, energy_vector="Electric")
        assert res["ton_km"] == 5000.0
        assert res["wtw_co2e_kg"] == 42.5
        assert res["ttw_co2e_kg"] == 0.0  # Zero tailpipe emissions for EV
        assert res["co2_saved_kg"] > 0.0

    def test_pdf_report_generation(self, tmp_path: Path) -> None:
        out_pdf = tmp_path / "test_certificate.pdf"
        data = {
            "shipment_id": "TEST-SHP-001",
            "carrier_name": "CleanCarrier",
            "distance_km": 450.0,
            "weight_tons": 12.0,
            "energy_source": "Bio-LNG",
        }
        pdf_path = generate_pdf_esg_report(data, str(out_pdf))
        assert Path(pdf_path).exists()
        assert Path(pdf_path).stat().st_size > 1000


class TestConsolidationAndControlTower:
    """Tests for LTL consolidation and Control Tower monitoring."""

    def test_ltl_consolidation(self) -> None:
        df_shipments = pd.DataFrame([
            {
                "shipment_id": "SHP-1",
                "origin_city": "Madrid",
                "destination_city": "Paris",
                "cargo_weight_kg": 2500.0,
                "distance_km": 1200.0,
                "creation_timestamp": "2026-05-01 08:00:00",
            },
            {
                "shipment_id": "SHP-2",
                "origin_city": "Madrid",
                "destination_city": "Paris",
                "cargo_weight_kg": 3000.0,
                "distance_km": 1200.0,
                "creation_timestamp": "2026-05-01 14:00:00",
            },
        ])
        opps = detect_ltl_consolidation_opportunities(df_shipments)
        assert len(opps) == 1
        assert opps.iloc[0]["financial_savings_usd"] > 0.0
        assert opps.iloc[0]["co2_reduction_kg"] > 0.0

    def test_control_tower_exception_monitoring(self, sample_carriers_df: pd.DataFrame) -> None:
        df_active = pd.DataFrame([
            {
                "shipment_id": "SHP-ERR-1",
                "carrier_id": "CAR-002",
                "cargo_type": "Refrigerated",
                "ambient_temp_c": 7.5,  # Exceeds tolerance
                "actual_delay_hours": 3.0,
                "weather_risk_index": 1.8,
            }
        ])
        monitored = monitor_active_shipments(df_active)
        assert monitored.iloc[0]["severity_level"] == "CRITICAL"
        assert "Cold Chain Deviation" in monitored.iloc[0]["exception_flags"]

        # Test reroute trigger
        reroute = trigger_automated_reroute("SHP-ERR-1", sample_carriers_df, df_active)
        assert reroute["status"] == "REROUTE_CONFIRMED"
        assert reroute["backup_carrier_id"] != "CAR-002"
