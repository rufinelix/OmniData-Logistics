"""Synthetic and production-grade dataset generator for OmniData-Logistics.

Generates:
1. data/carrier_rates.csv (10 comprehensive real-world carrier profiles)
2. data/refueling_stations.csv (20 corridor stations across European transport routes)
3. data/synthetic_shipments.csv (2,000 realistic historical shipment logs)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CARRIERS_DATA = [
    {
        "carrier_id": "CAR-001",
        "carrier_name": "EcoTrans European Logistics",
        "base_rate_usd": 180.0,
        "rate_per_km": 1.45,
        "fuel_surcharge_pct": 0.12,
        "otd_historical": 0.965,
        "in_full_historical": 0.985,
        "co2_g_per_ton_km": 28.5,
        "energy_source": "Bio-LNG",
        "effective_range_km": 1100.0,
        "fleet_capacity_trucks": 85,
        "noise_pollution_db": 68.0,
    },
    {
        "carrier_id": "CAR-002",
        "carrier_name": "VoltFreight Heavy Transport",
        "base_rate_usd": 240.0,
        "rate_per_km": 1.65,
        "fuel_surcharge_pct": 0.05,
        "otd_historical": 0.940,
        "in_full_historical": 0.990,
        "co2_g_per_ton_km": 8.2,
        "energy_source": "Electric",
        "effective_range_km": 420.0,
        "fleet_capacity_trucks": 45,
        "noise_pollution_db": 58.5,
    },
    {
        "carrier_id": "CAR-003",
        "carrier_name": "Continental Road Express",
        "base_rate_usd": 150.0,
        "rate_per_km": 1.30,
        "fuel_surcharge_pct": 0.18,
        "otd_historical": 0.920,
        "in_full_historical": 0.955,
        "co2_g_per_ton_km": 62.0,
        "energy_source": "Diesel",
        "effective_range_km": 1800.0,
        "fleet_capacity_trucks": 220,
        "noise_pollution_db": 76.5,
    },
    {
        "carrier_id": "CAR-004",
        "carrier_name": "HydroHaul Green Corridors",
        "base_rate_usd": 290.0,
        "rate_per_km": 1.85,
        "fuel_surcharge_pct": 0.08,
        "otd_historical": 0.950,
        "in_full_historical": 0.980,
        "co2_g_per_ton_km": 4.5,
        "energy_source": "Hydrogen",
        "effective_range_km": 750.0,
        "fleet_capacity_trucks": 30,
        "noise_pollution_db": 62.0,
    },
    {
        "carrier_id": "CAR-005",
        "carrier_name": "Nordic Clean Lines",
        "base_rate_usd": 195.0,
        "rate_per_km": 1.50,
        "fuel_surcharge_pct": 0.10,
        "otd_historical": 0.975,
        "in_full_historical": 0.992,
        "co2_g_per_ton_km": 18.0,
        "energy_source": "HVO",
        "effective_range_km": 1500.0,
        "fleet_capacity_trucks": 110,
        "noise_pollution_db": 71.0,
    },
    {
        "carrier_id": "CAR-006",
        "carrier_name": "BlueCorridor Gas Freight",
        "base_rate_usd": 170.0,
        "rate_per_km": 1.38,
        "fuel_surcharge_pct": 0.14,
        "otd_historical": 0.935,
        "in_full_historical": 0.965,
        "co2_g_per_ton_km": 44.0,
        "energy_source": "LNG",
        "effective_range_km": 1250.0,
        "fleet_capacity_trucks": 140,
        "noise_pollution_db": 70.0,
    },
    {
        "carrier_id": "CAR-007",
        "carrier_name": "Iberia Prime Logistics",
        "base_rate_usd": 155.0,
        "rate_per_km": 1.32,
        "fuel_surcharge_pct": 0.17,
        "otd_historical": 0.910,
        "in_full_historical": 0.945,
        "co2_g_per_ton_km": 64.5,
        "energy_source": "Diesel",
        "effective_range_km": 1750.0,
        "fleet_capacity_trucks": 190,
        "noise_pollution_db": 77.0,
    },
    {
        "carrier_id": "CAR-008",
        "carrier_name": "Alpine EcoShuttle",
        "base_rate_usd": 225.0,
        "rate_per_km": 1.60,
        "fuel_surcharge_pct": 0.06,
        "otd_historical": 0.960,
        "in_full_historical": 0.988,
        "co2_g_per_ton_km": 9.5,
        "energy_source": "Electric",
        "effective_range_km": 450.0,
        "fleet_capacity_trucks": 55,
        "noise_pollution_db": 59.0,
    },
    {
        "carrier_id": "CAR-009",
        "carrier_name": "Rhine-Alpine Green Cargo",
        "base_rate_usd": 185.0,
        "rate_per_km": 1.42,
        "fuel_surcharge_pct": 0.11,
        "otd_historical": 0.955,
        "in_full_historical": 0.978,
        "co2_g_per_ton_km": 24.0,
        "energy_source": "Bio-LNG",
        "effective_range_km": 1150.0,
        "fleet_capacity_trucks": 95,
        "noise_pollution_db": 67.5,
    },
    {
        "carrier_id": "CAR-010",
        "carrier_name": "Apex Sustainable Freight",
        "base_rate_usd": 205.0,
        "rate_per_km": 1.52,
        "fuel_surcharge_pct": 0.09,
        "otd_historical": 0.970,
        "in_full_historical": 0.989,
        "co2_g_per_ton_km": 16.5,
        "energy_source": "HVO",
        "effective_range_km": 1400.0,
        "fleet_capacity_trucks": 75,
        "noise_pollution_db": 70.5,
    },
]

STATIONS_DATA = [
    {"station_id": "ST-01", "station_name": "Milano Cargo Hub Supercharger", "corridor_name": "Mediterranean-Corridor", "latitude": 45.4642, "longitude": 9.1900, "fuel_type": "Electric", "charger_power_kw": 350.0, "avg_detour_km": 4.2},
    {"station_id": "ST-02", "station_name": "Lyon Sud Megawatt Depot", "corridor_name": "Rhone-Alps-Corridor", "latitude": 45.7640, "longitude": 4.8357, "fuel_type": "Electric", "charger_power_kw": 400.0, "avg_detour_km": 6.1},
    {"station_id": "ST-03", "station_name": "Barcelona Port LNG & Bio-LNG", "corridor_name": "Mediterranean-Corridor", "latitude": 41.3879, "longitude": 2.1699, "fuel_type": "Bio-LNG", "charger_power_kw": 0.0, "avg_detour_km": 5.0},
    {"station_id": "ST-04", "station_name": "Zaragoza Plaza Logistics LNG", "corridor_name": "Atlantic-Mediterranean-Link", "latitude": 41.6488, "longitude": -0.8891, "fuel_type": "LNG", "charger_power_kw": 0.0, "avg_detour_km": 3.8},
    {"station_id": "ST-05", "station_name": "Madrid San Fernando H2 Station", "corridor_name": "Central-Iberian-Hub", "latitude": 40.4168, "longitude": -3.7038, "fuel_type": "Hydrogen", "charger_power_kw": 0.0, "avg_detour_km": 7.5},
    {"station_id": "ST-06", "station_name": "Frankfurt Gateway Fast Charger", "corridor_name": "Rhine-Alpine-Corridor", "latitude": 50.1109, "longitude": 8.6821, "fuel_type": "Electric", "charger_power_kw": 300.0, "avg_detour_km": 3.2},
    {"station_id": "ST-07", "station_name": "Rotterdam Maasvlakte HVO Depot", "corridor_name": "North-Sea-Baltic", "latitude": 51.9244, "longitude": 4.4777, "fuel_type": "HVO", "charger_power_kw": 0.0, "avg_detour_km": 2.5},
    {"station_id": "ST-08", "station_name": "Munich Ost Megacharger", "corridor_name": "Scandinavian-Mediterranean", "latitude": 48.1351, "longitude": 11.5820, "fuel_type": "Electric", "charger_power_kw": 350.0, "avg_detour_km": 5.8},
    {"station_id": "ST-09", "station_name": "Stuttgart H2 Heavy Refueler", "corridor_name": "Rhine-Danube-Corridor", "latitude": 48.7758, "longitude": 9.1829, "fuel_type": "Hydrogen", "charger_power_kw": 0.0, "avg_detour_km": 8.0},
    {"station_id": "ST-10", "station_name": "Antwerp Gateway LNG Station", "corridor_name": "North-Sea-Mediterranean", "latitude": 51.2194, "longitude": 4.4025, "fuel_type": "LNG", "charger_power_kw": 0.0, "avg_detour_km": 4.0},
    {"station_id": "ST-11", "station_name": "Paris Nord Garonor Bio-LNG", "corridor_name": "Atlantic-Corridor", "latitude": 48.8566, "longitude": 2.3522, "fuel_type": "Bio-LNG", "charger_power_kw": 0.0, "avg_detour_km": 6.5},
    {"station_id": "ST-12", "station_name": "Bordeaux Sud HVO Distribution", "corridor_name": "Atlantic-Corridor", "latitude": 44.8378, "longitude": -0.5792, "fuel_type": "HVO", "charger_power_kw": 0.0, "avg_detour_km": 3.9},
    {"station_id": "ST-13", "station_name": "Valencia Riba-Roja Electric Park", "corridor_name": "Mediterranean-Corridor", "latitude": 39.4699, "longitude": -0.3763, "fuel_type": "Electric", "charger_power_kw": 300.0, "avg_detour_km": 4.5},
    {"station_id": "ST-14", "station_name": "Basel Border Clean Fuel Hub", "corridor_name": "Rhine-Alpine-Corridor", "latitude": 47.5596, "longitude": 7.5886, "fuel_type": "HVO", "charger_power_kw": 0.0, "avg_detour_km": 2.8},
    {"station_id": "ST-15", "station_name": "Innsbruck Alpine H2 Gateway", "corridor_name": "Brenner-Corridor", "latitude": 47.2692, "longitude": 11.4041, "fuel_type": "Hydrogen", "charger_power_kw": 0.0, "avg_detour_km": 9.2},
    {"station_id": "ST-16", "station_name": "Verona Quadrante Europa LNG", "corridor_name": "Brenner-Corridor", "latitude": 45.4384, "longitude": 10.9916, "fuel_type": "LNG", "charger_power_kw": 0.0, "avg_detour_km": 3.1},
    {"station_id": "ST-17", "station_name": "Dijon Bourgogne Fast Charger", "corridor_name": "Rhone-Saone-Link", "latitude": 47.3220, "longitude": 5.0415, "fuel_type": "Electric", "charger_power_kw": 250.0, "avg_detour_km": 5.4},
    {"station_id": "ST-18", "station_name": "Mannheim Intermodal Bio-LNG", "corridor_name": "Rhine-Alpine-Corridor", "latitude": 49.4875, "longitude": 8.4660, "fuel_type": "Bio-LNG", "charger_power_kw": 0.0, "avg_detour_km": 4.1},
    {"station_id": "ST-19", "station_name": "Toulouse Eurocentre HVO Depot", "corridor_name": "Pyrenean-Corridor", "latitude": 43.6047, "longitude": 1.4442, "fuel_type": "HVO", "charger_power_kw": 0.0, "avg_detour_km": 3.6},
    {"station_id": "ST-20", "station_name": "Strasbourg Port Electric Charger", "corridor_name": "Rhine-Frontier-Link", "latitude": 48.5734, "longitude": 7.7521, "fuel_type": "Electric", "charger_power_kw": 300.0, "avg_detour_km": 4.0},
]

CITIES = [
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Barcelona", "lat": 41.3879, "lon": 2.1699},
    {"name": "Valencia", "lat": 39.4699, "lon": -0.3763},
    {"name": "Zaragoza", "lat": 41.6488, "lon": -0.8891},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Lyon", "lat": 45.7640, "lon": 4.8357},
    {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
    {"name": "Munich", "lat": 48.1351, "lon": 11.5820},
    {"name": "Rotterdam", "lat": 51.9244, "lon": 4.4777},
    {"name": "Milan", "lat": 45.4642, "lon": 9.1900},
]

CARGO_TYPES = ["General", "Refrigerated", "Hazmat", "High-Value", "Bulk"]


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two GPS points using Haversine formula."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c * 1.25)  # 1.25 winding factor for road travel


def generate_carrier_rates(output_path: Path) -> pd.DataFrame:
    """Generate carrier_rates.csv with 10 real-world carrier profiles."""
    df = pd.DataFrame(CARRIERS_DATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_refueling_stations(output_path: Path) -> pd.DataFrame:
    """Generate refueling_stations.csv with 20 corridor stations."""
    df = pd.DataFrame(STATIONS_DATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_synthetic_shipments(output_path: Path, n_records: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate 2,000 realistic shipment logs."""
    np.random.seed(seed)
    carrier_ids = [c["carrier_id"] for c in CARRIERS_DATA]
    carrier_otd_map = {c["carrier_id"]: c["otd_historical"] for c in CARRIERS_DATA}
    carrier_in_full_map = {c["carrier_id"]: c["in_full_historical"] for c in CARRIERS_DATA}

    records = []
    base_start_date = datetime(2026, 1, 1, 6, 0, 0)

    for i in range(1, n_records + 1):
        shipment_id = f"SHP-{i:05d}"
        order_id = f"ORD-{100000 + i}"
        carrier_id = str(np.random.choice(carrier_ids))

        # Select two distinct cities
        idx1, idx2 = np.random.choice(len(CITIES), size=2, replace=False)
        origin = CITIES[idx1]
        dest = CITIES[idx2]

        distance_km = round(_haversine_distance(origin["lat"], origin["lon"], dest["lat"], dest["lon"]), 1)
        cargo_type = str(np.random.choice(CARGO_TYPES, p=[0.45, 0.25, 0.10, 0.10, 0.10]))
        cargo_weight_kg = round(float(np.random.uniform(1200.0, 24000.0)), 1)
        ordered_qty = int(np.random.randint(20, 500))

        # Timestamp generation
        days_offset = float(np.random.uniform(0, 270))
        hour_offset = float(np.random.uniform(6, 20))
        creation_dt = base_start_date + timedelta(days=days_offset, hours=hour_offset)

        # Expected speed approx 70 km/h + mandatory driver breaks
        transit_hours_nominal = max(3.0, distance_km / 70.0 + (distance_km // 400.0) * 0.75)
        promised_dt = creation_dt + timedelta(hours=transit_hours_nominal + float(np.random.uniform(2.0, 8.0)))

        # Environmental risk factors
        weather_risk_index = round(float(np.random.exponential(scale=0.45)), 2)
        traffic_delay_minutes = round(float(np.random.gamma(shape=2.5, scale=12.0)), 1)

        # Delay computation influenced by carrier reliability, distance, weather & traffic
        carrier_otd = carrier_otd_map[carrier_id]
        is_delayed_carrier = np.random.random() > carrier_otd
        base_delay_hrs = 0.0
        if is_delayed_carrier:
            base_delay_hrs = float(np.random.exponential(scale=3.5))

        weather_delay_hrs = max(0.0, (weather_risk_index - 0.7) * 2.5) if weather_risk_index > 0.7 else 0.0
        traffic_delay_hrs = traffic_delay_minutes / 60.0

        # Total actual delay in hours (negative or zero means on-time)
        actual_delay_hours = round(
            float(np.random.normal(loc=-1.2, scale=0.6)) if not is_delayed_carrier
            else (base_delay_hrs + weather_delay_hrs + traffic_delay_hrs * 0.5),
            2
        )

        actual_dt = promised_dt + timedelta(hours=max(-6.0, actual_delay_hours))

        # In-full fulfillment
        carrier_infull = carrier_in_full_map[carrier_id]
        if np.random.random() <= carrier_infull:
            delivered_qty_ok = ordered_qty
        else:
            damaged = int(np.random.randint(1, max(2, int(ordered_qty * 0.15))))
            delivered_qty_ok = ordered_qty - damaged

        records.append({
            "shipment_id": shipment_id,
            "order_id": order_id,
            "carrier_id": carrier_id,
            "origin_city": origin["name"],
            "destination_city": dest["name"],
            "origin_lat": origin["lat"],
            "origin_lon": origin["lon"],
            "dest_lat": dest["lat"],
            "dest_lon": dest["lon"],
            "distance_km": distance_km,
            "cargo_weight_kg": cargo_weight_kg,
            "cargo_type": cargo_type,
            "creation_timestamp": creation_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "promised_delivery_timestamp": promised_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_delivery_timestamp": actual_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ordered_qty": ordered_qty,
            "delivered_qty_ok": delivered_qty_ok,
            "actual_delay_hours": actual_delay_hours,
            "weather_risk_index": weather_risk_index,
            "traffic_delay_minutes": traffic_delay_minutes,
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """Generate all baseline datasets."""
    print("Generating baseline production datasets...")
    rates_file = DATA_DIR / "carrier_rates.csv"
    stations_file = DATA_DIR / "refueling_stations.csv"
    shipments_file = DATA_DIR / "synthetic_shipments.csv"

    df_rates = generate_carrier_rates(rates_file)
    print(f"Generated {len(df_rates)} carrier profiles at {rates_file}")

    df_stations = generate_refueling_stations(stations_file)
    print(f"Generated {len(df_stations)} refueling stations at {stations_file}")

    df_shipments = generate_synthetic_shipments(shipments_file, n_records=2000)
    print(f"Generated {len(df_shipments)} shipment logs at {shipments_file}")


if __name__ == "__main__":
    main()
