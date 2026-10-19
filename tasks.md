# OmniData-Logistics Implementation Roadmap

- [x] Phase 1: System Environment, Directory Structure & API Credentials
- [x] Phase 2: Dynamic Route Constraints & JSON Configuration Manager
- [x] Phase 3: Production Dataset Ingestion Engine & Carrier Onboarding
- [x] Phase 4: Core Logistics Metrics Engine & ISO 14083 ESG Reporting
- [x] Phase 5: Predictive ML Model & Monte Carlo Risk Engine
- [x] Phase 6: Live External API Connectors (NREL, OpenChargeMap, TomTom)
- [x] Phase 7: Geocoded Weather Routing & LTL Consolidation Engine
- [x] Phase 8: Alternative Fuel Range & Refueling Constraints Optimizer
- [x] Phase 9: Multi-Criteria Decision Making (MCDM) Optimization Engine
- [x] Phase 10: Control Tower Exception & Automated Rerouting Engine
- [x] Phase 11: History Analytics & Business Intelligence Engine
- [x] Phase 12: Route Constraint Management UI Tab
- [x] Phase 13: High-Performance Streamlit Dashboard UI Architecture
- [x] Phase 14: Pipeline Unit Testing & Automated Validation
- [x] Phase 15: Comprehensive Cross-File Code Audit & Performance Refactoring
- [x] Phase 16: Enterprise GitHub Repository & Documentation Setup

---

## Detailed Phase Instructions

### Phase 1: System Environment, Directory Structure & API Credentials
Act as a Senior AI Architect. Set up a modular Python application named 'OmniData-Logistics' following PEP 8 standards, zero mock code, and pure production-grade data pipelines.

Execution Steps:
1. Create virtual environment `venv`, activate it, and create `requirements.txt` containing: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `plotly`, `folium`, `streamlit-folium`, `requests`, `reportlab`, `python-dotenv`, `pytest`. Install all packages.
2. Create an `.env.example` file with exact environment variable placeholders:
   OPEN_METEO_API_KEY='your_open_meteo_api_key_here'
   OPENCHARGE_MAP_API_KEY='your_opencharge_map_api_key_here'
   HUGGING_FACE_API_KEY='your_hugging_face_api_key_here'
   TOMTOM_TRAFFIC_API_KEY='your_tomtom_traffic_api_key_here'
   MAPBOX_API_KEY='your_mapbox_api_key_here'
   NREL_AFDC_API_KEY='your_nrel_afdc_api_key_here'
3. Create `.env` copied from `.env.example`.
4. Create the exact project directory layout:
   - OmniData-Logistics/
     ├── .env.example
     ├── .env
     ├── data/
     │   ├── synthetic_shipments.csv
     │   ├── carrier_rates.csv
     │   ├── refueling_stations.csv
     │   └── route_constraints_config.json
     ├── src/
     │   ├── __init__.py
     │   ├── data_loader.py
     │   ├── route_config.py
     │   ├── metrics.py
     │   ├── ml_engine.py
     │   ├── risk_simulation.py
     │   ├── consolidation.py
     │   ├── esg_reporting.py
     │   ├── geo_routing.py
     │   ├── range_optimizer.py
     │   ├── optimizer.py
     │   ├── control_tower.py
     │   ├── history_analytics.py
     │   └── external_apis/
     │       ├── __init__.py
     │       ├── stations.py
     │       └── traffic.py
     ├── tests/
     │   ├── __init__.py
     │   ├── test_pipeline.py
     │   └── test_external_apis.py
     ├── app.py
     ├── requirements.txt
     └── README.md
5. All code, comments, docstrings, and UI strings MUST be written in English.

---

### Phase 2: Dynamic Route Constraints & JSON Configuration Manager
Act as a Senior Software Engineer. Implement `src/route_config.py` using `python-dotenv` and standard `json` modules following PEP 8.

Execution Steps:
1. Create `data/route_constraints_config.json` with exact key definitions:
   - `max_transit_time_hours`: float (default: 48.0)
   - `max_detour_km_allowed`: float (default: 30.0)
   - `max_refueling_delay_minutes`: float (default: 45.0)
   - `safety_battery_buffer_pct`: float (default: 0.15)
   - `green_premium_pct_limit`: float (default: 0.15)
   - `shadow_carbon_price_per_ton`: float (default: 80.0)
   - `hazmat_restricted_corridors`: list[str] (e.g., ["MAD-BCN-Tunnel", "ALP-Pass"])
   - `refrigerated_temp_tolerance_c`: float (default: 2.0)
2. Implement class `RouteConfigManager` in `src/route_config.py` with type-annotated methods:
   - `load_config() -> dict[str, Any]`: Reads JSON configuration file.
   - `update_config(new_params: dict[str, Any]) -> None`: Mutates configuration dictionary in-memory and persists to `route_constraints_config.json`.
   - `reset_to_defaults() -> dict[str, Any]`: Restores factory default JSON values.

---

### Phase 3: Production Dataset Ingestion Engine & Carrier Onboarding
Act as a Senior AI Engineer. Implement `src/data_generator.py` and `src/data_loader.py` to populate production-grade initial datasets and build the dynamic onboarding engine.

Execution Steps:
1. Generate and save initial production-spec CSV files in `data/`:
   - `carrier_rates.csv` (10 real-world carrier profiles): `carrier_id` (str), `carrier_name` (str), `base_rate_usd` (float), `rate_per_km` (float), `fuel_surcharge_pct` (float), `otd_historical` (float), `in_full_historical` (float), `co2_g_per_ton_km` (float), `energy_source` (str: Diesel, LNG, Bio-LNG, Electric, HVO, Hydrogen), `effective_range_km` (float), `fleet_capacity_trucks` (int), `noise_pollution_db` (float).
   - `refueling_stations.csv` (20 real corridor stations): `station_id` (str), `station_name` (str), `corridor_name` (str), `latitude` (float), `longitude` (float), `fuel_type` (str), `charger_power_kw` (float), `avg_detour_km` (float).
   - `synthetic_shipments.csv` (2,000 real shipment logs): `shipment_id`, `order_id`, `carrier_id`, `origin_city`, `destination_city`, `origin_lat`, `origin_lon`, `dest_lat`, `dest_lon`, `distance_km`, `cargo_weight_kg`, `cargo_type`, `creation_timestamp`, `promised_delivery_timestamp`, `actual_delivery_timestamp`, `ordered_qty`, `delivered_qty_ok`, `actual_delay_hours`, `weather_risk_index`, `traffic_delay_minutes`.
2. Implement `append_new_carrier(carrier_data: dict[str, Any]) -> None` in `src/data_loader.py` to append onboarding entries to `carrier_rates.csv` cleanly.
3. Execute `src/data_generator.py` via CLI to create initial datasets.

---

### Phase 4: Core Logistics Metrics Engine & ISO 14083 ESG Reporting Module
Act as a Senior AI Engineer. Implement `src/metrics.py` and `src/esg_reporting.py` using vectorized Pandas/NumPy operations and ReportLab.

Execution Steps:
1. In `src/metrics.py`, write vectorized PEP 8 functions accepting Pandas DataFrames:
   - `calculate_otd(df: pd.DataFrame) -> pd.Series`: Vectorized check `actual_delay_hours <= 0`.
   - `calculate_in_full(df: pd.DataFrame) -> pd.Series`: Vectorized check `delivered_qty_ok == ordered_qty`.
   - `calculate_otif(df: pd.DataFrame) -> pd.Series`: Vectorized Boolean AND of OTD and In-Full.
   - `calculate_financial_cost(df: pd.DataFrame) -> pd.Series`: `base_rate + (distance_km * rate_per_km * (1 + fuel_surcharge_pct))`.
2. In `src/esg_reporting.py`:
   - `calculate_iso14083_emissions(weight_tons: float, distance_km: float, energy_vector: str) -> dict[str, float]`: Computes WTW GHG emissions (kg CO2e) and emission intensity (gCO2e / tkm).
   - `generate_pdf_esg_report(shipment_data: dict[str, Any], output_path: str) -> str`: Uses ReportLab canvas API to build a downloadable PDF carbon certificate.

---

### Phase 5: Predictive Machine Learning Model & Monte Carlo Risk Engine
Act as a Senior AI Engineer. Implement `src/ml_engine.py` and `src/risk_simulation.py` engineered for real-time execution speed.

Execution Steps:
1. In `src/ml_engine.py`:
   - Train an `XGBoostRegressor` on `synthetic_shipments.csv` predicting `actual_delay_hours` from: `distance_km`, `cargo_weight_kg`, `carrier_id`, `weather_risk_index`, `traffic_delay_minutes`, `day_of_week`.
   - Save trained model pipeline using `joblib` to `data/ml_delay_model.joblib`.
2. In `src/risk_simulation.py`:
   - Implement `run_monte_carlo_simulation(carrier_id: str, distance_km: float, weather_index: float, iterations: int = 1000) -> dict[str, float]`.
   - Use vectorized `numpy.random.normal()` sampling over velocity and traffic delay variance.
   - Return dict keys: `p95_eta_hours` (float), `confidence_interval_90` (tuple[float, float]), `sla_compliance_probability` (float).

---

### Phase 6: Live External API Connectors (NREL AFDC, OpenChargeMap & TomTom Traffic)
Act as a Senior AI Engineer. Implement `src/external_apis/stations.py` and `src/external_apis/traffic.py` for real-time external API fetching.

Execution Steps:
1. In `src/external_apis/stations.py`:
   - Implement `fetch_real_lng_stations(latitude: float, longitude: float, radius_km: float = 100.0) -> pd.DataFrame`: Connects to NREL AFDC REST API using `NREL_AFDC_API_KEY` to retrieve operational LNG/CNG/HY stations.
   - Implement `fetch_real_electric_charging(latitude: float, longitude: float, distance_km: float = 50.0) -> pd.DataFrame`: Connects to OpenChargeMap API using `OPENCHARGE_MAP_API_KEY` filtering high-power commercial chargers (>= 150 kW).
2. In `src/external_apis/traffic.py`:
   - Implement `fetch_traffic_incidents_and_delay(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> dict[str, float]`: Connects to TomTom Routing & Traffic API using `TOMTOM_TRAFFIC_API_KEY` with `travelMode='truck'` to retrieve real-time congestion delay minutes and traffic severity index.
3. If an API key is missing or invalid in `.env`, the function must log an explicit warning and raise an `APIConfigurationError` or gracefully fall back to local dataset queries without crashing the execution pipeline.

---

### Phase 7: Geocoded Weather API Routing & LTL Consolidation Engine
Act as a Senior Software Engineer. Implement `src/geo_routing.py` and `src/consolidation.py`.

Execution Steps:
1. In `src/geo_routing.py`:
   - Implement `fetch_waypoint_weather(origin_coords: tuple[float, float], dest_coords: tuple[float, float]) -> dict[str, Any]`: Queries Open-Meteo API using `OPEN_METEO_API_KEY` for precipitation, temperature, and wind along route coordinates.
   - Render interactive Folium maps with color-coded status markers (Green = clear, Red = adverse weather).
2. In `src/consolidation.py`:
   - Implement `detect_ltl_consolidation_opportunities(df_shipments: pd.DataFrame) -> pd.DataFrame`.
   - Filter orders with `cargo_weight_kg < 5000` sharing origin/destination regions within a 48-hour pickup window.
   - Calculate financial savings ($) and CO2 reduction (kg) from merging orders into multi-stop FTL shipments.

---

### Phase 8: Alternative Fuel Range & Refueling Constraints Optimizer
Act as a Senior AI Engineer. Implement `src/range_optimizer.py` integrating live API feeds and `RouteConfigManager` rules.

Execution Steps:
1. Implement `validate_route_autonomy(carrier_row: pd.Series, route_distance_km: float, fuel_type: str, config: dict[str, Any]) -> dict[str, Any]`.
2. Compute usable range applying `safety_battery_buffer_pct` (e.g., 85% usable range).
3. If `route_distance_km > usable_range_km`, fetch live refueling/recharging stations along the corridor via `src/external_apis/stations.py`.
4. Apply disqualification rules against `RouteConfigManager`:
   - If `detour_km > config['max_detour_km_allowed']` -> Flag: 'Disqualified: Refueling Detour Exceeds Limit'.
   - If `delay_mins > config['max_refueling_delay_minutes']` -> Flag: 'Disqualified: Charging Delay Exceeds Limit'.
   - If cost with detours > `base_cost * (1 + config['green_premium_pct_limit'])` -> Flag: 'Disqualified: Green Premium Exceeded'.
5. Return dict: `is_viable` (bool), `status_flag` (str), `added_cost_usd` (float), `added_time_mins` (float), `detour_km` (float).

---

### Phase 9: Multi-Criteria Decision Making (MCDM) Optimization Engine
Act as a Senior AI Engineer. Implement `src/optimizer.py` containing class `LogisticsOptimizer`.

Execution Steps:
1. Read dynamic limits via `RouteConfigManager`.
2. Apply Hard Constraints:
   - Exclude carriers where `energy_source` is not in allowed user multiselect.
   - Exclude carriers failing thresholds: `min_otif_pct`, `min_otd_pct`, `max_co2_g_tkm`, `max_budget_usd`.
   - Exclude carriers flagged as 'Disqualified' by `range_optimizer.py`.
3. Apply Soft Weights Normalization:
   - Take 5 user weights: `cost`, `speed`, `otd`, `otif`, `co2`.
   - Normalize active weights so they sum to 1.0. If weight == 0, mark as '(Silenced)' and re-scale active weights.
4. Compute Min-Max scaling (0.0 to 1.0) and output `composite_score` (0-100) per carrier in descending order.

---

### Phase 10: Control Tower Exception & Automated Rerouting Engine
Act as a Senior Software Engineer. Implement `src/control_tower.py` following PEP 8.

Execution Steps:
1. Implement `monitor_active_shipments(df_active: pd.DataFrame) -> pd.DataFrame`.
2. Scan active in-transit shipments and flag exceptions:
   - 'Delayed Pickup Risk' if `actual_pickup > promised_pickup`.
   - 'Cold Chain Deviation' if `cargo_type == 'Refrigerated'` and temp exceeds threshold.
   - 'Severe Weather Delay' if `weather_risk_index > 1.5`.
3. Implement `trigger_automated_reroute(shipment_id: str, df_carriers: pd.DataFrame) -> dict[str, Any]`: Executes `LogisticsOptimizer` to recommend an immediate secondary backup carrier.

---

### Phase 11: History Analytics & Business Intelligence Engine
Act as a Senior AI Engineer. Implement `src/history_analytics.py` for dynamic historical log processing.

Execution Steps:
1. Implement `filter_historical_shipments(df: pd.DataFrame, start_date: str, end_date: str, carriers: list[str], cargo_types: list[str], energy_sources: list[str]) -> pd.DataFrame`.
2. Implement `compute_aggregated_metrics(df_filtered: pd.DataFrame) -> dict[str, float]`: Returns total shipments, total spend ($), average cost/km, aggregate OTD %, In-Full %, OTIF %, and total CO2 (kg).
3. Implement Plotly visualization functions:
   - `plot_time_series_trend()`: Monthly/Weekly OTIF % vs Avg Cost line chart.
   - `plot_cost_vs_risk_matrix()`: Cost vs Delay scatter plot grouped by carrier.
   - `plot_emissions_by_fuel()`: CO2 breakdown pie chart.
   - `plot_delay_distribution()`: Delay distribution histogram.

---

### Phase 12: Route Constraint Management UI Tab
Act as a Senior Software Engineer. Build the 'Route Rules & Constraint Configuration' tab in `app.py`.

Execution Steps:
1. Create interface form using `RouteConfigManager`:
   - Number inputs & sliders for: `max_detour_km_allowed`, `max_refueling_delay_minutes`, `safety_battery_buffer_pct`, `green_premium_pct_limit`, `shadow_carbon_price_per_ton`.
   - Multiselect input for `hazmat_restricted_corridors`.
2. Connect 'Save Configuration' button to `RouteConfigManager.update_config()` to persist changes instantly to `route_constraints_config.json`.
3. Connect 'Reset to Factory Defaults' button to `RouteConfigManager.reset_to_defaults()`.

---

### Phase 13: High-Performance Streamlit Dashboard UI Architecture
Act as a Senior AI Engineer. Implement `app.py` combining all modules with `@st.cache_data` and `@st.cache_resource`.

Execution Steps:
1. Load environment variables at startup using `python-dotenv`: `load_dotenv()`.
2. Build 8 navigation tabs:
   - Tab 1: 'Shipment Quote & Carrier Optimizer' (Inputs, KPI cards, Recommended carrier badge, Plotly comparative charts, Folium route map).
   - Tab 2: 'Route Rules & Constraint Configuration' (Interactive constraint manager).
   - Tab 3: 'Risk & Monte Carlo Analytics' (On-demand 1,000 iteration simulation & P95 SLA risk chart).
   - Tab 4: 'LTL Consolidation Hub' (Consolidation candidates table & merge action).
   - Tab 5: 'Control Tower & Exception Management' (Live exception map & backup carrier reroute button).
   - Tab 6: 'Historical Analytics & BI Dashboard' (Date filters, KPI cards, dynamic chart switcher, CSV export).
   - Tab 7: 'Carrier Management & Onboarding' (Carrier form & CSV ingestion pipeline).
   - Tab 8: 'ESG & Sustainability Reporting' (ISO 14083 metrics & PDF certificate download).

---

### Phase 14: Pipeline Unit Testing & Automated Validation
Act as a Senior Software Engineer. Implement `tests/test_pipeline.py` and `tests/test_external_apis.py` using `pytest`.

Execution Steps:
1. Test `RouteConfigManager`: Assert JSON read, update, and reset operations.
2. Test `metrics.py`: Assert vectorized calculations for OTD, In-Full, OTIF, and costs match manual arithmetic.
3. Test `range_optimizer.py`: Assert disqualification triggers when detour > max detour limit.
4. Test `optimizer.py`: Assert weight normalization equals 1.0 and weight == 0 silences variables.
5. Test `src/external_apis/`: Verify API connectors handle environment variables cleanly.
6. Execute test suite via terminal: `pytest tests/` and verify zero failures.

---

### Phase 15: Comprehensive Cross-File Code Audit & Quality Assurance
Act as a Principal Software Engineer and QA Architect. Conduct a cross-file audit of 'OmniData-Logistics'.

Execution Steps:
1. Review every module (`src/*.py`, `app.py`, `tests/*.py`) for strict PEP 8 compliance, line lengths, docstrings, and typing.
2. Verify environment variable handling: Ensure `os.getenv()` is used everywhere with string fallbacks, and zero API keys are hardcoded.
3. Verify signature alignment: Ensure function calls across `app.py`, `optimizer.py`, `range_optimizer.py`, and `external_apis/` match exact parameter names and types.
4. Verify Streamlit caching (`@st.cache_data`, `@st.cache_resource`) to prevent cache desynchronization.
5. Output a Code Audit Report detailing line-by-line fixes for any detected bug or performance issue.

---

### Phase 16: Enterprise GitHub Repository & Documentation Setup
Act as a Senior AI Engineer. Write `README.md` for the GitHub repository in English.

Execution Steps:
1. Write Title, Badges, and Executive Summary of OmniData-Logistics.
2. Include ASCII System Architecture Diagram showing decoupled pipeline services.
3. Detail mathematical formulations for OTIF %, Composite Carrier Score (CCS), and ISO 14083 Carbon Accounting.
4. Document environment variable configuration (`.env.example` to `.env` setup guide).
5. Provide step-by-step installation commands: `python -m venv venv`, `pip install -r requirements.txt`, `pytest tests/`, and `streamlit run app.py`.