# OmniData-Logistics 🚚🌱

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-brightgreen.svg)](https://peps.python.org/pep-0008/)
[![ISO Standard: 14083:2023](https://img.shields.io/badge/Standard-ISO%2014083%3A2023-emerald.svg)](https://www.iso.org/standard/78864.html)
[![GLEC Framework: v3.0](https://img.shields.io/badge/Framework-GLEC%20v3.0-teal.svg)](https://www.smartfreightcentre.org/en/glec-framework/)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-19%2F19%20passing-success.svg)](https://docs.pytest.org/)

**OmniData-Logistics** is an autonomous, production-grade enterprise logistics optimization and ESG control tower platform. Engineered for multinational shippers, 3PL/4PL carriers, and supply chain operators, the platform couples real-time environmental routing, machine learning delay forecasting, stochastic Monte Carlo risk quantification, autonomous LTL consolidation, and audit-ready ISO 14083 greenhouse gas accounting.

---

## 🏛️ System Architecture

```text
+-----------------------------------------------------------------------------------------------------+
|                                 OMNIDATA-LOGISTICS ENTERPRISE PLATFORM                              |
+-----------------------------------------------------------------------------------------------------+
                                                  |
           +--------------------------------------+--------------------------------------+
           |                                                                             |
           v                                                                             v
+-----------------------+                                                   +-------------------------+
| EXTERNAL TELEMETRY    |                                                   | OPERATIONAL DATA STORE  |
| - Open-Meteo Weather  |                                                   | - carrier_rates.csv     |
| - TomTom Truck Traffic|                                                   | - refueling_stations.csv|
| - NREL AFDC Fuel Hubs |                                                   | - synthetic_shipments   |
| - OpenChargeMap (EV)  |                                                   | - route_constraints.json|
+-----------------------+                                                   +-------------------------+
           |                                                                             |
           +--------------------------------------+--------------------------------------+
                                                  |
                                                  v
+-----------------------------------------------------------------------------------------------------+
|                                      CORE ENGINE PIPELINES                                          |
|                                                                                                     |
|  +--------------------+   +--------------------+   +--------------------+   +--------------------+  |
|  |   RouteConfig      |   |   MetricsEngine    |   |   Predictive ML    |   |   Monte Carlo      |  |
|  |  JSON Manager &    |   |   Vectorized OTD,  |   |  XGBoost Regressor |   | Stochastic Risk    |  |
|  |  Rule Persistence  |   |   In-Full & OTIF   |   | Delay Forecasting  |   | P95 SLA Simulation |  |
|  +--------------------+   +--------------------+   +--------------------+   +--------------------+  |
|            |                        |                        |                        |             |
|  +--------------------+   +--------------------+   +--------------------+   +--------------------+  |
|  |   RangeOptimizer   |   |   MCDM Optimizer   |   |  LTL Consolidation |   |  Control Tower     |  |
|  |   Battery Buffers, |   | Min-Max Normalized |   | Multi-Order 48-Hr  |   | 24/7 Monitoring &  |  |
|  | Detours & Charging |   | CCS Carrier Ranks  |   | FTL Carbon Savings |   | Automated Reroute  |  |
|  +--------------------+   +--------------------+   +--------------------+   +--------------------+  |
|                                                                                                     |
|  +-----------------------------------------------------------------------------------------------+  |
|  |                         ISO 14083 ESG Carbon Accounting Engine                                |  |
|  |             WTW / WTT / TTW Quantification & ReportLab PDF Certificate Generation            |  |
|  +-----------------------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+-----------------------------------------------------------------------------------------------------+
|                            HIGH-PERFORMANCE STREAMLIT DASHBOARD (app.py)                            |
|  [Tab 1: Quote & MCDM] [Tab 2: Constraints] [Tab 3: Risk MC] [Tab 4: Consolidation]                 |
|  [Tab 5: Control Tower] [Tab 6: History BI] [Tab 7: Onboarding] [Tab 8: ISO 14083 ESG Report]      |
+-----------------------------------------------------------------------------------------------------+
```

---

## 📐 Mathematical Formulations

### 1. On-Time In-Full (OTIF) Performance
Logistics operational fulfillment evaluates temporal punctuality against volumetric integrity:

$$\text{OTD}_i = \mathbb{I}(\text{actual\_delay\_hours}_i \le 0.0)$$

$$\text{In-Full}_i = \mathbb{I}(\text{delivered\_qty}_i = \text{ordered\_qty}_i)$$

$$\text{OTIF}_i = \text{OTD}_i \land \text{In-Full}_i$$

$$\text{Fleet OTIF (\%)} = \left(\frac{1}{N} \sum_{i=1}^{N} \text{OTIF}_i\right) \times 100$$

### 2. Composite Carrier Score (CCS) — MCDM Normalization
For $K$ normalized criteria with user priority weights $w_k \ge 0$ such that $\sum_{k \in \mathcal{A}} w_k = 1.0$:

$$\text{Score}_{\text{Cost}} = 1.0 - \frac{\text{Cost}_i - \min(\text{Cost})}{\max(\text{Cost}) - \min(\text{Cost})}$$

$$\text{Score}_{\text{Speed}} = 1.0 - \frac{\text{Duration}_i - \min(\text{Duration})}{\max(\text{Duration}) - \min(\text{Duration})}$$

$$\text{Score}_{\text{OTD}} = \frac{\text{OTD}_i - \min(\text{OTD})}{\max(\text{OTD}) - \min(\text{OTD})}$$

$$\text{Score}_{\text{OTIF}} = \frac{\text{OTIF}_i - \min(\text{OTIF})}{\max(\text{OTIF}) - \min(\text{OTIF})}$$

$$\text{Score}_{\text{CO}_2} = 1.0 - \frac{\text{CO}_{2, i} - \min(\text{CO}_2)}{\max(\text{CO}_2) - \min(\text{CO}_2)}$$

$$\text{CCS}_i = 100 \times \sum_{k \in \{\text{Cost, Speed, OTD, OTIF, CO}_2\}} w_k \cdot \text{Score}_{k, i}$$

### 3. ISO 14083:2023 Greenhouse Gas Accounting
Transport activity is quantified in metric ton-kilometers ($t \cdot km$):

$$\text{Activity} = M_{\text{cargo}} \times D_{\text{route}} \quad [t \cdot km]$$

$$\text{GHG}_{\text{WTW}} = \text{GHG}_{\text{WTT}} + \text{GHG}_{\text{TTW}} = \frac{\text{Activity} \times I_{\text{WTW}}}{1000} \quad [\text{kg CO}_2\text{e}]$$

Where $I_{\text{WTW}}$ denotes the Well-to-Wheel emission intensity in $\text{g CO}_2\text{e} / t \cdot km$:
- **Electric (Renewable Grid):** $8.5 \text{ g CO}_2\text{e}/tkm$
- **Green Hydrogen:** $4.5 \text{ g CO}_2\text{e}/tkm$
- **HVO (Hydrotreated Vegetable Oil):** $17.5 \text{ g CO}_2\text{e}/tkm$
- **Bio-LNG:** $26.0 \text{ g CO}_2\text{e}/tkm$
- **Fossil LNG:** $44.0 \text{ g CO}_2\text{e}/tkm$
- **Diesel (EU Baseline):** $63.0 \text{ g CO}_2\text{e}/tkm$

---

## ⚙️ Environment Configuration

The application operates with zero hardcoded API keys and includes robust, graceful local fallback catalog routines if external connectors are unconfigured.

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Configure optional API credentials in `.env`:
   ```ini
   OPEN_METEO_API_KEY='your_open_meteo_api_key_here'
   OPENCHARGE_MAP_API_KEY='your_opencharge_map_api_key_here'
   TOMTOM_TRAFFIC_API_KEY='your_tomtom_traffic_api_key_here'
   NREL_AFDC_API_KEY='your_nrel_afdc_api_key_here'
   ```

---

## 🚀 Quickstart Installation Guide

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Enter Repository
```bash
git clone https://github.com/your-username/OmniData-Logistics.git
cd OmniData-Logistics
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit Platform
```bash
streamlit run app.py
```

---

## 📦 Directory Structure

```text
OmniData-Logistics/
├── .env.example                       # API key placeholder template
├── .env                               # Active local environment file
├── requirements.txt                   # Production dependencies
├── README.md                          # Enterprise documentation
├── tasks.md                           # Implementation roadmap & status
├── app.py                             # High-performance 8-tab Streamlit UI
├── data/
│   ├── carrier_rates.csv              # Carrier catalog (10 real-world fleets)
│   ├── refueling_stations.csv         # European corridor alternative stations
│   ├── synthetic_shipments.csv        # 2,000 historical telemetry shipment records
│   ├── route_constraints_config.json  # Dynamic JSON route boundaries
│   └── ml_delay_model.joblib          # Trained XGBoost delay predictor artifact
├── src/
│   ├── __init__.py
│   ├── route_config.py                # RouteConfigManager JSON persistence
│   ├── data_generator.py              # Production synthetic dataset generator
│   ├── data_loader.py                 # Resilient ingestion & carrier onboarding
│   ├── metrics.py                     # Vectorized OTD, In-Full, OTIF, financial cost
│   ├── ml_engine.py                   # XGBoost delay prediction pipeline
│   ├── risk_simulation.py             # Vectorized Monte Carlo P95 ETA simulation
│   ├── consolidation.py               # LTL 48-hr multi-order consolidation engine
│   ├── esg_reporting.py               # ISO 14083 GHG quantification & ReportLab PDF
│   ├── geo_routing.py                 # Open-Meteo weather routing & Folium maps
│   ├── range_optimizer.py             # Alternative fuel autonomy & detour rules
│   ├── optimizer.py                   # Multi-Criteria Decision Making (MCDM)
│   ├── control_tower.py               # 24/7 Exception monitoring & automated reroute
│   ├── history_analytics.py           # BI aggregations & Plotly visualizations
│   └── external_apis/
│       ├── __init__.py
│       ├── stations.py                # NREL AFDC & OpenChargeMap connectors
│       └── traffic.py                 # TomTom commercial truck traffic connector
└── tests/
    ├── __init__.py
    ├── test_pipeline.py               # Core services & algorithm unit tests
    └── test_external_apis.py          # API connector & fallback unit tests
```

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
