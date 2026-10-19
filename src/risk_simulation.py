"""Monte Carlo risk simulation engine.

Runs high-speed vectorized stochastic simulations over transit velocity, traffic congestion,
weather volatility, and carrier historical variances to estimate P95 ETA, confidence intervals,
and SLA compliance probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CARRIER_RATES_PATH = Path(__file__).resolve().parent.parent / "data" / "carrier_rates.csv"
_CACHED_CARRIERS_DF: pd.DataFrame | None = None


def _get_carrier_reliability(carrier_id: str) -> float:
    """Retrieve historical OTD rate for a carrier or fallback to 0.95."""
    global _CACHED_CARRIERS_DF
    if _CACHED_CARRIERS_DF is None and CARRIER_RATES_PATH.exists():
        try:
            _CACHED_CARRIERS_DF = pd.read_csv(CARRIER_RATES_PATH)
        except Exception:
            _CACHED_CARRIERS_DF = None

    if _CACHED_CARRIERS_DF is not None and "carrier_id" in _CACHED_CARRIERS_DF.columns:
        match = _CACHED_CARRIERS_DF[_CACHED_CARRIERS_DF["carrier_id"] == carrier_id]
        if not match.empty and "otd_historical" in match.columns:
            return float(match["otd_historical"].values[0])
    return 0.95


def run_monte_carlo_simulation(
    carrier_id: str,
    distance_km: float,
    weather_index: float,
    iterations: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute vectorized Monte Carlo simulation for transit time variance.

    Args:
        carrier_id: Carrier identifier code (e.g. 'CAR-001').
        distance_km: Total trip distance in kilometers.
        weather_index: Weather severity risk index (0.0=clear, >1.0=stormy).
        iterations: Number of stochastic sampling iterations (default: 1000).
        random_seed: Seed for repeatable simulations.

    Returns:
        dict[str, Any]: Simulation results containing:
            - 'p95_eta_hours': 95th percentile transit duration.
            - 'confidence_interval_90': Tuple of (5th percentile, 95th percentile).
            - 'sla_compliance_probability': Probability of completing within nominal SLA.
            - 'mean_eta_hours': Average transit duration.
            - 'median_eta_hours': Median transit duration.
            - 'simulated_etas': Numpy array of all simulated durations.
    """
    np.random.seed(random_seed)
    dist = max(10.0, float(distance_km))
    weather = max(0.0, float(weather_index))
    iters = max(100, int(iterations))

    # Carrier historical baseline
    otd_rate = _get_carrier_reliability(carrier_id)

    # 1. Vectorized Velocity Sampling (normal distribution around 72 km/h, sigma 6.5 km/h)
    velocities = np.random.normal(loc=72.0, scale=6.5, size=iters)
    velocities = np.clip(velocities, 45.0, 90.0)  # Heavy vehicle speed limiter
    base_driving_hours = dist / velocities

    # 2. Mandatory European Driver Rest Break compliance (45 min break per 4.5 hours driving)
    break_periods = dist / 70.0 // 4.5
    break_hours = break_periods * 0.75

    # 3. Traffic delay stochastic modeling (Gamma distribution)
    traffic_delays_hours = np.random.gamma(shape=2.2, scale=0.35, size=iters)

    # 4. Weather impact variance
    weather_impact_base = max(0.0, (weather - 0.5) * 1.8) if weather > 0.5 else 0.0
    weather_delays_hours = np.maximum(
        0.0,
        np.random.normal(loc=weather_impact_base, scale=max(0.1, weather_impact_base * 0.4), size=iters),
    )

    # 5. Carrier operational jitter (lower historical OTD increases variance)
    carrier_unreliability = max(0.01, 1.0 - otd_rate)
    carrier_jitter = np.random.exponential(scale=carrier_unreliability * 8.0, size=iters)

    # Aggregate simulated total transit ETA in hours
    simulated_etas = base_driving_hours + break_hours + traffic_delays_hours + weather_delays_hours + carrier_jitter

    # SLA Benchmark: Nominal speed (68 km/h) + breaks + 2.5 hr operational buffer
    nominal_sla_hours = (dist / 68.0) + break_hours + 2.5
    sla_compliance_prob = float(np.mean(simulated_etas <= nominal_sla_hours))

    p05 = float(np.percentile(simulated_etas, 5))
    p50 = float(np.percentile(simulated_etas, 50))
    p95 = float(np.percentile(simulated_etas, 95))
    mean_val = float(np.mean(simulated_etas))

    return {
        "p95_eta_hours": round(p95, 2),
        "confidence_interval_90": (round(p05, 2), round(p95, 2)),
        "sla_compliance_probability": round(sla_compliance_prob, 3),
        "mean_eta_hours": round(mean_val, 2),
        "median_eta_hours": round(p50, 2),
        "nominal_sla_hours": round(nominal_sla_hours, 2),
        "simulated_etas": simulated_etas,
    }
