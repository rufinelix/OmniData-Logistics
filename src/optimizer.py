"""Multi-Criteria Decision Making (MCDM) Optimization Engine.

Provides multi-objective carrier ranking utilizing Min-Max normalization,
flexible user priority weighting, and rigorous multi-tier hard constraint filtering
informed by dynamic RouteConfigManager rules and range constraints.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from src.range_optimizer import validate_route_autonomy
from src.route_config import RouteConfigManager

logger = logging.getLogger(__name__)


class LogisticsOptimizer:
    """Enterprise multi-criteria optimization engine for carrier selection."""

    def __init__(self, config_manager: RouteConfigManager | None = None) -> None:
        """Initialize optimizer with RouteConfigManager instance."""
        self.config_manager = config_manager or RouteConfigManager()

    def normalize_weights(self, user_weights: dict[str, float]) -> dict[str, tuple[float, bool]]:
        """Normalize user weights to sum to 1.0, tracking silenced criteria.

        Args:
            user_weights: Raw user weights for cost, speed, otd, otif, co2.

        Returns:
            dict[str, tuple[float, bool]]: Map of criteria name to (normalized_weight, is_silenced).
        """
        raw_keys = ["cost", "speed", "otd", "otif", "co2"]
        extracted = {k: max(0.0, float(user_weights.get(k, 1.0))) for k in raw_keys}

        active_sum = sum(v for v in extracted.values() if v > 0.0)

        if active_sum == 0.0:
            # All weights set to 0 -> fall back to equal distribution
            return {k: (0.20, False) for k in raw_keys}

        result = {}
        for k in raw_keys:
            raw_v = extracted[k]
            if raw_v == 0.0:
                result[k] = (0.0, True)  # Silenced
            else:
                result[k] = (round(raw_v / active_sum, 4), False)

        return result

    def optimize_carrier_selection(
        self,
        df_carriers: pd.DataFrame,
        distance_km: float,
        cargo_weight_kg: float,
        cargo_type: str = "General",
        user_weights: dict[str, float] | None = None,
        hard_constraints: dict[str, Any] | None = None,
        predicted_delays: dict[str, float] | None = None,
        origin_coords: tuple[float, float] = (40.4168, -3.7038),
    ) -> pd.DataFrame:
        """Rank and evaluate all candidate carriers across multi-criteria objectives.

        Args:
            df_carriers: DataFrame of candidate carrier profiles.
            distance_km: Transit route distance in kilometers.
            cargo_weight_kg: Shipment payload weight in kilograms.
            cargo_type: Cargo classification (e.g. 'General', 'Refrigerated', 'Hazmat').
            user_weights: Desired priority weights (cost, speed, otd, otif, co2).
            hard_constraints: User threshold filters (min_otd, min_otif, max_budget, allowed_energy).
            predicted_delays: Optional dict of carrier_id -> predicted delay hours from ML engine.
            origin_coords: Coordinate of origin for refueling distance calculations.

        Returns:
            pd.DataFrame: Comprehensive carrier comparison table with composite_score,
                          rank, eligibility flag, and disqualification reason.
        """
        if df_carriers.empty:
            return pd.DataFrame()

        config = self.config_manager.load_config()
        weights_config = self.normalize_weights(user_weights or {})
        constraints = hard_constraints or {}

        # Extract thresholds
        allowed_energy = constraints.get("allowed_energy_sources", [])
        min_otd_pct = float(constraints.get("min_otd_pct", 0.0))
        min_otif_pct = float(constraints.get("min_otif_pct", 0.0))
        max_budget_usd = float(constraints.get("max_budget_usd", 1e9))
        max_co2_intensity = float(constraints.get("max_co2_g_tkm", 1e9))

        # Corridor Hazmat check
        hazmat_restricted = config.get("hazmat_restricted_corridors", [])
        is_hazmat_shipment = cargo_type.lower() == "hazmat"

        evaluated_rows: list[dict[str, Any]] = []

        for _, carrier in df_carriers.iterrows():
            cid = str(carrier["carrier_id"])
            cname = str(carrier["carrier_name"])
            energy = str(carrier["energy_source"])
            base_rate = float(carrier.get("base_rate_usd", carrier.get("base_rate", 180.0)))
            rate_km = float(carrier["rate_per_km"])
            surcharge = float(carrier["fuel_surcharge_pct"])
            otd_pct = round(float(carrier["otd_historical"]) * 100.0, 1)
            otif_pct = round(float(carrier["in_full_historical"]) * float(carrier["otd_historical"]) * 100.0, 1)
            co2_intensity = float(carrier["co2_g_per_ton_km"])

            # 1. Base Transportation Cost
            base_cost = base_rate + (distance_km * rate_km * (1.0 + surcharge))

            # 2. Check Autonomy & Refueling Detours
            autonomy_result = validate_route_autonomy(
                carrier_row=carrier,
                route_distance_km=distance_km,
                fuel_type=energy,
                config=config,
                origin_coords=origin_coords,
            )

            total_financial_cost = round(base_cost + autonomy_result["added_cost_usd"], 2)

            # 3. Transit duration (driving hours + mandatory break + refueling delay + predicted ML delay)
            nominal_driving_hours = distance_km / 70.0
            break_hours = (distance_km / 70.0 // 4.5) * 0.75
            refueling_hours = autonomy_result["added_time_mins"] / 60.0
            ml_delay = float(predicted_delays.get(cid, 0.0)) if predicted_delays else 0.0

            total_duration_hours = round(max(1.0, nominal_driving_hours + break_hours + refueling_hours + max(0.0, ml_delay)), 2)

            # 4. Shadow Carbon Cost Calculation
            shadow_price = float(config.get("shadow_carbon_price_per_ton", 80.0))
            ton_km = (cargo_weight_kg / 1000.0) * distance_km
            co2_emissions_kg = (ton_km * co2_intensity) / 1000.0
            shadow_carbon_cost = round((co2_emissions_kg / 1000.0) * shadow_price, 2)
            total_economic_cost = round(total_financial_cost + shadow_carbon_cost, 2)

            # 5. Evaluate Hard Constraints
            is_eligible = True
            disqualification_reason = "Eligible"

            if allowed_energy and energy not in allowed_energy:
                is_eligible = False
                disqualification_reason = f"Disqualified: Energy source '{energy}' not in selected criteria"
            elif otd_pct < min_otd_pct:
                is_eligible = False
                disqualification_reason = f"Disqualified: OTD ({otd_pct}%) below required minimum ({min_otd_pct}%)"
            elif otif_pct < min_otif_pct:
                is_eligible = False
                disqualification_reason = f"Disqualified: OTIF ({otif_pct}%) below required minimum ({min_otif_pct}%)"
            elif total_financial_cost > max_budget_usd:
                is_eligible = False
                disqualification_reason = f"Disqualified: Cost (${total_financial_cost:,.2f}) exceeds budget (${max_budget_usd:,.2f})"
            elif co2_intensity > max_co2_intensity:
                is_eligible = False
                disqualification_reason = f"Disqualified: CO2 intensity ({co2_intensity} g/tkm) exceeds limit ({max_co2_intensity} g/tkm)"
            elif not autonomy_result["is_viable"]:
                is_eligible = False
                disqualification_reason = autonomy_result["status_flag"]

            evaluated_rows.append({
                "carrier_id": cid,
                "carrier_name": cname,
                "energy_source": energy,
                "financial_cost_usd": total_financial_cost,
                "shadow_carbon_cost_usd": shadow_carbon_cost,
                "total_economic_cost_usd": total_economic_cost,
                "transit_duration_hours": total_duration_hours,
                "otd_pct": otd_pct,
                "otif_pct": otif_pct,
                "co2_g_per_ton_km": co2_intensity,
                "co2_emissions_kg": round(co2_emissions_kg, 2),
                "refueling_detour_km": autonomy_result["detour_km"],
                "refueling_delay_mins": autonomy_result["added_time_mins"],
                "stops_required": autonomy_result.get("stops_required", 0),
                "is_eligible": is_eligible,
                "disqualification_reason": disqualification_reason,
            })

        df_eval = pd.DataFrame(evaluated_rows)

        # Min-Max Normalization & Composite Scoring across eligible carriers
        eligible_mask = df_eval["is_eligible"]
        df_eligible = df_eval[eligible_mask]

        if not df_eligible.empty:
            cost_min, cost_max = df_eligible["financial_cost_usd"].min(), df_eligible["financial_cost_usd"].max()
            speed_min, speed_max = df_eligible["transit_duration_hours"].min(), df_eligible["transit_duration_hours"].max()
            otd_min, otd_max = df_eligible["otd_pct"].min(), df_eligible["otd_pct"].max()
            otif_min, otif_max = df_eligible["otif_pct"].min(), df_eligible["otif_pct"].max()
            co2_min, co2_max = df_eligible["co2_g_per_ton_km"].min(), df_eligible["co2_g_per_ton_km"].max()

            def _scale(val: float, mn: float, mx: float, invert: bool = False) -> float:
                if mx == mn:
                    return 1.0
                ratio = (val - mn) / (mx - mn)
                return 1.0 - ratio if invert else ratio

            w_cost = weights_config["cost"][0]
            w_speed = weights_config["speed"][0]
            w_otd = weights_config["otd"][0]
            w_otif = weights_config["otif"][0]
            w_co2 = weights_config["co2"][0]

            scores = []
            for idx, r in df_eval.iterrows():
                if not r["is_eligible"]:
                    scores.append(0.0)
                    continue

                s_cost = _scale(r["financial_cost_usd"], cost_min, cost_max, invert=True)
                s_speed = _scale(r["transit_duration_hours"], speed_min, speed_max, invert=True)
                s_otd = _scale(r["otd_pct"], otd_min, otd_max, invert=False)
                s_otif = _scale(r["otif_pct"], otif_min, otif_max, invert=False)
                s_co2 = _scale(r["co2_g_per_ton_km"], co2_min, co2_max, invert=True)

                composite = (
                    (s_cost * w_cost)
                    + (s_speed * w_speed)
                    + (s_otd * w_otd)
                    + (s_otif * w_otif)
                    + (s_co2 * w_co2)
                ) * 100.0

                scores.append(round(composite, 2))

            df_eval["composite_score"] = scores
        else:
            df_eval["composite_score"] = 0.0

        # Sort: Eligible first by composite_score descending, then disqualified
        df_eval = df_eval.sort_values(
            by=["is_eligible", "composite_score"], ascending=[False, False]
        ).reset_index(drop=True)

        # Assign ranks
        ranks = []
        cur_rank = 1
        for is_elig in df_eval["is_eligible"]:
            if is_elig:
                ranks.append(cur_rank)
                cur_rank += 1
            else:
                ranks.append(None)
        df_eval["rank"] = ranks

        return df_eval
