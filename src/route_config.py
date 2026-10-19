"""Route constraints configuration manager module.

This module provides persistent storage and runtime access to route-level
operational constraints and sustainability thresholds across the OmniData-Logistics platform.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "route_constraints_config.json"

FACTORY_DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "max_transit_time_hours": 48.0,
    "max_detour_km_allowed": 30.0,
    "max_refueling_delay_minutes": 45.0,
    "safety_battery_buffer_pct": 0.15,
    "green_premium_pct_limit": 0.15,
    "shadow_carbon_price_per_ton": 80.0,
    "hazmat_restricted_corridors": ["MAD-BCN-Tunnel", "ALP-Pass"],
    "refrigerated_temp_tolerance_c": 2.0,
}


class RouteConfigManager:
    """Manages reading, mutating, and restoring route constraint parameters."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize RouteConfigManager with configuration file path."""
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Read JSON configuration file. Creates default if file does not exist.

        Returns:
            dict[str, Any]: Active configuration dictionary.
        """
        if not self.config_path.exists():
            logger.warning("Configuration file %s not found. Initializing with defaults.", self.config_path)
            self._config = dict(FACTORY_DEFAULT_CONSTRAINTS)
            self._save_to_disk()
            return self._config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read configuration from %s: %s. Falling back to defaults.", self.config_path, exc)
            self._config = dict(FACTORY_DEFAULT_CONSTRAINTS)

        # Ensure all required keys exist
        updated = False
        for key, value in FACTORY_DEFAULT_CONSTRAINTS.items():
            if key not in self._config:
                self._config[key] = value
                updated = True

        if updated:
            self._save_to_disk()

        return dict(self._config)

    def update_config(self, new_params: dict[str, Any]) -> None:
        """Mutate configuration dictionary in-memory and persist to disk.

        Args:
            new_params (dict[str, Any]): Dictionary of updated parameters.
        """
        self._config.update(new_params)
        self._save_to_disk()
        logger.info("Configuration successfully updated and persisted to %s.", self.config_path)

    def reset_to_defaults(self) -> dict[str, Any]:
        """Restore factory default JSON values and persist to disk.

        Returns:
            dict[str, Any]: Reset configuration dictionary.
        """
        self._config = dict(FACTORY_DEFAULT_CONSTRAINTS)
        self._save_to_disk()
        logger.info("Configuration reset to factory defaults.")
        return dict(self._config)

    def _save_to_disk(self) -> None:
        """Internal helper to write the active configuration to JSON disk file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)
