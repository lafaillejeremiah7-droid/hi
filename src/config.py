"""Configuration loading and management.

Loads config/default.yaml as the base configuration, then overlays
config/custom.yaml if it exists. Exposes the merged config as a dict.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml lives)."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve().parent.parent
    if (current / "pyproject.toml").exists():
        return current
    # Fallback: use CWD
    return Path.cwd()


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML files.

    Args:
        config_path: Optional path to a specific config file.
                     If None, loads config/default.yaml with optional
                     config/custom.yaml overlay.

    Returns:
        Merged configuration dictionary.
    """
    project_root = get_project_root()
    config_dir = project_root / "config"

    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = project_root / path
        with open(path) as f:
            return yaml.safe_load(f)

    # Load default config
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default config not found at {default_path}. "
            "Ensure config/default.yaml exists in the project root."
        )

    with open(default_path) as f:
        config = yaml.safe_load(f)

    # Overlay custom config if it exists
    custom_path = config_dir / "custom.yaml"
    if custom_path.exists():
        with open(custom_path) as f:
            custom = yaml.safe_load(f)
            if custom:
                config = _deep_merge(config, custom)

    return config


def get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract data section from config."""
    return config.get("data", {})


def get_costs_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract costs section from config."""
    return config.get("costs", {})


def get_monte_carlo_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract Monte Carlo section from config."""
    return config.get("monte_carlo", {})


def get_strategy_config(config: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    """Extract strategy-specific config section.

    Args:
        config: Full configuration dict.
        strategy_name: Name of the strategy (e.g., 'order_flow_strategy').

    Returns:
        Strategy configuration dict.
    """
    return config.get(strategy_name, {})
