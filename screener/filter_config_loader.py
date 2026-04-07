"""Loads, validates, and provides typed accessors for config/filter_config.yaml.

This is the single permitted path to filter_config.yaml for all business-logic
modules. No other module may call yaml.safe_load() directly (the sole exception
is data_acquisition/api_config.py, which must load config before this module is
importable without creating a circular dependency).

Usage
-----
    from screener.filter_config_loader import get_config

    cfg = get_config()
    min_cap = cfg["universe"]["min_market_cap_usd"]

The config is loaded once per process and cached at module level. Call
``reload_config()`` in tests that need a fresh load after patching the file path.

Key exports
-----------
get_config() -> dict
    Return the validated config dict (cached after first call).
reload_config() -> dict
    Force a re-read from disk and reset the cache. Intended for tests only.
ConfigValidationError
    Raised when required keys are absent or weights don't sum to 1.0.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "filter_config.yaml"

# Module-level cache — populated on first call to get_config().
_config_cache: dict[str, Any] | None = None

# Required top-level sections. Missing any of these raises ConfigValidationError.
_REQUIRED_SECTIONS = (
    "universe",
    "data_sources",
    "exclusions",
    "hard_filters",
    "soft_scores",
    "valuation",
    "recommendations",
    "sell_signals",
    "output",
    "logging",
)

# Weight keys that must sum to 1.0 (within tolerance).
_WEIGHT_KEYS = (
    "roe",
    "gross_margin",
    "sga_ratio",
    "eps_growth",
    "debt_conservatism",
    "owner_earnings_growth",
    "capital_efficiency",
    "buyback",
    "retained_earnings_return",
    "interest_coverage",
)
_WEIGHT_TOLERANCE = 1e-6


class ConfigValidationError(Exception):
    """Raised when filter_config.yaml is structurally invalid.

    Parameters
    ----------
    message:
        Human-readable description of what failed validation.
    """


def get_config() -> dict[str, Any]:
    """Return the parsed and validated filter_config.yaml as a nested dict.

    Loads from disk on the first call and caches the result for the lifetime
    of the process. Subsequent calls return the cached dict.

    Returns
    -------
    dict
        Full contents of filter_config.yaml.

    Raises
    ------
    FileNotFoundError
        If config/filter_config.yaml does not exist.
    ConfigValidationError
        If required sections are missing or soft_score weights don't sum to 1.0.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_and_validate()
    return _config_cache


def reload_config() -> dict[str, Any]:
    """Force a fresh read from disk, bypassing the cache.

    Intended for use in tests that patch config values or the config file path.
    Not safe to call from multiple threads simultaneously.

    Returns
    -------
    dict
        Freshly loaded and validated config.
    """
    global _config_cache
    _config_cache = None
    return get_config()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_validate() -> dict[str, Any]:
    """Read filter_config.yaml and run structural validation.

    Returns
    -------
    dict
        Validated config dict.

    Raises
    ------
    FileNotFoundError
        If the config file is not found at the expected path.
    ConfigValidationError
        If required sections are absent or weights are invalid.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"filter_config.yaml not found at {_CONFIG_PATH}. "
            "Ensure the config/ directory exists at the project root."
        )

    with _CONFIG_PATH.open("r") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ConfigValidationError(
            "filter_config.yaml did not parse as a YAML mapping (dict). "
            f"Got: {type(config).__name__}"
        )

    _validate_required_sections(config)
    _validate_soft_score_weights(config)

    logger.debug("filter_config.yaml loaded and validated from %s", _CONFIG_PATH)
    return config


def _validate_required_sections(config: dict[str, Any]) -> None:
    """Raise ConfigValidationError if any top-level section is missing.

    Parameters
    ----------
    config:
        The raw parsed YAML dict.

    Raises
    ------
    ConfigValidationError
        Lists all missing sections in a single error.
    """
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ConfigValidationError(
            f"filter_config.yaml is missing required top-level section(s): "
            f"{missing}. Check config/filter_config.yaml."
        )


def _validate_soft_score_weights(config: dict[str, Any]) -> None:
    """Raise ConfigValidationError if soft_scores weights do not sum to 1.0.

    Parameters
    ----------
    config:
        The raw parsed YAML dict (must already have 'soft_scores' key).

    Raises
    ------
    ConfigValidationError
        If the total weight deviates from 1.0 by more than _WEIGHT_TOLERANCE.
    """
    soft = config.get("soft_scores", {})
    total = 0.0
    found: list[str] = []
    for key in _WEIGHT_KEYS:
        section = soft.get(key, {})
        weight = section.get("weight")
        if weight is not None:
            total += float(weight)
            found.append(key)

    if found and abs(total - 1.0) > _WEIGHT_TOLERANCE:
        raise ConfigValidationError(
            f"soft_scores weights must sum to 1.0, but sum is {total:.6f}. "
            f"Keys found: {found}. "
            "Adjust weights in config/filter_config.yaml."
        )
