"""Loads, validates, and provides typed accessors for config/filter_config.yaml.

This is the single permitted path to filter_config.yaml for all business-logic
modules. No other module may call yaml.safe_load() directly (the sole exception
is data_acquisition/api_config.py, which must load config before this module is
importable without creating a circular dependency).

Data Lineage Contract
---------------------
Upstream source:
    - ``config/filter_config.yaml`` — single source of truth for every
      threshold, weight, and parameter used anywhere in the codebase.
      Loaded via ``yaml.safe_load()`` on first access; cached in
      ``_config_cache`` for the process lifetime.

Downstream consumers (every module that reads thresholds or weights):
    - ``metrics_engine.profitability`` → ``get_threshold("hard_filters.min_avg_roe")``
    - ``metrics_engine.composite_score`` → soft_scores section weights/breakpoints
    - ``screener.hard_filters`` → hard_filters section thresholds
    - ``screener.soft_filters`` → soft_scores section breakpoints
    - ``screener.exclusions`` → exclusions section lists
    - ``metrics_engine.leverage``, ``metrics_engine.growth``,
      ``metrics_engine.valuation``, ``metrics_engine.returns``
      → various hard_filters and valuation thresholds
    - ``data_acquisition.universe`` → universe section (min_market_cap_usd, exchanges)
    - ``data_acquisition.data_quality`` → data_quality section

Config dependencies:
    - Requires ``config/filter_config.yaml`` at project root.
    - No environment variables or external services.

Validation rules:
    - All 10 required top-level sections must exist (see ``_REQUIRED_SECTIONS``).
    - Soft score weights (10 criteria) must sum to 1.0 within tolerance.

Usage
-----
    from screener.filter_config_loader import get_config, get_threshold

    cfg = get_config()
    min_cap = cfg["universe"]["min_market_cap_usd"]

    roe_floor = get_threshold("hard_filters.min_avg_roe")   # → 0.15

The config is loaded once per process and cached at module level. Call
``reload_config()`` in tests that need a fresh load after patching the file path.

Key exports
-----------
load_config() -> dict
    Load and return the validated config dict (alias for get_config).
get_config() -> dict
    Return the validated config dict (cached after first call).
get_threshold(path) -> Any
    Dot-notation accessor: get_threshold("hard_filters.min_avg_roe") → 0.15.
reload_config() -> dict
    Force a re-read from disk and reset the cache. Intended for tests only.
ConfigError
    Base exception for all config-related errors.
ConfigValidationError(ConfigError)
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


class ConfigError(Exception):
    """Raised when a requested config key path does not exist.

    Parameters
    ----------
    message:
        Human-readable description of the missing path, including the full
        dot-notation path that was requested and the segment that failed.
    """


class ConfigValidationError(ConfigError):
    """Raised when filter_config.yaml is structurally invalid.

    Inherits from ``ConfigError`` so callers that catch ``ConfigError`` will
    also catch validation failures (missing sections, invalid weights, etc.).

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


def load_config() -> dict[str, Any]:
    """Load and return the validated filter_config.yaml as a nested dict.

    Convenience alias for :func:`get_config`.  Prefer ``get_config`` in new
    code; ``load_config`` is provided for API consistency with project docs.

    Returns
    -------
    dict
        Full contents of filter_config.yaml (cached after first call).

    Raises
    ------
    FileNotFoundError
        If config/filter_config.yaml does not exist.
    ConfigValidationError
        If required sections are missing or soft_score weights don't sum to 1.0.
    """
    return get_config()


def get_threshold(path: str) -> Any:
    """Return a config value via a dot-notation key path.

    Parameters
    ----------
    path:
        Dot-separated path into the config dict, e.g.
        ``"hard_filters.min_avg_roe"`` or ``"soft_scores.roe.weight"``.

    Returns
    -------
    Any
        The value at the specified path.

    Raises
    ------
    ConfigError
        If any segment of *path* is not present in the config.

    Examples
    --------
    >>> get_threshold("hard_filters.min_avg_roe")
    0.15
    >>> get_threshold("soft_scores.roe.weight")
    0.15
    """
    config = get_config()
    parts = path.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(
                f"Config key not found: '{path}' (failed at segment '{part}'). "
                "Check config/filter_config.yaml."
            )
        current = current[part]
    return current


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
