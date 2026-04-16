"""Registry and dispatch for standardized strategy scan interfaces."""

from __future__ import annotations

from typing import Callable

from stock_ana.strategies.api import (
    scan_ma_squeeze,
    scan_main_rally_pullback_setups,
    scan_momentum,
    scan_rs_acceleration,
    scan_rs_trap_alert,
    scan_triangle_ascending,
    scan_triangle_kde_setups,
    scan_triangle_parallel_channel,
    scan_triangle_rising_wedge,
    scan_triangle_vcp_setups,
    scan_vegas_mid_pullbacks,
    scan_vegas_touches,
    scan_vcp_setups,
)
from stock_ana.strategies.contracts import StrategyKind


STRATEGY_SCAN_REGISTRY: dict[str, Callable[..., object]] = {
    "vegas": scan_vegas_touches,
    "vegas_mid": scan_vegas_mid_pullbacks,
    "ma_squeeze": scan_ma_squeeze,
    "momentum": scan_momentum,
    "main_rally_pullback": scan_main_rally_pullback_setups,
    "rs_acceleration": scan_rs_acceleration,
    "rs_trap": scan_rs_trap_alert,
    "vcp": scan_vcp_setups,
    "triangle_ascending": scan_triangle_ascending,
    "triangle_parallel_channel": scan_triangle_parallel_channel,
    "triangle_rising_wedge": scan_triangle_rising_wedge,
    "triangle_kde": scan_triangle_kde_setups,
    "triangle_vcp": scan_triangle_vcp_setups,
}

STRATEGY_KIND_REGISTRY: dict[str, StrategyKind] = {
    "vegas": "pattern",
    "vegas_mid": "stateful_signal",
    "ma_squeeze": "stateful_signal",
    "momentum": "stateful_signal",
    "main_rally_pullback": "pattern",
    "rs_acceleration": "stateful_signal",
    "rs_trap": "stateful_signal",
    "vcp": "pattern",
    "triangle_ascending": "pattern",
    "triangle_parallel_channel": "pattern",
    "triangle_rising_wedge": "pattern",
    "triangle_kde": "pattern",
    "triangle_vcp": "pattern",
}


def list_registered_strategies() -> list[str]:
    """Return all strategy names that can be dispatched through the registry."""
    return sorted(STRATEGY_SCAN_REGISTRY.keys())


def get_strategy_kind(strategy: str) -> StrategyKind:
    """Return the normalized kind for one registered strategy name."""
    kind = STRATEGY_KIND_REGISTRY.get(strategy)
    if kind is None:
        names = ", ".join(sorted(STRATEGY_KIND_REGISTRY.keys()))
        raise ValueError(f"未知策略: {strategy}. 可用策略: {names}")
    return kind


def scan_strategy(strategy: str, **kwargs):
    """Dispatch one registered scan function by name with passthrough arguments."""
    runner = STRATEGY_SCAN_REGISTRY.get(strategy)
    if runner is None:
        names = ", ".join(list_registered_strategies())
        raise ValueError(f"未知策略: {strategy}. 可用策略: {names}")
    return runner(**kwargs)
