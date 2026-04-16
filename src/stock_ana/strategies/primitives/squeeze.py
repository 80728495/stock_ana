"""Reusable squeeze-pattern measurements shared across breakout strategies."""

from __future__ import annotations

import pandas as pd


def normalized_price_range(seg: pd.Series) -> float:
    """Compute normalized price-range width for a recent closing-price segment."""
    if len(seg) < 2:
        return 1.0
    mean_value = seg.mean()
    if mean_value == 0:
        return 1.0
    return float((seg.max() - seg.min()) / mean_value)


def compute_ma_squeeze_ratio(ma_values: list[float]) -> float:
    """Compute the max/min spread ratio across a set of moving-average values."""
    ma_max = max(ma_values)
    ma_min = min(ma_values)
    return ma_max / ma_min if ma_min > 0 else 999.0


def compute_volume_trend_ratio(
    volume: pd.Series,
    *,
    short_window: int = 5,
    long_window: int = 20,
) -> float:
    """Compare short-window average volume versus a longer baseline window."""
    short_mean = volume.iloc[-(short_window + 1):-1].mean()
    long_mean = volume.iloc[-(long_window + 1):-1].mean()
    return float(short_mean / long_mean) if long_mean > 0 else 999.0


def is_recent_crossover(
    fast: pd.Series,
    slow: pd.Series,
    *,
    window: int,
) -> bool:
    """Return whether fast crossed above slow inside the recent window."""
    for offset in range(1, window + 1):
        idx = -offset
        idx_prev = idx - 1
        if abs(idx_prev) > len(fast):
            continue
        if fast.iloc[idx] > slow.iloc[idx] and fast.iloc[idx_prev] <= slow.iloc[idx_prev]:
            return True
    return False
