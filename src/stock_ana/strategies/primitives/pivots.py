"""Pivot extraction helpers shared by multiple pattern-recognition strategies."""

from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema


def argrel_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract swing high/low indices with scipy argrelextrema."""
    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    return hi_idx, lo_idx


def multiscale_argrel_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    orders: tuple[int, ...] = (3, 6),
) -> tuple[np.ndarray, np.ndarray]:
    """Merge pivot candidates from multiple argrelextrema scales."""
    hi_sets: list[np.ndarray] = []
    lo_sets: list[np.ndarray] = []
    for order in orders:
        hi_idx, lo_idx = argrel_pivots(highs, lows, order=order)
        hi_sets.append(hi_idx)
        lo_sets.append(lo_idx)
    hi_merged = np.unique(np.concatenate(hi_sets)) if hi_sets else np.array([], dtype=int)
    lo_merged = np.unique(np.concatenate(lo_sets)) if lo_sets else np.array([], dtype=int)
    return hi_merged, lo_merged


def zigzag_indices(
    highs: np.ndarray,
    lows: np.ndarray,
    threshold_pct: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ZigZag-confirmed high/low indices using percent reversal threshold."""
    n = len(highs)
    if n < 5:
        return np.array([], dtype=int), np.array([], dtype=int)

    hi_list: list[int] = []
    lo_list: list[int] = []
    trend = 0
    candidate_high_idx, candidate_high_val = 0, highs[0]
    candidate_low_idx, candidate_low_val = 0, lows[0]

    for i in range(1, n):
        if trend == 0:
            if highs[i] > candidate_high_val:
                candidate_high_idx, candidate_high_val = i, highs[i]
            if lows[i] < candidate_low_val:
                candidate_low_idx, candidate_low_val = i, lows[i]
            if candidate_high_val > 0 and (candidate_high_val - lows[i]) / candidate_high_val * 100 >= threshold_pct:
                hi_list.append(candidate_high_idx)
                trend = -1
                candidate_low_idx, candidate_low_val = i, lows[i]
            elif candidate_low_val > 0 and (highs[i] - candidate_low_val) / candidate_low_val * 100 >= threshold_pct:
                lo_list.append(candidate_low_idx)
                trend = 1
                candidate_high_idx, candidate_high_val = i, highs[i]
        elif trend == 1:
            if highs[i] > candidate_high_val:
                candidate_high_idx, candidate_high_val = i, highs[i]
            if candidate_high_val > 0 and (candidate_high_val - lows[i]) / candidate_high_val * 100 >= threshold_pct:
                hi_list.append(candidate_high_idx)
                trend = -1
                candidate_low_idx, candidate_low_val = i, lows[i]
        else:
            if lows[i] < candidate_low_val:
                candidate_low_idx, candidate_low_val = i, lows[i]
            if candidate_low_val > 0 and (highs[i] - candidate_low_val) / candidate_low_val * 100 >= threshold_pct:
                lo_list.append(candidate_low_idx)
                trend = 1
                candidate_high_idx, candidate_high_val = i, highs[i]

    return np.array(hi_list, dtype=int), np.array(lo_list, dtype=int)


def zigzag_points(
    highs: np.ndarray,
    lows: np.ndarray,
    threshold_pct: float = 5.0,
) -> list[dict]:
    """Return alternating ZigZag pivot points as H/L dicts."""
    n = len(highs)
    if n < 5:
        return []

    pivots: list[dict] = []
    trend = 0
    candidate_high_idx, candidate_high_val = 0, highs[0]
    candidate_low_idx, candidate_low_val = 0, lows[0]

    for i in range(1, n):
        if trend == 0:
            if highs[i] > candidate_high_val:
                candidate_high_idx, candidate_high_val = i, highs[i]
            if lows[i] < candidate_low_val:
                candidate_low_idx, candidate_low_val = i, lows[i]
            if candidate_high_val > 0 and (candidate_high_val - lows[i]) / candidate_high_val * 100 >= threshold_pct:
                pivots.append({"type": "H", "iloc": candidate_high_idx, "value": float(candidate_high_val)})
                trend = -1
                candidate_low_idx, candidate_low_val = i, lows[i]
            elif candidate_low_val > 0 and (highs[i] - candidate_low_val) / candidate_low_val * 100 >= threshold_pct:
                pivots.append({"type": "L", "iloc": candidate_low_idx, "value": float(candidate_low_val)})
                trend = 1
                candidate_high_idx, candidate_high_val = i, highs[i]
        elif trend == 1:
            if highs[i] > candidate_high_val:
                candidate_high_idx, candidate_high_val = i, highs[i]
            if candidate_high_val > 0 and (candidate_high_val - lows[i]) / candidate_high_val * 100 >= threshold_pct:
                pivots.append({"type": "H", "iloc": candidate_high_idx, "value": float(candidate_high_val)})
                trend = -1
                candidate_low_idx, candidate_low_val = i, lows[i]
        else:
            if lows[i] < candidate_low_val:
                candidate_low_idx, candidate_low_val = i, lows[i]
            if candidate_low_val > 0 and (highs[i] - candidate_low_val) / candidate_low_val * 100 >= threshold_pct:
                pivots.append({"type": "L", "iloc": candidate_low_idx, "value": float(candidate_low_val)})
                trend = 1
                candidate_high_idx, candidate_high_val = i, highs[i]

    return pivots


def merge_pivots_with_zigzag(
    highs: np.ndarray,
    lows: np.ndarray,
    zz_threshold_pct: float = 5.0,
    orders: tuple[int, ...] = (3, 6),
    proximity: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge argrelextrema candidates with ZigZag-confirmed pivots."""
    ar_hi, ar_lo = multiscale_argrel_pivots(highs, lows, orders=orders)
    zz_hi, zz_lo = zigzag_indices(highs, lows, threshold_pct=zz_threshold_pct)

    def _filter_by_zz(ar_idx: np.ndarray, zz_idx: np.ndarray) -> np.ndarray:
        if len(zz_idx) == 0:
            return ar_idx
        if len(ar_idx) == 0:
            return zz_idx
        kept: set[int] = set()
        for z in zz_idx:
            dists = np.abs(ar_idx.astype(int) - int(z))
            nearby = ar_idx[dists <= proximity]
            if len(nearby) > 0:
                kept.add(int(nearby[0]))
            else:
                kept.add(int(z))
        return np.array(sorted(kept), dtype=int)

    merged_hi = _filter_by_zz(ar_hi, zz_hi)
    merged_lo = _filter_by_zz(ar_lo, zz_lo)
    return merged_hi, merged_lo
