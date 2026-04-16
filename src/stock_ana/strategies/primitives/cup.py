"""Reusable cup-body primitives shared by VCP logic and research scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.pivots import zigzag_points


def poly_smoothness(closes: np.ndarray, degree: int = 3) -> float:
    """Return polynomial-fit $R^2$ as a smoothness proxy for a price base."""
    n = len(closes)
    if n < 10:
        return 0.0
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, closes, degree)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((closes - fitted) ** 2)
    ss_tot = np.sum((closes - closes.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def bottom_channel_ratio(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    low_val: float,
    amplitude: float,
) -> tuple[float, float]:
    """Measure bottom-channel height ratio and dwell ratio for anti-V validation."""
    if amplitude <= 0 or len(closes) == 0:
        return 1.0, 0.0
    bottom_threshold = low_val + amplitude * 0.30
    mask = closes <= bottom_threshold
    dwell = float(np.sum(mask)) / len(closes)
    if not np.any(mask):
        return 0.0, 0.0
    ch_height = float(highs[mask].max() - lows[mask].min())
    return ch_height / amplitude, dwell


def analyze_cup_base_waves(
    base_highs: np.ndarray,
    base_lows: np.ndarray,
    depth_pct: float,
) -> dict:
    """Extract ZigZag wave structure inside a cup body for contraction checks."""
    threshold = max(3.5, min(depth_pct * 0.25, 6.0))
    pivots = zigzag_points(base_highs, base_lows, threshold)

    troughs = [pivot for pivot in pivots if pivot["type"] == "L"]
    peaks = [pivot for pivot in pivots if pivot["type"] == "H"]

    waves_detail: list[dict] = []
    wave_depths: list[float] = []

    for index in range(len(pivots) - 1):
        if pivots[index]["type"] == "H" and pivots[index + 1]["type"] == "L":
            high_value = pivots[index]["value"]
            low_value = pivots[index + 1]["value"]
            depth = (high_value - low_value) / high_value * 100 if high_value > 0 else 0
            waves_detail.append(
                {
                    "high_idx": pivots[index]["iloc"],
                    "low_idx": pivots[index + 1]["iloc"],
                    "high_val": float(high_value),
                    "low_val": float(low_value),
                }
            )
            wave_depths.append(round(depth, 2))

    is_contracting = False
    if len(wave_depths) >= 2:
        strict = all(
            wave_depths[idx] > wave_depths[idx + 1]
            for idx in range(len(wave_depths) - 1)
        )
        relaxed = (
            all(
                wave_depths[idx + 1] < wave_depths[idx] * 1.10
                for idx in range(len(wave_depths) - 1)
            )
            and wave_depths[-1] < wave_depths[0] * 0.80
        )
        is_contracting = strict or relaxed

    return {
        "pivots": pivots,
        "trough_count": len(troughs),
        "peak_count": len(peaks),
        "wave_depths": wave_depths,
        "is_contracting": is_contracting,
        "waves_detail": waves_detail,
    }


def find_cup_structure(
    df: pd.DataFrame,
    search_start_iloc: int,
    search_end_iloc: int,
    data_end_iloc: int,
    symmetry_tol: float = 0.05,
    min_depth_pct: float = 15.0,
    max_depth_pct: float = 33.0,
    min_weeks: int = 7,
    max_weeks: int = 65,
) -> dict | None:
    """Find a reusable P1/P2/P3 cup-body structure without handle logic."""
    n = len(df)
    start = max(0, search_start_iloc)
    end = min(n, search_end_iloc)
    if end - start < 35:
        return None

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    max_cup_bars = max_weeks * 5
    min_cup_bars = min_weeks * 5

    best: dict | None = None
    best_score = float("inf")

    half_window = max(min_cup_bars // 2, 20)
    p1_candidates: list[tuple[int, float]] = []
    for idx in range(start, end):
        win_left = max(start, idx - half_window)
        win_right = min(end, idx + half_window + 1)
        if highs[idx] >= highs[win_left:win_right].max():
            p1_candidates.append((idx, float(highs[idx])))

    for p1_iloc, p1_val in p1_candidates:
        p3_search_left = p1_iloc + min_cup_bars
        p3_search_right = min(end, p1_iloc + max_cup_bars + 1)
        if p3_search_left >= p3_search_right:
            continue

        for p3_iloc in range(p3_search_left, p3_search_right):
            p3_val = float(highs[p3_iloc])
            symmetry_pct = (p3_val - p1_val) / p1_val
            if abs(symmetry_pct) > symmetry_tol:
                continue

            region_highs = highs[p1_iloc : p3_iloc + 1]
            if p1_val < region_highs.max():
                continue

            half_local = min(10, (p3_iloc - p1_iloc) // 6)
            half_local = max(half_local, 3)
            local_left = max(p1_iloc, p3_iloc - half_local)
            local_right = min(end, p3_iloc + half_local + 1)
            if p3_val < highs[local_left:local_right].max():
                continue

            region_lows = lows[p1_iloc : p3_iloc + 1]
            p2_offset = int(np.argmin(region_lows))
            p2_iloc = p1_iloc + p2_offset
            p2_val = float(region_lows[p2_offset])

            depth_pct = (p1_val - p2_val) / p1_val * 100
            if not (min_depth_pct <= depth_pct <= max_depth_pct):
                continue

            cup_days = p3_iloc - p1_iloc
            max_gap = max(35, cup_days // 4)
            if data_end_iloc - p3_iloc > max_gap:
                continue

            cup_closes = closes[p1_iloc : p3_iloc + 1]
            amplitude = p1_val - p2_val
            if amplitude <= 0:
                continue

            bottom_threshold = p2_val + amplitude * 0.30
            dwell = float(np.sum(cup_closes <= bottom_threshold)) / len(cup_closes)
            if dwell < 0.08:
                continue

            x = np.arange(len(cup_closes), dtype=float)
            coeffs = np.polyfit(x, cup_closes, 2)
            a = coeffs[0]
            vertex_x = -coeffs[1] / (2 * a) if a != 0 else len(x) / 2
            is_u_shape = a > 0 and len(x) * 0.20 <= vertex_x <= len(x) * 0.80
            if not is_u_shape:
                continue

            score = abs(depth_pct - (min_depth_pct + max_depth_pct) / 2.0)
            if score < best_score:
                best_score = score
                best = {
                    "p1_iloc": p1_iloc,
                    "p1_val": float(p1_val),
                    "p2_iloc": p2_iloc,
                    "p2_val": float(p2_val),
                    "p3_iloc": p3_iloc,
                    "p3_val": float(p3_val),
                    "depth_pct": round(depth_pct, 2),
                    "symmetry_pct": round(symmetry_pct * 100, 2),
                    "cup_days": cup_days,
                    "dwell_ratio": round(dwell, 3),
                    "is_u_shape": is_u_shape,
                }

    if best is None:
        return None

    p1_iloc = best["p1_iloc"]
    p3_iloc = best["p3_iloc"]
    wave_info = analyze_cup_base_waves(
        highs[p1_iloc : p3_iloc + 1],
        lows[p1_iloc : p3_iloc + 1],
        best["depth_pct"],
    )
    best["wave_info"] = wave_info
    return best


def check_cup_ma_trend(
    df: pd.DataFrame,
    ref_close: float,
    ref_iloc: int,
) -> tuple[bool, str]:
    """Validate the MA trend template used before cup-body and handle checks."""
    closes = df["close"].values.astype(float)
    n = len(closes)
    if n < 210:
        return False, "数据不足210根K线"

    idx = min(ref_iloc, n - 1)
    ref_val = ref_close

    def _sma(period: int, at: int) -> float | None:
        start = at - period + 1
        if start < 0:
            return None
        return float(np.mean(closes[start : at + 1]))

    sma50 = _sma(50, idx)
    sma150 = _sma(150, idx)
    sma200 = _sma(200, idx)

    if sma50 is None or sma150 is None or sma200 is None:
        return False, "均线数据不足"
    if ref_val <= sma150:
        return False, f"价格{ref_val:.2f}≤SMA150({sma150:.2f})"
    if ref_val <= sma200:
        return False, f"价格{ref_val:.2f}≤SMA200({sma200:.2f})"
    if sma50 <= sma150:
        return False, f"SMA50({sma50:.2f})≤SMA150({sma150:.2f})"
    if sma150 <= sma200:
        return False, f"SMA150({sma150:.2f})≤SMA200({sma200:.2f})"

    slope_window = 20
    sma200_series: list[float] = []
    for k in range(idx - slope_window + 1, idx + 1):
        if k < 199:
            sma200_series = []
            break
        sma200_series.append(float(np.mean(closes[k - 199 : k + 1])))

    if len(sma200_series) < slope_window:
        return False, "SMA200序列不足以计算斜率"

    x = np.arange(slope_window, dtype=float)
    y = np.array(sma200_series)
    slope_coef = float(np.polyfit(x, y, 1)[0])
    if slope_coef <= 0:
        return False, f"SMA200斜率={slope_coef:.4f}≤0（下行趋势）"

    lookback_22 = idx - 22
    if lookback_22 < 199:
        return False, "SMA200上升持续时间数据不足"
    sma200_22ago = float(np.mean(closes[lookback_22 - 199 : lookback_22 + 1]))
    if sma200 <= sma200_22ago:
        return False, f"SMA200未能持续上升≥1个月(当前{sma200:.2f}≤22日前{sma200_22ago:.2f})"

    prior_start = max(0, idx - 252)
    prior_lows = df["low"].values.astype(float)[prior_start : idx + 1]
    if len(prior_lows) < 60:
        return False, "前期数据不足60根K线"
    prior_52w_low = float(np.min(prior_lows))
    if prior_52w_low <= 0:
        return False, "前期低点无效"
    prior_advance = (ref_val - prior_52w_low) / prior_52w_low * 100
    if prior_advance < 30.0:
        return False, f"前期涨幅{prior_advance:.1f}%<30%（动能不足）"

    return True, "OK"
