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


def swing_pivots(
    df: "pd.DataFrame",
    threshold_pct: float = 7.0,
) -> list[dict]:
    """
    用原始 K 线 high/low 做 ZigZag，识别小波段高低转折点。

    与 detect_ema8_swings 的区别：
      - 直接使用 raw high/low，不经过 EMA8 平滑
      - 粒度更细，适合蜡烛图形态分析（顶底识别）
      - 默认 threshold=7%：过滤日内噪音，保留有意义的波段转折

    Args:
        df:             包含 open/high/low/close 列的 DataFrame，index 为日期
        threshold_pct:  判定转折所需的最小反向幅度（%），默认 7.0

    Returns:
        list of dict，每项包含：
          - type:  "H"（高点）或 "L"（低点）
          - iloc:  在 df 中的整数位置
          - value: 对应的 high 或 low 价格
          - date:  日期字符串 (YYYY-MM-DD)
    """
    import pandas as pd

    cols = [c.lower() for c in df.columns]
    highs = df[df.columns[cols.index("high")]].values.astype(float)
    lows  = df[df.columns[cols.index("low")]].values.astype(float)
    idx   = df.index

    raw = zigzag_points(highs, lows, threshold_pct=threshold_pct)
    for p in raw:
        dt = idx[p["iloc"]]
        p["date"] = str(dt.date()) if hasattr(dt, "date") else str(dt)
    return raw


def swing_current_state(
    df: "pd.DataFrame",
    threshold_pct: float = 7.0,
) -> dict:
    """
    返回 ZigZag 右侧"进行中"波段的当前状态，解决 zigzag 右侧盲区问题。

    ZigZag 只在价格反向运动足够幅度后才"确认"一个 pivot，因此最后若干根
    K 线始终处于待定状态。本函数追踪最后一个已确认 pivot 之后的方向和
    当前候选极值，让调用方可以判断当前最新 K 线处于哪个波段方向中。

    Args:
        df:             包含 high/low 列的 DataFrame
        threshold_pct:  与 swing_pivots 保持一致

    Returns:
        dict 包含：
          - confirmed_pivots: 已确认的 pivot 列表（同 swing_pivots）
          - last_pivot:       最后一个已确认 pivot dict（或 None）
          - trend:            当前进行中的方向，"up" | "down" | "unknown"
                              "up"   = 最后确认是低点，正在寻找下一个高点
                              "down" = 最后确认是高点，正在寻找下一个低点
          - candidate_iloc:   当前候选极值的 iloc（尚未确认的当前段极值）
          - candidate_value:  当前候选极值的价格
          - candidate_date:   当前候选极值的日期字符串
          - pct_from_last:    候选极值相对于最后确认 pivot 的变化幅度（%）
                              正数=上涨，负数=下跌
          - pct_to_confirm:   还需要再反向多少 % 才能确认当前候选为 pivot
                              即距离"触发下一个转折"还差多远
    """
    import pandas as pd

    cols = [c.lower() for c in df.columns]
    highs = df[df.columns[cols.index("high")]].values.astype(float)
    lows  = df[df.columns[cols.index("low")]].values.astype(float)
    idx   = df.index
    n = len(highs)

    def _date_str(i: int) -> str:
        dt = idx[i]
        return str(dt.date()) if hasattr(dt, "date") else str(dt)

    confirmed: list[dict] = []
    trend = 0
    cand_hi_i, cand_hi_v = 0, highs[0]
    cand_lo_i, cand_lo_v = 0, lows[0]

    for i in range(1, n):
        if trend == 0:
            if highs[i] > cand_hi_v:
                cand_hi_i, cand_hi_v = i, highs[i]
            if lows[i] < cand_lo_v:
                cand_lo_i, cand_lo_v = i, lows[i]
            if cand_hi_v > 0 and (cand_hi_v - lows[i]) / cand_hi_v * 100 >= threshold_pct:
                confirmed.append({"type": "H", "iloc": cand_hi_i, "value": float(cand_hi_v), "date": _date_str(cand_hi_i)})
                trend = -1
                cand_lo_i, cand_lo_v = i, lows[i]
            elif cand_lo_v > 0 and (highs[i] - cand_lo_v) / cand_lo_v * 100 >= threshold_pct:
                confirmed.append({"type": "L", "iloc": cand_lo_i, "value": float(cand_lo_v), "date": _date_str(cand_lo_i)})
                trend = 1
                cand_hi_i, cand_hi_v = i, highs[i]
        elif trend == 1:
            if highs[i] > cand_hi_v:
                cand_hi_i, cand_hi_v = i, highs[i]
            if cand_hi_v > 0 and (cand_hi_v - lows[i]) / cand_hi_v * 100 >= threshold_pct:
                confirmed.append({"type": "H", "iloc": cand_hi_i, "value": float(cand_hi_v), "date": _date_str(cand_hi_i)})
                trend = -1
                cand_lo_i, cand_lo_v = i, lows[i]
        else:  # trend == -1
            if lows[i] < cand_lo_v:
                cand_lo_i, cand_lo_v = i, lows[i]
            if cand_lo_v > 0 and (highs[i] - cand_lo_v) / cand_lo_v * 100 >= threshold_pct:
                confirmed.append({"type": "L", "iloc": cand_lo_i, "value": float(cand_lo_v), "date": _date_str(cand_lo_i)})
                trend = 1
                cand_hi_i, cand_hi_v = i, highs[i]

    last_pivot = confirmed[-1] if confirmed else None

    # 根据当前 trend 方向确定候选极值
    if trend == 1:
        # 上升段：追踪候选高点
        current_trend = "up"
        cand_i, cand_v = cand_hi_i, cand_hi_v
    elif trend == -1:
        # 下降段：追踪候选低点
        current_trend = "down"
        cand_i, cand_v = cand_lo_i, cand_lo_v
    else:
        current_trend = "unknown"
        cand_i, cand_v = cand_hi_i, cand_hi_v

    # 计算候选极值相对于最后确认 pivot 的涨跌幅
    if last_pivot and last_pivot["value"] > 0:
        pct_from_last = (cand_v - last_pivot["value"]) / last_pivot["value"] * 100
    else:
        pct_from_last = 0.0

    # 当前最新收盘价距离"触发下一个反转"还差多少
    last_close = float(df[df.columns[cols.index("close")]].iloc[-1])
    if current_trend == "up" and cand_v > 0:
        # 需要从候选高点跌 threshold_pct 才能确认高点
        trigger_price = cand_v * (1 - threshold_pct / 100)
        pct_to_confirm = (last_close - trigger_price) / trigger_price * 100
    elif current_trend == "down" and cand_v > 0:
        # 需要从候选低点涨 threshold_pct 才能确认低点
        trigger_price = cand_v * (1 + threshold_pct / 100)
        pct_to_confirm = (trigger_price - last_close) / last_close * 100
    else:
        pct_to_confirm = float("nan")

    return {
        "confirmed_pivots": confirmed,
        "last_pivot":        last_pivot,
        "trend":             current_trend,
        "candidate_iloc":    int(cand_i),
        "candidate_value":   float(cand_v),
        "candidate_date":    _date_str(cand_i),
        "pct_from_last":     round(pct_from_last, 2),
        "pct_to_confirm":    round(pct_to_confirm, 2) if not np.isnan(pct_to_confirm) else None,
    }


def trend_series_from_pivots(
    df: "pd.DataFrame",
    confirmed_pivots: list[dict],
    current_trend: str,
) -> "pd.Series":
    """
    根据已确认的 ZigZag pivot 列表，为 df 中每根 K 线标记趋势方向。

    规则：
      - L → H 段（含两端）：标记为 "up"
      - H → L 段（含两端）：标记为 "down"
      - 最后一个 pivot 之后到末尾：使用 current_trend（swing_current_state 提供）

    Args:
        df:                DataFrame，index 为日期，用于对齐长度
        confirmed_pivots:  swing_current_state["confirmed_pivots"]
        current_trend:     swing_current_state["trend"]，"up" | "down" | "unknown"

    Returns:
        pd.Series，index 同 df，值为 "up" | "down" | "unknown"
    """
    import pandas as pd

    trend = pd.Series("unknown", index=df.index, dtype=str)

    if not confirmed_pivots:
        return trend

    for i in range(len(confirmed_pivots) - 1):
        p1 = confirmed_pivots[i]
        p2 = confirmed_pivots[i + 1]
        seg = "up" if p1["type"] == "L" else "down"
        trend.iloc[p1["iloc"]: p2["iloc"] + 1] = seg

    # 最后一个 pivot 之后使用实时方向
    last_iloc = confirmed_pivots[-1]["iloc"]
    trend.iloc[last_iloc:] = current_trend

    return trend


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
