"""
主升浪上升阶段回调检测策略（Vegas 回踩版）

实现机制：
  门控：check_vegas_gate（统一，与 vegas_mid 共用同一套条件）
  检测：detect_vegas_pullback（状态机，touch -> 两日站稳，与 vegas_mid 共用同一逻辑）
  范围：MID（EMA34/55/60）+ LONG（EMA144/169/200）均可触发
        mid_only=True 时退化为仅检测中期线，与 vegas_mid 等价

与 vegas_mid 的关系：
  vegas_mid 策略 = 本策略 + mid_only=True（只踩中期线的严格子集）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.vegas_zones import (
    ALL_VEGAS_EMAS,
    MID_EMAS,
    LONG_EMAS,
    compute_vegas_emas,
)
from stock_ana.strategies.primitives.vegas_pullback import (
    check_vegas_gate,
    detect_vegas_pullback,
)

_MIN_HISTORY = 252  # 需要足够历史数据支撑年低涨幅及 EMA 稳定计算


def screen_main_rally_pullback(
    df,
    trend_days=63,
    high_lookback=126,
    prior_above_days=55,
    retrace_days=5,
    rise_min_pct=30.0,
    touch_margin=0.01,
    cooldown=10,
    mid_only=False,
):
    """检测当日是否出现 Vegas 回调站稳信号。

    门控条件（check_vegas_gate）：
      1. EMA 多头排列：EMA34 > EMA55 > EMA144 > EMA169
      2. EMA 持续上升：EMA55 上涨天数 >= 80%，EMA34 >= 72%，净增
      3. 强势筛选：当前价较一年低点涨幅 >= rise_min_pct

    检测机制（detect_vegas_pullback，状态机）：
      low 触线 -> 收盘站回 -> 次日仍站上 -> 确认（两日站稳）
      单日触碰即收回视为强势止跌，直接确认

    Parameters
    ----------
    mid_only : 仅检测踩中期 Vegas（EMA34/55/60），等价于 vegas_mid 策略。
               默认 False，中期 + 长期（EMA144/169/200）均可触发。

    Returns
    -------
    命中返回 dict（包含判定细节），否则返回 None。
    """
    if df is None or len(df) < _MIN_HISTORY:
        return None

    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    if "close" not in x.columns:
        return None
    if "high" not in x.columns:
        x["high"] = x["close"]
    if "low" not in x.columns:
        x["low"] = x["close"]

    close_arr = x["close"].astype(float).values
    high_arr = x["high"].astype(float).values
    low_arr = x["low"].astype(float).values
    n = len(close_arr)

    emas = compute_vegas_emas(x["close"].astype(float))

    # 门控检查（当前最后一根 K 线）
    gate = check_vegas_gate(
        bar=n - 1,
        close=close_arr,
        high=high_arr,
        low=low_arr,
        emas=emas,
        trend_days=trend_days,
        rise_min_pct=rise_min_pct,
    )
    if not gate["passed"]:
        return None

    # 状态机扫描（全历史，取今日信号）
    spans = MID_EMAS if mid_only else ALL_VEGAS_EMAS
    signals = detect_vegas_pullback(
        close_arr, low_arr, emas,
        spans=spans,
        touch_margin=touch_margin,
        cooldown=cooldown,
    )

    # entry_bar == n 表示今日确认、次日入场
    today_signals = [s for s in signals if s["entry_bar"] == n]
    if not today_signals:
        return None

    sig = today_signals[-1]

    # 辅助指标（保持与旧版输出字段的一致性）
    curr_close = float(close_arr[-1])
    ema_vals = {span: float(emas[span][-1]) for span in ALL_VEGAS_EMAS}

    actual_lookback = min(high_lookback, n - 1)
    ref_high = float(np.max(high_arr[-actual_lookback:]))
    pullback_pct = (ref_high - curr_close) / ref_high * 100 if ref_high > 0 else 0.0

    zone_spans = MID_EMAS if sig["support_type"] == "mid_vegas" else LONG_EMAS
    zone_vals = [ema_vals[s] for s in zone_spans]

    def _vs_pct(span):
        v = ema_vals[span]
        return round((curr_close / v - 1.0) * 100, 2) if v > 0 else 0.0

    return {
        "pattern": f"vegas_pullback_{sig['support_type']}",
        "ema34": round(ema_vals[34], 3),
        "ema55": round(ema_vals[55], 3),
        "ema60": round(ema_vals[60], 3),
        "ema144": round(ema_vals[144], 3),
        "ema169": round(ema_vals[169], 3),
        "ema200": round(ema_vals[200], 3),
        "ema34_up_ratio": gate["ema34_up_ratio"],
        "ema55_up_ratio": gate["ema55_up_ratio"],
        "ema55_3m_change_pct": gate["ema55_net_up_pct"],
        "pullback_pct_from_high": round(pullback_pct, 2),
        "close_vs_ema34_pct": _vs_pct(34),
        "close_vs_ema55_pct": _vs_pct(55),
        "close_vs_ema60_pct": _vs_pct(60),
        "close_vs_ema200_pct": _vs_pct(200),
        "rise_from_1y_low_pct": gate["rise_from_1y_low_pct"],
        "support_type": sig["support_type"],
        "support_band": sig["support_band"],
        "support_zone_low": round(min(zone_vals), 3),
        "support_zone_high": round(max(zone_vals), 3),
        "touch_bar": sig["touch_bar"],
        "confirm_bar": sig["confirm_bar"],
        "entry_bar": sig["entry_bar"],
        "touch_low": sig["touch_low"],
        "ema_at_touch": sig["ema_at_touch"],
    }
