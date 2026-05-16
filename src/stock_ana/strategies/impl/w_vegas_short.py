"""Weekly Vegas Short (w_ema_8/21) pullback strategy.

Channel mapping
---------------
  Short Vegas  : w_ema_8 / w_ema_21   — pullback target (回踩目标通道)
  Mid Vegas    : w_ema_34 / w_ema_55  — trend guard (趋势守卫, no Long channel on weekly)

Structure gate (all four must pass)
------------------------------------
  1. short_above_mid   : max(w_ema_8, w_ema_21) > max(w_ema_34, w_ema_55)
  2. price_above_mid   : close > max(w_ema_34, w_ema_55)
  3. mid_rising        : Mid upper slope over MID_SLOPE_WINDOW weeks > 0
  4. price_above_mid_Nw: all closes over PRICE_ABOVE_MID_WINDOW weeks > mid upper

Pullback detection reuses detect_vegas_pullback() from primitives unchanged.
The state machine behaviour (touch_margin, pierce tolerance, cooldown) is
identical to the daily strategy; only the time granularity is different.

Public API
----------
Constants:
    SHORT_EMAS, MID_EMAS_W, MID_SLOPE_WINDOW, PRICE_ABOVE_MID_WINDOW

EMA helpers:
    compute_w_short_emas(close_s) -> dict[int, ndarray]

Signal detection:
    detect_w_short_touch_and_hold(close, low, emas) -> list[dict]
    detect_w_short_touch_immediate(close, low, emas) -> list[dict]

Structure validation:
    check_w_short_structure(entry_bar, close, emas) -> dict

Scoring / classification:
    score_w_pullback(market, mid_slope_pct, short_mid_gap_pct) -> (int, dict)
    classify_signal(score) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.vegas_pullback import detect_vegas_pullback

# ─────────────────────────────────────────────────────────────
# Channel constants (weekly frame)
# ─────────────────────────────────────────────────────────────

SHORT_EMAS: list[int] = [8, 21]        # 回踩目标通道
MID_EMAS_W: list[int] = [34, 55]       # 趋势守卫 (weekly 不设 Long)
ALL_W_EMAS: list[int] = SHORT_EMAS + MID_EMAS_W

# Structure gate windows (in weeks)
MID_SLOPE_WINDOW: int = 10        # Mid 斜率窗口（10周 ≈ 2.5 个月）
PRICE_ABOVE_MID_WINDOW: int = 8   # 价格需持续高于 Mid 的最少周数（≈2 个月）

# Pullback detection parameters
_TOUCH_MARGIN: float = 0.02   # 2% 容差，与日线一致
_COOLDOWN: int = 5            # 周线冷却期：5 周
_ABOVE_LOOKBACK: int = 10     # 前置检查：10 周内至少 60% close > EMA


# ─────────────────────────────────────────────────────────────
# EMA computation (weekly close)
# ─────────────────────────────────────────────────────────────

def compute_w_short_emas(close_s: pd.Series) -> dict[int, np.ndarray]:
    """计算 Short + Mid Vegas 周线 EMA，返回 {span: ndarray}。

    与日线版本使用相同的 ewm(span, adjust=False) 计算方式，
    保证策略层在日/周之间切换时行为一致。
    """
    return {
        s: close_s.ewm(span=s, adjust=False).mean().values
        for s in ALL_W_EMAS
    }


# ─────────────────────────────────────────────────────────────
# Signal Detection  (zero look-ahead)
# ─────────────────────────────────────────────────────────────

def detect_w_short_touch_and_hold(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = _TOUCH_MARGIN,
    cooldown: int = _COOLDOWN,
) -> list[dict]:
    """Short Vegas (w_ema_8/21) 周线触碰 + 站稳确认扫描。

    状态机逻辑委托给 detect_vegas_pullback()，与日线 Mid Vegas 完全一致：
      - 前置检查（above_lookback=10 周）：防止从下方长期修复后的假信号
      - 站稳确认：单周收回 或 ≤2 周刺破后两周站稳
      - 冷却期：cooldown=5 周
    """
    return detect_vegas_pullback(
        close, low, emas,
        spans=SHORT_EMAS,
        touch_margin=touch_margin,
        cooldown=cooldown,
        above_lookback=_ABOVE_LOOKBACK,
    )


def detect_w_short_touch_immediate(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = _TOUCH_MARGIN,
    cooldown: int = _COOLDOWN,
) -> list[dict]:
    """Short Vegas 触碰即出（无需站稳确认）。

    触碰当日即发出信号（entry_bar == touch_bar），适合配合人工研判。
    与日线 detect_mid_touch_immediate 逻辑对称。
    """
    n = len(close)
    raw_signals: list[dict] = []

    for span in SHORT_EMAS:
        ema = emas[span]
        band = f"w_ema{span}"
        last_emitted = -999

        for i in range(1, n):
            if i - last_emitted < cooldown:
                continue

            ev = ema[i]
            lo = low[i]

            if lo <= ev * (1 + touch_margin):
                lb_start = max(0, i - _ABOVE_LOOKBACK)
                above_cnt = int(np.sum(close[lb_start:i] >= ema[lb_start:i]))
                if above_cnt < (i - lb_start) * 0.6:
                    continue

                raw_signals.append(dict(
                    touch_bar=i, confirm_bar=i, entry_bar=i,
                    support_type="short_vegas_w", support_band=band,
                    ema_span=span,
                    touch_low=float(lo), ema_at_touch=float(ev),
                ))
                last_emitted = i

    raw_signals.sort(key=lambda s: s["entry_bar"])
    signals: list[dict] = []
    last_entry = -999
    for sig in raw_signals:
        if sig["entry_bar"] - last_entry >= cooldown:
            signals.append(sig)
            last_entry = sig["entry_bar"]

    return signals


# ─────────────────────────────────────────────────────────────
# Structure Validation
# ─────────────────────────────────────────────────────────────

def check_w_short_structure(
    entry_bar: int,
    close: np.ndarray,
    emas: dict[int, np.ndarray],
) -> dict:
    """周线 Short Vegas 结构门控（全部基于 entry_bar 及之前数据，零前瞻）。

    Hard gate（全部通过才算 passed）:
      1. Short 上沿 > Mid 上沿（通道有序排列）
      2. 入场价 > Mid 上沿（价格在趋势守卫上方）
      3. Mid 上升（最近 MID_SLOPE_WINDOW 周斜率 > 0）
      4. 过去 PRICE_ABOVE_MID_WINDOW 周收盘始终高于 Mid 上沿

    辅助参考（不参与过滤）:
      gap_enough: short/mid 间距 >= 5%
      mid_slope_strong: mid 斜率 >= 2%

    Returns dict with keys:
        passed, short_above_mid, price_above_mid, mid_rising,
        price_above_mid_Nw, gap_enough, mid_slope_strong,
        short_upper, mid_upper, mid_slope_pct, short_mid_gap_pct
    """
    short_upper = max(emas[s][entry_bar] for s in SHORT_EMAS)
    mid_upper = max(emas[s][entry_bar] for s in MID_EMAS_W)

    short_above_mid = short_upper > mid_upper
    price_above_mid = close[entry_bar] > mid_upper

    if entry_bar >= MID_SLOPE_WINDOW:
        mid_now = mid_upper
        mid_prev = max(emas[s][entry_bar - MID_SLOPE_WINDOW] for s in MID_EMAS_W)
        mid_slope_pct = (mid_now / mid_prev - 1) * 100 if mid_prev > 0 else 0.0
        mid_rising = mid_slope_pct > 0
    else:
        mid_slope_pct = 0.0
        mid_rising = False

    short_mid_gap_pct = (short_upper / mid_upper - 1) * 100 if mid_upper > 0 else 0.0
    gap_enough = short_mid_gap_pct >= 5.0
    mid_slope_strong = mid_slope_pct >= 2.0

    if entry_bar >= PRICE_ABOVE_MID_WINDOW:
        lookback_start = entry_bar - PRICE_ABOVE_MID_WINDOW
        mid_upper_arr = np.maximum(
            emas[34][lookback_start:entry_bar + 1],
            emas[55][lookback_start:entry_bar + 1],
        )
        price_above_mid_nw = bool(
            np.all(close[lookback_start:entry_bar + 1] > mid_upper_arr)
        )
    else:
        price_above_mid_nw = False

    passed = short_above_mid and price_above_mid and mid_rising and price_above_mid_nw

    return {
        "passed": passed,
        "short_above_mid": short_above_mid,
        "price_above_mid": price_above_mid,
        "mid_rising": mid_rising,
        "price_above_mid_nw": price_above_mid_nw,
        "gap_enough": gap_enough,
        "mid_slope_strong": mid_slope_strong,
        "short_upper": round(short_upper, 3),
        "mid_upper": round(mid_upper, 3),
        "mid_slope_pct": round(mid_slope_pct, 2),
        "short_mid_gap_pct": round(short_mid_gap_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
# Scoring & Classification
# ─────────────────────────────────────────────────────────────

def score_w_pullback(
    market: str,
    mid_slope_pct: float,
    short_mid_gap_pct: float,
) -> tuple[int, dict[str, int]]:
    """周线 Short Vegas 回踩打分（简化版，优先待策略跑稳后再校准）。

    Factors
    -------
    mkt        : HK = +1, US = 0
    mid_slope  : slope >= 2% = +1, else 0      (Mid 斜率强劲)
    gap        : short/mid 间距 >= 5% = +1, else 0  (通道展开充分)

    Max score: HK=3, US=2. Classification:
        >= 2 → BUY, >= 3 → STRONG_BUY, else HOLD

    Returns:
        (total_score, detail_dict)
    """
    details: dict[str, int] = {}
    details["mkt"] = 1 if market == "HK" else 0
    details["mid_slope"] = 1 if mid_slope_pct >= 2.0 else 0
    details["gap"] = 1 if short_mid_gap_pct >= 5.0 else 0
    total = sum(details.values())
    return total, details


def classify_signal(score: int) -> str:
    """Weekly signal classification.

    score >= 3  → STRONG_BUY
    score >= 2  → BUY
    score >= 1  → HOLD
    score <  1  → AVOID
    """
    if score >= 3:
        return "STRONG_BUY"
    elif score >= 2:
        return "BUY"
    elif score >= 1:
        return "HOLD"
    else:
        return "AVOID"
