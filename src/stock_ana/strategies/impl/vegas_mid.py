"""Mid Vegas (EMA34/55/60) pullback detection, scoring, and structure checking.

Core strategy logic for detecting Mid Vegas channel touchbacks within major
upwaves.  Extracted from the backtest module so scan, api, and workflow layers
can all reference this single implementation without going through backtest.

Signal detection delegates to primitives.vegas_pullback.detect_vegas_pullback
(spans=MID_EMAS), ensuring the same state-machine logic is shared with
main_rally_pullback (which uses ALL_VEGAS_EMAS).

Public API
----------
Constants:
    MID_EMAS, LONG_EMAS, LONG_SLOPE_WINDOW, PRICE_ABOVE_LONG_WINDOW

EMA helpers:
    compute_vegas_emas(close_s) -> dict[int, ndarray]

Signal detection (zero look-ahead):
    detect_mid_touch_and_hold(close, low, emas) -> list[dict]  # Mid-only wrapper

Unified gate (re-exported from primitives):
    check_vegas_gate(bar, close, high, low, emas, ...) -> dict

Structure validation (legacy, no high/low required):
    check_mid_vegas_structure(entry_bar, close, emas) -> dict

Wave context:
    find_wave_context(waves, bar_idx) -> dict | None
    backward_consecutive_count(waves, wave_number, min_rise) -> int

Scoring / classification:
    score_pullback(sub_number, wave_rise_pct, wave_number,
                   market, consecutive_wave_count, mid_long_gap_pct)
        -> tuple[int, dict[str, int]]
    classify_signal(score) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ana.strategies.primitives.vegas_zones import (
    MID_EMAS,
    LONG_EMAS,
    LONG_SLOPE_WINDOW,
    PRICE_ABOVE_LONG_WINDOW,
    compute_vegas_emas,  # re-export：外部调用者无需改导入路径
)
from stock_ana.strategies.primitives.vegas_pullback import (
    check_vegas_gate,          # re-export：统一门控，供新调用者使用
    detect_vegas_pullback,     # 底层状态机，detect_mid_touch_and_hold 的实现基础
)


# ─────────────────────────────────────────────────────────────
# Signal Detection  (zero look-ahead)
# ─────────────────────────────────────────────────────────────

def detect_mid_touch_and_hold(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = 0.02,
    cooldown: int = 10,
) -> list[dict]:
    """Mid Vegas (EMA34/55/60) 触碰 + 站稳信号扫描（Mid-only 包装）。

    委托给 primitives.vegas_pullback.detect_vegas_pullback(spans=MID_EMAS)。
    默认 touch_margin=0.02（2% 容差，捕获 low 接近但未实际触碰的近距回踩）。
    启用 above_lookback=10，排除从下方长期恢复后才碰到 mid 的假信号。
    返回字段与原版完全兼容，新增 support_type="mid_vegas" 字段。
    """
    return detect_vegas_pullback(
        close, low, emas,
        spans=MID_EMAS,
        touch_margin=touch_margin,
        cooldown=cooldown,
        above_lookback=10,
    )


def detect_mid_touch_immediate(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = 0.02,
    cooldown: int = 10,
) -> list[dict]:
    """Mid Vegas 触碰信号扫描——触碰即发出信号，无需站稳确认。

    前置条件与 detect_mid_touch_and_hold 完全相同（above_lookback=10），
    但跳过 1-2 日站稳确认，在触碰当日直接发出信号：
        entry_bar == confirm_bar == touch_bar

    用于"触碰策略"：只要 low 接近 Mid Vegas，即输出信号供人工研判，
    不依赖后续收盘站稳，因此会产生更多信号（含部分假信号）。
    """
    n = len(close)
    raw_signals: list[dict] = []

    for span in MID_EMAS:
        ema = emas[span]
        band = f"ema{span}"
        last_emitted = -999

        for i in range(1, n):
            if i - last_emitted < cooldown:
                continue

            ev = ema[i]
            lo = low[i]

            if lo <= ev * (1 + touch_margin):
                # 与 detect_mid_touch_and_hold 相同的前置检查
                lb_start = max(0, i - 10)
                above_cnt = int(np.sum(close[lb_start:i] >= ema[lb_start:i]))
                if above_cnt < (i - lb_start) * 0.6:
                    continue

                raw_signals.append(dict(
                    touch_bar=i, confirm_bar=i, entry_bar=i,
                    support_type="mid_vegas", support_band=band,
                    ema_span=span,
                    touch_low=float(lo), ema_at_touch=float(ev),
                ))
                last_emitted = i

    # 跨 span 去重：同一次回调只保留最新一条（与 detect_vegas_pullback 一致）
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

def check_mid_vegas_structure(
    entry_bar: int,
    close: np.ndarray,
    emas: dict[int, np.ndarray],
) -> dict:
    """检查入场时的结构条件（全部基于入场当日及之前的数据，零前瞻）。

    Hard gate 条件（全部通过才算 passed）:
      1. Mid Vegas 上沿 > Long Vegas 上沿
      2. 入场价 > Long Vegas 上沿
      3. Long Vegas 上升（最近 LONG_SLOPE_WINDOW 天斜率 > 0）
      4. 过去 PRICE_ABOVE_LONG_WINDOW 天收盘始终高于 Long Vegas 上沿

    辅助参考列（不参与过滤）:
      gap_enough: mid/long 间距 >= 5%
      long_slope_strong: long 斜率 >= 2%

    Returns dict with keys:
        passed, mid_above_long, price_above_long, long_rising,
        price_above_long_3m, gap_enough, long_slope_strong,
        mid_upper, long_upper, long_slope_pct, mid_long_gap_pct
    """
    mid_upper = max(emas[s][entry_bar] for s in MID_EMAS)
    long_upper = max(emas[s][entry_bar] for s in LONG_EMAS)

    mid_above_long = mid_upper > long_upper
    price_above_long = close[entry_bar] > long_upper

    if entry_bar >= LONG_SLOPE_WINDOW:
        long_now = long_upper
        long_prev = max(emas[s][entry_bar - LONG_SLOPE_WINDOW] for s in LONG_EMAS)
        long_slope_pct = (long_now / long_prev - 1) * 100
        long_rising = long_slope_pct > 0
    else:
        long_slope_pct = 0.0
        long_rising = False

    mid_long_gap_pct = (mid_upper / long_upper - 1) * 100 if long_upper > 0 else 0.0
    gap_enough = mid_long_gap_pct >= 5.0
    long_slope_strong = long_slope_pct >= 2.0

    if entry_bar >= PRICE_ABOVE_LONG_WINDOW:
        lookback_start = entry_bar - PRICE_ABOVE_LONG_WINDOW
        long_upper_arr = np.maximum(
            np.maximum(
                emas[144][lookback_start:entry_bar + 1],
                emas[169][lookback_start:entry_bar + 1],
            ),
            emas[200][lookback_start:entry_bar + 1],
        )
        price_above_long_3m = bool(
            np.all(close[lookback_start:entry_bar + 1] > long_upper_arr)
        )
    else:
        price_above_long_3m = False

    passed = mid_above_long and price_above_long and long_rising and price_above_long_3m

    return {
        "passed": passed,
        "mid_above_long": mid_above_long,
        "price_above_long": price_above_long,
        "long_rising": long_rising,
        "price_above_long_3m": price_above_long_3m,
        "gap_enough": gap_enough,
        "long_slope_strong": long_slope_strong,
        "mid_upper": round(mid_upper, 3),
        "long_upper": round(long_upper, 3),
        "long_slope_pct": round(long_slope_pct, 2),
        "mid_long_gap_pct": round(mid_long_gap_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
# Wave Context
# ─────────────────────────────────────────────────────────────

def find_wave_context(waves: list[dict], bar_idx: int) -> dict | None:
    """找到 bar_idx 所处的大浪上下文（只用已知信息）。

    Returns the wave dict from ``analyze_wave_structure`` output, or None.
    """
    best = None
    for w in waves:
        sp = w["start_pivot"]["iloc"]
        ep = w["end_pivot"]["iloc"] if w["end_pivot"] else float("inf")
        if sp <= bar_idx <= ep:
            best = w
    if best is None:
        # 可能在最后一个未结束的浪中
        for w in reversed(waves):
            if w["start_pivot"]["iloc"] <= bar_idx:
                best = w
                break
    return best


def backward_consecutive_count(
    waves: list[dict],
    wave_number: int,
    min_rise: float = 20.0,
) -> int:
    """从 wave_number 往前数有多少连续且有效（rise≥min_rise）的浪（含自身）。

    只使用已完成浪的信息（end_pivot 已确定、rise_pct 已知），无未来信息泄露。
    当前浪（最后一个）不要求 rise≥min_rise（因为还在进行中），但前驱浪必须满足。
    """
    wave_map = {w["wave_number"]: w for w in waves}
    count = 0
    wn = wave_number

    while wn in wave_map:
        w = wave_map[wn]
        if count == 0:
            count = 1
            wn -= 1
            continue
        if w["end_pivot"] is None:
            break
        if w["rise_pct"] < min_rise:
            break
        next_w = wave_map.get(wn + 1)
        if next_w is None:
            break
        if w["end_pivot"]["iloc"] != next_w["start_pivot"]["iloc"]:
            break
        count += 1
        wn -= 1

    return count


# ─────────────────────────────────────────────────────────────
# Orderly Pullback Detection
# ─────────────────────────────────────────────────────────────

def check_orderly_pullback(
    close: np.ndarray,
    emas: dict[int, np.ndarray],
    prev_touch_bar: int,
    curr_touch_bar: int,
    min_bars: int = 5,
) -> bool:
    """两次 Mid Vegas 触碰之间是否形成有序回踩。

    有序回踩 = 两次触碰之间只有一个显著局部高点 + mid EMA 上升。
    代表"上升途中的单次呼吸"——最理想的中继回踩形态。

    Conditions
    ----------
    1. 两次 touch 之间只有一个显著局部高点（合并 ≤3 bar 相邻 peak，
       显著 = 高于两端较低者 × 1.01）
    2. 期间 Vegas mid (EMA55) 上升
    3. 当前 touch 价格 ≥ 上一次 touch 价格（隐含于 mid 上升）
    """
    if prev_touch_bar < 0 or curr_touch_bar - prev_touch_bar < min_bars:
        return False

    segment = close[prev_touch_bar : curr_touch_bar + 1]
    n_seg = len(segment)
    if n_seg < 3:
        return False

    # ── 1. 局部高点计数 ──
    peaks: list[int] = []
    for i in range(1, n_seg - 1):
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            peaks.append(i)
    if not peaks:
        return False

    # 合并相邻 peak（间距 ≤ 3 bar → 取较高者计为一个 peak）
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] <= 3:
            if segment[p] > segment[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # 只保留显著 peak：高于两端较低者 × 1.01
    base = min(segment[0], segment[-1])
    significant = [p for p in merged if segment[p] > base * 1.01]

    if len(significant) != 1:
        return False

    # ── 2. Mid EMA 上升 ──
    mid_prev = float(emas[55][prev_touch_bar])
    mid_curr = float(emas[55][curr_touch_bar])
    if mid_curr <= mid_prev:
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Scoring & Classification
# ─────────────────────────────────────────────────────────────

def score_pullback(
    sub_number: int,
    wave_rise_pct: float,
    wave_number: int,
    market: str,
    consecutive_wave_count: int,
    mid_long_gap_pct: float,
    orderly_pullback: bool = False,
) -> tuple[int, dict[str, int]]:
    """对升浪中一次 Mid Vegas 回踩打分（入场前提：结构条件已在调用前确认通过）。

    Factors — v2 (re-calibrated on new wave structure, 2026-04)
    -------
    mkt        : HK = +1, US = 0
                 (HK 63d wr=71.9% vs US 53.3%)
    sub_pos    : sub=1 = +2, sub=2~3 = 0, sub=4 = -1, sub>=5 = -2
                 (sub1 wr21=56.7%, sub4=-1.78%, sub5=-4.55%)
    wave_rise  : <30% = +2, 30-100% = +1, 101-200% = 0, >200% = -2
                 (wave_rise<30% wr21=64.2%, >200% wr21=41.5%)
    three_wave : consecutive_wave_count ≥ 3 → +1  (kept as trend filter)
    [removed]  : wave_num==3, ml_gap, orderly_pullback — no predictive power

    Returns:
        (total_score, detail_dict)
    """
    details: dict[str, int] = {}

    details["mkt"] = 1 if market == "HK" else 0

    if sub_number == 1:
        details["sub_pos"] = 2
    elif sub_number <= 3:
        details["sub_pos"] = 0
    elif sub_number == 4:
        details["sub_pos"] = -1
    else:
        details["sub_pos"] = -2

    if wave_rise_pct < 30:
        details["wave_rise"] = 2
    elif wave_rise_pct <= 100:
        details["wave_rise"] = 1
    elif wave_rise_pct <= 200:
        details["wave_rise"] = 0
    else:
        details["wave_rise"] = -2

    details["three_wave"] = 1 if consecutive_wave_count >= 3 else 0

    total = sum(details.values())
    return total, details


def classify_signal(score: int) -> str:
    """Map a composite factor score into the strategy's discrete action label.

    score >= 4  → STRONG_BUY
    score >= 2  → BUY
    score >= 0  → HOLD
    score <  0  → AVOID
    """
    if score >= 4:
        return "STRONG_BUY"
    elif score >= 2:
        return "BUY"
    elif score >= 0:
        return "HOLD"
    else:
        return "AVOID"
