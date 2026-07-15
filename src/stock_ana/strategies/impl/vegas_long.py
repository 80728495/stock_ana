"""Long Vegas (EMA144/169/200) pullback detection, gating, stats and scoring.

Two strategy layers live here:

1. Observation layer (legacy):
   detect_long_touch_immediate — single-bar touch detection, no scoring,
   consumed by vegas_mid_scan as "long_touch" OBSERVE signals.

2. Wave-pullback strategy (vegas_long):
   大浪回踩策略——上涨周期中，价格从显著高于 Long Vegas 的浪顶回踩到
   Long Vegas 通道并止跌回弹时触发。核心假设：强势上涨股的大浪回踩
   通常止于 Long Vegas 通道而不会更深，且越早期的回踩越可靠。

   Components (all zero look-ahead at signal time):
     detect_long_touch_and_hold  — 触碰 + 站稳回弹确认（状态机复用
                                   primitives.vegas_pullback，spans=LONG_EMAS）
     locate_wave_pullback        — 把触碰映射到它终结的大浪，给出回踩序次
                                   与「是否真实大浪回踩」判定
     check_long_wave_structure   — 上涨周期门控（LV 上升、多头排列、
                                   上方运行占比、年内涨幅、浪顶落差）
     compute_lv_respect_stats    — 历史统计：该标的过去大浪回踩是否
                                   显著以 Long Vegas 为回踩节点
     score_long_pullback         — 回踩序次 / 历史尊重率 / 趋势强度打分
     classify_long_signal        — score → STRONG_BUY/BUY/HOLD/AVOID

Public API
----------
detect_long_touch_immediate(close, low, emas, ...) -> list[dict]
detect_long_touch_and_hold(close, low, emas, ...) -> list[dict]
locate_wave_pullback(waves, touch_bar, tol) -> dict
check_long_wave_structure(entry_bar, close, low, emas, ...) -> dict
compute_lv_respect_stats(waves, close, emas, ...) -> dict
score_long_pullback(...) -> tuple[int, dict[str, int]]
classify_long_signal(score) -> str
"""

from __future__ import annotations

import numpy as np

from stock_ana.strategies.primitives.vegas_zones import (
    LONG_EMAS,
    LONG_SLOPE_WINDOW,
)
from stock_ana.strategies.primitives.vegas_pullback import detect_vegas_pullback


def detect_long_touch_immediate(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = 0.02,
    cooldown: int = 10,
    above_lookback: int = 10,
) -> list[dict]:
    """Long Vegas 触碰信号扫描——low 触及 Long EMA 且当日收盘站住。

    与 Mid Vegas touch_immediate 类似，但目标线为 EMA144/169/200。
    仅保留 close >= Long EMA 的 bar（收盘站住），不接受跌破后次日收回。

    Parameters
    ----------
    touch_margin : low ≤ EMA × (1 + margin) 视为"触及"。0.02 = 2% 容差。
    cooldown : 同一 span 两次信号之间最少间隔 bar 数。
    above_lookback : 「回踩」语义门回看窗口（pullback_precondition：
                     前一根收盘在上方 + 80% 占比 + 从更高处跌下来）。

    Returns
    -------
    list of signal dicts, each containing:
        touch_bar, confirm_bar, entry_bar,
        support_type ("long_vegas"), support_band (e.g. "ema144"),
        ema_span, touch_low, ema_at_touch
    """
    from stock_ana.strategies.primitives.vegas_pullback import pullback_precondition

    n = len(close)
    raw_signals: list[dict] = []

    for span in LONG_EMAS:
        ema = emas[span]
        band = f"ema{span}"
        last_emitted = -999

        for i in range(1, n):
            if i - last_emitted < cooldown:
                continue

            ev = ema[i]
            lo = low[i]
            c = close[i]

            # 触及 Long EMA 且收盘站住
            if lo <= ev * (1 + touch_margin) and c >= ev:
                # 「回踩」语义门：从上方跌下来（拒绝跌穿后涨回的上穿触碰）
                if above_lookback > 0 and not pullback_precondition(
                    i, close, ema, above_lookback, from_above_pct=0.04,
                ):
                    continue

                raw_signals.append(dict(
                    touch_bar=i, confirm_bar=i, entry_bar=i,
                    support_type="long_vegas", support_band=band,
                    ema_span=span,
                    touch_low=float(lo), ema_at_touch=float(ev),
                ))
                last_emitted = i

    # 跨 span 去重：同一次回调只保留最新一条
    raw_signals.sort(key=lambda s: s["entry_bar"])
    signals: list[dict] = []
    last_entry = -999
    for sig in raw_signals:
        if sig["entry_bar"] - last_entry >= cooldown:
            signals.append(sig)
            last_entry = sig["entry_bar"]

    return signals


# ─────────────────────────────────────────────────────────────
# Wave-Pullback Strategy: Signal Detection  (zero look-ahead)
# ─────────────────────────────────────────────────────────────

def detect_long_touch_and_hold(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    touch_margin: float = 0.02,
    depart_pct: float = 0.08,
    above_lookback: int = 20,
) -> list[dict]:
    """Long Vegas 触碰 + 站稳回弹信号扫描（Long-only 包装）。

    委托给 primitives.vegas_pullback.detect_vegas_pullback(spans=LONG_EMAS)，
    与 Mid Vegas 共用同一状态机：
      - 单日触碰收回（low 靠近 LV，close ≥ LV）→ 当日确认
      - 短暂刺破（≤ 2 日 close < LV）→ 两日站稳确认
    「回踩」语义门（pullback_precondition，above_lookback=20）：触碰前一根
    收盘在 LV 上方 + 近 20 日 80% 在上方 + 窗口内最高收盘 ≥ LV×1.04
    ——从上方跌下来，拒绝跌穿后涨回的上穿触碰 / 下跌途中的反抽。
    去重用"离开-再回踩/越踩越深"规则，depart_pct=8%（大浪回踩节奏比
    mid 慢、幅度更大，离开阈值相应放宽）。
    """
    return detect_vegas_pullback(
        close, low, emas,
        spans=LONG_EMAS,
        touch_margin=touch_margin,
        depart_pct=depart_pct,
        above_lookback=above_lookback,
        from_above_pct=0.04,
    )


# ─────────────────────────────────────────────────────────────
# Wave-Pullback Strategy: Map a touch to the wave it terminates
# ─────────────────────────────────────────────────────────────

def locate_wave_pullback(
    waves: list[dict],
    touch_bar: int,
    tol: int = 15,
) -> dict:
    """把一次 Long Vegas 触碰映射到它所「终结」的大浪。

    大浪结构中，每个浪的 end_pivot 就是一次 LV 回踩（终结本浪、启动下一浪）。
    ``wave_number`` 是该浪在连续升浪链中的位置，因此「本次回踩是第几次」=
    被终结浪的 wave_number。

    判定顺序（零前瞻，只用 touch_bar 及之前的浪信息）:
      1. 触碰接近某已完成浪的 end_pivot（|Δiloc| ≤ tol）→ 该浪终结于此。
         注意不能用 find_wave_context：共享边界处它会返回「下一浪」，
         导致序次 +1 偏差（第 1 次回踩被误报成第 2 次）。
      2. 否则，若最后一个浪仍在进行（end_pivot=None）且触碰在其峰之后
         → 实时回踩尚未被 EMA8 zigzag 确认，正在终结这个进行中的浪。
      3. 否则 → 非浪终结型触碰（浪内小回踩 / 建底期触碰）→ 不是目标形态。

    Returns dict with keys:
        seq          : 被终结浪的 wave_number；非浪终结型 = 0
        is_wave_end  : 是否对应真实的大浪回踩（浪终点）
        wave         : 被终结的浪 dict（或 None）
        wave_rise_pct: 被终结浪的 start→peak 涨幅（非浪终结型 = 0.0）
    """
    none_result = {"seq": 0, "is_wave_end": False, "wave": None, "wave_rise_pct": 0.0}
    if not waves:
        return none_result

    # 1) 就近匹配已完成浪的 end_pivot
    best = None
    best_dist = tol + 1
    for w in waves:
        ep = w.get("end_pivot")
        if ep is None:
            continue
        d = abs(ep["iloc"] - touch_bar)
        if d <= tol and d < best_dist:
            best, best_dist = w, d
    if best is not None:
        return {
            "seq": best["wave_number"],
            "is_wave_end": True,
            "wave": best,
            "wave_rise_pct": float(best.get("rise_pct", 0.0)),
        }

    # 2) 进行中的末浪：峰已过、尚未确认回踩低点
    last = waves[-1]
    if last.get("end_pivot") is None and touch_bar >= last["peak_pivot"]["iloc"]:
        return {
            "seq": last["wave_number"],
            "is_wave_end": True,
            "wave": last,
            "wave_rise_pct": float(last.get("rise_pct", 0.0)),
        }

    return none_result


# ─────────────────────────────────────────────────────────────
# Wave-Pullback Strategy: Structure Gate
# ─────────────────────────────────────────────────────────────

def check_long_wave_structure(
    entry_bar: int,
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    above_window: int = 60,
    above_min_ratio: float = 0.6,
    rise_min_pct: float = 30.0,
    peak_gap_min_pct: float = 10.0,
) -> dict:
    """上涨周期门控：在 entry_bar 处评估，零前瞻。

    Hard gate 条件（全部通过才算 passed）:
      1. long_rising      : Long Vegas 上沿最近 LONG_SLOPE_WINDOW 天斜率 > 0
      2. long_order       : EMA144 > EMA169（长期通道多头排列）
      3. above_ratio_ok   : 过去 above_window 天中 ≥ above_min_ratio 比例
                            收盘在 Long Vegas 上沿之上（处于上涨周期而非震荡）
      4. rise_1y_ok       : 现价较过去 252 日最低价涨幅 ≥ rise_min_pct
      5. peak_gap_ok      : 过去 above_window 天收盘最高点高于当前 LV 上沿
                            ≥ peak_gap_min_pct（确认是"从大涨幅高点回踩"）

    辅助参考（不参与过滤）:
      long_slope_strong: LV 斜率 ≥ 2%

    Returns dict with keys:
        passed, long_rising, long_order, above_ratio_ok, rise_1y_ok,
        peak_gap_ok, long_slope_strong, long_upper, long_slope_pct,
        above_ratio, rise_from_1y_low_pct, peak_gap_pct
    """
    lv_arr = np.maximum(
        np.maximum(emas[144], emas[169]), emas[200]
    )
    long_upper = float(lv_arr[entry_bar])

    # 1) LV 斜率
    if entry_bar >= LONG_SLOPE_WINDOW:
        long_prev = float(lv_arr[entry_bar - LONG_SLOPE_WINDOW])
        long_slope_pct = (long_upper / long_prev - 1) * 100 if long_prev > 0 else 0.0
    else:
        long_slope_pct = 0.0
    long_rising = long_slope_pct > 0
    long_slope_strong = long_slope_pct >= 2.0

    # 2) 长期通道多头排列
    long_order = bool(emas[144][entry_bar] > emas[169][entry_bar])

    # 3) 上方运行占比
    win_start = max(0, entry_bar - above_window + 1)
    win_close = close[win_start : entry_bar + 1]
    win_lv = lv_arr[win_start : entry_bar + 1]
    above_ratio = float(np.mean(win_close >= win_lv)) if len(win_close) else 0.0
    above_ratio_ok = above_ratio >= above_min_ratio

    # 4) 较年低点涨幅
    low_start = max(0, entry_bar - 252 + 1)
    low_1y = float(np.min(low[low_start : entry_bar + 1]))
    rise_from_1y_low_pct = (
        (float(close[entry_bar]) / low_1y - 1) * 100 if low_1y > 0 else 0.0
    )
    rise_1y_ok = rise_from_1y_low_pct >= rise_min_pct

    # 5) 浪顶落差：回踩前的高点须显著高于通道
    peak_close = float(np.max(win_close)) if len(win_close) else 0.0
    peak_gap_pct = (peak_close / long_upper - 1) * 100 if long_upper > 0 else 0.0
    peak_gap_ok = peak_gap_pct >= peak_gap_min_pct

    passed = bool(
        long_rising and long_order and above_ratio_ok and rise_1y_ok and peak_gap_ok
    )

    return {
        "passed": passed,
        "long_rising": long_rising,
        "long_order": long_order,
        "above_ratio_ok": above_ratio_ok,
        "rise_1y_ok": rise_1y_ok,
        "peak_gap_ok": peak_gap_ok,
        "long_slope_strong": long_slope_strong,
        "long_upper": round(long_upper, 3),
        "long_slope_pct": round(long_slope_pct, 2),
        "above_ratio": round(above_ratio, 3),
        "rise_from_1y_low_pct": round(rise_from_1y_low_pct, 2),
        "peak_gap_pct": round(peak_gap_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
# Wave-Pullback Strategy: Historical LV-Respect Statistics
# ─────────────────────────────────────────────────────────────

def compute_lv_respect_stats(
    waves: list[dict],
    close: np.ndarray,
    emas: dict[int, np.ndarray],
    pending_margin: int = 20,
    min_events: int = 2,
    min_rate: float = 0.6,
    hold_window: int = 40,
    breach_days: int = 3,
    breach_margin: float = 0.97,
) -> dict:
    """统计该标的历史大浪回踩对 Long Vegas 通道的"尊重率"。

    事件定义：每个已完成大浪（end_pivot 非 None）的终点即一次 LV 回踩。
      - held（尊重）: 回踩点之后 hold_window 根 K 线内，收盘从未连续
        breach_days 日低于 LV × breach_margin —— 即回踩止于通道，未更深
      - breach（跌破）: 上述深破发生（synthetic 截断浪天然属于此类，
        因为 synthetic end 本身就是深破起点）

    注意"尊重"只要求止跌于通道，不要求下一浪立刻展开——真实数据中
    浪与浪之间常有 1-3 个月横盘衔接，横盘不等于跌破。

    最近 pending_margin 根 K 线内结束的最后一浪视为"进行中回踩"（可能
    正是当前信号本身），不计入统计，避免用当下事件污染历史口径。

    Returns dict with keys:
        n_events, n_held, n_breach, respect_rate, qualified
        qualified = n_events >= min_events and respect_rate >= min_rate
    """
    n_bars = len(close)
    lv_arr = np.maximum(np.maximum(emas[144], emas[169]), emas[200])

    def _breached_after(start_iloc: int) -> bool:
        end = min(n_bars, start_iloc + hold_window + 1)
        consec = 0
        for bi in range(start_iloc, end):
            lv = float(lv_arr[bi])
            if lv > 0 and float(close[bi]) < lv * breach_margin:
                consec += 1
                if consec >= breach_days:
                    return True
            else:
                consec = 0
        return False

    n_events = 0
    n_held = 0
    n_breach = 0

    for i, w in enumerate(waves):
        ep = w.get("end_pivot")
        if ep is None:
            continue
        is_last = i == len(waves) - 1
        if is_last and ep["iloc"] >= n_bars - pending_margin:
            continue  # 进行中回踩，不计入

        n_events += 1
        if ep.get("synthetic") or _breached_after(ep["iloc"]):
            n_breach += 1
        else:
            n_held += 1

    respect_rate = n_held / n_events if n_events > 0 else 0.0
    qualified = n_events >= min_events and respect_rate >= min_rate

    return {
        "n_events": n_events,
        "n_held": n_held,
        "n_breach": n_breach,
        "respect_rate": round(respect_rate, 3),
        "qualified": qualified,
    }


# ─────────────────────────────────────────────────────────────
# Wave-Pullback Strategy: Scoring & Classification
# ─────────────────────────────────────────────────────────────

def score_long_pullback(
    pullback_seq: int,
    respect_rate: float,
    respect_n: int,
    long_slope_strong: bool,
    wave_rise_pct: float,
) -> tuple[int, dict[str, int]]:
    """对一次 Long Vegas 大浪回踩打分（前提：结构与统计门槛已在调用前确认）。

    Factors — v1 (structural prior, 待回测校准)
    -------
    seq        : 第 1 次 LV 回踩 = +2，第 2 次 = +1，第 3 次 = -1，
                 第 4 次及以后 = -2，未知(0) = 0
                 （越靠后的回踩，下跌概率越高——用户经验先验）
    history    : respect_rate >= 0.75 且事件 >= 3 → +2；
                 respect_rate >= 0.6  且事件 >= 2 → +1；其余 0
    slope      : LV 斜率 >= 2% → +1
    wave_rise  : 本浪涨幅 30-150% = +1（健康主升）；> 250% = -2（透支）；
                 其余 0

    Returns:
        (total_score, detail_dict)
    """
    details: dict[str, int] = {}

    if pullback_seq == 1:
        details["seq"] = 2
    elif pullback_seq == 2:
        details["seq"] = 1
    elif pullback_seq == 3:
        details["seq"] = -1
    elif pullback_seq >= 4:
        details["seq"] = -2
    else:
        details["seq"] = 0

    if respect_rate >= 0.75 and respect_n >= 3:
        details["history"] = 2
    elif respect_rate >= 0.6 and respect_n >= 2:
        details["history"] = 1
    else:
        details["history"] = 0

    details["slope"] = 1 if long_slope_strong else 0

    if wave_rise_pct > 250:
        details["wave_rise"] = -2
    elif 30 <= wave_rise_pct <= 150:
        details["wave_rise"] = 1
    else:
        details["wave_rise"] = 0

    total = sum(details.values())
    return total, details


def classify_long_signal(score: int) -> str:
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
