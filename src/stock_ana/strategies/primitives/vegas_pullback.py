"""Vegas pullback strategy primitives: unified gate and state-machine detection.

Shared by all Vegas pullback variants (Mid-only, Mid+Long, etc.) so that
detection logic and gate conditions are defined in exactly one place.

Public API
----------
check_vegas_gate(bar, close, high, low, emas, trend_days, rise_min_pct) -> dict
    Evaluate the unified entry-gate conditions at a specific bar index.

detect_vegas_pullback(close, low, emas, spans, touch_margin, cooldown) -> list[dict]
    State-machine scanner: touch → two-bar 站稳 confirmation (zero look-ahead).
"""

from __future__ import annotations

import numpy as np

from stock_ana.strategies.primitives.vegas_zones import MID_EMAS, ALL_VEGAS_EMAS


# ─────────────────────────────────────────────────────────────
# Unified Gate Conditions
# ─────────────────────────────────────────────────────────────

def check_vegas_gate(
    bar: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    trend_days: int = 63,
    rise_min_pct: float = 30.0,
) -> dict:
    """统一门控：在 bar 处评估，零前瞻。

    Hard gate（全部通过才算 passed）:
      1. EMA 多头排列：EMA34 > EMA55 > EMA144 > EMA169
      2. EMA 持续上升：EMA55 上涨天数占比 ≥ 80%，EMA34 ≥ 72%，净值正增
                       （均在过去 trend_days 窗口内计算）
      3. 强势筛选：当前价较过去 252 交易日最低价涨幅 ≥ rise_min_pct

    Returns dict with keys:
        passed, cond_ema_order, cond_ema_up, cond_1y_rise,
        ema34_now, ema55_now, ema144_now, ema169_now,
        ema55_up_ratio, ema34_up_ratio, ema55_net_up_pct,
        rise_from_1y_low_pct
    """
    ema34_v = float(emas[34][bar])
    ema55_v = float(emas[55][bar])
    ema144_v = float(emas[144][bar])
    ema169_v = float(emas[169][bar])
    curr_close = float(close[bar])

    # 1) EMA 多头排列
    cond_ema_order = bool(ema34_v > ema55_v > ema144_v > ema169_v)

    # 2) EMA 持续上升
    start = max(0, bar - trend_days + 1)
    ema55_win = emas[55][start : bar + 1]
    ema34_win = emas[34][start : bar + 1]
    if len(ema55_win) > 1:
        up55 = float((np.diff(ema55_win) > 0).mean())
        up34 = float((np.diff(ema34_win) > 0).mean())
        net_up = float(ema55_win[-1] / ema55_win[0] - 1.0) if ema55_win[0] > 0 else -1.0
    else:
        up55 = up34 = 0.0
        net_up = -1.0
    cond_ema_up = bool((up55 >= 0.80) and (up34 >= 0.72) and (net_up > 0))

    # 3) 强势筛选（较年低点涨幅）
    low_start = max(0, bar - 252 + 1)
    low_1y = float(np.min(low[low_start : bar + 1]))
    rise_pct = (curr_close / low_1y - 1.0) * 100 if low_1y > 0 else 0.0
    cond_1y_rise = bool(rise_pct >= rise_min_pct)

    passed = bool(cond_ema_order and cond_ema_up and cond_1y_rise)
    return {
        "passed": passed,
        "cond_ema_order": cond_ema_order,
        "cond_ema_up": cond_ema_up,
        "cond_1y_rise": cond_1y_rise,
        "ema34_now": round(ema34_v, 3),
        "ema55_now": round(ema55_v, 3),
        "ema144_now": round(ema144_v, 3),
        "ema169_now": round(ema169_v, 3),
        "ema55_up_ratio": round(up55, 3),
        "ema34_up_ratio": round(up34, 3),
        "ema55_net_up_pct": round(net_up * 100, 2),
        "rise_from_1y_low_pct": round(rise_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
# Unified State-Machine Detection  (zero look-ahead)
# ─────────────────────────────────────────────────────────────

def detect_vegas_pullback(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    spans: list[int] | None = None,
    touch_margin: float = 0.01,
    cooldown: int = 10,
    above_lookback: int = 0,
) -> list[dict]:
    """逐日扫描 Vegas 均线触碰 + 站稳确认信号（零前瞻）。

    确认机制（"站稳"）：
      - 单日触碰收回（low 靠近线，close ≥ EMA）→ 当日即确认
      - 短暂刺破（≤ 2 交易日 close < EMA）→ 两日站稳确认

    above_lookback > 0 时启用"上方运行"前置检查：
      触发前的 above_lookback 个交易日中，至少 60% 的日子 close ≥ EMA，
      确保信号来自"上方回踩"而非"下方修复"。
      典型用于 Mid Vegas 中继（above_lookback=10），排除从下方长期恢复
      后才碰到 mid 的假信号。

    同一次回调多条 EMA 同时触发 → 冷却去重，只保留最新一条。

    Parameters
    ----------
    spans : 待检测的 EMA 周期列表，默认为 MID_EMAS (34/55/60)。
            传入 ALL_VEGAS_EMAS 则同时检测长期 Vegas (144/169/200)。
    touch_margin : low ≤ EMA × (1 + touch_margin) 即视为"靠近"。
                   0.01 → 1% 容差；0.02 → 2% 容差（捕获近距未触）。
    above_lookback : 前置检查回看天数，0 = 不检查（兼容旧行为）。

    Returns
    -------
    list of signal dicts with keys:
        touch_bar, confirm_bar, entry_bar,
        support_type ("mid_vegas" | "long_vegas"),
        support_band (e.g. "ema55"),
        ema_span, touch_low, ema_at_touch
    """
    if spans is None:
        spans = MID_EMAS
    n = len(close)
    raw_signals: list[dict] = []

    for span in spans:
        ema = emas[span]
        band = f"ema{span}"
        support_type = "mid_vegas" if span in MID_EMAS else "long_vegas"

        state = None
        touch_bar = -1
        broke_below = 0
        last_confirmed = -999

        for i in range(200, n):
            if i - last_confirmed < cooldown:
                state = None
                continue

            ev = ema[i]
            c = close[i]
            lo = low[i]

            if state is None:
                if lo <= ev * (1 + touch_margin):
                    # ── 前置检查：最近是否在 EMA 上方运行 ──
                    if above_lookback > 0:
                        lb_start = max(0, i - above_lookback)
                        above_cnt = int(np.sum(close[lb_start:i] >= ema[lb_start:i]))
                        if above_cnt < (i - lb_start) * 0.6:
                            # 大部分时间在下方 → 不是上方回踩，跳过
                            continue

                    if c >= ev:
                        # 单日触碰收回 — 强势止跌，直接确认
                        entry_bar = i + 1
                        if entry_bar <= n:
                            raw_signals.append(dict(
                                touch_bar=i, confirm_bar=i, entry_bar=entry_bar,
                                support_type=support_type, support_band=band,
                                ema_span=span,
                                touch_low=float(lo), ema_at_touch=float(ev),
                            ))
                        last_confirmed = i
                    else:
                        state = "touched"
                        touch_bar = i
                        broke_below = 1

            elif state == "touched":
                if c >= ev:
                    state = "confirm_1"
                else:
                    broke_below += 1
                    if broke_below > 2:
                        state = None

            elif state == "confirm_1":
                if c >= ev:
                    # 两日站稳确认
                    entry_bar = i + 1
                    if entry_bar <= n:
                        raw_signals.append(dict(
                            touch_bar=touch_bar, confirm_bar=i, entry_bar=entry_bar,
                            support_type=support_type, support_band=band,
                            ema_span=span,
                            touch_low=float(low[touch_bar]),
                            ema_at_touch=float(ema[touch_bar]),
                        ))
                    last_confirmed = i
                    state = None
                else:
                    state = None

    # 全跨度去重：同一次回调只保留最新一条
    raw_signals.sort(key=lambda s: s["entry_bar"])
    signals: list[dict] = []
    last_entry = -999
    for sig in raw_signals:
        if sig["entry_bar"] - last_entry < cooldown:
            continue
        last_entry = sig["entry_bar"]
        signals.append(sig)

    return signals
