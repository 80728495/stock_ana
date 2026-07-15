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
# Pullback Precondition  (「回踩」语义门：从上方跌下来 + regime)
# ─────────────────────────────────────────────────────────────

def pullback_precondition(
    i: int,
    close: np.ndarray,
    ema: np.ndarray,
    above_lookback: int,
    above_min_ratio: float = 0.8,
    from_above_pct: float = 0.02,
    regime_mask: np.ndarray | None = None,
) -> bool:
    """触碰 bar i 是否满足「回踩」语义（零前瞻，供各检测器共用）。

    「回踩」= 价格从高于当前的位置跌下来触碰均线。跌穿后重新涨回的
    上穿触碰不是回踩。四个条件全过才 True：

      1. regime_mask[i] 为 True（None = 不检查）。mid 触碰用它要求
         中期均线运行在长期均线之上（min(EMA34,55) > max(EMA144,169,200)）。
      2. 触碰前一根收盘 ≥ 前一根 EMA —— 价格自上而下接近均线；
         跌穿后涨回的上穿触碰，前一根收盘必在均线之下，被此条拒绝。
      3. 过去 above_lookback 根收盘 ≥ EMA 的占比 ≥ above_min_ratio
         （0.8，旧值 0.6 太松：10 根里 4 根在下方也能过）。
      4. 过去 above_lookback 根内最高收盘 ≥ EMA×(1+from_above_pct)
         —— 确实从更高处跌下来，而非一直贴线横爬。
    """
    if regime_mask is not None and not bool(regime_mask[i]):
        return False
    if i < 1 or close[i - 1] < ema[i - 1]:
        return False
    lb_start = max(0, i - above_lookback)
    if lb_start >= i:
        return False
    win_close = close[lb_start:i]
    win_ema = ema[lb_start:i]
    if float(np.mean(win_close >= win_ema)) < above_min_ratio:
        return False
    if float(np.max(win_close)) < float(ema[i]) * (1 + from_above_pct):
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Unified State-Machine Detection  (zero look-ahead)
# ─────────────────────────────────────────────────────────────

def detect_vegas_pullback(
    close: np.ndarray,
    low: np.ndarray,
    emas: dict[int, np.ndarray],
    spans: list[int] | None = None,
    touch_margin: float = 0.01,
    depart_pct: float = 0.05,
    deepen_pct: float = 0.03,
    min_gap_bars: int = 3,
    above_lookback: int = 0,
    above_min_ratio: float = 0.8,
    from_above_pct: float = 0.02,
    regime_mask: np.ndarray | None = None,
) -> list[dict]:
    """逐日扫描 Vegas 均线触碰 + 站稳确认信号（零前瞻）。

    确认机制（"站稳"）：
      - 单日触碰收回（low 靠近线，close ≥ EMA）→ 当日即确认
      - 短暂刺破（≤ 2 交易日 close < EMA）→ 两日站稳确认

    above_lookback > 0 时启用「回踩」语义门（pullback_precondition）：
      ① regime_mask（如 mid>long 多头排列）② 触碰前一根收盘在 EMA 上方
      （自上而下接近，拒绝跌穿后涨回的上穿触碰）③ 上方占比 ≥
      above_min_ratio（0.8）④ 窗口内最高收盘 ≥ EMA×(1+from_above_pct)
      （确实从更高处跌下来）。regime_mask 单独传入时即使
      above_lookback=0 也生效。

    去重（"离开-再回踩 / 越踩越深"规则，取代固定冷却期）：
      一次回踩常在均线附近磨几天，属同一事件。仅当满足以下之一才认第二次
      为新回踩，否则合并：
        1. 价格在两次触碰之间**明显离开过通道**（某根收盘 ≥ 通道上沿 ×
           (1+depart_pct)）——回踩已结束、价格重新走高又跌回；
        2. 本次触碰**创了明显更低的低点**（touch_low < 上次 ×(1-deepen_pct)）
           ——同一轮下跌里逐步加深的支撑测试（先擦 ema34 再深探 ema55），
           越深的那次才是真正的抄底点，应单列。
      这比"数 N 根 K 线"精确：锚定"回踩是否结束/是否加深"的经济事实，行情
      快速运行时不会把两次不同的回踩误并（旧固定冷却期的缺陷）。
      min_gap_bars 作最小间隔地板，防相邻 1-2 日重复。

    Parameters
    ----------
    spans : 待检测的 EMA 周期列表，默认为 MID_EMAS (34/55)。
    touch_margin : low ≤ EMA × (1 + touch_margin) 即视为"靠近"。
    depart_pct : 判定"离开通道"的阈值（收盘高出通道上沿的比例）。0.05 = 5%。
    min_gap_bars : 两次保留信号的最小 bar 间隔地板。
    above_lookback : 前置检查回看天数，0 = 不检查。

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

        for i in range(1, n):
            ev = ema[i]
            c = close[i]
            lo = low[i]

            if state is None:
                if lo <= ev * (1 + touch_margin):
                    # ── 「回踩」语义门：从上方跌下来 + regime ──
                    if above_lookback > 0:
                        if not pullback_precondition(
                            i, close, ema, above_lookback,
                            above_min_ratio=above_min_ratio,
                            from_above_pct=from_above_pct,
                            regime_mask=regime_mask,
                        ):
                            continue
                    elif regime_mask is not None and not bool(regime_mask[i]):
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
                    state = None
                else:
                    state = None

    # ── 去重：同一次回踩合并，价格离开通道后再回踩才算新事件 ──
    channel_top = emas[spans[0]].astype(float).copy()
    for s in spans[1:]:
        channel_top = np.maximum(channel_top, emas[s].astype(float))

    raw_signals.sort(key=lambda s: s["entry_bar"])
    signals: list[dict] = []
    last_kept: dict | None = None
    for sig in raw_signals:
        if last_kept is None:
            signals.append(sig)
            last_kept = sig
            continue
        if sig["entry_bar"] - last_kept["entry_bar"] < min_gap_bars:
            continue
        seg_lo = last_kept["touch_bar"] + 1
        seg_hi = sig["touch_bar"] + 1
        departed = bool(
            seg_hi > seg_lo
            and np.any(close[seg_lo:seg_hi] >= channel_top[seg_lo:seg_hi] * (1 + depart_pct))
        )
        deeper = sig["touch_low"] < last_kept["touch_low"] * (1 - deepen_pct)
        if departed or deeper:
            signals.append(sig)
            last_kept = sig
        # 否则视为同一次回踩的延续，跳过

    return signals
