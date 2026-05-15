"""Long Vegas (EMA144/169/200) pullback touch detection.

Detects bars where price pulls back to the Long Vegas zone during an
established uptrend and holds above it on a closing basis.

Signal condition (single-bar, no look-ahead):
  - low ≤ Long EMA × (1 + touch_margin)   — price reached the zone
  - close ≥ Long EMA                       — closed above (held support)
  - above_lookback check                   — came from above, not repairing from below

This module only provides detection.  Scoring is intentionally omitted;
signals are labelled "OBSERVE" and raw contextual features are emitted
for future statistical calibration.

Public API
----------
detect_long_touch_immediate(close, low, emas, ...) -> list[dict]
"""

from __future__ import annotations

import numpy as np

from stock_ana.strategies.primitives.vegas_zones import LONG_EMAS


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
    above_lookback : 触碰前 N 根 K 线中，close >= EMA 的比例须 >= 60%，
                     确保价格从上方回落而非从下方修复。

    Returns
    -------
    list of signal dicts, each containing:
        touch_bar, confirm_bar, entry_bar,
        support_type ("long_vegas"), support_band (e.g. "ema144"),
        ema_span, touch_low, ema_at_touch
    """
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
                # 前置检查：最近是否在 EMA 上方运行
                if above_lookback > 0:
                    lb_start = max(0, i - above_lookback)
                    above_cnt = int(np.sum(close[lb_start:i] >= ema[lb_start:i]))
                    if above_cnt < (i - lb_start) * 0.6:
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
