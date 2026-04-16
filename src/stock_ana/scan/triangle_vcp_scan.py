"""宏观峰值起点的 VCP 三角形单股历史扫描模块。

为回测层提供单股轮诂扫描：结合宏观峰值与 ZigZag 过滤，
识别具有波幅递减特征的 VCP 式三角形整理形态。

Exports:
    scan_historical_triangle_vcp(ticker, df, step, min_data, min_gap_days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger  # noqa: F401

from stock_ana.strategies.primitives.peaks import find_macro_peaks
from stock_ana.strategies.impl.triangle_vcp import screen_triangle_vcp

FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月


def scan_historical_triangle_vcp(
    ticker: str,
    df: pd.DataFrame,
    step: int = 5,
    min_data: int = 260,
    min_gap_days: int = 20,
) -> list[dict]:
    """
    在单只股票历史中，从每个前高出发检测 VCP 三角形。

    算法：
      1. 调用 find_macro_peaks 获取全部宏观前高
      2. 对每个时间点 t (从 min_data 开始，步长 step)：
         a. 找 t 之前最近的前高 peak
         b. 取 [peak, t] 区间，调用 screen_triangle_vcp
         c. 如果检出 → 记录信号 + 计算前瞻收益
    """
    signals: list[dict] = []
    total = len(df)

    if total < min_data + 5:
        return signals

    # 获取前高
    peaks_df = find_macro_peaks(df, min_gap_days=65, min_drawdown_pct=10.0)
    if peaks_df.empty:
        return signals

    peak_ilocs = [df.index.get_loc(d) for d in peaks_df.index]
    peak_prices = peaks_df["high"].values.astype(float)

    last_signal_iloc = -999

    for cutoff in range(min_data, total - 1, step):
        if cutoff - last_signal_iloc < min_gap_days:
            continue

        # 找 cutoff 之前最近的前高
        latest_peak_pos = -1
        for j in range(len(peak_ilocs) - 1, -1, -1):
            if peak_ilocs[j] <= cutoff:
                latest_peak_pos = j
                break
        if latest_peak_pos < 0:
            continue

        peak_iloc = peak_ilocs[latest_peak_pos]
        peak_price = float(peak_prices[latest_peak_pos])

        # 前高到当前点至少 25 天
        gap = cutoff - peak_iloc
        if gap < 25:
            continue

        # 截取到 cutoff 的数据
        df_slice = df.iloc[:cutoff + 1].copy()

        try:
            result = screen_triangle_vcp(
                df_slice,
                peak_iloc=peak_iloc,
                require_trend=True,
            )
        except Exception:
            continue

        if result is None:
            continue

        signal_date = df.index[cutoff]
        entry_idx = cutoff + 1
        if entry_idx >= total:
            continue

        entry_price = float(df.iloc[entry_idx]["close"])
        remaining = df.iloc[entry_idx:]

        # ── 前瞻收益 ──
        fwd_returns: dict = {}
        for fwd in FORWARD_DAYS:
            key = f"{fwd}d"
            if len(remaining) > fwd:
                exit_price = float(remaining.iloc[fwd]["close"])
                ret = (exit_price - entry_price) / entry_price * 100
                period_closes = remaining.iloc[:fwd + 1]["close"].values.astype(float)
                peak_acc = np.maximum.accumulate(period_closes)
                drawdowns = (period_closes - peak_acc) / peak_acc * 100
                max_dd = float(np.min(drawdowns))
                max_gain = float(
                    (np.max(period_closes) - entry_price) / entry_price * 100
                )
                fwd_returns[key] = {
                    "return_pct": round(ret, 2),
                    "max_drawdown_pct": round(max_dd, 2),
                    "max_gain_pct": round(max_gain, 2),
                }
            else:
                actual_days = len(remaining) - 1
                if actual_days > 0:
                    exit_price = float(remaining.iloc[-1]["close"])
                    ret = (exit_price - entry_price) / entry_price * 100
                    fwd_returns[key] = {
                        "return_pct": round(ret, 2),
                        "actual_days": actual_days,
                        "note": "数据不足",
                    }

        signals.append({
            "ticker": ticker,
            "signal_date": str(signal_date.date()),
            "signal_iloc": cutoff,
            "entry_date": str(df.index[entry_idx].date()),
            "entry_price": round(entry_price, 2),
            "peak_date": str(df.index[peak_iloc].date()),
            "peak_iloc": peak_iloc,
            "peak_price": round(peak_price, 2),
            "gap_days": gap,
            # 形态信息
            "pattern": result["pattern"],
            "period": result["period"],
            "window_start_iloc": result["window_start"],
            "convergence_ratio": result["convergence_ratio"],
            "spread_contraction": result["spread_contraction"],
            "vol_contraction": result["vol_contraction"],
            "position_in_channel": result["position_in_channel"],
            "score": result["score"],
            "upper_slope_pct": result["upper_slope_pct"],
            "lower_slope_pct": result["lower_slope_pct"],
            # 绘图用
            "resistance_slope": result["resistance"]["slope"],
            "resistance_intercept": result["resistance"]["intercept"],
            "support_slope": result["support"]["slope"],
            "support_intercept": result["support"]["intercept"],
            # 前瞻收益
            **fwd_returns,
        })

        last_signal_iloc = cutoff

    return signals
