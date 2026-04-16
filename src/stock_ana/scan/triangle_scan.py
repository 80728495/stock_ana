"""上升收敛三角形 / 橔形单股历史扫描模块。

为回测层提供单股轮诂扫描：给定一只股票全量历史数据，逐步踪对 OLS 回归
拟合阻力线与支撇线，识别三种修正形态：上升三角形、平行通道、上升橔形。

Exports:
    scan_historical_triangles(ticker, df, step, min_data, min_gap_days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger  # noqa: F401

from stock_ana.strategies.api import screen_triangle_ascending

FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月


def scan_historical_triangles(
    ticker: str,
    df: pd.DataFrame,
    step: int = 5,
    min_data: int = 260,
    min_gap_days: int = 20,
) -> list[dict]:
    """
    在单只股票历史中滑动窗口检测所有上升收敛形态。

    Args:
        ticker: 股票代码
        df: 完整历史 DataFrame
        step: 扫描步长
        min_data: 最少历史数据天数（check_trend_template 需要 260）
        min_gap_days: 同一股票两次信号最小间隔天数

    Returns:
        信号列表
    """
    signals: list[dict] = []
    total = len(df)

    if total < min_data + 5:
        return signals

    last_signal_idx = -999

    for cutoff in range(min_data, total - 1, step):
        if cutoff - last_signal_idx < min_gap_days:
            continue

        df_slice = df.iloc[: cutoff + 1].copy()

        try:
            decision = screen_triangle_ascending(df_slice)
        except Exception:
            continue

        if not decision.passed:
            continue
        result = decision.features

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
                period_closes = remaining.iloc[: fwd + 1]["close"].values.astype(float)
                peak = np.maximum.accumulate(period_closes)
                drawdowns = (period_closes - peak) / peak * 100
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

        signals.append(
            {
                "ticker": ticker,
                "signal_date": str(signal_date.date()),
                "signal_iloc": cutoff,
                "entry_date": str(df.index[entry_idx].date()),
                "entry_price": round(entry_price, 2),
                # ── 形态信息 ──
                "pattern": result["pattern"],
                "period": result["period"],
                "window_start_iloc": cutoff - result["period"],
                "convergence_ratio": result["convergence_ratio"],
                "convergence_angle_deg": result["convergence_angle_deg"],
                "spread_contraction": result["spread_contraction"],
                "vol_contraction": result["vol_contraction"],
                "position_in_channel": result["position_in_channel"],
                "score": result["score"],
                # 绘图用
                "resistance_slope": result["resistance"]["slope"],
                "resistance_intercept": result["resistance"]["intercept"],
                "support_slope": result["support"]["slope"],
                "support_intercept": result["support"]["intercept"],
                # ── 前瞻收益 ──
                **fwd_returns,
            }
        )

        last_signal_idx = cutoff

    return signals
