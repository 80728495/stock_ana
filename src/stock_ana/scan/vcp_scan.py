"""
VCP（Volatility Contraction Pattern 波动收缩形态）单股历史扫描模块。

为回测层提供单股轮诂扫描：结合显式杯身结构识别与 ZigZag 多波收缩验证，
刡除不符巴星主升走势的标的，提取每次有效 VCP 形成事件。

Exports:
    scan_historical_vcps(ticker, df, step, min_data, min_gap_days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger  # noqa: F401

from stock_ana.strategies.api import screen_vcp_setup

FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月


def scan_historical_vcps(
    ticker: str,
    df: pd.DataFrame,
    step: int = 5,
    min_data: int = 260,
    min_gap_days: int = 20,
) -> list[dict]:
    """
    在单只股票的完整历史中，滑动窗口检测所有 VCP 出现的位置。

    Args:
        ticker: 股票代码
        df: 完整历史数据 DataFrame
        step: 扫描步长（每隔多少交易日检测一次）
        min_data: 最少需要的历史数据天数（check_trend_template 需要 260）
        min_gap_days: 同一只股票两次 VCP 信号之间的最小间隔天数

    Returns:
        VCP 信号列表，每项包含形态信息和前瞻收益
    """
    signals = []
    total = len(df)

    if total < min_data + 5:
        return signals

    last_signal_idx = -999  # 上次信号的 iloc 位置

    for cutoff in range(min_data, total - 1, step):
        # 去重：距上次信号太近则跳过
        if cutoff - last_signal_idx < min_gap_days:
            continue

        df_slice = df.iloc[:cutoff + 1].copy()

        try:
            decision = screen_vcp_setup(df_slice)
        except Exception:
            continue

        if not decision.passed:
            continue
        result = decision.features

        # ── 提取 VCP 起止信息 ──
        base_high_iloc = cutoff - result["base_days"]
        base_high_date = df.index[base_high_iloc]
        signal_date = df.index[cutoff]

        # ── 计算前瞻收益 ──
        entry_idx = cutoff + 1  # 信号日的下一个交易日入场
        if entry_idx >= total:
            continue

        entry_price = df.iloc[entry_idx]["close"]
        remaining = df.iloc[entry_idx:]

        fwd_returns = {}
        for fwd in FORWARD_DAYS:
            key = f"{fwd}d"
            if len(remaining) > fwd:
                exit_price = remaining.iloc[fwd]["close"]
                ret = (exit_price - entry_price) / entry_price * 100
                period_closes = remaining.iloc[:fwd + 1]["close"].values
                peak = np.maximum.accumulate(period_closes)
                drawdowns = (period_closes - peak) / peak * 100
                max_dd = float(np.min(drawdowns))
                max_gain = float((np.max(period_closes) - entry_price) / entry_price * 100)
                fwd_returns[key] = {
                    "return_pct": round(ret, 2),
                    "max_drawdown_pct": round(max_dd, 2),
                    "max_gain_pct": round(max_gain, 2),
                }
            else:
                actual_days = len(remaining) - 1
                if actual_days > 0:
                    exit_price = remaining.iloc[-1]["close"]
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
            "entry_price": round(float(entry_price), 2),
            # VCP 形态信息
            "base_high_date": str(base_high_date.date()),
            "base_high": result["base_high"],
            "base_days": result["base_days"],
            "wave_count": result["wave_count"],
            "depths": result["depths"],
            "tightness": result["tightness"],
            "vol_ratio": result["vol_ratio"],
            "distance_to_pivot_pct": result["distance_to_pivot_pct"],
            # 前瞻收益
            **fwd_returns,
        })

        last_signal_idx = cutoff

    return signals
