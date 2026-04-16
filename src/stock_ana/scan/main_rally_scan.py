"""主升浪回调（Vegas风格）单股历史扫描模块。

此文件为回测层提供单股轮诂扫描：
给定一只股票全量历史数据，逐步步进，在每个时间点识别主升浪回调信号。
同时调用 analyze_wave_structure 识别浪结构，记录当前信号属于第几浪、共几浪。

Exports:
    scan_one_symbol(market, symbol, name, df, analysis_start, step, min_gap_days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger  # noqa: F401

from stock_ana.strategies.impl.main_rally_pullback import screen_main_rally_pullback
from stock_ana.strategies.primitives.wave import analyze_wave_structure

FORWARD_DAYS = [5, 10, 21, 63]  # 1周 / 2周 / 1月 / 3月


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize one raw OHLCV frame to lower-case columns and a sorted datetime index."""
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    x.index = pd.to_datetime(x.index)
    x.index.name = "date"
    return x.sort_index()


def _compute_forward(full_df: pd.DataFrame, entry_idx: int) -> dict[str, float]:
    """从 entry_idx 开始计算各持有期收益。"""
    out: dict[str, float] = {}
    closes = full_df["close"].astype(float).values
    if entry_idx >= len(closes):
        return out

    entry = closes[entry_idx]
    if entry <= 0:
        return out

    for fwd in FORWARD_DAYS:
        key = f"ret_{fwd}d"
        if entry_idx + fwd < len(closes):
            out[key] = round((closes[entry_idx + fwd] / entry - 1.0) * 100, 2)
        else:
            out[key] = np.nan

    out["ret_to_end"] = round((closes[-1] / entry - 1.0) * 100, 2)
    return out


def scan_one_symbol(
    market: str,
    symbol: str,
    name: str,
    df: pd.DataFrame,
    analysis_start: pd.Timestamp,
    step: int = 3,
    min_gap_days: int = 12,
) -> list[dict]:
    """对单只股票做历史滚动回测。"""
    x = _prepare_price_frame(df)

    if len(x) < 80:   # EMA 递推从第1天开始，实际最低约需 78 天数据
        return []

    hits: list[dict] = []
    last_signal_idx = -999

    scan_start = min(300, max(80, len(x) // 3))
    for cutoff in range(scan_start, len(x) - 2, step):
        if x.index[cutoff] < analysis_start:
            continue
        if cutoff - last_signal_idx < min_gap_days:
            continue

        sub = x.iloc[:cutoff + 1]
        sig = screen_main_rally_pullback(sub)
        if sig is None:
            continue

        entry_idx = cutoff + 1
        entry_price = float(x.iloc[entry_idx]["close"])
        row = {
            "market": market,
            "symbol": symbol,
            "name": name,
            "signal_date": str(x.index[cutoff].date()),
            "signal_iloc": cutoff,
            "entry_date": str(x.index[entry_idx].date()),
            "entry_iloc": entry_idx,
            "entry_price": round(entry_price, 4),
            **sig,
        }

        # 波浪结构：记录信号发生时属于第几浪（0 = 无法识别浪结构）
        wave_result = analyze_wave_structure(sub)
        row["wave_number"] = wave_result.get("current_wave_number", 0)

        row.update(_compute_forward(x, entry_idx))

        hits.append(row)
        last_signal_idx = cutoff

    return hits
