"""Reusable relative-strength calculations shared across RS-based strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rs_line(
    df_stock: pd.DataFrame,
    df_market: pd.DataFrame,
) -> pd.Series | None:
    """Compute relative-strength line by aligning stock and benchmark closes."""
    idx_stock = df_stock.index
    idx_market = df_market.index
    if hasattr(idx_stock, "freq"):
        idx_stock = idx_stock._with_freq(None)
    if hasattr(idx_market, "freq"):
        idx_market = idx_market._with_freq(None)

    common_idx = idx_stock.intersection(idx_market)
    if len(common_idx) < 200:
        return None

    stock_close = df_stock.loc[common_idx, "close"]
    market_close = df_market.loc[common_idx, "close"]
    return stock_close / market_close


def compute_rs_rank_63d(
    stock_data: dict[str, pd.DataFrame],
    df_market: pd.DataFrame,
) -> dict[str, float]:
    """Rank stocks by aligned 63-day return percentile versus a benchmark set."""
    returns_63d: dict[str, float] = {}

    for ticker, df in stock_data.items():
        common_idx = df.index.intersection(df_market.index)
        if len(common_idx) < 63:
            continue
        aligned_close = df.loc[common_idx, "close"]
        if len(aligned_close) < 63:
            continue
        ret = (aligned_close.iloc[-1] / aligned_close.iloc[-63] - 1) * 100
        returns_63d[ticker] = ret

    if not returns_63d:
        return {}

    all_returns = np.array(list(returns_63d.values()))
    ranks: dict[str, float] = {}
    for ticker, ret in returns_63d.items():
        ranks[ticker] = round(float(np.mean(all_returns <= ret)) * 100, 1)
    return ranks


def compute_rs_rank_at_cutoff(
    stock_data: dict[str, pd.DataFrame],
    df_market: pd.DataFrame,
    cutoff_idx: int,
) -> dict[str, float]:
    """Compute RS percentile ranks using each stock's data truncated at a cutoff."""
    returns_63d: dict[str, float] = {}

    for ticker, df in stock_data.items():
        df_cut = df.iloc[:cutoff_idx]
        if len(df_cut) < 63:
            continue
        common_idx = df_cut.index.intersection(df_market.index)
        if len(common_idx) < 63:
            continue
        aligned_close = df_cut.loc[common_idx, "close"]
        if len(aligned_close) < 63:
            continue
        ret = (aligned_close.iloc[-1] / aligned_close.iloc[-63] - 1) * 100
        returns_63d[ticker] = ret

    if not returns_63d:
        return {}

    all_returns = np.array(list(returns_63d.values()))
    ranks: dict[str, float] = {}
    for ticker, ret in returns_63d.items():
        ranks[ticker] = round(float(np.mean(all_returns <= ret)) * 100, 1)
    return ranks
